# SPDX-License-Identifier: GPL-3.0-or-later
"""Mesh extraction and GPU passes: camera-space reference render and UV-space projection bake."""

import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix


class BridgeError(Exception):
    pass


# ---------------------------------------------------------------- geometry

GEO_KEYS = ("pos", "nor", "uv", "target")


def _loose_parts(nverts, edge_verts):
    """Connected-component label per vertex, compacted to 0..k-1."""
    labels = np.arange(nverts, dtype=np.int64)
    a, b = edge_verts[0::2].astype(np.int64), edge_verts[1::2].astype(np.int64)
    if a.size:
        while True:
            la, lb = labels[a], labels[b]
            if np.array_equal(la, lb):
                break
            low = np.minimum(la, lb)
            np.minimum.at(labels, la, low)
            np.minimum.at(labels, lb, low)
            while True:
                jumped = labels[labels]
                if np.array_equal(jumped, labels):
                    break
                labels = jumped
    _uniq, compact = np.unique(labels, return_inverse=True)
    return compact.astype(np.int32), int(_uniq.size)


def _part_names(mesh, obj, labels, count):
    """Name each part after its heaviest vertex group, sampling at most 256 vertices per part."""
    group_names = [g.name for g in obj.vertex_groups]
    names = []
    order = np.argsort(labels, kind="stable")
    bounds = np.searchsorted(labels[order], np.arange(count + 1))
    for part in range(count):
        members = order[bounds[part]:bounds[part + 1]]
        name = ""
        if group_names and members.size:
            step = max(1, members.size // 256)
            totals = {}
            for vi in members[::step]:
                for g in mesh.vertices[int(vi)].groups:
                    if g.weight > 0.0 and g.group < len(group_names):
                        totals[g.group] = totals.get(g.group, 0.0) + g.weight
            if totals:
                name = group_names[max(totals, key=totals.get)]
        names.append(name or f"Part {part + 1:02d}")
    seen = {}
    for i, name in enumerate(names):
        if name in seen:
            seen[name] += 1
            names[i] = f"{name} {seen[name]}"
        else:
            seen[name] = 1
    return names


def extract_geometry(obj, depsgraph, uv_name, material_indices, with_parts=False, keep_verts=None,
                     keep_faces=None):
    """Evaluated world-space triangle corners: dict of pos, nor, uv, target arrays.

    With `with_parts`, also adds per-corner `part` ids, `part_names` and world-space `edges`.
    `keep_faces` (bool per face) or `keep_verts` (bool per vertex, all of a face's vertices) limit
    the result to those faces.
    """
    eval_obj = obj.evaluated_get(depsgraph)
    matrix = np.array(eval_obj.matrix_world, dtype=np.float64)
    mesh = eval_obj.to_mesh()
    try:
        mesh.calc_loop_triangles()
        ntri = len(mesh.loop_triangles)
        if ntri == 0:
            raise BridgeError("Mesh has no faces")
        uv_layer = mesh.uv_layers.get(uv_name) if uv_name else None
        if uv_layer is None:
            uv_layer = next((l for l in mesh.uv_layers if l.active_render), mesh.uv_layers.active)
        if uv_layer is None:
            raise BridgeError("Mesh has no UV map")

        tri_loops = np.empty(ntri * 3, np.int32)
        mesh.loop_triangles.foreach_get("loops", tri_loops)
        mat_index = np.empty(ntri, np.int32)
        mesh.loop_triangles.foreach_get("material_index", mat_index)
        nloops = len(mesh.loops)
        vert_index = np.empty(nloops, np.int32)
        mesh.loops.foreach_get("vertex_index", vert_index)
        co = np.empty(len(mesh.vertices) * 3, np.float32)
        mesh.vertices.foreach_get("co", co)
        normals = np.empty(nloops * 3, np.float32)
        mesh.corner_normals.foreach_get("vector", normals)
        uv = np.empty(nloops * 2, np.float32)
        uv_layer.uv.foreach_get("vector", uv)
        if keep_faces is not None:
            if len(keep_faces) != len(mesh.polygons):
                raise BridgeError("Isolating faces needs modifiers that keep the face count")
            tri_face = np.empty(ntri, np.int32)
            mesh.loop_triangles.foreach_get("polygon_index", tri_face)
        if with_parts or keep_verts is not None or keep_faces is not None:
            edge_verts = np.empty(len(mesh.edges) * 2, np.int32)
            mesh.edges.foreach_get("vertices", edge_verts)
        if with_parts:
            labels, count = _loose_parts(len(mesh.vertices), edge_verts)
            part_names = _part_names(mesh, eval_obj, labels, count)
    finally:
        eval_obj.to_mesh_clear()

    co = co.reshape(-1, 3).astype(np.float64) @ matrix[:3, :3].T + matrix[:3, 3]
    normal_matrix = np.linalg.inv(matrix[:3, :3]).T
    normals = normals.reshape(-1, 3).astype(np.float64) @ normal_matrix.T
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)

    if keep_verts is not None or keep_faces is not None:
        keep_tri = np.ones(ntri, bool)
        if keep_verts is not None:
            if len(keep_verts) != len(co):
                raise BridgeError("Isolating parts needs modifiers that keep the vertex count")
            keep_tri &= keep_verts[vert_index[tri_loops].reshape(-1, 3)].all(axis=1)
        if keep_faces is not None:
            keep_tri &= keep_faces[tri_face]
        if not keep_tri.any():
            raise BridgeError("The isolated faces have no triangles")
        tri_loops = tri_loops.reshape(-1, 3)[keep_tri].ravel()
        mat_index = mat_index[keep_tri]
        ntri = int(keep_tri.sum())
        used = np.zeros(len(co), bool)
        used[vert_index[tri_loops]] = True
        pairs = edge_verts.reshape(-1, 2)
        edge_verts = pairs[used[pairs].all(axis=1)].ravel()

    if material_indices is None:
        target = np.ones(ntri, bool)
    else:
        target = np.isin(mat_index, np.fromiter(material_indices, np.int32))
    geo = {
        "pos": co[vert_index[tri_loops]].astype(np.float32),
        "nor": normals[tri_loops].astype(np.float32),
        "uv": uv.reshape(-1, 2)[tri_loops].astype(np.float32),
        "target": np.repeat(target, 3).astype(np.float32),
    }
    if with_parts:
        geo["part"] = labels[vert_index[tri_loops]].astype(np.float32)
        geo["part_names"] = part_names
        geo["edges"] = co[edge_verts].astype(np.float32)
    return geo


# ---------------------------------------------------------------- shaders

_shaders = {}


def _view_shader():
    if "view" in _shaders:
        return _shaders["view"]
    iface = gpu.types.GPUStageInterfaceInfo("pbridge_view_iface")
    iface.smooth("VEC2", "v_uv")
    iface.smooth("FLOAT", "v_depth")
    iface.flat("FLOAT", "v_target")
    iface.flat("FLOAT", "v_part")
    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant("MAT4", "viewMatrix")
    info.push_constant("MAT4", "projMatrix")
    info.push_constant("FLOAT", "layerMode")
    info.sampler(0, "FLOAT_2D", "baseTex")
    info.vertex_in(0, "VEC3", "pos")
    info.vertex_in(1, "VEC2", "uv")
    info.vertex_in(2, "FLOAT", "target")
    info.vertex_in(3, "FLOAT", "part")
    info.vertex_out(iface)
    info.fragment_out(0, "VEC4", "outColor")
    info.fragment_out(1, "VEC4", "outDepth")
    info.fragment_out(2, "VEC4", "outPart")
    info.vertex_source(
        "void main() {\n"
        "  vec4 vp = viewMatrix * vec4(pos, 1.0);\n"
        "  v_depth = -vp.z;\n"
        "  v_uv = uv;\n"
        "  v_target = target;\n"
        "  v_part = part;\n"
        "  gl_Position = projMatrix * vp;\n"
        "}\n")
    info.fragment_source(
        "void main() {\n"
        "  vec4 c = texture(baseTex, v_uv);\n"
        "  if (layerMode > 0.5) {\n"
        "    outColor = v_target > 0.5 ? vec4(c.rgb / max(c.a, 1e-6), c.a) : vec4(0.0);\n"
        "  } else {\n"
        "    outColor = vec4(v_target > 0.5 ? c.rgb : vec3(0.5), 1.0);\n"
        "  }\n"
        "  outDepth = vec4(v_depth, 1.0, 0.0, 1.0);\n"
        "  outPart = vec4(v_part + 1.0, 0.0, 0.0, 1.0);\n"
        "}\n")
    _shaders["view"] = gpu.shader.create_from_info(info)
    return _shaders["view"]


def _wire_shader():
    if "wire" in _shaders:
        return _shaders["wire"]
    iface = gpu.types.GPUStageInterfaceInfo("pbridge_wire_iface")
    iface.smooth("FLOAT", "v_depth")
    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant("MAT4", "viewMatrix")
    info.push_constant("MAT4", "projMatrix")
    info.push_constant("FLOAT", "depthEps")
    info.sampler(0, "FLOAT_2D", "depthTex")
    info.vertex_in(0, "VEC3", "pos")
    info.vertex_out(iface)
    info.fragment_out(0, "VEC4", "outColor")
    info.vertex_source(
        "void main() {\n"
        "  vec4 vp = viewMatrix * vec4(pos, 1.0);\n"
        "  v_depth = -vp.z;\n"
        "  gl_Position = projMatrix * vp;\n"
        "}\n")
    info.fragment_source(
        "void main() {\n"
        "  ivec2 size = textureSize(depthTex, 0);\n"
        "  ivec2 px = clamp(ivec2(gl_FragCoord.xy), ivec2(0), size - 1);\n"
        "  float d = -1e30;\n"
        "  for (int y = -1; y <= 1; y++) {\n"
        "    for (int x = -1; x <= 1; x++) {\n"
        "      vec4 t = texelFetch(depthTex, clamp(px + ivec2(x, y), ivec2(0), size - 1), 0);\n"
        "      if (t.g > 0.5) { d = max(d, t.r); }\n"
        "    }\n"
        "  }\n"
        "  if (d < -1e29 || v_depth > d + abs(d) * 0.004 + depthEps) { discard; }\n"
        "  outColor = vec4(1.0);\n"
        "}\n")
    _shaders["wire"] = gpu.shader.create_from_info(info)
    return _shaders["wire"]


def _bake_shader():
    if "bake" in _shaders:
        return _shaders["bake"]
    iface = gpu.types.GPUStageInterfaceInfo("pbridge_bake_iface")
    iface.smooth("VEC3", "v_world")
    iface.smooth("VEC3", "v_nor")
    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant("MAT4", "viewMatrix")
    info.push_constant("MAT4", "projMatrix")
    info.push_constant("VEC4", "params")  # isPersp, cosStart, cosEnd, depth tolerance
    info.sampler(0, "FLOAT_2D", "paintTex")
    info.sampler(1, "FLOAT_2D", "depthTex")
    info.vertex_in(0, "VEC3", "pos")
    info.vertex_in(1, "VEC3", "nor")
    info.vertex_in(2, "VEC2", "uv")
    info.vertex_out(iface)
    info.fragment_out(0, "VEC4", "outColor")
    info.vertex_source(
        "void main() {\n"
        "  v_world = pos;\n"
        "  v_nor = nor;\n"
        "  gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);\n"
        "}\n")
    info.fragment_source(
        "void main() {\n"
        "  outColor = vec4(0.0);\n"
        "  vec4 vp = viewMatrix * vec4(v_world, 1.0);\n"
        "  vec4 clip = projMatrix * vp;\n"
        "  if (clip.w <= 0.0) { return; }\n"
        "  vec2 suv = clip.xy / clip.w * 0.5 + 0.5;\n"
        "  if (any(lessThan(suv, vec2(0.0))) || any(greaterThan(suv, vec2(1.0)))) { return; }\n"
        "  ivec2 size = textureSize(depthTex, 0);\n"
        "  ivec2 px = clamp(ivec2(suv * vec2(size)), ivec2(0), size - 1);\n"
        "  float d = -1e30;\n"
        "  for (int y = -1; y <= 1; y++) {\n"
        "    for (int x = -1; x <= 1; x++) {\n"
        "      vec4 t = texelFetch(depthTex, clamp(px + ivec2(x, y), ivec2(0), size - 1), 0);\n"
        "      if (t.g > 0.5) { d = max(d, t.r); }\n"
        "    }\n"
        "  }\n"
        "  float mine = -vp.z;\n"
        "  if (d < -1e29 || mine > d + abs(d) * 0.01 + params.w) { return; }\n"
        "  vec3 n = normalize(mat3(viewMatrix) * v_nor);\n"
        "  vec3 dir = params.x > 0.5 ? normalize(-vp.xyz) : vec3(0.0, 0.0, 1.0);\n"
        "  float facing = abs(dot(n, dir));\n"
        "  float w = clamp((facing - params.z) / max(params.y - params.z, 1e-4), 0.0, 1.0);\n"
        "  outColor = texture(paintTex, suv) * w;\n"
        "}\n")
    _shaders["bake"] = gpu.shader.create_from_info(info)
    return _shaders["bake"]


# ---------------------------------------------------------------- helpers

def _to_texture(array, fmt="RGBA32F"):
    h, w = array.shape[:2]
    flat = np.ascontiguousarray(array, dtype=np.float32).ravel()
    return gpu.types.GPUTexture((w, h), format=fmt, data=gpu.types.Buffer("FLOAT", flat.size, flat))


def _read_texture(tex, w, h):
    buf = tex.read()
    buf.dimensions = w * h * 4
    return np.asarray(buf, dtype=np.float32).reshape(h, w, 4).copy()


def _depth_eps(points):
    """Absolute depth tolerance scaled to the model size; view depth can be negative in ortho views."""
    if len(points) == 0:
        return 1e-4
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))) * 0.002 + 1e-6


def _matrix(values):
    return Matrix([list(row) for row in np.asarray(values, dtype=np.float64).reshape(4, 4)])


def _view_pass(geo, view, proj, width, height, base_rgba=None, layer_mode=False):
    """Render the mesh from the capture view. Returns (color, linear depth, part id) textures."""
    color = gpu.types.GPUTexture((width, height), format="RGBA32F")
    depth_color = gpu.types.GPUTexture((width, height), format="RGBA32F")
    part_tex = gpu.types.GPUTexture((width, height), format="RGBA32F")
    zbuf = gpu.types.GPUTexture((width, height), format="DEPTH_COMPONENT32F")
    if base_rgba is None:
        base_rgba = np.full((1, 1, 4), 0.5, np.float32)
    base_tex = _to_texture(base_rgba)
    part = geo.get("part")
    if part is None:
        part = np.zeros(len(geo["pos"]), np.float32)

    shader = _view_shader()
    batch = batch_for_shader(shader, "TRIS", {
        "pos": geo["pos"], "uv": geo["uv"], "target": geo["target"], "part": part})
    fb = gpu.types.GPUFrameBuffer(depth_slot=zbuf, color_slots=(color, depth_color, part_tex))
    with fb.bind():
        fb.clear(color=(0.0, 0.0, 0.0, 0.0), depth=1.0)
        gpu.state.depth_test_set("LESS_EQUAL")
        gpu.state.depth_mask_set(True)
        gpu.state.face_culling_set("NONE")
        gpu.state.blend_set("NONE")
        shader.bind()
        shader.uniform_float("viewMatrix", _matrix(view))
        shader.uniform_float("projMatrix", _matrix(proj))
        shader.uniform_float("layerMode", 1.0 if layer_mode else 0.0)
        shader.uniform_sampler("baseTex", base_tex)
        batch.draw(shader)
        gpu.state.depth_test_set("NONE")
        gpu.state.depth_mask_set(False)
    return color, depth_color, part_tex


def render_layer(geo, view, proj, width, height, layer_rgba):
    """One texture layer (straight RGBA, bottom row first) seen from the capture view.

    Returns straight RGBA float32 H x W x 4, top row first; alpha is the layer's own alpha on
    visible faces and 0 elsewhere.
    """
    premult = layer_rgba.copy()
    premult[:, :, :3] *= premult[:, :, 3:4]
    color, _depth, _part = _view_pass(geo, view, proj, width, height, premult, layer_mode=True)
    return _read_texture(color, width, height)[::-1]


def _wire_pass(edges, depth_tex, view, proj, width, height):
    out = gpu.types.GPUTexture((width, height), format="RGBA32F")
    shader = _wire_shader()
    batch = batch_for_shader(shader, "LINES", {"pos": edges})
    fb = gpu.types.GPUFrameBuffer(color_slots=(out,))
    with fb.bind():
        fb.clear(color=(0.0, 0.0, 0.0, 0.0))
        gpu.state.depth_test_set("NONE")
        gpu.state.blend_set("NONE")
        shader.bind()
        shader.uniform_float("viewMatrix", _matrix(view))
        shader.uniform_float("projMatrix", _matrix(proj))
        shader.uniform_float("depthEps", _depth_eps(edges) * 0.5)
        shader.uniform_sampler("depthTex", depth_tex)
        batch.draw(shader)
    return out


def render_capture(geo, view, proj, width, height, base_rgba):
    """Render the capture view; every array is top row first.

    Returns (color H x W x 4 float32, part id H x W int32 with 0 = background, wire coverage H x W or None).
    """
    color, depth, part = _view_pass(geo, view, proj, width, height, base_rgba)
    rgba = _read_texture(color, width, height)[::-1]
    ids = np.rint(_read_texture(part, width, height)[::-1, :, 0]).astype(np.int32)
    wire = None
    edges = geo.get("edges")
    if edges is not None and len(edges):
        wire_tex = _wire_pass(edges, depth, view, proj, width, height)
        wire = _read_texture(wire_tex, width, height)[::-1, :, 3]
    return rgba, ids, wire


def project_paint(geo, view, proj, is_persp, paint_premul, tex_w, tex_h, cos_start, cos_end):
    """Project a premultiplied camera-space paint image (top row first) into UV space.

    Returns the premultiplied tex_h x tex_w x 4 contribution, bottom row first.
    """
    height, width = paint_premul.shape[:2]
    _color, depth_tex, _part = _view_pass(geo, view, proj, width, height)
    paint_tex = _to_texture(paint_premul[::-1])

    keep = geo["target"] > 0.5
    pos, nor, uv = geo["pos"][keep], geo["nor"][keep], geo["uv"][keep]
    if len(pos) == 0:
        raise BridgeError("No faces use the base color image")
    shader = _bake_shader()
    content = {"pos": pos, "nor": nor, "uv": uv}
    tris = batch_for_shader(shader, "TRIS", content)
    edge_index = (np.arange(len(pos), dtype=np.int32).reshape(-1, 3)[:, [0, 1, 1, 2, 2, 0]]).reshape(-1, 2)
    edges = batch_for_shader(shader, "LINES", content, indices=edge_index)

    out = gpu.types.GPUTexture((tex_w, tex_h), format="RGBA32F")
    fb = gpu.types.GPUFrameBuffer(color_slots=(out,))
    with fb.bind():
        fb.clear(color=(0.0, 0.0, 0.0, 0.0))
        gpu.state.depth_test_set("NONE")
        gpu.state.face_culling_set("NONE")
        gpu.state.blend_set("NONE")
        shader.bind()
        shader.uniform_float("viewMatrix", _matrix(view))
        shader.uniform_float("projMatrix", _matrix(proj))
        shader.uniform_float("params", (1.0 if is_persp else 0.0, cos_start, cos_end, _depth_eps(pos)))
        shader.uniform_sampler("paintTex", paint_tex)
        shader.uniform_sampler("depthTex", depth_tex)
        edges.draw(shader)
        tris.draw(shader)
    return _read_texture(out, tex_w, tex_h)


_cover = None


def _cover_shader():
    global _cover
    if _cover is None:
        info = gpu.types.GPUShaderCreateInfo()
        info.vertex_in(0, "VEC2", "uv")
        info.fragment_out(0, "VEC4", "outColor")
        info.vertex_source("void main() { gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0); }\n")
        info.fragment_source("void main() { outColor = vec4(1.0); }\n")
        _cover = gpu.shader.create_from_info(info)
    return _cover


def uv_coverage(uv_corners, w, h):
    """Texels covered by the UV triangles (corners as N*3 x 2), including their edges."""
    shader = _cover_shader()
    n = len(uv_corners)
    edge_index = np.arange(n, dtype=np.int32).reshape(-1, 3)[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    tris = batch_for_shader(shader, "TRIS", {"uv": uv_corners})
    edges = batch_for_shader(shader, "LINES", {"uv": uv_corners}, indices=edge_index)
    out = gpu.types.GPUTexture((w, h), format="RGBA32F")
    fb = gpu.types.GPUFrameBuffer(color_slots=(out,))
    with fb.bind():
        fb.clear(color=(0.0, 0.0, 0.0, 0.0))
        gpu.state.depth_test_set("NONE")
        gpu.state.face_culling_set("NONE")
        gpu.state.blend_set("NONE")
        shader.bind()
        edges.draw(shader)
        tris.draw(shader)
    return _read_texture(out, w, h)[:, :, 0] > 0.5


def uv_edges(uv_corners, w, h):
    """Coverage (0/1 float, bottom row first) of the UV triangles' edges, for a UV wireframe layer."""
    shader = _cover_shader()
    n = len(uv_corners)
    edge_index = np.arange(n, dtype=np.int32).reshape(-1, 3)[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    edges = batch_for_shader(shader, "LINES", {"uv": uv_corners}, indices=edge_index)
    out = gpu.types.GPUTexture((w, h), format="RGBA32F")
    fb = gpu.types.GPUFrameBuffer(color_slots=(out,))
    with fb.bind():
        fb.clear(color=(0.0, 0.0, 0.0, 0.0))
        gpu.state.depth_test_set("NONE")
        gpu.state.blend_set("NONE")
        shader.bind()
        edges.draw(shader)
    return _read_texture(out, w, h)[:, :, 0]


_NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def bleed(img, cover, steps):
    """Grow a premultiplied H x W x 4 image from UV islands into the gutter around them, `steps` texels.

    Texels inside islands are never changed; gutter texels take the average of their filled neighbours.
    """
    hit = img[:, :, 3] > 0
    if steps <= 0 or not hit.any():
        return img
    h, w = hit.shape
    ys, xs = np.nonzero(hit)
    y0, y1 = max(int(ys.min()) - steps - 1, 0), min(int(ys.max()) + steps + 2, h)
    x0, x1 = max(int(xs.min()) - steps - 1, 0), min(int(xs.max()) + steps + 2, w)
    sub = img[y0:y1, x0:x1].copy()
    valid = cover[y0:y1, x0:x1].copy()
    sh, sw = valid.shape
    for _ in range(steps):
        acc = np.zeros_like(sub)
        cnt = np.zeros((sh, sw), np.float32)
        for dy, dx in _NEIGHBOURS:
            dst = (slice(max(dy, 0), sh + min(dy, 0)), slice(max(dx, 0), sw + min(dx, 0)))
            src = (slice(max(-dy, 0), sh + min(-dy, 0)), slice(max(-dx, 0), sw + min(-dx, 0)))
            m = valid[src].astype(np.float32)
            acc[dst] += sub[src] * m[:, :, None]
            cnt[dst] += m
        grow = ~valid & (cnt > 0)
        if not grow.any():
            break
        sub[grow] = acc[grow] / cnt[grow][:, None]
        valid |= grow
    out = img.copy()
    out[y0:y1, x0:x1] = sub
    return out


def pad_gutter(img, cover, steps):
    """Fill texels outside the UV islands, up to `steps` texels away, with the nearest island texel."""
    if steps <= 0 or cover.all() or not cover.any():
        return img
    h, w = cover.shape
    nearest = np.where(cover, np.arange(h * w, dtype=np.int64).reshape(h, w), -1)
    for _ in range(steps):
        grown = nearest.copy()
        for dst, src in (((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
                         ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
                         ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
                         ((slice(None), slice(None, -1)), (slice(None), slice(1, None)))):
            take = (grown[dst] < 0) & (nearest[src] >= 0)
            grown[dst][take] = nearest[src][take]
        if np.array_equal(grown, nearest):
            break
        nearest = grown
    fill = ~cover & (nearest >= 0)
    out = img.copy()
    flat = out.reshape(h * w, -1)
    flat[fill.ravel()] = img.reshape(h * w, -1)[nearest[fill]]
    return out
