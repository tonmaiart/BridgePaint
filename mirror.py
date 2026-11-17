# SPDX-License-Identifier: GPL-3.0-or-later
"""Mirror texture content across a local axis of the mesh's rest shape."""

import bmesh
import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from .render import BridgeError, _read_texture, _to_texture, pad_gutter, uv_coverage

_shader = None


def _mirror_shader():
    global _shader
    if _shader is not None:
        return _shader
    iface = gpu.types.GPUStageInterfaceInfo("pbridge_mirror_iface")
    iface.smooth("VEC2", "v_muv")
    iface.smooth("VEC3", "v_lpos")
    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant("VEC4", "params")  # axis, source sign, mode (0 sample, 1 position)
    info.sampler(0, "FLOAT_2D", "srcTex")
    info.vertex_in(0, "VEC2", "uv")
    info.vertex_in(1, "VEC2", "muv")
    info.vertex_in(2, "VEC3", "lpos")
    info.vertex_out(iface)
    info.fragment_out(0, "VEC4", "outColor")
    info.fragment_out(1, "VEC4", "outMask")
    info.vertex_source(
        "void main() {\n"
        "  v_muv = muv;\n"
        "  v_lpos = lpos;\n"
        "  gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);\n"
        "}\n")
    info.fragment_source(
        "void main() {\n"
        "  float c = params.x < 0.5 ? v_lpos.x : (params.x < 1.5 ? v_lpos.y : v_lpos.z);\n"
        "  outColor = vec4(0.0);\n"
        "  outMask = vec4(0.0);\n"
        "  if (c * params.y >= 0.0) { return; }\n"
        "  outMask = vec4(1.0);\n"
        "  outColor = params.z > 0.5 ? vec4(v_lpos, 1.0) : texture(srcTex, v_muv);\n"
        "}\n")
    _shader = gpu.shader.create_from_info(info)
    return _shader


def _rest_triangles(obj, uv_name, material_indices):
    if obj.mode != "EDIT":
        return _mesh_triangles(obj.data, uv_name, material_indices)
    # In Edit Mode the mesh's own UVs and positions are not readable, so read a copy of the edit mesh.
    tmp = bpy.data.meshes.new("__paint_bridge_edit_copy")
    try:
        bmesh.from_edit_mesh(obj.data).to_mesh(tmp)
        return _mesh_triangles(tmp, uv_name, material_indices)
    finally:
        bpy.data.meshes.remove(tmp)


def _mesh_triangles(mesh, uv_name, material_indices):
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
    tri_verts = np.empty(ntri * 3, np.int32)
    mesh.loop_triangles.foreach_get("vertices", tri_verts)
    mat_index = np.empty(ntri, np.int32)
    mesh.loop_triangles.foreach_get("material_index", mat_index)
    co = np.empty(len(mesh.vertices) * 3, np.float32)
    mesh.vertices.foreach_get("co", co)
    uv = np.empty(len(mesh.loops) * 2, np.float32)
    uv_layer.uv.foreach_get("vector", uv)

    keep = np.ones(ntri, bool) if material_indices is None else         np.isin(mat_index, np.fromiter(material_indices, np.int32))
    tri_verts = tri_verts.reshape(-1, 3)[keep]
    tri_loops = tri_loops.reshape(-1, 3)[keep]
    if len(tri_verts) == 0:
        raise BridgeError("No faces use the base color image")
    co = co.reshape(-1, 3).astype(np.float64)
    return co, tri_verts, uv.reshape(-1, 2)[tri_loops].astype(np.float64)


def _bilinear(img, uv):
    """Sample an H x W x C image (bottom row first) at UV coordinates, clamped to the edge."""
    h, w = img.shape[:2]
    x = np.clip(uv[:, 0] * w - 0.5, 0, w - 1)
    y = np.clip(uv[:, 1] * h - 0.5, 0, h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bottom = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


class MirrorMap:
    """Maps every texel to the texel at its mirrored surface point, across one local axis."""

    def __init__(self, obj, uv_name, material_indices, axis):
        self.axis = axis
        co, self.tri_verts, self.tri_uv = _rest_triangles(obj, uv_name, material_indices)
        self.co = co
        self.bvh = BVHTree.FromPolygons([Vector(v) for v in co], self.tri_verts.tolist(), all_triangles=True)
        self.flip = np.ones(3)
        self.flip[axis] = -1.0
        far = float(np.linalg.norm(co.max(axis=0) - co.min(axis=0))) * 0.01

        self.corners = co[self.tri_verts]
        centroid = self.corners.mean(axis=1, keepdims=True)
        # Nudge each probe toward its own face so a mirrored vertex on a UV seam picks the matching side.
        probe = (self.corners + (centroid - self.corners) * 1e-3) * self.flip
        ntri = len(self.tri_verts)
        self.muv = np.empty((ntri, 3, 2))
        dist = np.empty((ntri, 3))
        for t in range(ntri):
            for k in range(3):
                uv, d = self._uv_at(Vector(probe[t, k]))
                self.muv[t, k] = uv if uv is not None else self.tri_uv[t, k]
                dist[t, k] = d

        def max_edge(tris):
            return np.max(np.linalg.norm(tris - np.roll(tris, 1, axis=1), axis=2), axis=1)

        self.split = (max_edge(self.muv) > max_edge(self.tri_uv) * 2.0 + 1e-3) | (dist.max(axis=1) > far)
        self._coverage = {}

    def _uv_at(self, point):
        loc, _normal, index, dist = self.bvh.find_nearest(point)
        if loc is None:
            return None, np.inf
        a, b, c = (self.co[v] for v in self.tri_verts[index])
        p = np.array(loc)
        v0, v1, v2 = b - a, c - a, p - a
        d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
        d20, d21 = v2 @ v0, v2 @ v1
        den = d00 * d11 - d01 * d01
        if abs(den) < 1e-20:
            weights = np.array([1.0, 0.0, 0.0])
        else:
            v = (d11 * d20 - d01 * d21) / den
            u = (d00 * d21 - d01 * d20) / den
            weights = np.array([1.0 - v - u, v, u])
        return weights @ self.tri_uv[index], dist

    def coverage(self, w, h):
        key = (w, h)
        if key not in self._coverage:
            self._coverage[key] = uv_coverage(self.tri_uv.reshape(-1, 2).astype(np.float32), w, h)
        return self._coverage[key]

    def _draw(self, src_tex, tris, mode, sign, w, h):
        if len(tris) == 0:
            return np.zeros((h, w, 4), np.float32), np.zeros((h, w), bool)
        shader = _mirror_shader()
        content = {
            "uv": self.tri_uv[tris].reshape(-1, 2).astype(np.float32),
            "muv": self.muv[tris].reshape(-1, 2).astype(np.float32),
            "lpos": self.corners[tris].reshape(-1, 3).astype(np.float32),
        }
        n = len(content["uv"])
        edge_index = np.arange(n, dtype=np.int32).reshape(-1, 3)[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
        tris_batch = batch_for_shader(shader, "TRIS", content)
        edges_batch = batch_for_shader(shader, "LINES", content, indices=edge_index)
        color = gpu.types.GPUTexture((w, h), format="RGBA32F")
        mask = gpu.types.GPUTexture((w, h), format="RGBA32F")
        fb = gpu.types.GPUFrameBuffer(color_slots=(color, mask))
        with fb.bind():
            fb.clear(color=(0.0, 0.0, 0.0, 0.0))
            gpu.state.depth_test_set("NONE")
            gpu.state.face_culling_set("NONE")
            gpu.state.blend_set("NONE")
            shader.bind()
            shader.uniform_float("params", (float(self.axis), sign, float(mode), 0.0))
            shader.uniform_sampler("srcTex", src_tex)
            edges_batch.draw(shader)
            tris_batch.draw(shader)
        return _read_texture(color, w, h), _read_texture(mask, w, h)[:, :, 0] > 0.5

    def sample(self, src, from_positive):
        """Mirror an H x W x 4 image (bottom row first) onto one side.

        Returns (image with the mirrored values on the target side and zeros elsewhere, target mask).
        """
        h, w = src.shape[:2]
        src_tex = _to_texture(src)
        sign = 1.0 if from_positive else -1.0
        out, mask = self._draw(src_tex, np.flatnonzero(~self.split), 0, sign, w, h)
        pos, exact = self._draw(src_tex, np.flatnonzero(self.split), 1, sign, w, h)
        pick = np.argwhere(exact)
        if len(pick):
            uvs = np.empty((len(pick), 2))
            for i, (y, x) in enumerate(pick):
                uv, _d = self._uv_at(Vector(pos[y, x, :3] * self.flip))
                uvs[i] = uv if uv is not None else ((x + 0.5) / w, (y + 0.5) / h)
            out[pick[:, 0], pick[:, 1]] = _bilinear(src, uvs)
            mask[pick[:, 0], pick[:, 1]] = True
        return out, mask

    def both_ways(self, src):
        """Every texel gets the value from its mirrored point (each side from the other)."""
        a, mask_a = self.sample(src, True)
        b, mask_b = self.sample(src, False)
        return a + b, mask_a | mask_b


def mirror(obj, base, uv_name, material_indices, axis, from_positive, bleed_px=0):
    """Copy one side of `base` (H x W x 4, bottom row first) onto the other. Returns (image, texel count)."""
    mmap = MirrorMap(obj, uv_name, material_indices, axis)
    h, w = base.shape[:2]
    sampled, mask = mmap.sample(base, from_positive)
    result = base.copy()
    result[mask, :3] = sampled[mask, :3]
    if bleed_px:
        result = pad_gutter(result, mmap.coverage(w, h), bleed_px)
    return result, int(mask.sum())
