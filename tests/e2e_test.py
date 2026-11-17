# SPDX-License-Identifier: GPL-3.0-or-later
"""End-to-end test. Run with a window (GPU needed):

    blender --factory-startup --python tests/e2e_test.py -- <output_dir>

Writes results.txt (and texture PNGs) to <output_dir>, then quits.
"""

import os
import struct
import sys
import traceback

import bpy
import numpy as np

ADDON_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else os.path.join(ADDON_PARENT, "pb_test_out")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, ADDON_PARENT)

log_lines = []


def log(msg):
    log_lines.append(str(msg))


def check(cond, msg):
    log(("PASS " if cond else "FAIL ") + msg)


def packbits(row):
    out = bytearray()
    i, n = 0, len(row)
    while i < n:
        j = i
        while j + 1 < n and row[j + 1] == row[i] and j - i < 127:
            j += 1
        if j > i:
            out += bytes((257 - (j - i + 1),)) + bytes((row[i],))
            i = j + 1
        else:
            k = i
            while k + 1 < n and row[k + 1] != row[k] and k - i < 127:
                k += 1
            out += bytes((k - i,)) + bytes(row[i:k + 1])
            i = k + 1
    return bytes(out)


def write_rle_psd(path, w, h, layers):
    """Write a PSD whose layers (name, full-canvas RGBA uint8, rle) are stored like Photoshop does."""
    import io
    f = io.BytesIO()
    f.write(b"8BPS" + struct.pack(">H6xHIIHH", 1, 4, h, w, 8, 3))
    f.write(struct.pack(">II", 0, 0))
    records = bytearray()
    chdata = bytearray()
    for name, rgba, rle in layers:
        records += struct.pack(">iiiiH", 0, 0, h, w, 4)
        for cid, c in ((-1, 3), (0, 0), (1, 1), (2, 2)):
            plane = np.ascontiguousarray(rgba[:, :, c])
            if rle:
                rows = [packbits(plane[y].tobytes()) for y in range(h)]
                body = struct.pack(">H", 1) + b"".join(struct.pack(">H", len(r)) for r in rows) + b"".join(rows)
            else:
                body = struct.pack(">H", 0) + plane.tobytes()
            records += struct.pack(">hI", cid, len(body))
            chdata += body
        nm = bytes((len(name),)) + name.encode()
        nm += b"\0" * ((4 - len(nm) % 4) % 4)
        extra = struct.pack(">II", 0, 0) + nm
        records += b"8BIMnorm" + struct.pack(">BBBB", 255, 0, 0, 0) + struct.pack(">I", len(extra)) + extra
    li = struct.pack(">h", -len(layers)) + records + chdata
    if len(li) % 2:
        li += b"\0"
    sec = struct.pack(">I", len(li)) + li + struct.pack(">I", 0)
    f.write(struct.pack(">I", len(sec)) + sec)
    f.write(struct.pack(">H", 0))
    for c in range(4):
        f.write(np.ascontiguousarray(layers[0][1][:, :, c]).tobytes())
    open(path, "wb").write(f.getvalue())


def canvas_of(doc, layer):
    """Full-canvas straight RGBA uint8 of a decoded layer."""
    out = np.zeros((doc.height, doc.width, 4), np.uint8)
    px = layer.pixels
    if 0 not in px or layer.width <= 0:
        return out
    rgba = np.dstack([px[c] for c in (0, 1, 2)] + [px.get(-1, np.ones_like(px[0]))])
    out[layer.top:layer.bottom, layer.left:layer.right] = (rgba * 255 + 0.5).astype(np.uint8)
    return out


def paint_on(psd, path, layer_name, paint, group=None, erase=False):
    """Paint (straight RGBA uint8 canvas) over every layer `layer_name` (inside folder `group`), like a brush.

    With `erase`, pixels where `paint` has alpha become transparent instead.
    """
    doc = psd.read_psd(path)
    tree = psd.layer_tree(doc)
    layers = [psd.layer_to_dict(doc, l) for l in doc.layers]
    hit = 0
    for i, layer in enumerate(doc.layers):
        if layer.section or layer.name != layer_name:
            continue
        if group is not None and not any(doc.layers[g].name == group for g in tree.get(i, (True, ()))[1]):
            continue
        cur = canvas_of(doc, layer).astype(np.float32) / 255.0
        pa = paint[:, :, 3:4].astype(np.float32) / 255.0
        if erase:
            cur[:, :, 3:4] *= (1.0 - pa)
        else:
            out_a = pa + cur[:, :, 3:4] * (1.0 - pa)
            rgb = paint[:, :, :3] / 255.0 * pa + cur[:, :, :3] * cur[:, :, 3:4] * (1.0 - pa)
            cur[:, :, :3] = np.where(out_a > 0, rgb / np.maximum(out_a, 1e-6), cur[:, :, :3])
            cur[:, :, 3:4] = out_a
        layers[i]["rgba"] = (cur * 255 + 0.5).astype(np.uint8)
        layers[i]["rect"] = (0, 0, doc.height, doc.width)
        hit += 1
    comp = np.zeros((doc.height, doc.width, 4), np.uint8)
    psd.write_psd(path, doc.width, doc.height, layers, comp)
    return hit


def save_png(image_name, filename):
    img = bpy.data.images[image_name]
    buf = np.empty(len(img.pixels), np.float32)
    img.pixels.foreach_get(buf)
    copy = bpy.data.images.new(image_name + "_out", *img.size, alpha=True)
    copy.pixels.foreach_set(buf)
    copy.filepath_raw = os.path.join(OUT, filename)
    copy.file_format = "PNG"
    copy.save()
    bpy.data.images.remove(copy)


def layer_dict(layer):
    px = layer.pixels
    rgba = np.dstack([px[c] for c in (0, 1, 2, -1)])
    d = {"name": layer.name, "rgba": (rgba * 255 + 0.5).astype(np.uint8),
         "rect": (layer.top, layer.left, layer.bottom, layer.right), "opacity": layer.opacity}
    if layer.mask and -2 in px:
        d["mask"] = (px[-2] * 255 + 0.5).astype(np.uint8)
        d["mask_rect"] = layer.mask[:4]
    return d


def rebuild_layer(layer):
    """Layer dict for write_psd from a decoded layer, keeping groups, masks and empty layers."""
    px = layer.pixels
    d = {"name": layer.name, "opacity": layer.opacity, "section": layer.section or None, "rgba": None}
    if 0 in px and layer.width > 0:
        rgba = np.dstack([px[c] for c in (0, 1, 2, -1)])
        d["rgba"] = (rgba * 255 + 0.5).astype(np.uint8)
        d["rect"] = (layer.top, layer.left, layer.bottom, layer.right)
    if layer.mask and -2 in px:
        d["mask"] = (px[-2] * 255 + 0.5).astype(np.uint8)
        d["mask_rect"] = layer.mask[:4]
    return d


def run_parts_test(s, obj, img, window, area, region, bridge, psd):
    from mathutils import Vector
    s.split_parts = True
    rv3d = area.spaces.active.region_3d
    toward_cam = rv3d.view_rotation @ Vector((0.0, 0.0, 1.0))
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=toward_cam * 1.5)
    cube = bpy.context.active_object
    for loop_uv in cube.data.uv_layers.active.uv:
        loop_uv.vector = (0.995, 0.995)
    cube.vertex_groups.new(name="Box").add(range(len(cube.data.vertices)), 1.0, "REPLACE")
    obj.vertex_groups.new(name="Body").add(range(len(obj.data.vertices)), 1.0, "REPLACE")
    with bpy.context.temp_override(active_object=obj, selected_editable_objects=[obj, cube]):
        bpy.ops.object.join()
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    before = np.empty(512 * 512 * 4, np.float32)
    img.pixels.foreach_get(before)
    before = before.reshape(512, 512, 4)

    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.paint_bridge.capture()
    check(result == {"FINISHED"}, f"parts capture {result}")
    item = s.captures[s.active_index]
    paths = bridge.capture_paths(item.directory, item.name)
    doc = psd.read_psd(paths["psd"])
    names = [l.name for l in doc.layers]
    expected = ["</Layer group>", "Base", "Box", "</Layer group>", "Base", "Body", "Wireframe"]
    check(names == expected, f"one folder per part, named from vertex groups {names}")
    sections = [l.section for l in doc.layers]
    check(sections == [3, 0, 2, 3, 0, 2, 0], f"folders written as groups {sections}")
    check(all(doc.layers[i].mask is not None and -2 in doc.layers[i].pixels for i in (2, 5)), "masks sit on the folders")
    check(all(doc.layers[i].mask is None for i in (1, 4)), "Base layers carry no mask of their own")
    wire = doc.layers[6]
    check(wire.lock_all and (wire.pixels[-1] > 0).sum() > 200, f"wireframe pixels {(wire.pixels[-1] > 0).sum()}")

    red_canvas = np.zeros((doc.height, doc.width, 4), np.uint8)
    red_canvas[:] = (255, 0, 0, 255)
    paint_on(psd, paths["psd"], "Base", red_canvas, group="Body")  # Body/Base: whole canvas red

    doc2 = psd.read_psd(paths["psd"])
    comp, _ = psd.flatten_layers(doc2, lambda l: l.name != "Wireframe")
    box = doc.layers[2]                              # the Box folder; its faces do not use the texture
    box_px = np.zeros((doc.height, doc.width), bool)
    mt, ml, mb, mr = box.mask[:4]
    box_px[mt:mb, ml:mr] = box.pixels[-2] > 0.5
    red = (comp[:, :, 0] > 0.99) & (comp[:, :, 1] < 0.01)
    check(box_px.sum() > 100 and not (red & box_px).any(), "Body folder mask keeps paint off the box")
    check(red.sum() > 1000 and not (red & (comp[:, :, 3] < 0.5)).any(), "Body folder mask keeps paint off the background")

    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.paint_bridge.merge()
    after = np.empty(512 * 512 * 4, np.float32)
    img.pixels.foreach_get(after)
    after = after.reshape(512, 512, 4)
    red_tex = (after[:, :, 0] > 0.95) & (after[:, :, 1] < 0.05) & (after[:, :, 2] < 0.05)
    changed = np.abs(after - before).max(axis=2) > 1e-3
    check(result == {"FINISHED"} and red_tex.sum() > 20000, f"parts merge red texels {int(red_tex.sum())}")
    check(changed.sum() < 512 * 512 * 0.6, f"parts merge leaves the far side ({int(changed.sum())} changed)")
    save_png("BaseTex", "after_parts_merge.png")
    import shutil
    shutil.copy(paths["psd"], os.path.join(OUT, "parts_capture.psd"))


def run_ortho_test(s, img, window, area, region, bridge, psd):
    """Ortho view with the view plane inside the mesh, so part of it has negative view depth."""
    s.split_parts = False
    rv3d = area.spaces.active.region_3d
    rv3d.view_perspective = "ORTHO"
    from mathutils import Vector
    rv3d.view_distance = 4.0
    rv3d.view_location = -(rv3d.view_rotation @ Vector((0.0, 0.0, 1.0))) * 4.0
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.paint_bridge.capture()
    item = s.captures[s.active_index]
    paths = bridge.capture_paths(item.directory, item.name)
    doc = psd.read_psd(paths["psd"])
    wire = doc.layers[1]
    meta, _p = bridge.load_meta(item.directory, item.name)
    geo = np.load(paths["geo"])
    view_z = np.c_[geo["pos"], np.ones(len(geo["pos"]))] @ np.array(meta["view"])[2]
    check((view_z > 0).any(), "ortho test has surfaces with negative view depth")
    check(result == {"FINISHED"} and (wire.pixels[-1] > 0).sum() > 500,
          f"ortho capture wireframe pixels {(wire.pixels[-1] > 0).sum()}")
    blue = np.zeros((doc.height, doc.width, 4), np.uint8)
    blue[:] = (0, 0, 255, 255)
    paint_on(psd, paths["psd"], "Base", blue)
    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.paint_bridge.merge()
    after = np.empty(512 * 512 * 4, np.float32)
    img.pixels.foreach_get(after)
    after = after.reshape(512, 512, 4)
    blue = (after[:, :, 2] > 0.95) & (after[:, :, 0] < 0.05) & (after[:, :, 1] < 0.05)
    check(result == {"FINISHED"} and blue.sum() > 10000, f"ortho merge blue texels {int(blue.sum())}")


def run_list_test(s, window, area, region, bridge, psd):
    import json
    import shutil
    from mathutils import Euler
    paint_bridge = sys.modules["paint_bridge"]
    scene = bpy.context.scene
    directory = bridge.project_directory(s)
    paint_bridge.sync_captures(scene)
    names = [i.name for i in s.captures]
    check(names == bridge.list_captures(directory) and len(names) == 3, f"list mirrors folder {names}")

    open(os.path.join(directory, "stray.psd"), "wb").write(b"x")
    src = bridge.capture_paths(directory, names[0])
    dst = bridge.capture_paths(directory, "copied_capture")
    for key in ("psd", "meta", "geo", "ref"):
        shutil.copy(src[key], dst[key])
    paint_bridge.sync_captures(scene)
    names2 = [i.name for i in s.captures]
    check("copied_capture" in names2 and "stray" not in names2, "new PSD with capture data appears, bare PSD does not")
    for key in ("psd", "meta", "geo", "ref"):
        os.remove(dst[key])
    paint_bridge.sync_captures(scene)
    check("copied_capture" not in [i.name for i in s.captures], "deleted PSD leaves the list")

    space = area.spaces.active
    rv3d = space.region_3d
    meta0, _ = bridge.load_meta(directory, names[1])
    rv3d.view_perspective = "PERSP"
    rv3d.view_rotation = Euler((0.3, 0.2, 2.5)).to_quaternion()
    rv3d.view_distance = 9.0
    s.active_index = 0
    with bpy.context.temp_override(window=window, area=area, region=region):
        s.active_index = 1
    vp = meta0["viewport"]
    same = (abs(rv3d.view_distance - vp["distance"]) < 1e-4 and
            max(abs(a - b) for a, b in zip(rv3d.view_rotation, vp["rotation"])) < 1e-4 and
            rv3d.view_perspective == vp["mode"])
    check(same, "selecting a capture snaps the view")

    stem = names[1]
    paths = bridge.capture_paths(directory, stem)
    rv3d.view_rotation = Euler((1.0, 0.0, 2.0)).to_quaternion()
    rv3d.view_distance = 5.0
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.paint_bridge.recapture("EXEC_DEFAULT")
    meta1 = json.load(open(paths["meta"]))
    check(result == {"FINISHED"} and abs(meta1["viewport"]["distance"] - 5.0) < 1e-4
          and not os.path.exists(paths["applied"]) and len(s.captures) == 3
          and s.captures[s.active_index].name == stem,
          "recapture overwrites the same PSD with the current view")
    check(psd.read_psd(paths["psd"]).layers[-1].name == "Wireframe", "recaptured PSD is readable")

    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.paint_bridge.capture(name="My View.psd")
    named = bridge.capture_paths(directory, "My View")
    check(result == {"FINISHED"} and os.path.exists(named["psd"])
          and s.captures[s.active_index].name == "My View", "capture uses the typed name")
    try:
        with bpy.context.temp_override(window=window, area=area, region=region):
            again = bpy.ops.paint_bridge.capture(name="My View")
    except RuntimeError:
        again = {"CANCELLED"}
    check(again == {"CANCELLED"}, "an existing name is refused instead of overwritten")

    removed = []
    real = bridge._send_to_recycle_bin
    bridge._send_to_recycle_bin = lambda files: [removed.append(f) or os.remove(f) for f in files]
    try:
        with bpy.context.temp_override(window=window, area=area, region=region):
            result = bpy.ops.paint_bridge.delete("EXEC_DEFAULT")
    finally:
        bridge._send_to_recycle_bin = real
    check(result == {"FINISHED"} and len(removed) == 4 and not os.path.exists(named["psd"])
          and "My View" not in [i.name for i in s.captures], "delete removes the PSD, its data and the list entry")


def run_mirror_test(s, bridge):
    """Flat grid: left half (x<0) and right half have different topology and UV layouts, and the
    right half is split into two UV islands at x=0.6. Mirroring +X onto -X must give each left
    texel the colour of the right-side point at (-x, y)."""
    pieces = [  # x0, x1, columns, uv(x, y)
        (-1.0, 0.0, 7, lambda x, y: ((x + 1) * 0.5, (y + 1) * 0.5)),
        (0.0, 0.6, 3, lambda x, y: (0.55 + x * 0.5, (y + 1) * 0.25)),
        (0.6, 1.0, 2, lambda x, y: (0.55 + (x - 0.6) * 0.5, 0.5 + (y + 1) * 0.25)),
    ]
    verts, faces, uvs = [], [], []
    for x0, x1, nx, uvf in pieces:
        ny = 6
        base = len(verts)
        for j in range(ny + 1):
            for i in range(nx + 1):
                verts.append((x0 + (x1 - x0) * i / nx, -1 + 2 * j / ny, 0.0))
        for j in range(ny):
            for i in range(nx):
                a0 = base + j * (nx + 1) + i
                faces.append((a0, a0 + 1, a0 + nx + 2, a0 + nx + 1))
                uvs.append(uvf)
    mesh = bpy.data.meshes.new("MirrorGrid")
    mesh.from_pydata(verts, [], faces)
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for poly, uvf in zip(mesh.polygons, uvs):
        for li in poly.loop_indices:
            x, y, _z = mesh.vertices[mesh.loops[li].vertex_index].co
            uv_layer.uv[li].vector = uvf(x, y)
    obj = bpy.data.objects.new("MirrorGrid", mesh)
    bpy.context.collection.objects.link(obj)

    n = 256
    tex = np.zeros((n, n, 4), np.float32)
    tex[:, :, 2] = 1.0
    tex[:, :, 3] = 1.0
    vv, uu = (np.mgrid[0:n, 0:n] + 0.5) / n
    r1 = (uu >= 0.55) & (uu <= 0.85) & (vv <= 0.5)
    r2 = (uu >= 0.55) & (uu <= 0.75) & (vv > 0.5)
    x1, y1 = (uu - 0.55) / 0.5, vv * 4 - 1
    x2, y2 = 0.6 + (uu - 0.55) / 0.5, (vv - 0.5) * 4 - 1
    tex[r1, 0], tex[r1, 1], tex[r1, 2] = x1[r1], (y1[r1] + 1) / 2, 0.0
    tex[r2, 0], tex[r2, 1], tex[r2, 2] = x2[r2], (y2[r2] + 1) / 2, 0.0
    img = bpy.data.images.new("MirrorTex", n, n, alpha=True, float_buffer=True)
    img.colorspace_settings.name = "Non-Color"
    img.pixels.foreach_set(tex.ravel())

    mat = bpy.data.materials.new("MirrorMat")
    tree = mat.node_tree
    bsdf = next(nd for nd in tree.nodes if nd.type == "BSDF_PRINCIPLED")
    node = tree.nodes.new("ShaderNodeTexImage")
    node.image = img
    tree.links.new(node.outputs["Color"], bsdf.inputs["Base Color"])
    mesh.materials.append(mat)

    s.image_override = None
    s.mirror_axis = "X"
    s.mirror_direction = "POS"
    with bpy.context.temp_override(active_object=obj, object=obj):
        result = bpy.ops.paint_bridge.mirror()
    out = np.empty(n * n * 4, np.float32)
    img.pixels.foreach_get(out)
    out = out.reshape(n, n, 4)
    left = (uu > 0.02) & (uu < 0.48) & (vv > 0.02) & (vv < 0.98)
    exp_r, exp_g = 1 - 2 * uu, vv
    err = np.maximum(np.abs(out[:, :, 0] - exp_r), np.abs(out[:, :, 1] - exp_g))[left]
    check(result == {"FINISHED"} and err.max() < 0.03, f"mirror +X to -X (max error {err.max():.4f})")
    seam = left & (uu > 0.15) & (uu < 0.21)
    check(np.abs(out[:, :, 0] - exp_r)[seam].max() < 0.03, "mirror across a source UV seam")
    check(np.array_equal(out[r1 | r2], tex[r1 | r2]), "source side untouched")
    backups = os.listdir(os.path.join(bridge.project_directory(s), "backup"))
    check(any(b.startswith("MirrorTex_") for b in backups), f"backup written {backups}")

    gutter = (uu > 0.505) & (uu < 0.545)           # between the left island and the right ones
    dirty = out.copy()
    dirty[gutter] = (1.0, 1.0, 1.0, 1.0)
    img.pixels.foreach_set(dirty.ravel())
    with bpy.context.temp_override(active_object=obj, object=obj):
        result = bpy.ops.paint_bridge.fill_seams()
    fixed = np.empty(n * n * 4, np.float32)
    img.pixels.foreach_get(fixed)
    fixed = fixed.reshape(n, n, 4)
    near = fixed[8:-8, 129:134, :3]                # first gutter texels right of the left island
    edge = fixed[8:-8, 127:128, :3]                # the left island's last column
    check(result == {"FINISHED"} and near.min(axis=2).max() < 0.85 and np.abs(near - edge).max() < 0.02,
          "Fill Seams replaces white gutter with the nearest island color")
    check(np.array_equal(fixed[~gutter & (uu < 0.5)], out[~gutter & (uu < 0.5)]), "Fill Seams leaves island texels alone")


def run_auto_mirror_test(s, window, area, region, bridge, psd):
    from mathutils import Quaternion
    obj = bpy.data.objects["MirrorGrid"]
    img = bpy.data.images["MirrorTex"]
    n = img.size[0]
    grey = np.full((n, n, 4), 0.5, np.float32)
    grey[:, :, 3] = 1.0
    img.pixels.foreach_set(grey.ravel())
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    rv3d = area.spaces.active.region_3d
    rv3d.view_perspective = "ORTHO"
    rv3d.view_rotation = Quaternion((1, 0, 0, 0))
    rv3d.view_location = (0, 0, 0)
    rv3d.view_distance = 3.0
    s.split_parts = False
    s.auto_mirror = True
    s.seam_bleed = 8
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.paint_bridge.capture()
    item = s.captures[s.active_index]
    paths = bridge.capture_paths(item.directory, item.name)
    meta, _ = bridge.load_meta(item.directory, item.name)
    W, H = meta["width"], meta["height"]
    V, P = np.array(meta["view"]), np.array(meta["proj"])

    def to_px(x, y):
        c = P @ V @ np.array([x, y, 0.0, 1.0])
        return (c[0] / c[3] * 0.5 + 0.5) * W, (1 - (c[1] / c[3] * 0.5 + 0.5)) * H

    cx, cy = to_px(0.3, 0.0)
    rx = to_px(0.7, 0.0)[0] - cx
    yy, xx = np.mgrid[0:H, 0:W]
    disc = ((xx - cx) / rx) ** 2 + ((yy - cy) / rx) ** 2 < 1.0
    paint = np.zeros((H, W, 4), np.uint8)
    paint[disc] = (255, 0, 0, 255)
    gx, gy = to_px(-0.3, 0.0)
    small = ((xx - gx) / rx) ** 2 + ((yy - gy) / rx) ** 2 < (0.15 / 0.4) ** 2
    paint[small] = (0, 255, 0, 255)                # the left side has its own paint here
    paint_on(psd, paths["psd"], "Base", paint)
    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.paint_bridge.merge()
    out = np.empty(n * n * 4, np.float32)
    img.pixels.foreach_get(out)
    out = out.reshape(n, n, 4)
    red = (out[:, :, 0] > 0.9) & (out[:, :, 1] < 0.1)
    green = (out[:, :, 1] > 0.9) & (out[:, :, 0] < 0.1)
    vv, uu = (np.mgrid[0:n, 0:n] + 0.5) / n
    left = (red | green) & (uu < 0.5)
    right = red & (uu > 0.55)
    ratio = left.sum() / max(right.sum(), 1)
    xl = (2 * uu[left] - 1).mean()                # left island: x = 2u - 1
    check(result == {"FINISHED"} and 1.6 < ratio < 2.4 and abs(xl + 0.3) < 0.06,
          f"auto mirror paints the other side (texel ratio {ratio:.2f}, mean x {xl:.3f})")
    def texel(u, v):
        return out[int(v * n), int(u * n), :3]
    check(texel(0.35, 0.5)[1] > 0.9 and texel(0.35, 0.5)[0] < 0.1, "own paint wins over the mirrored paint")
    check(texel(0.225, 0.5)[0] > 0.9, "unpainted area on the left receives the mirrored paint")
    check(texel(0.70, 0.25)[0] > 0.9 and texel(0.70, 0.25)[1] < 0.1, "right side keeps its paint over the mirrored green")
    edge = out[n // 2, 126:128, 0]                 # left island's last texels at y = 0, painted
    gutter = out[n // 2, 128:132, 0]               # first gutter texels past that edge
    check(edge.min() > 0.9 and gutter.min() > 0.9, f"seam bleed fills the gutter (min red {gutter.min():.2f})")
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.paint_bridge.revert()
    back = np.empty(n * n * 4, np.float32)
    img.pixels.foreach_get(back)
    check(np.abs(back.reshape(n, n, 4) - grey).max() < 1.01 / 255, "revert removes the mirrored paint and bleed too")
    s.auto_mirror = False


def run_isolation_test(s, window, area, region, bridge, psd):
    """In Edit Mode, faces hidden with H are left out of Capture and Open File."""
    import bmesh
    from mathutils import Euler
    pb = sys.modules["paint_bridge"]
    from paint_bridge import parts
    obj = bpy.data.objects["Sphere"]
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    s.split_parts = True
    s.auto_mirror = False
    rv3d = area.spaces.active.region_3d
    rv3d.view_perspective = "PERSP"
    rv3d.view_location = (0, 0, 0)
    rv3d.view_distance = 4.0
    rv3d.view_rotation = Euler((1.2, 0.0, 0.5)).to_quaternion()
    ov = dict(window=window, area=area, region=region)

    def call(op, *args, **kw):
        with bpy.context.temp_override(**ov):
            bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
        with bpy.context.temp_override(**ov):
            return op(*args, **kw)

    def hide_part(name, show_others=True):
        """Hide one part's faces in Edit Mode (and unhide everything else)."""
        labels, names = parts.part_info(obj)
        bm = bmesh.from_edit_mesh(obj.data)
        target = names.index(name)
        for f in bm.faces:
            f.hide_set(False)
        for f in bm.faces:
            f.select_set(bool(labels[f.verts[0].index] == target))
        for f in bm.faces:
            if labels[f.verts[0].index] == target:
                f.hide_set(True)
        bmesh.update_edit_mesh(obj.data)

    call(bpy.ops.paint_bridge.capture)
    full = s.captures[s.active_index]
    full_doc = psd.read_psd(bridge.capture_paths(full.directory, full.name)["psd"])
    full_body = full_doc.layers[[l.name for l in full_doc.layers].index("Body") - 1]
    check([l.name for l in full_doc.layers if l.section == 2] == ["Box", "Body"], "Object Mode capture has every part")

    result = call(bpy.ops.paint_bridge.set_mode, mode="VERTEX")
    check(result == {"FINISHED"} and obj.mode == "EDIT", "Select Vertex enters Edit Mode")
    hide_part("Box")
    result = call(bpy.ops.paint_bridge.capture)
    item = s.captures[s.active_index]
    paths = bridge.capture_paths(item.directory, item.name)
    doc = psd.read_psd(paths["psd"])
    groups = [l.name for l in doc.layers if l.section == 2]
    body = doc.layers[[l.name for l in doc.layers].index("Body") - 1]
    check(result == {"FINISHED"} and "Body" in item.name and groups == ["Body"],
          f"capture with Box hidden has only the Body folder ({item.name}, {groups})")
    check((body.pixels[-1] > 0.5).sum() > (full_body.pixels[-1] > 0.5).sum() + 100,
          "hidden faces are left out, so the box no longer covers the body")

    opened = []
    real_open = pb.open_in_paint_app
    pb.open_in_paint_app = lambda path, context: opened.append(path)
    try:
        s.active_index = [c.name for c in s.captures].index(full.name)
        result = call(bpy.ops.paint_bridge.open_file)
    finally:
        pb.open_in_paint_app = real_open
    sub = opened[-1] if opened else ""
    sub_doc = psd.read_psd(sub) if sub else None
    check(result == {"FINISHED"} and sub.endswith("@Body.psd") and sub_doc is not None
          and [l.name for l in sub_doc.layers if l.section == 2] == ["Body"]
          and sub_doc.layers[-1].name == "Wireframe", f"Open File with Box hidden opens a Body-only PSD {sub}")

    s.active_index = [c.name for c in s.captures].index(item.name)
    pb.open_in_paint_app = lambda path, context: opened.append(path)
    try:
        call(bpy.ops.paint_bridge.open_file)
    finally:
        pb.open_in_paint_app = real_open
    body_sub = opened[-1]
    sdoc = psd.read_psd(body_sub)
    green = np.zeros((sdoc.height, sdoc.width, 4), np.uint8)
    green[:] = (0, 255, 0, 255)
    paint_on(psd, body_sub, "Base", green)
    img = bpy.data.images["BaseTex"]
    s.auto_mirror = True
    try:
        result = call(bpy.ops.paint_bridge.merge)
    except RuntimeError as exc:
        result = str(exc)
    s.auto_mirror = False
    px = np.empty(len(img.pixels), np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(img.size[1], img.size[0], 4)
    greens = ((px[:, :, 1] > 0.95) & (px[:, :, 0] < 0.05)).sum()
    check(result == {"FINISHED"} and greens > 20000,
          f"Edit Mode merge with Auto Mirror picks up the per-part PSD ({result}, {greens} texels)")

    result = call(bpy.ops.paint_bridge.set_mode, mode="PART")
    check(result == {"FINISHED"} and obj.mode == "EDIT" and pb._picking, "Select Part enters Edit Mode and starts picking")
    hide_part("Body")
    call(bpy.ops.paint_bridge.capture)
    box_item = s.captures[s.active_index]
    box_paths = bridge.capture_paths(box_item.directory, box_item.name)
    groups = [l.name for l in psd.read_psd(box_paths["psd"]).layers if l.section == 2]
    check(groups == ["Box"], f"capture with Body hidden has only the Box folder ({groups})")
    call(bpy.ops.paint_bridge.set_mode, mode="OBJECT")
    check(obj.mode == "OBJECT", "Object button returns to Object Mode")
    s.active_index = [c.name for c in s.captures].index(box_item.name)
    call(bpy.ops.paint_bridge.recapture, "EXEC_DEFAULT")
    groups = [l.name for l in psd.read_psd(box_paths["psd"]).layers if l.section == 2]
    check(groups == ["Box"], f"Recapture in Object Mode keeps the capture's own isolation ({groups})")

    with bpy.context.temp_override(**ov):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.reveal()
        bpy.ops.object.mode_set(mode="OBJECT")
    s.active_index = [c.name for c in s.captures].index(item.name)
    real = bridge._send_to_recycle_bin
    bridge._send_to_recycle_bin = lambda files: [os.remove(f) for f in files]
    try:
        call(bpy.ops.paint_bridge.delete, "EXEC_DEFAULT")
    finally:
        bridge._send_to_recycle_bin = real
    check(not os.path.exists(body_sub) and not os.path.exists(paths["psd"]), "delete also removes per-part PSDs")


def run_refresh_test(s, window, area, region, bridge, psd):
    """Merging one capture redraws the other captures' layers, keeping what was painted on them."""
    from mathutils import Euler
    obj = bpy.data.objects["Sphere"]
    bpy.context.view_layer.objects.active = obj
    if obj.mode != "OBJECT":
        with bpy.context.temp_override(window=window, area=area, region=region):
            bpy.ops.object.mode_set(mode="OBJECT")
    s.split_parts = True
    s.auto_mirror = False
    s.refresh_others = True
    rv3d = area.spaces.active.region_3d
    rv3d.view_perspective = "PERSP"
    rv3d.view_location = (0, 0, 0)
    rv3d.view_distance = 4.0

    def capture_at(rot):
        rv3d.view_rotation = Euler(rot).to_quaternion()
        with bpy.context.temp_override(window=window, area=area, region=region):
            bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
        with bpy.context.temp_override(window=window, area=area, region=region):
            bpy.ops.paint_bridge.capture()
        item = s.captures[s.active_index]
        return item.name, bridge.capture_paths(item.directory, item.name)

    a_name, a_paths = capture_at((1.2, 0.0, 0.5))
    b_name, b_paths = capture_at((1.2, 0.0, 1.1))

    doc_b = psd.read_psd(b_paths["psd"])
    base_i = [l.name for l in doc_b.layers].index("Body") - 1
    base_b = canvas_of(doc_b, doc_b.layers[base_i])
    ys, xs = np.nonzero(base_b[:, :, 3] > 0)
    ey, ex = int(ys[len(ys) // 2]), int(xs[len(xs) // 2])
    gy, gx = int(ys[len(ys) // 4]), int(xs[len(xs) // 4])
    edit = np.zeros_like(base_b)
    edit[ey, ex] = (0, 0, 255, 255)
    edit[gy - 3:gy + 4, gx - 3:gx + 4] = (255, 0, 255, 255)
    paint_on(psd, b_paths["psd"], "Base", edit, group="Body")
    layer_count = len(doc_b.layers)

    red = np.zeros((psd.read_psd(a_paths["psd"]).height, psd.read_psd(a_paths["psd"]).width, 4), np.uint8)
    red[:] = (255, 0, 0, 255)
    paint_on(psd, a_paths["psd"], "Base", red, group="Body")
    s.active_index = [c.name for c in s.captures].index(a_name)
    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.paint_bridge.merge()
    doc_b2 = psd.read_psd(b_paths["psd"])
    base2 = canvas_of(doc_b2, doc_b2.layers[base_i])
    reds = ((base2[:, :, 0] > 240) & (base2[:, :, 1] < 15) & (base2[:, :, 3] > 128)).sum()
    check(result == {"FINISHED"} and reds > 1000, f"other capture's layer shows the merged paint ({reds} px)")
    check(tuple(base2[ey, ex, :3]) == (0, 0, 255), "a pixel painted on the other capture is kept")
    block = base2[gy - 3:gy + 4, gx - 3:gx + 4]
    check((block[:, :, 0] > 250).all() and (block[:, :, 2] > 250).all() and (block[:, :, 1] < 5).all()
          and len(doc_b2.layers) == layer_count,
          "the other capture's painted block and layer structure stay")
    item_b = s.captures[[c.name for c in s.captures].index(b_name)]
    check(abs(float(item_b.mtime_text) - os.path.getmtime(b_paths["psd"])) < 1e-3, "refreshing does not trigger Auto Merge")


def run_main_test(s, window, area, region, bridge, psd):
    """Layers added in Edit Main show up in captures; merges go back into the matching main layer."""
    from mathutils import Euler
    pb = sys.modules["paint_bridge"]
    from paint_bridge import main as mainpsd
    obj = bpy.data.objects["Sphere"]
    img = bpy.data.images["BaseTex"]
    bpy.context.view_layer.objects.active = obj
    ov = dict(window=window, area=area, region=region)
    directory = bridge.project_directory(s)
    main_file = mainpsd.main_path(directory, img)

    opened = []
    real_open = pb.open_in_paint_app
    pb.open_in_paint_app = lambda path, context: opened.append(path)
    try:
        with bpy.context.temp_override(**ov):
            result = bpy.ops.paint_bridge.edit_main()
    finally:
        pb.open_in_paint_app = real_open
    check(result == {"FINISHED"} and opened and opened[-1] == main_file, "Edit Main opens the main PSD")

    mdoc = psd.read_psd(main_file)
    layers = [psd.layer_to_dict(mdoc, l) for l in mdoc.layers]
    yellow = np.zeros((mdoc.height, mdoc.width, 4), np.uint8)
    yellow[:, :256] = (255, 255, 0, 255)            # left half of the UV square, top rows first
    layers.insert(len(layers) - 1, {"name": "Detail", "rgba": yellow})
    psd.write_psd(main_file, mdoc.width, mdoc.height, layers, np.zeros((mdoc.height, mdoc.width, 4), np.uint8))
    os.utime(main_file, (os.path.getmtime(main_file) + 5, os.path.getmtime(main_file) + 5))
    first = bridge.changed_mains(s)
    second = bridge.changed_mains(s)
    px = np.empty(len(img.pixels), np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(img.size[1], img.size[0], 4)
    check(first == [] and second == [img] and (px[:, 10, 0] > 0.95).all() and (px[:, 10, 2] < 0.05).all(),
          "saving the main PSD outside Blender updates the texture")

    s.split_parts = True
    rv3d = area.spaces.active.region_3d
    rv3d.view_perspective = "PERSP"
    rv3d.view_location = (0, 0, 0)
    rv3d.view_distance = 4.0
    rv3d.view_rotation = Euler((1.2, 0.0, 0.5)).to_quaternion()
    with bpy.context.temp_override(**ov):
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
    with bpy.context.temp_override(**ov):
        bpy.ops.paint_bridge.capture()
    item = s.captures[s.active_index]
    paths = bridge.capture_paths(item.directory, item.name)
    doc = psd.read_psd(paths["psd"])
    names = [l.name for l in doc.layers]
    check(names == ["</Layer group>", "Base", "Detail", "Box", "</Layer group>", "Base", "Detail", "Body", "Wireframe"],
          f"each part folder holds the main PSD's layers {names}")

    mdoc = psd.read_psd(main_file)
    base_before = canvas_of(mdoc, mdoc.layers[0])
    green = np.zeros((doc.height, doc.width, 4), np.uint8)
    green[:] = (0, 255, 0, 255)
    paint_on(psd, paths["psd"], "Detail", green, group="Body")
    with bpy.context.temp_override(**ov):
        result = bpy.ops.paint_bridge.merge()
    mdoc = psd.read_psd(main_file)
    detail = canvas_of(mdoc, mdoc.layers[[l.name for l in mdoc.layers].index("Detail")])
    painted = (detail[:, :, 1] > 250) & (detail[:, :, 0] < 5) & (detail[:, :, 3] > 250)
    check(result == {"FINISHED"} and painted.sum() > 20000, f"paint on Detail lands on the main Detail layer ({painted.sum()})")
    check(np.array_equal(canvas_of(mdoc, mdoc.layers[0]), base_before), "the main Base layer is untouched")

    paint_on(psd, paths["psd"], "Detail", green, group="Body", erase=True)
    with bpy.context.temp_override(**ov):
        bpy.ops.paint_bridge.merge()
    mdoc = psd.read_psd(main_file)
    detail = canvas_of(mdoc, mdoc.layers[[l.name for l in mdoc.layers].index("Detail")])
    check((detail[:, :, 3][painted] < 5).mean() > 0.95, "erasing on the capture erases the main layer there")

    mdoc = psd.read_psd(main_file)
    layers = [psd.layer_to_dict(mdoc, l) for l in mdoc.layers]
    layers[[l.name for l in mdoc.layers].index("Detail")]["opacity"] = 128
    psd.write_psd(main_file, mdoc.width, mdoc.height, layers, np.zeros((mdoc.height, mdoc.width, 4), np.uint8))
    with bpy.context.temp_override(**ov):
        bpy.ops.paint_bridge.capture()
    item2 = s.captures[s.active_index]
    doc2 = psd.read_psd(bridge.capture_paths(item2.directory, item2.name)["psd"])
    opac = [l.opacity for l in doc2.layers if l.name == "Detail"]
    result = bridge.merge(item2.directory, item2.name, s)
    check(opac and all(o == 128 for o in opac) and result["texels"] == 0,
          f"layer opacity carries over and an untouched capture merges nothing ({opac}, {result['texels']})")

    # Edit Capture: merge unmerged paint, redraw at the capture's own angle, open it.
    s.active_index = [c.name for c in s.captures].index(item2.name)
    cpaths = bridge.capture_paths(item2.directory, item2.name)
    meta_before = bridge.load_meta(item2.directory, item2.name)[0]
    cdoc = psd.read_psd(cpaths["psd"])
    magenta = np.zeros((cdoc.height, cdoc.width, 4), np.uint8)
    magenta[:] = (255, 0, 255, 255)
    paint_on(psd, cpaths["psd"], "Base", magenta, group="Body")
    check(bridge.has_unmerged(item2.directory, item2.name), "unmerged paint is detected")
    rv3d.view_rotation = Euler((0.2, 0.0, 2.8)).to_quaternion()
    with bpy.context.temp_override(**ov):
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
    opened = []
    pb.open_in_paint_app = lambda path, context: opened.append(path)
    try:
        with bpy.context.temp_override(**ov):
            result = bpy.ops.paint_bridge.open_file()
    finally:
        pb.open_in_paint_app = real_open
    meta_after = bridge.load_meta(item2.directory, item2.name)[0]
    mdoc = psd.read_psd(main_file)
    main_base = canvas_of(mdoc, mdoc.layers[0])
    on_main = ((main_base[:, :, 0] > 250) & (main_base[:, :, 1] < 5) & (main_base[:, :, 2] > 250)).sum()
    cdoc2 = psd.read_psd(cpaths["psd"])
    body_base = canvas_of(cdoc2, cdoc2.layers[[l.name for l in cdoc2.layers].index("Body") - 2])
    redrawn = ((body_base[:, :, 0] > 250) & (body_base[:, :, 1] < 5) & (body_base[:, :, 2] > 250)).sum()
    check(result == {"FINISHED"} and on_main > 20000, f"Edit Capture merges unmerged paint first ({on_main} texels)")
    new_rot = list(rv3d.view_rotation)
    check(meta_after["view"] != meta_before["view"]
          and max(abs(a - b) for a, b in zip(meta_after["viewport"]["rotation"], new_rot)) < 1e-4,
          "Edit Capture recaptures from the viewport's current angle")
    check(redrawn > 1000 and opened == [cpaths["psd"]] and not bridge.has_unmerged(item2.directory, item2.name),
          f"the reopened capture shows the latest texture and has nothing left to merge ({redrawn})")


def run_panel_draw_test(window, area):
    """Draw the sidebar on the Paint Bridge tab; draw errors show up in the console log."""
    space = area.spaces.active
    space.show_region_ui = True
    ui = next(r for r in area.regions if r.type == "UI")
    try:
        ui.active_panel_category = "Paint Bridge"
    except (AttributeError, TypeError):
        pass
    with bpy.context.temp_override(window=window, area=area):
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=2)
    results = []
    for name in ("PBRIDGE_PT_mirror_popup", "PBRIDGE_PT_fill_seams_popup", "PBRIDGE_PT_options"):
        region = next(r for r in area.regions if r.type == "WINDOW")
        with bpy.context.temp_override(window=window, area=area, region=region):
            results.append(bpy.ops.wm.call_panel(name=name, keep_open=False))
            bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
    check(all(r == {"INTERFACE"} for r in results),
          f"Mirror..., Fill Seams... and Options popups open {results}")


def run():
    import paint_bridge
    from paint_bridge import bridge, psd

    paint_bridge.register()
    scene = bpy.context.scene
    s = scene.paint_bridge
    s.project_dir = os.path.join(OUT, "project")
    s.split_parts = False

    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=1.0)
    obj = bpy.context.active_object

    img = bpy.data.images.new("BaseTex", 512, 512, alpha=True)
    px = np.zeros((512, 512, 4), np.float32)
    yy, xx = np.mgrid[0:512, 0:512]
    checker = ((xx // 32 + yy // 32) % 2).astype(np.float32)
    px[:, :, 0] = 0.2 + 0.5 * checker
    px[:, :, 1] = 0.6
    px[:, :, 2] = 0.8 - 0.5 * checker
    px[:, :, 3] = 1.0
    img.pixels.foreach_set(px.ravel())
    img.filepath_raw = os.path.join(OUT, "BaseTex.png")
    img.file_format = "PNG"
    img.save()
    original = np.empty(512 * 512 * 4, np.float32)
    img.pixels.foreach_get(original)
    original = original.reshape(512, 512, 4)

    mat = bpy.data.materials.new("Mat")
    tree = mat.node_tree
    bsdf = next(n for n in tree.nodes if n.type == "BSDF_PRINCIPLED")
    tex = tree.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    obj.data.materials.append(mat)

    window = bpy.context.window_manager.windows[0]
    area = next(a for a in window.screen.areas if a.type == "VIEW_3D")
    region = next(r for r in area.regions if r.type == "WINDOW")
    rv3d = area.spaces.active.region_3d
    rv3d.view_perspective = "PERSP"
    rv3d.view_distance = 4.0
    rv3d.view_location = (0, 0, 0)
    from mathutils import Euler
    rv3d.view_rotation = Euler((1.2, 0.0, 0.5)).to_quaternion()
    area.spaces.active.shading.type = "MATERIAL"

    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.paint_bridge.capture()
    check(result == {"FINISHED"}, f"capture operator {result}")
    item = s.captures[0]
    paths = bridge.capture_paths(item.directory, item.name)
    check(os.path.exists(paths["psd"]), "psd written " + paths["psd"])

    doc = psd.read_psd(paths["psd"])
    names = [l.name for l in doc.layers]
    check(names == ["Base", "Wireframe"], f"capture layers mirror the main PSD {names}")
    from paint_bridge import main as mainpsd
    main_file = mainpsd.main_path(item.directory, img)
    mdoc = psd.read_psd(main_file)
    check([l.name for l in mdoc.layers] == ["Base", "UV Wireframe"] and mdoc.layers[1].lock_all
          and (mdoc.width, mdoc.height) == (512, 512), "main PSD created from the texture with a UV wireframe")
    base_alpha = doc.layers[0].pixels[-1]
    coverage = float((base_alpha > 0.5).mean())
    check(0.05 < coverage < 0.95, f"Base layer silhouette coverage {coverage:.3f}")
    log(f"capture size {doc.width}x{doc.height}")

    h, w = doc.height, doc.width
    yy, xx = np.mgrid[0:h, 0:w]
    disc = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) < (min(w, h) * 0.12) ** 2
    base8 = canvas_of(doc, doc.layers[0])
    base8[disc & (base8[:, :, 3] > 0)] = (255, 0, 0, 255)
    write_rle_psd(paths["psd"], w, h, [("Base", base8, True), ("Wireframe", canvas_of(doc, doc.layers[1]), False)])
    doc2 = psd.read_psd(paths["psd"])
    check(np.array_equal(canvas_of(doc2, doc2.layers[0]), base8), "RLE-compressed capture layer round-trips")

    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.paint_bridge.merge()
    check(result == {"FINISHED"}, f"merge operator {result}")
    after = np.empty(512 * 512 * 4, np.float32)
    img.pixels.foreach_get(after)
    after = after.reshape(512, 512, 4)
    red = (after[:, :, 0] > 0.95) & (after[:, :, 1] < 0.05) & (after[:, :, 2] < 0.05)
    check(red.sum() > 500, f"red texels after merge: {int(red.sum())}")
    changed = np.abs(after - original).max(axis=2) > 1e-3
    check(changed.sum() < 512 * 512 * 0.5, f"changed texels {int(changed.sum())} (back side untouched)")
    mdoc = psd.read_psd(main_file)
    main_base = canvas_of(mdoc, mdoc.layers[0])[::-1]
    check(((main_base[:, :, 0] > 240) & (main_base[:, :, 1] < 15))[red].mean() > 0.95,
          "merge writes the paint into the main PSD's Base layer")
    save_png("BaseTex", "after_merge.png")

    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.paint_bridge.merge()
    again = np.empty(512 * 512 * 4, np.float32)
    img.pixels.foreach_get(again)
    check(np.abs(again.reshape(512, 512, 4) - after).max() < 2 / 255, "re-merge is idempotent")
    reloaded = bpy.data.images.load(os.path.join(OUT, "BaseTex.png"))
    disk = np.empty(512 * 512 * 4, np.float32)
    reloaded.pixels.foreach_get(disk)
    check(np.abs(disk.reshape(512, 512, 4) - again.reshape(512, 512, 4)).max() < 1.5 / 255, "master texture file saved")

    with bpy.context.temp_override(window=window, area=area, region=region):
        result = bpy.ops.paint_bridge.revert()
    reverted = np.empty(512 * 512 * 4, np.float32)
    img.pixels.foreach_get(reverted)
    check(result == {"FINISHED"} and np.abs(reverted.reshape(512, 512, 4) - original).max() < 1.5 / 255,
          "revert restores the original texture")
    mdoc = psd.read_psd(main_file)
    check(np.abs(canvas_of(mdoc, mdoc.layers[0])[::-1].astype(int)[:, :, :3]
                 - (original[:, :, :3] * 255 + 0.5).astype(int)).max() <= 1, "revert restores the main PSD too")

    run_parts_test(s, obj, img, window, area, region, bridge, psd)
    run_ortho_test(s, img, window, area, region, bridge, psd)
    run_list_test(s, window, area, region, bridge, psd)
    run_mirror_test(s, bridge)
    run_auto_mirror_test(s, window, area, region, bridge, psd)
    run_isolation_test(s, window, area, region, bridge, psd)
    run_refresh_test(s, window, area, region, bridge, psd)
    run_main_test(s, window, area, region, bridge, psd)
    run_panel_draw_test(window, area)
    paint_bridge.unregister()


def main():
    try:
        run()
    except Exception:
        log("FAIL exception\n" + traceback.format_exc())
    with open(os.path.join(OUT, "results.txt"), "w") as f:
        f.write("\n".join(log_lines) + "\n")
    bpy.ops.wm.quit_blender()


bpy.app.timers.register(main, first_interval=1.0)
