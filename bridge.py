# SPDX-License-Identifier: GPL-3.0-or-later
"""Capture a view to PSD and merge an edited PSD back into the base color texture."""

import json
import math
import os
import re
import sys
import time

import bpy
import numpy as np
from mathutils import Matrix

from . import main as mainpsd
from . import parts, psd
from .mirror import MirrorMap
from .mirror import _rest_triangles as rest_triangles
from .render import (GEO_KEYS, BridgeError, extract_geometry, pad_gutter, project_paint, render_capture,
                     render_layer, uv_coverage, uv_edges)

REFERENCE_LAYER = "Blender Reference"
PAINT_LAYER = "Paint"
WIREFRAME_LAYER = "Wireframe"
BASE_LAYER = "Base"
GROUP_END = "</Layer group>"
MAX_CAPTURE_SIZE = 8192


# ---------------------------------------------------------------- material lookup

def _upstream_image_node(socket, depth=0):
    if depth > 24 or not socket.is_linked:
        return None
    for link in socket.links:
        node = link.from_node
        if node.type == "TEX_IMAGE" and node.image is not None:
            return node
        for inp in node.inputs:
            found = _upstream_image_node(inp, depth + 1)
            if found is not None:
                return found
    return None


def base_color_node(material):
    tree = getattr(material, "node_tree", None) if material else None
    if tree is None:
        return None
    for node in tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            socket = node.inputs.get("Base Color")
        elif node.type == "BSDF_DIFFUSE":
            socket = node.inputs.get("Color")
        else:
            continue
        found = _upstream_image_node(socket) if socket else None
        if found is not None:
            return found
    return None


def _uv_of_node(node):
    vector = node.inputs.get("Vector")
    if vector is not None and vector.is_linked:
        src = vector.links[0].from_node
        if src.type == "UVMAP" and src.uv_map:
            return src.uv_map
    return ""


def resolve_source(obj, override_image=None):
    """Return (image, uv_name, material_indices or None) for an object's base color texture."""
    if obj is None or obj.type != "MESH":
        raise BridgeError("Select a mesh object")
    image, uv_name, indices = override_image, "", set()
    for index, slot in enumerate(obj.material_slots):
        node = base_color_node(slot.material)
        if node is None:
            continue
        if image is None:
            image = node.image
        if node.image == image:
            indices.add(index)
            uv_name = uv_name or _uv_of_node(node)
    if image is None:
        raise BridgeError("No Base Color image texture found in the object's materials")
    return image, uv_name, (indices or None)


# ---------------------------------------------------------------- image io

def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((np.maximum(c, 0.0) + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.maximum(c, 0.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1.0 / 2.4) - 0.055)


def _is_linear(image):
    return image.is_float and not image.colorspace_settings.is_data


def read_image(image):
    """Image pixels as display-referred RGBA float32, bottom row first."""
    w, h = image.size
    if w == 0 or h == 0:
        image.reload()
        w, h = image.size
    if w == 0 or h == 0:
        path = os.path.normpath(bpy.path.abspath(image.filepath)) if image.filepath else ""
        if path and not os.path.exists(path):
            raise BridgeError(f"Image file not found: {path}")
        raise BridgeError(f"Image '{image.name}' could not be loaded ({path or 'no file path'})")
    ch = image.channels
    if ch not in (3, 4):
        raise BridgeError(f"Image '{image.name}' must be RGB or RGBA")
    px = np.empty(w * h * ch, np.float32)
    image.pixels.foreach_get(px)
    px = px.reshape(h, w, ch)
    if ch == 3:
        px = np.dstack([px, np.ones((h, w, 1), np.float32)])
    if _is_linear(image):
        px[:, :, :3] = _linear_to_srgb(px[:, :, :3])
    return px


def write_image(image, rgba):
    out = rgba.copy()
    if _is_linear(image):
        out[:, :, :3] = _srgb_to_linear(out[:, :, :3])
    else:
        np.clip(out, 0.0, 1.0, out=out)
    out = out[:, :, :image.channels]
    image.pixels.foreach_set(np.ascontiguousarray(out, dtype=np.float32).ravel())
    image.update()


def image_missing(image):
    return image.source == "FILE" and image.packed_file is None and image.size[0] == 0


def create_texture_file(image, width, height, color):
    """Write a solid-color PNG to //textures/ and point `image` at it, keeping every material link."""
    blend_dir = os.path.dirname(bpy.data.filepath)
    if not blend_dir:
        raise BridgeError("Save the .blend file first")
    folder = os.path.join(blend_dir, "textures")
    os.makedirs(folder, exist_ok=True)
    stem = os.path.splitext(bpy.path.basename(image.filepath) or image.name)[0]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem) or "BaseColor"
    path = os.path.join(folder, stem + ".png")
    n = 1
    while os.path.exists(path):
        path = os.path.join(folder, f"{stem}_{n:02d}.png")
        n += 1

    tmp = bpy.data.images.new("__paint_bridge_new", width, height, alpha=True)
    try:
        px = np.empty((height, width, 4), np.float32)
        px[:, :] = (color[0], color[1], color[2], 1.0)
        tmp.pixels.foreach_set(px.ravel())
        tmp.filepath_raw = path
        tmp.file_format = "PNG"
        tmp.save()
    finally:
        bpy.data.images.remove(tmp)

    image.source = "FILE"
    image.filepath = bpy.path.relpath(path)
    image.colorspace_settings.name = "sRGB"
    image.reload()
    return path


def backup_image(image, settings):
    """Save the image's current pixels to <project>/backup/ and return the file path."""
    folder = os.path.join(project_directory(settings), "backup")
    os.makedirs(folder, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.splitext(image.name)[0]) or "texture"
    ext, fmt = (".exr", "OPEN_EXR") if image.is_float else (".png", "PNG")
    path = os.path.join(folder, f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}{ext}")
    w, h = image.size
    px = np.empty(w * h * image.channels, np.float32)
    image.pixels.foreach_get(px)
    px = px.reshape(h, w, image.channels)
    if image.channels == 3:
        px = np.dstack([px, np.ones((h, w, 1), np.float32)])
    tmp = bpy.data.images.new("__paint_bridge_backup", w, h, alpha=True, float_buffer=image.is_float)
    try:
        tmp.pixels.foreach_set(np.ascontiguousarray(px).ravel())
        tmp.filepath_raw = path
        tmp.file_format = fmt
        tmp.save()
    finally:
        bpy.data.images.remove(tmp)
    return path


def save_master(image):
    """Write the image back to where it lives; returns a short status or None."""
    if image.packed_file is not None:
        image.pack()
        return "packed into .blend"
    if image.source == "FILE" and image.filepath:
        image.save()
        return "saved to " + bpy.path.abspath(image.filepath)
    return None


# ---------------------------------------------------------------- capture files

def project_directory(settings, create=True):
    path = bpy.path.abspath(settings.project_dir or "//paint_bridge/")
    if path.startswith("//") or not os.path.isabs(path):
        raise BridgeError("Save the .blend file first, or set an absolute Project Folder")
    path = os.path.normpath(path)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def list_captures(directory):
    """Stems of the PSDs in `directory` that have capture data next to them, sorted by name."""
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    found = set(names)
    return sorted((n[:-4] for n in names
                   if n.lower().endswith(".psd") and n[:-4] + ".json" in found), key=str.lower)


def capture_paths(directory, stem):
    base = os.path.join(directory, stem)
    return {
        "psd": base + ".psd",
        "meta": base + ".json",
        "geo": base + ".geo.npz",
        "ref": base + ".ref.npz",
        "applied": base + ".applied.npz",
    }


def _unique_stem(directory, obj, image, tag=""):
    label = tag or os.path.splitext(image.name)[0]
    clean = re.sub(r"[^A-Za-z0-9_-]+", "_", f"{obj.name}_{label}").strip("_")
    n = 1
    while os.path.exists(os.path.join(directory, f"{clean}_{n:03d}.psd")):
        n += 1
    return f"{clean}_{n:03d}"


def _viewport_state(space, scene):
    rv3d = space.region_3d
    return {
        "mode": rv3d.view_perspective,
        "location": list(rv3d.view_location),
        "rotation": list(rv3d.view_rotation),
        "distance": rv3d.view_distance,
        "lens": space.lens,
        "camera": scene.camera.name if scene.camera else "",
    }


def snap_view(space, scene, meta):
    """Put a 3D View back to the angle a capture was taken from."""
    rv3d = space.region_3d
    state = meta.get("viewport")
    if state is None:
        rv3d.view_perspective = "PERSP" if meta["perspective"] else "ORTHO"
        rv3d.view_matrix = Matrix(meta["view"])
        return
    camera = bpy.data.objects.get(state["camera"]) if state["camera"] else None
    if state["mode"] == "CAMERA" and camera is not None:
        scene.camera = camera
        rv3d.view_perspective = "CAMERA"
        return
    rv3d.view_perspective = "PERSP" if state["mode"] == "CAMERA" else state["mode"]
    rv3d.view_location = state["location"]
    rv3d.view_rotation = state["rotation"]
    rv3d.view_distance = state["distance"]
    space.lens = state["lens"]


def _view_from_context(context, scale):
    space = context.space_data
    if space is None or space.type != "VIEW_3D":
        raise BridgeError("Run from a 3D Viewport")
    rv3d = space.region_3d
    scene = context.scene
    if rv3d.view_perspective == "CAMERA" and scene.camera is not None:
        cam = scene.camera
        if cam.type == "CAMERA" and cam.data.type == "PANO":
            raise BridgeError("Panoramic cameras are not supported")
        r = scene.render
        w = max(4, int(r.resolution_x * r.resolution_percentage / 100))
        h = max(4, int(r.resolution_y * r.resolution_percentage / 100))
        depsgraph = context.evaluated_depsgraph_get()
        proj = cam.calc_matrix_camera(depsgraph, x=w, y=h, scale_x=r.pixel_aspect_x, scale_y=r.pixel_aspect_y)
        view = cam.matrix_world.inverted()
        persp = not (cam.type == "CAMERA" and cam.data.type == "ORTHO")
    else:
        region = next((r for r in context.area.regions if r.type == "WINDOW"), None)
        if region is None:
            raise BridgeError("No viewport region found")
        w, h = int(region.width * scale), int(region.height * scale)
        view, proj, persp = rv3d.view_matrix, rv3d.window_matrix, rv3d.is_perspective
    longest = max(w, h)
    if longest > MAX_CAPTURE_SIZE:
        f = MAX_CAPTURE_SIZE / longest
        w, h = int(w * f), int(h * f)
    return [list(row) for row in view], [list(row) for row in proj], bool(persp), max(w, 4), max(h, 4)


def _part_boxes(ids, names):
    """(part name, (top, left, bottom, right), inside mask) for each part seen in a part-id render."""
    h, w = ids.shape
    fg = np.flatnonzero(ids.ravel())
    if fg.size == 0:
        return []
    pid = ids.ravel()[fg]
    ys, xs = fg // w, fg % w
    count = int(pid.max()) + 1
    top = np.full(count, h)
    left = np.full(count, w)
    bottom = np.zeros(count, np.int64)
    right = np.zeros(count, np.int64)
    np.minimum.at(top, pid, ys)
    np.minimum.at(left, pid, xs)
    np.maximum.at(bottom, pid, ys + 1)
    np.maximum.at(right, pid, xs + 1)

    boxes = []
    for p in np.unique(pid):
        t, l, b, r = int(top[p]), int(left[p]), int(bottom[p]), int(right[p])
        name = names[p - 1] if 0 < p <= len(names) else f"Part {p:02d}"
        boxes.append((name, (t, l, b, r), ids[t:b, l:r] == p))
    boxes.sort(key=lambda box: box[0].lower(), reverse=True)
    return boxes


def _group(name, rect, inside, members):
    t, l, b, r = rect
    return ([{"name": GROUP_END, "rgba": None, "section": 3}] + members +
            [{"name": name, "rgba": None, "section": 2,
              "mask": np.where(inside, 255, 0).astype(np.uint8), "mask_rect": rect}])


def _part_layers(ref8, ids, names):
    """A closed group folder per visible part, masked to its silhouette, holding Base and Paint layers."""
    layers = []
    for name, (t, l, b, r), inside in _part_boxes(ids, names):
        crop = ref8[t:b, l:r].copy()
        crop[:, :, 3] = np.where(inside, 255, 0)
        layers += _group(name, (t, l, b, r), inside, [
            {"name": BASE_LAYER, "rgba": crop, "rect": (t, l, b, r)},
            {"name": PAINT_LAYER, "rgba": None}])
    return layers


def _main_layers(renders, ids, names, split):
    """Capture layers that mirror the main PSD's layers, one folder per part when `split`."""
    if not split:
        return [{"name": layer.name, "rgba": r8, "opacity": layer.opacity, "hidden": layer.hidden}
                for layer, r8 in renders]
    out = []
    for name, (t, l, b, r), inside in _part_boxes(ids, names):
        members = []
        for layer, r8 in renders:
            crop = r8[t:b, l:r].copy()
            crop[:, :, 3] = np.where(inside, crop[:, :, 3], 0)
            members.append({"name": layer.name, "rgba": crop, "rect": (t, l, b, r),
                            "opacity": layer.opacity, "hidden": layer.hidden})
        out += _group(name, (t, l, b, r), inside, members)
    return out


def _wire_layer(wire):
    rows = np.flatnonzero(wire.max(axis=1) > 0)
    cols = np.flatnonzero(wire.max(axis=0) > 0)
    if rows.size == 0:
        return None
    t, b, l, r = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    crop = np.zeros((b - t, r - l, 4), np.uint8)
    crop[:, :, 3] = (np.clip(wire[t:b, l:r], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return {"name": WIREFRAME_LAYER, "rgba": crop, "rect": (t, l, b, r), "opacity": 140, "locked": True}


def clean_name(name):
    """A capture name usable as a file name on every platform, without the .psd extension."""
    name = name.strip()
    if name.lower().endswith(".psd"):
        name = name[:-4]
    return re.sub(r'[\\/:*?"<>|]+', "_", name).strip(" .")


def default_name(context, settings):
    obj = context.active_object
    image, _uv, _idx = resolve_source(obj, settings.image_override)
    return _unique_stem(project_directory(settings), obj, image, parts.isolation(obj)[1])


def _isolation_of(obj, meta, paths):
    """(vertex mask, face mask, description) a capture was isolated to, read back for Recapture."""
    desc = meta.get("isolate") or {"kind": parts.ALL}
    kind = desc["kind"]
    if kind == parts.VISIBLE:
        data = np.load(paths["geo"])
        mask = np.zeros(int(data["iso_face_count"]), bool)
        mask[data["iso_faces"]] = True
        return None, mask, desc
    if kind == parts.PART:
        labels, names = parts.part_info(obj)
        if desc["part"] not in names:
            raise BridgeError(f"Part '{desc['part']}' is not on '{obj.name}'")
        return labels == names.index(desc["part"]), None, desc
    if kind == parts.SELECTION:
        verts = np.load(paths["geo"])["iso_verts"]
        mask = np.zeros(len(obj.data.vertices), bool)
        mask[verts[verts < len(mask)]] = True
        return mask, None, desc
    return None, None, desc


def capture(context, settings, stem=None, name=None, keep_view=False):
    """Render the current view and write the PSD with its sidecar files. Returns (stem, directory).

    With `stem`, the capture of that name is overwritten for its own object, image and isolation.
    With `name`, a new capture is written under that name. In Edit Mode, faces hidden in the
    viewport are left out, so they neither show nor block the rest. With `keep_view`, a recapture
    reuses the capture's own camera and size instead of the viewport's.
    """
    directory = project_directory(settings)
    if name:
        name = clean_name(name)
        if not name:
            raise BridgeError("Enter a name for the capture")
        if os.path.exists(capture_paths(directory, name)["psd"]):
            raise BridgeError(f"'{name}.psd' already exists; choose another name or use Recapture")
    if stem:
        meta, _paths = load_meta(directory, stem)
        obj = bpy.data.objects.get(meta["object"])
        image = bpy.data.images.get(meta["image"])
        if obj is None or image is None:
            raise BridgeError(f"Object '{meta['object']}' or image '{meta['image']}' is not in this .blend")
        keep_verts, keep_faces, iso_desc = _isolation_of(obj, meta, _paths)
        tag = ""
    else:
        obj, image = context.active_object, settings.image_override
        keep_faces, tag, iso_desc = parts.isolation(obj)
        keep_verts = None
    image, uv_name, indices = resolve_source(obj, image)
    if keep_view and stem:
        view, proj, persp, w, h = meta["view"], meta["proj"], meta["perspective"], meta["width"], meta["height"]
        viewport = meta.get("viewport")
    else:
        view, proj, persp, w, h = _view_from_context(context, settings.capture_scale)
        viewport = _viewport_state(context.space_data, context.scene)

    geo = extract_geometry(obj, context.evaluated_depsgraph_get(), uv_name, indices, with_parts=True,
                           keep_verts=keep_verts, keep_faces=keep_faces)
    base = read_image(image)
    ref, ids, wire = render_capture(geo, view, proj, w, h, base)
    ref8 = (np.clip(ref, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    ref8[:, :, 3] = np.where(ids > 0, 255, 0)

    main_file = ensure_main(settings, image, obj)
    mdoc = psd.read_psd(main_file)
    if (mdoc.width, mdoc.height) != tuple(image.size):
        raise BridgeError(f"{os.path.basename(main_file)} is {mdoc.width}x{mdoc.height} but the texture is "
                          f"{image.size[0]}x{image.size[1]}")
    renders = []
    for _index, layer in mainpsd.leaf_layers(mdoc):
        rgba = render_layer(geo, view, proj, w, h, mainpsd.layer_canvas(mdoc, layer)[::-1])
        renders.append((layer, (np.clip(rgba, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)))
    if not renders:
        raise BridgeError(f"{os.path.basename(main_file)} has no paintable layers")
    layers = _main_layers(renders, ids, geo["part_names"], settings.split_parts)
    layout = "main"
    if settings.wireframe_layer and wire is not None:
        wire_layer = _wire_layer(wire)
        if wire_layer:
            layers.append(wire_layer)

    stem = stem or name or _unique_stem(directory, obj, image, tag)
    paths = capture_paths(directory, stem)
    tmp = paths["psd"] + ".tmp"
    psd.write_psd(tmp, w, h, layers, ref8)
    try:
        os.replace(tmp, paths["psd"])
    except OSError:
        os.remove(tmp)
        raise BridgeError(f"Could not overwrite {paths['psd']}; close it in other apps and try again")
    for stale in [paths["applied"]] + sub_psds(directory, stem):
        if os.path.exists(stale):
            os.remove(stale)
    extra = {}
    if keep_faces is not None:
        extra = {"iso_faces": np.flatnonzero(keep_faces), "iso_face_count": np.array(len(keep_faces))}
    elif keep_verts is not None:
        extra = {"iso_verts": np.flatnonzero(keep_verts)}
    np.savez_compressed(paths["geo"], **{k: geo[k] for k in GEO_KEYS}, **extra)
    np.savez_compressed(paths["ref"], ref=ref8, **{f"layer_{k}": r8 for k, (_l, r8) in enumerate(renders)})
    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump({
            "version": 1,
            "layout": layout,
            "main": os.path.basename(main_file),
            "layers": [layer.name for layer, _r8 in renders],
            "isolate": iso_desc,
            "object": obj.name,
            "image": image.name,
            "uv": uv_name,
            "view": view,
            "proj": proj,
            "perspective": persp,
            "viewport": viewport,
            "width": w,
            "height": h,
            "texture_size": list(image.size),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=1)
    return stem, directory


def sub_psds(directory, stem):
    """Per-part PSDs written for a capture by Open File, oldest first."""
    prefix = stem + "@"
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    found = [os.path.join(directory, n) for n in names if n.startswith(prefix) and n.lower().endswith(".psd")]
    return sorted(found, key=os.path.getmtime)


def part_psd(directory, stem, part_names):
    """Path of a PSD holding only the given parts' folders of a capture, written on first use."""
    meta, paths = load_meta(directory, stem)
    if meta.get("layout") not in ("groups", "main"):
        raise BridgeError("This capture has no part folders; turn on Layer per Part and recapture it")
    tag = clean_name("+".join(part_names))[:80] or "parts"
    path = os.path.join(directory, f"{stem}@{tag}.psd")
    if os.path.exists(path):
        return path
    doc = psd.read_psd(paths["psd"])
    wanted = set(part_names)
    layers = []
    for header, start, group_name in psd.group_spans(doc):
        if group_name in wanted:
            layers += [psd.layer_to_dict(doc, doc.layers[i]) for i in range(start, header + 1)]
    if not layers:
        raise BridgeError(f"{', '.join(part_names)} is not in capture '{stem}'")
    layers += [psd.layer_to_dict(doc, l) for l in doc.layers if l.name == WIREFRAME_LAYER]
    ref8 = np.load(paths["ref"])["ref"]
    psd.write_psd(path, doc.width, doc.height, layers, ref8)
    return path


def load_meta(directory, stem):
    paths = capture_paths(directory, stem)
    if not os.path.exists(paths["meta"]):
        raise BridgeError(f"Capture data missing: {paths['meta']}")
    with open(paths["meta"], encoding="utf-8") as f:
        return json.load(f), paths


# ---------------------------------------------------------------- merge

def _paint_from_psd(path, meta, paths):
    """Premultiplied paint (top row first) and a list of warnings."""
    warnings = []
    skip = (REFERENCE_LAYER, WIREFRAME_LAYER)
    doc = psd.read_psd(path, decode_filter=lambda l: l.name not in skip)
    if (doc.width, doc.height) != (meta["width"], meta["height"]):
        raise BridgeError(f"PSD canvas is {doc.width}x{doc.height}, capture was "
                          f"{meta['width']}x{meta['height']}; do not crop or resize it")
    if meta.get("layout") == "groups":
        return _paint_from_groups(doc, paths)
    if any(l.name == REFERENCE_LAYER for l in doc.layers):
        paint, odd = psd.flatten_layers(doc, lambda l: l.name not in skip)
        if odd:
            warnings.append("Blend modes treated as Normal: " + ", ".join(odd))
        return paint, warnings

    # Part layers, or a flattened file: keep only pixels that differ from the captured render.
    if doc.layers:
        comp, odd = psd.flatten_layers(doc, lambda l: l.name != WIREFRAME_LAYER)
        if odd:
            warnings.append("Blend modes treated as Normal: " + ", ".join(odd))
        alpha = comp[:, :, 3:4]
        rgb = np.where(alpha > 0, comp[:, :, :3] / np.maximum(alpha, 1e-6), 0.0)
    elif doc.composite is not None:
        rgb, alpha = doc.composite[:, :, :3], doc.composite[:, :, 3:4]
    else:
        raise BridgeError("PSD has no readable image data")
    ref = np.load(paths["ref"])["ref"].astype(np.float32) / 255.0
    changed = (np.abs(rgb - ref[:, :, :3]).max(axis=2) > 0.75 / 255.0)[:, :, None]
    a = alpha * changed
    if meta.get("layout") != "parts":
        warnings.append(f"'{REFERENCE_LAYER}' layer not found; merged only pixels that changed")
    return np.dstack([rgb * a, a]).astype(np.float32), warnings


def _paint_from_groups(doc, paths):
    """Paint layers as they are (transparent stays transparent), plus any direct edits to Base layers."""
    warnings = []
    paint, odd = psd.flatten_layers(doc, lambda l: l.name not in (BASE_LAYER, WIREFRAME_LAYER))
    base, odd_base = psd.flatten_layers(doc, lambda l: l.name == BASE_LAYER)
    for mode in odd + odd_base:
        if mode not in odd:
            odd.append(mode)
    if odd:
        warnings.append("Blend modes treated as Normal: " + ", ".join(odd))
    alpha = base[:, :, 3:4]
    rgb = np.where(alpha > 0, base[:, :, :3] / np.maximum(alpha, 1e-6), 0.0)
    ref = np.load(paths["ref"])["ref"].astype(np.float32) / 255.0
    edited = ((np.abs(rgb - ref[:, :, :3]).max(axis=2) > 0.75 / 255.0) & (alpha[:, :, 0] > 0.5))[:, :, None]
    edits = np.dstack([rgb * edited, edited.astype(np.float32)])
    return (paint + edits * (1.0 - paint[:, :, 3:4])).astype(np.float32), warnings


def _restore_applied(base, paths):
    if not os.path.exists(paths["applied"]):
        return False
    data = np.load(paths["applied"])
    if tuple(data["shape"]) != base.shape:
        return False
    base.reshape(-1, 4)[data["idx"]] = data["orig"]
    return True


def merge(directory, stem, settings):
    """Project the PSD paint onto the texture. Re-merging replaces this capture's previous result."""
    meta, paths = load_meta(directory, stem)
    image = bpy.data.images.get(meta["image"])
    if image is None:
        raise BridgeError(f"Image '{meta['image']}' is not in this .blend")
    if not os.path.exists(paths["psd"]):
        raise BridgeError(f"PSD not found: {paths['psd']}")
    if meta.get("layout") == "main":
        return _merge_main(directory, stem, meta, paths, settings, image)
    if os.path.exists(mainpsd.main_path(directory, image)):
        raise BridgeError(f"'{stem}' was captured before the main PSD existed; Recapture it to merge "
                          "into the main PSD")

    paint, warnings = _paint_from_psd(paths["psd"], meta, paths)
    for sub in sub_psds(directory, stem):
        doc = psd.read_psd(sub, decode_filter=lambda l: l.name != WIREFRAME_LAYER)
        if (doc.width, doc.height) != (meta["width"], meta["height"]):
            raise BridgeError(f"{os.path.basename(sub)} was cropped or resized; it cannot be merged")
        sub_paint, sub_warnings = _paint_from_groups(doc, paths)
        paint = sub_paint + paint * (1.0 - sub_paint[:, :, 3:4])
        warnings += [w for w in sub_warnings if w not in warnings]
    geo = dict(np.load(paths["geo"]))
    base = read_image(image)
    _restore_applied(base, paths)
    tex_h, tex_w = base.shape[:2]

    contrib = project_paint(
        geo, meta["view"], meta["proj"], meta["perspective"], paint, tex_w, tex_h,
        math.cos(settings.fade_start), math.cos(settings.fade_end))

    if settings.auto_mirror:
        obj = bpy.data.objects.get(meta["object"])
        if obj is None:
            raise BridgeError(f"Object '{meta['object']}' is not in this .blend; turn off Auto Mirror")
        _img, uv_name, indices = resolve_source(obj, image)
        mmap = MirrorMap(obj, uv_name, indices, "XYZ".index(settings.mirror_axis))
        mirrored, _mask = mmap.both_ways(contrib)
        contrib = contrib + mirrored * (1.0 - contrib[:, :, 3:4])
    flat_base = base.reshape(-1, 4)
    flat_add = contrib.reshape(-1, 4)
    idx = np.flatnonzero(flat_add[:, 3] > 1e-4).astype(np.int64)
    np.savez_compressed(paths["applied"], idx=idx, orig=flat_base[idx].copy(), shape=np.array(base.shape))
    a = flat_add[idx, 3:4]
    flat_base[idx, :3] = flat_base[idx, :3] * (1.0 - a) + flat_add[idx, :3]
    base = fill_seams(bpy.data.objects.get(meta["object"]), image, base, settings.seam_bleed)
    write_image(image, base)

    saved = save_master(image) if settings.save_master else None
    return {"texels": int(idx.size), "warnings": warnings, "saved": saved, "image": image}


def revert(directory, stem, settings):
    meta, paths = load_meta(directory, stem)
    image = bpy.data.images.get(meta["image"])
    if image is None:
        raise BridgeError(f"Image '{meta['image']}' is not in this .blend")
    if meta.get("layout") == "main":
        _revert_main(directory, meta, paths, settings, image)
        return image
    base = read_image(image)
    if not _restore_applied(base, paths):
        raise BridgeError("Nothing to revert for this capture")
    base = fill_seams(bpy.data.objects.get(meta["object"]), image, base, settings.seam_bleed)
    write_image(image, base)
    os.remove(paths["applied"])
    if settings.save_master:
        save_master(image)
    return image


def fill_seams(obj, image, base, steps):
    """Pad the gutter around every UV island of `obj` that uses `image` with the nearest island color."""
    if steps <= 0 or obj is None or obj.type != "MESH":
        return base
    _img, uv_name, indices = resolve_source(obj, image)
    _co, _verts, tri_uv = rest_triangles(obj, uv_name, indices)
    h, w = base.shape[:2]
    cover = uv_coverage(tri_uv.reshape(-1, 2).astype(np.float32), w, h)
    return pad_gutter(base, cover, steps)


def _send_to_recycle_bin(paths):
    import ctypes
    from ctypes import wintypes

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT), ("pFrom", wintypes.LPCWSTR),
                    ("pTo", wintypes.LPCWSTR), ("fFlags", ctypes.c_ushort),
                    ("fAnyOperationsAborted", wintypes.BOOL), ("hNameMappings", ctypes.c_void_p),
                    ("lpszProgressTitle", wintypes.LPCWSTR)]

    fo_delete, flags = 3, 0x40 | 0x10 | 0x04 | 0x400  # allow undo, no confirmation, silent, no error UI
    op = SHFILEOPSTRUCTW(None, fo_delete, "\0".join(paths) + "\0", None, flags, False, None, None)
    if ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op)) != 0 or op.fAnyOperationsAborted:
        raise OSError("Could not move the files to the Recycle Bin")


def delete_capture(directory, stem):
    """Delete a capture's PSD and its sidecar files (to the Recycle Bin on Windows). Returns the file count."""
    files = [p for p in capture_paths(directory, stem).values() if os.path.exists(p)]
    files += sub_psds(directory, stem)
    if not files:
        return 0
    if sys.platform == "win32":
        _send_to_recycle_bin([os.path.abspath(p) for p in files])
    else:
        for path in files:
            os.remove(path)
    return len(files)


def refresh_capture(directory, stem, image):
    """Re-render a capture's Base layers from the current texture, keeping all paint and edits.

    Only captures of `image` are touched. Returns how many PSD files changed.
    """
    meta, paths = load_meta(directory, stem)
    if meta["image"] != image.name or not os.path.exists(paths["psd"]):
        return 0
    if meta.get("layout") == "main":
        return _refresh_main(directory, stem, meta, paths)
    geo = dict(np.load(paths["geo"]))
    old8 = np.load(paths["ref"])["ref"]
    rgba, _ids, _wire = render_capture(geo, meta["view"], meta["proj"], meta["width"], meta["height"],
                                       read_image(image))
    new8 = (np.clip(rgba, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    new8[:, :, 3] = old8[:, :, 3]
    names = {BASE_LAYER, REFERENCE_LAYER}
    changed = 0
    for path in [paths["psd"]] + sub_psds(directory, stem):
        if psd.refresh_layers(path, names, old8, new8):
            changed += 1
    np.savez_compressed(paths["ref"], ref=new8)
    return changed


def is_merged(directory, stem):
    return os.path.exists(capture_paths(directory, stem)["applied"])


# ---------------------------------------------------------------- main PSD

_main_seen = {}
_main_pending = {}


def _remember_main(path):
    try:
        _main_seen[path] = os.path.getmtime(path)
    except OSError:
        pass


def find_object_for_image(image):
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        try:
            found, _uv, _idx = resolve_source(obj, None)
        except BridgeError:
            continue
        if found == image:
            return obj
    return None


def ensure_main(settings, image, obj=None):
    """Path of the image's main PSD, written from the current texture the first time."""
    path = mainpsd.main_path(project_directory(settings), image)
    if not os.path.exists(path):
        rgba = read_image(image)
        wire = None
        obj = obj or find_object_for_image(image)
        if obj is not None:
            try:
                _img, uv_name, indices = resolve_source(obj, image)
                _co, _verts, tri_uv = rest_triangles(obj, uv_name, indices)
                wire = uv_edges(tri_uv.reshape(-1, 2).astype(np.float32), rgba.shape[1], rgba.shape[0])
            except BridgeError:
                wire = None
        mainpsd.create(path, image.name, rgba, wire)
    if path not in _main_seen:
        _remember_main(path)
    return path


def apply_main(path, image, settings, obj=None):
    """Rebuild the texture from the main PSD. Returns the save status."""
    doc = psd.read_psd(path, decode_filter=lambda l: l.name != mainpsd.UV_WIRE_LAYER)
    base = read_image(image)
    if (doc.height, doc.width) != base.shape[:2]:
        raise BridgeError(f"{os.path.basename(path)} is {doc.width}x{doc.height} but the texture is "
                          f"{base.shape[1]}x{base.shape[0]}")
    comp = mainpsd.flatten(doc)[::-1]
    a = comp[:, :, 3:4]
    base[:, :, :3] = base[:, :, :3] * (1.0 - a) + comp[:, :, :3]
    base = fill_seams(obj or find_object_for_image(image), image, base, settings.seam_bleed)
    write_image(image, base)
    _remember_main(path)
    return save_master(image) if settings.save_master else None


def changed_mains(settings):
    """Images whose main PSD was saved outside Blender since it was last read, after applying them."""
    try:
        directory = project_directory(settings, create=False)
    except BridgeError:
        return []
    applied = []
    for path in mainpsd.list_mains(directory):
        try:
            mtime, size = os.path.getmtime(path), os.path.getsize(path)
        except OSError:
            continue
        seen = _main_seen.get(path)
        if seen is None:
            _main_seen[path] = mtime
            continue
        if mtime <= seen + 1e-3:
            continue
        if _main_pending.get(path) != (mtime, size):
            _main_pending[path] = (mtime, size)
            continue
        _main_pending.pop(path, None)
        image = bpy.data.images.get(mainpsd.image_of(path))
        if image is None:
            _main_seen[path] = mtime
            continue
        apply_main(path, image, settings)
        applied.append(image)
    return applied


def _layer_changes(doc, names, refs, only_groups):
    """Per main-layer name: (premultiplied new pixels where changed, change mask) in one capture PSD.

    With `only_groups`, changes count only inside the PSD's own folders (a per-part PSD).
    """
    tree = psd.layer_tree(doc)
    region = None
    if only_groups:
        region = np.zeros((doc.height, doc.width), bool)
        for header, _start, _name in psd.group_spans(doc):
            mask = psd._group_mask(doc, doc.layers[header])
            if mask is not None:
                region |= mask > 0.5
    out = {}
    for k, name in enumerate(names):
        members = [i for i, l in enumerate(doc.layers) if l.section == 0 and l.name == name]
        if not members or not all(tree.get(i, (False, ()))[0] for i in members):
            continue
        current, _odd = psd.flatten_layers(doc, lambda l, n=name: l.name == n, opacity=False)
        ref = refs[k].astype(np.float32) / 255.0
        ref[:, :, :3] *= ref[:, :, 3:4]
        changed = np.abs(current - ref).max(axis=2) > 1.5 / 255.0
        if region is not None:
            changed &= region
        if changed.any():
            out[name] = (current * changed[:, :, None], changed.astype(np.float32))
    return out


def has_unmerged(directory, stem):
    """True when a main-layout capture (or one of its per-part PSDs) holds paint not merged yet."""
    meta, paths = load_meta(directory, stem)
    if meta.get("layout") != "main" or not os.path.exists(paths["psd"]):
        return False
    names = meta["layers"]
    refs_npz = np.load(paths["ref"])
    refs = [refs_npz[f"layer_{k}"] for k in range(len(names))]
    applied = _load_applied_main(paths)
    for path, only_groups in [(paths["psd"], False)] + [(p, True) for p in sub_psds(directory, stem)]:
        doc = psd.read_psd(path, decode_filter=lambda l: l.name != WIREFRAME_LAYER)
        if (doc.width, doc.height) != (meta["width"], meta["height"]):
            continue
        changes = _layer_changes(doc, names, refs, only_groups)
        if changes and (not applied or os.path.getmtime(path) > os.path.getmtime(paths["applied"])):
            return True
    return False


def _load_applied_main(paths):
    if not os.path.exists(paths["applied"]):
        return {}
    data = np.load(paths["applied"])
    if "names" not in data:
        return {}
    return {str(n): (data[f"idx_{k}"], data[f"orig_{k}"]) for k, n in enumerate(data["names"])}


def _merge_main(directory, stem, meta, paths, settings, image):
    names = meta["layers"]
    refs_npz = np.load(paths["ref"])
    refs = [refs_npz[f"layer_{k}"] for k in range(len(names))]
    warnings = []

    doc = psd.read_psd(paths["psd"], decode_filter=lambda l: l.name != WIREFRAME_LAYER)
    if (doc.width, doc.height) != (meta["width"], meta["height"]):
        raise BridgeError(f"PSD canvas is {doc.width}x{doc.height}, capture was "
                          f"{meta['width']}x{meta['height']}; do not crop or resize it")
    changes = _layer_changes(doc, names, refs, False)
    for sub in sub_psds(directory, stem):
        sdoc = psd.read_psd(sub, decode_filter=lambda l: l.name != WIREFRAME_LAYER)
        if (sdoc.width, sdoc.height) != (meta["width"], meta["height"]):
            raise BridgeError(f"{os.path.basename(sub)} was cropped or resized; it cannot be merged")
        for name, (paint, mask) in _layer_changes(sdoc, names, refs, True).items():
            if name in changes:
                old_paint, old_mask = changes[name]
                changes[name] = (old_paint * (1.0 - mask[:, :, None]) + paint, np.maximum(old_mask, mask))
            else:
                changes[name] = (paint, mask)

    main_file = os.path.join(directory, meta["main"])
    if not os.path.exists(main_file):
        raise BridgeError(f"Main PSD not found: {main_file}")
    mdoc = psd.read_psd(main_file)
    tex_w, tex_h = mdoc.width, mdoc.height
    canvases = {}

    def canvas_index(name):
        index, layer = mainpsd.find_layer(mdoc, name)
        if index is not None and index not in canvases:
            canvases[index] = mainpsd.layer_canvas(mdoc, layer)[::-1].copy()
        return index

    for name, (idx, orig) in _load_applied_main(paths).items():
        index = canvas_index(name)
        if index is not None:
            canvases[index].reshape(-1, 4)[idx] = orig

    geo = dict(np.load(paths["geo"]))
    mmap = None
    if settings.auto_mirror:
        obj = bpy.data.objects.get(meta["object"])
        if obj is None:
            raise BridgeError(f"Object '{meta['object']}' is not in this .blend; turn off Auto Mirror")
        _img, uv_name, indices = resolve_source(obj, image)
        mmap = MirrorMap(obj, uv_name, indices, "XYZ".index(settings.mirror_axis))
    cos_start, cos_end = math.cos(settings.fade_start), math.cos(settings.fade_end)

    applied, texels = {}, 0
    for name, (paint, mask) in changes.items():
        index = canvas_index(name)
        if index is None:
            warnings.append(f"Layer '{name}' is no longer in the main PSD; its paint was skipped")
            continue
        p = project_paint(geo, meta["view"], meta["proj"], meta["perspective"], paint, tex_w, tex_h,
                          cos_start, cos_end)
        q = project_paint(geo, meta["view"], meta["proj"], meta["perspective"],
                          np.repeat(mask[:, :, None], 4, axis=2), tex_w, tex_h, cos_start, cos_end)[:, :, 3]
        if mmap is not None:
            p_m, _ = mmap.both_ways(p)
            q_m, _ = mmap.both_ways(np.repeat(q[:, :, None], 4, axis=2))
            p = p + p_m * (1.0 - q[:, :, None])
            q = q + q_m[:, :, 3] * (1.0 - q)
        layer = canvases[index]
        idx = np.flatnonzero(q.ravel() > 1e-4)
        applied[name] = (idx, layer.reshape(-1, 4)[idx].copy())
        prem = layer.copy()
        prem[:, :, :3] *= prem[:, :, 3:4]
        prem = prem * (1.0 - q[:, :, None]) + p
        alpha = prem[:, :, 3:4]
        layer[:, :, :3] = np.where(alpha > 1e-6, prem[:, :, :3] / np.maximum(alpha, 1e-6), layer[:, :, :3])
        layer[:, :, 3] = prem[:, :, 3]
        texels += idx.size

    if applied:
        keys = {}
        for k, (name, (idx, orig)) in enumerate(applied.items()):
            keys[f"idx_{k}"], keys[f"orig_{k}"] = idx, orig
        np.savez_compressed(paths["applied"], names=np.array(list(applied)), **keys)
    elif os.path.exists(paths["applied"]):
        os.remove(paths["applied"])
    if canvases:
        mainpsd.write_layers(main_file, mdoc, {i: c[::-1] for i, c in canvases.items()})
    saved = apply_main(main_file, image, settings, bpy.data.objects.get(meta["object"]))
    return {"texels": int(texels), "warnings": warnings, "saved": saved, "image": image}


def _revert_main(directory, meta, paths, settings, image):
    applied = _load_applied_main(paths)
    if not applied:
        raise BridgeError("Nothing to revert for this capture")
    main_file = os.path.join(directory, meta["main"])
    mdoc = psd.read_psd(main_file)
    canvases = {}
    for name, (idx, orig) in applied.items():
        index, layer = mainpsd.find_layer(mdoc, name)
        if index is None:
            continue
        canvas = mainpsd.layer_canvas(mdoc, layer)[::-1].copy()
        canvas.reshape(-1, 4)[idx] = orig
        canvases[index] = canvas[::-1]
    if canvases:
        mainpsd.write_layers(main_file, mdoc, canvases)
    os.remove(paths["applied"])
    apply_main(main_file, image, settings, bpy.data.objects.get(meta["object"]))


def _refresh_main(directory, stem, meta, paths):
    """Redraw a main-layout capture's layers from the main PSD, keeping pixels painted since."""
    main_file = os.path.join(directory, meta["main"])
    if not os.path.exists(main_file):
        return 0
    mdoc = psd.read_psd(main_file)
    geo = dict(np.load(paths["geo"]))
    refs_npz = dict(np.load(paths["ref"]))
    pairs = {}
    for k, name in enumerate(meta["layers"]):
        _index, layer = mainpsd.find_layer(mdoc, name)
        if layer is None or f"layer_{k}" not in refs_npz:
            continue
        rgba = render_layer(geo, meta["view"], meta["proj"], meta["width"], meta["height"],
                            mainpsd.layer_canvas(mdoc, layer)[::-1])
        new8 = (np.clip(rgba, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        pairs[name] = (refs_npz[f"layer_{k}"], new8)
        refs_npz[f"layer_{k}"] = new8
    changed = 0
    for path in [paths["psd"]] + sub_psds(directory, stem):
        if psd.refresh_layer_map(path, pairs):
            changed += 1
    np.savez_compressed(paths["ref"], **refs_npz)
    return changed


def mirror_main(path, obj, image, uv_name, indices, axis, from_positive, settings):
    """Mirror every layer of the main PSD, then rebuild the texture from it. Returns the texel count."""
    mdoc = psd.read_psd(path)
    mmap = MirrorMap(obj, uv_name, indices, axis)
    canvases, count = {}, 0
    for index, layer in mainpsd.leaf_layers(mdoc):
        canvas = mainpsd.layer_canvas(mdoc, layer)[::-1].copy()
        prem = canvas.copy()
        prem[:, :, :3] *= prem[:, :, 3:4]
        sampled, mask = mmap.sample(prem, from_positive)
        prem[mask] = sampled[mask]
        alpha = prem[:, :, 3:4]
        canvas[:, :, :3] = np.where(alpha > 1e-6, prem[:, :, :3] / np.maximum(alpha, 1e-6), canvas[:, :, :3])
        canvas[:, :, 3] = prem[:, :, 3]
        canvases[index] = canvas[::-1]
        count = int(mask.sum())
    mainpsd.write_layers(path, mdoc, canvases)
    apply_main(path, image, settings, obj)
    return count
