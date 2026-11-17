# SPDX-License-Identifier: GPL-3.0-or-later
"""The main PSD: the texture's layers in UV space, the source the texture is built from."""

import json
import os
import re

import numpy as np

from . import psd

UV_WIRE_LAYER = "UV Wireframe"
FIRST_LAYER = "Base"


def main_path(directory, image):
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.splitext(image.name)[0]) or "texture"
    return os.path.join(directory, f"{stem}_main.psd")


def _info_path(path):
    return path + ".json"


def image_of(path):
    """Name of the image a main PSD belongs to."""
    try:
        with open(_info_path(path), encoding="utf-8") as f:
            return json.load(f).get("image", "")
    except (OSError, ValueError):
        return ""


def list_mains(directory):
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    return [os.path.join(directory, n) for n in names if n.endswith("_main.psd")]


def create(path, image_name, rgba, uv_wire):
    """Write a main PSD from texture pixels (straight RGBA float, bottom row first)."""
    h, w = rgba.shape[:2]
    top = (np.clip(rgba[::-1], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    layers = [{"name": FIRST_LAYER, "rgba": top}]
    if uv_wire is not None and uv_wire.any():
        wire = np.zeros((h, w, 4), np.uint8)
        wire[:, :, 3] = (np.clip(uv_wire[::-1], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        layers.append({"name": UV_WIRE_LAYER, "rgba": wire, "opacity": 128, "locked": True})
    psd.write_psd(path, w, h, layers, top)
    with open(_info_path(path), "w", encoding="utf-8") as f:
        json.dump({"image": image_name}, f)


def leaf_layers(doc):
    """(index, layer) of the pixel layers a texture is painted on, bottom-first."""
    return [(i, l) for i, l in enumerate(doc.layers) if l.section == 0 and l.name != UV_WIRE_LAYER]


def find_layer(doc, name):
    for index, layer in leaf_layers(doc):
        if layer.name == name:
            return index, layer
    return None, None


def layer_canvas(doc, layer):
    """A layer's own pixels as straight RGBA float32 over the whole canvas, top row first."""
    out = np.zeros((doc.height, doc.width, 4), np.float32)
    px = layer.pixels
    if 0 not in px or layer.width <= 0 or layer.height <= 0:
        return out
    alpha = px.get(-1, np.ones((layer.height, layer.width), np.float32))
    y0, x0 = max(layer.top, 0), max(layer.left, 0)
    y1, x1 = min(layer.bottom, doc.height), min(layer.right, doc.width)
    if y1 <= y0 or x1 <= x0:
        return out
    sy, sx = slice(y0 - layer.top, y1 - layer.top), slice(x0 - layer.left, x1 - layer.left)
    for c in range(3):
        out[y0:y1, x0:x1, c] = px.get(c, px[0])[sy, sx]
    out[y0:y1, x0:x1, 3] = alpha[sy, sx]
    return out


def write_layers(path, doc, canvases):
    """Replace whole layers (index -> straight RGBA float canvas, top row first), keeping the rest."""
    replacements = {}
    for index, canvas in canvases.items():
        rgba = (np.clip(canvas, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        replacements[index] = ((0, 0, doc.height, doc.width), rgba)
    psd.replace_layer_pixels(path, doc, replacements)


def flatten(doc):
    """Premultiplied composite of the main PSD without the UV wireframe, top row first."""
    return psd.flatten_layers(doc, lambda l: l.name != UV_WIRE_LAYER)[0]
