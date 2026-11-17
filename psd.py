# SPDX-License-Identifier: GPL-3.0-or-later
"""Minimal RGB PSD writer and reader (8/16-bit, raw/RLE/ZIP channels)."""

import os
import struct
import zlib

import numpy as np


# ---------------------------------------------------------------- writing

def _pascal_name(name):
    raw = name.encode("latin-1", "replace")[:255]
    data = bytes((len(raw),)) + raw
    pad = (4 - len(data) % 4) % 4
    return data + b"\0" * pad


def _info_block(key, payload):
    pad = (4 - len(payload) % 4) % 4
    payload += b"\0" * pad
    return b"8BIM" + key + struct.pack(">I", len(payload)) + payload


def _unicode_name(name):
    encoded = name.encode("utf-16-be")
    return _info_block(b"luni", struct.pack(">I", len(name)) + encoded)


def write_psd(path, width, height, layers, composite):
    """Write an 8-bit RGB PSD.

    `layers` is bottom-first; each is a dict with `name`, `rgba` (h x w x 4 uint8,
    top row first, or None for an empty layer), optional `rect` (top, left, bottom, right)
    placing `rgba` on the canvas (default: full canvas), optional `mask` (uint8 2D layer mask)
    with `mask_rect`, optional `opacity` (0-255), `locked`, `hidden`, and `section`
    (1 open group, 2 closed group, 3 group end; a group's children sit between its end and itself).
    `composite` is the flattened H x W x 4 uint8 image.
    """
    records = bytearray()
    channel_data = bytearray()
    for layer in layers:
        rgba = layer.get("rgba")
        if rgba is None:
            top = left = bottom = right = 0
            planes = [b"", b"", b"", b""]
        else:
            top, left, bottom, right = layer.get("rect", (0, 0, height, width))
            planes = [np.ascontiguousarray(rgba[:, :, 3]).tobytes()] + \
                [np.ascontiguousarray(rgba[:, :, c]).tobytes() for c in range(3)]
        ids = [-1, 0, 1, 2]
        mask = layer.get("mask")
        if mask is not None:
            ids.append(-2)
            planes.append(np.ascontiguousarray(mask, dtype=np.uint8).tobytes())
        records += struct.pack(">iiiiH", top, left, bottom, right, len(ids))
        for channel_id, plane in zip(ids, planes):
            records += struct.pack(">hI", channel_id, 2 + len(plane))
            channel_data += struct.pack(">H", 0) + plane
        flags = 0x02 if layer.get("hidden") else 0x00
        records += b"8BIMnorm" + struct.pack(">BBBB", int(layer.get("opacity", 255)), 0, flags, 0)
        if mask is not None:
            mt, ml, mb, mr = layer["mask_rect"]
            mask_data = struct.pack(">iiiiBBH", mt, ml, mb, mr, 0, 0, 0)
        else:
            mask_data = b""
        extra = struct.pack(">I", len(mask_data)) + mask_data + struct.pack(">I", 0) + \
            _pascal_name(layer["name"]) + _unicode_name(layer["name"])
        if layer.get("locked"):
            extra += _info_block(b"lspf", struct.pack(">I", 0x80000000))
        section = layer.get("section")
        if section:
            data = struct.pack(">I", section) + (b"8BIMpass" if section in (1, 2) else b"")
            extra += _info_block(b"lsct", data)
        records += struct.pack(">I", len(extra)) + extra

    layer_info = struct.pack(">h", -len(layers)) + records + channel_data
    if len(layer_info) % 2:
        layer_info += b"\0"
    layer_section = struct.pack(">I", len(layer_info)) + layer_info + struct.pack(">I", 0)

    with open(path, "wb") as f:
        f.write(b"8BPS" + struct.pack(">H6xHIIHH", 1, 4, height, width, 8, 3))
        f.write(struct.pack(">I", 0))  # color mode data
        f.write(struct.pack(">I", 0))  # image resources
        f.write(struct.pack(">I", len(layer_section)))
        f.write(layer_section)
        f.write(struct.pack(">H", 0))
        for c in range(4):
            f.write(np.ascontiguousarray(composite[:, :, c]).tobytes())


# ---------------------------------------------------------------- reading

class PsdError(Exception):
    pass


class PsdLayer:
    __slots__ = ("name", "top", "left", "bottom", "right", "opacity", "hidden",
                 "blend", "channels", "mask", "section", "pixels", "lock_all", "len_fields", "spans",
                 "record")

    def __init__(self):
        self.name = ""
        self.channels = []  # (id, length)
        self.mask = None    # (top, left, bottom, right, default, disabled)
        self.section = 0    # 1/2 group open/closed, 3 group end
        self.pixels = {}    # channel id -> float32 2D array
        self.lock_all = False
        self.len_fields = []  # file offset of each channel's length field
        self.spans = []       # (file offset, length) of each channel's data
        self.record = (0, 0, 0)  # file offsets: record start, end of channel entries, record end

    @property
    def width(self):
        return self.right - self.left

    @property
    def height(self):
        return self.bottom - self.top


class PsdDocument:
    def __init__(self):
        self.width = 0
        self.height = 0
        self.depth = 8
        self.layers = []       # bottom-first
        self.composite = None  # H x W x 4 float32, top row first
        self.pos = {}          # file offsets of the layer section, for in-place rewrites


class _Reader:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def read(self, n):
        if self.pos + n > len(self.data):
            raise PsdError("Unexpected end of PSD file")
        chunk = self.data[self.pos:self.pos + n]
        self.pos += n
        return chunk

    def unpack(self, fmt):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.read(size))


def _unpackbits(data, expected):
    out = bytearray()
    i = 0
    n = len(data)
    while i < n and len(out) < expected:
        h = data[i]
        i += 1
        if h < 128:
            count = h + 1
            out += data[i:i + count]
            i += count
        elif h > 128:
            out += data[i:i + 1] * (257 - h)
            i += 1
    if len(out) < expected:
        out += b"\0" * (expected - len(out))
    return bytes(out[:expected])


def _decode_plane(comp, payload, width, height, depth):
    bpc = depth // 8
    row_bytes = width * bpc
    expected = row_bytes * height
    if comp == 0:
        raw = payload[:expected]
    elif comp == 1:
        raw = _unpackbits(payload[height * 2:], expected)
    elif comp in (2, 3):
        raw = zlib.decompress(payload)[:expected]
    else:
        raise PsdError(f"Unsupported channel compression {comp}")

    if depth == 8:
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width)
        if comp == 3:
            arr = np.cumsum(arr, axis=1, dtype=np.uint8)
        return arr.astype(np.float32) / 255.0
    arr = np.frombuffer(raw, dtype=">u2").reshape(height, width)
    if comp == 3:
        arr = np.cumsum(arr.astype(np.uint16), axis=1, dtype=np.uint16)
    return arr.astype(np.float32) / 65535.0


def _parse_layer_info(r, end, doc, decode_filter):
    if r.pos >= end:
        return
    doc.pos["count"] = r.pos
    (count,) = r.unpack(">h")
    count = abs(count)
    doc.pos["records"] = r.pos
    layers = []
    for _ in range(count):
        layer = PsdLayer()
        record_start = r.pos
        layer.top, layer.left, layer.bottom, layer.right, nch = r.unpack(">iiiiH")
        for _c in range(nch):
            layer.channels.append(r.unpack(">hI"))
            layer.len_fields.append(r.pos - 4)
        entries_end = r.pos
        sig = r.read(4)
        if sig != b"8BIM":
            raise PsdError("Corrupt layer record")
        layer.blend = r.read(4).decode("latin-1")
        layer.opacity, _clip, flags, _filler = r.unpack(">BBBB")
        layer.hidden = bool(flags & 0x02)
        (extra_len,) = r.unpack(">I")
        extra_end = r.pos + extra_len

        (mask_len,) = r.unpack(">I")
        mask_end = r.pos + mask_len
        if mask_len >= 18:
            mt, ml, mb, mr, default, mflags = r.unpack(">iiiiBB")
            layer.mask = (mt, ml, mb, mr, default, bool(mflags & 0x02))
        r.pos = mask_end
        (ranges_len,) = r.unpack(">I")
        r.pos += ranges_len
        (name_len,) = r.unpack(">B")
        layer.name = r.read(name_len).decode("latin-1", "replace")
        r.pos += (4 - (1 + name_len) % 4) % 4

        while r.pos + 12 <= extra_end:
            sig = r.read(4)
            if sig not in (b"8BIM", b"8B64"):
                break
            key = r.read(4)
            (length,) = r.unpack(">I")
            block_end = r.pos + length
            if key == b"luni" and length >= 4:
                (chars,) = r.unpack(">I")
                layer.name = r.read(chars * 2).decode("utf-16-be", "replace")
            elif key in (b"lsct", b"lsdk") and length >= 4:
                (layer.section,) = r.unpack(">I")
            elif key == b"lspf" and length >= 4:
                (lock,) = r.unpack(">I")
                layer.lock_all = bool(lock & 0x80000000)
            r.pos = block_end
        r.pos = extra_end
        layer.record = (record_start, entries_end, extra_end)
        layers.append(layer)

    doc.pos["channels"] = r.pos
    for layer in layers:
        wanted = decode_filter(layer)
        for channel_id, length in layer.channels:
            start = r.pos
            layer.spans.append((start, length))
            r.pos += length
            if not wanted or length < 2:
                continue
            (comp,) = struct.unpack(">H", r.data[start:start + 2])
            if channel_id == -2 and layer.mask:
                w = layer.mask[3] - layer.mask[1]
                h = layer.mask[2] - layer.mask[0]
            elif channel_id < -1:
                continue
            else:
                w, h = layer.width, layer.height
            if w <= 0 or h <= 0:
                continue
            layer.pixels[channel_id] = _decode_plane(comp, r.data[start + 2:start + length], w, h, doc.depth)
    doc.pos["channels_end"] = r.pos
    doc.layers = layers


def read_psd(path, decode_filter=lambda layer: True):
    """Parse a PSD; only layers accepted by `decode_filter` get their pixels decoded."""
    with open(path, "rb") as f:
        data = f.read()
    r = _Reader(data)
    doc = PsdDocument()
    sig, version = r.read(4), r.unpack(">H")[0]
    if sig != b"8BPS":
        raise PsdError("Not a PSD file")
    if version != 1:
        raise PsdError("PSB (large document) files are not supported, save as PSD")
    r.pos += 6
    nchannels, doc.height, doc.width, doc.depth, mode = r.unpack(">HIIHH")
    if mode != 3:
        raise PsdError("Only RGB color mode is supported")
    if doc.depth not in (8, 16):
        raise PsdError(f"{doc.depth}-bit PSD is not supported, use 8 or 16-bit")

    (cm_len,) = r.unpack(">I")
    r.pos += cm_len
    (res_len,) = r.unpack(">I")
    r.pos += res_len

    doc.pos["lm_len"] = r.pos
    (lm_len,) = r.unpack(">I")
    lm_end = r.pos + lm_len
    doc.pos["lm_end"] = lm_end
    if lm_len:
        doc.pos["li_len"] = r.pos
        (li_len,) = r.unpack(">I")
        li_end = r.pos + li_len
        doc.pos["li_end"] = li_end
        if li_len:
            _parse_layer_info(r, li_end, doc, decode_filter)
            doc.pos["in_main"] = True
        r.pos = li_end
        (gm_len,) = r.unpack(">I")
        r.pos += gm_len
        while r.pos + 12 <= lm_end:
            sig = r.read(4)
            if sig not in (b"8BIM", b"8B64"):
                break
            key = r.read(4)
            (length,) = r.unpack(">I")
            block_end = r.pos + length
            if key in (b"Lr16", b"Lr32", b"Layr") and not doc.layers:
                _parse_layer_info(r, block_end, doc, decode_filter)
            r.pos = block_end
    r.pos = lm_end

    if r.pos + 2 <= len(data):
        (comp,) = r.unpack(">H")
        w, h = doc.width, doc.height
        bpc = doc.depth // 8
        planes = []
        try:
            if comp == 1:
                counts = np.frombuffer(r.read(nchannels * h * 2), dtype=">u2").astype(np.int64)
                for c in range(min(nchannels, 4)):
                    n = int(counts[c * h:(c + 1) * h].sum())
                    planes.append(_decode_plane(1, b"\0" * (h * 2) + r.read(n), w, h, doc.depth))
            elif comp == 0:
                for c in range(min(nchannels, 4)):
                    planes.append(_decode_plane(0, r.read(w * h * bpc), w, h, doc.depth))
            elif comp in (2, 3):
                pass
        except PsdError:
            planes = []
        if len(planes) >= 3:
            alpha = planes[3] if len(planes) >= 4 else np.ones((h, w), np.float32)
            doc.composite = np.dstack(planes[:3] + [alpha])
    return doc


def _layer_rgba(doc, layer, opacity=True):
    """Return (premultiplied H x W x 4 canvas array) for one decoded layer, or None."""
    if layer.width <= 0 or layer.height <= 0 or 0 not in layer.pixels:
        return None
    h, w = doc.height, doc.width
    ly = layer.pixels
    alpha = ly.get(-1, np.ones((layer.height, layer.width), np.float32))
    if layer.mask and -2 in ly and not layer.mask[5]:
        mt, ml, mb, mr, default, _ = layer.mask
        full = np.full((layer.height, layer.width), default / 255.0, np.float32)
        y0, x0 = max(mt, layer.top), max(ml, layer.left)
        y1, x1 = min(mb, layer.bottom), min(mr, layer.right)
        if y1 > y0 and x1 > x0:
            full[y0 - layer.top:y1 - layer.top, x0 - layer.left:x1 - layer.left] = \
                ly[-2][y0 - mt:y1 - mt, x0 - ml:x1 - ml]
        alpha = alpha * full

    canvas = np.zeros((h, w, 4), np.float32)
    y0, x0 = max(layer.top, 0), max(layer.left, 0)
    y1, x1 = min(layer.bottom, h), min(layer.right, w)
    if y1 <= y0 or x1 <= x0:
        return None
    sy, sx = slice(y0 - layer.top, y1 - layer.top), slice(x0 - layer.left, x1 - layer.left)
    a = alpha[sy, sx] * (layer.opacity / 255.0 if opacity else 1.0)
    for c in range(3):
        canvas[y0:y1, x0:x1, c] = ly.get(c, np.zeros_like(alpha))[sy, sx] * a
    canvas[y0:y1, x0:x1, 3] = a
    return canvas


def _group_mask(doc, layer):
    """Canvas-sized mask of a group folder, or None when it has no enabled mask."""
    if not layer.mask or -2 not in layer.pixels or layer.mask[5]:
        return None
    mt, ml, mb, mr, default, _ = layer.mask
    canvas = np.full((doc.height, doc.width), default / 255.0, np.float32)
    y0, x0 = max(mt, 0), max(ml, 0)
    y1, x1 = min(mb, doc.height), min(mr, doc.width)
    if y1 > y0 and x1 > x0:
        canvas[y0:y1, x0:x1] = layer.pixels[-2][y0 - mt:y1 - mt, x0 - ml:x1 - ml]
    return canvas


def layer_tree(doc):
    """Map pixel-layer index -> (visible, enclosing group indices), honouring hidden groups."""
    tree = {}
    stack = []
    for index in range(len(doc.layers) - 1, -1, -1):
        layer = doc.layers[index]
        if layer.section in (1, 2):
            stack.append(index)
        elif layer.section == 3:
            if stack:
                stack.pop()
        else:
            hidden = layer.hidden or any(doc.layers[g].hidden for g in stack)
            tree[index] = (not hidden, tuple(stack))
    return tree


def layer_to_dict(doc, layer):
    """A decoded layer as a `write_psd` layer dict (8-bit), keeping groups, masks, locks and empty layers."""
    px = layer.pixels
    d = {"name": layer.name, "opacity": layer.opacity, "hidden": layer.hidden, "rgba": None,
         "section": layer.section or None, "locked": layer.lock_all}
    if 0 in px and layer.width > 0 and layer.height > 0:
        alpha = px.get(-1, np.ones((layer.height, layer.width), np.float32))
        rgba = np.dstack([px.get(c, np.zeros_like(alpha)) for c in (0, 1, 2)] + [alpha])
        d["rgba"] = (np.clip(rgba, 0, 1) * 255 + 0.5).astype(np.uint8)
        d["rect"] = (layer.top, layer.left, layer.bottom, layer.right)
    if layer.mask and -2 in px:
        d["mask"] = (np.clip(px[-2], 0, 1) * 255 + 0.5).astype(np.uint8)
        d["mask_rect"] = layer.mask[:4]
    return d


def group_spans(doc):
    """Top-level group folders as (header index, first index, name), bottom-first."""
    spans, stack = [], []
    for index, layer in enumerate(doc.layers):
        if layer.section == 3:
            stack.append(index)
        elif layer.section in (1, 2) and stack:
            start = stack.pop()
            if not stack:
                spans.append((index, start, layer.name))
    return spans


def replace_layer_pixels(path, doc, replacements):
    """Rewrite `path` with new RGBA pixels for some layers, copying every other byte as it was.

    `doc` is the document read from `path`. `replacements` maps a layer index to an 8-bit
    H x W x 4 array of the layer's current size, or to ((top, left, bottom, right), array) to also
    move or grow the layer. A layer without an alpha channel gains one. Only 8-bit files whose
    layers sit in the main layer section work.
    """
    if doc.depth != 8 or not doc.pos.get("in_main"):
        raise PsdError("Only 8-bit PSD files can be updated in place; save it as 8-bit")
    with open(path, "rb") as f:
        data = f.read()
    pos = doc.pos
    records = bytearray()
    channels = bytearray()
    planes = {-1: 3, 0: 0, 1: 1, 2: 2}
    for index, layer in enumerate(doc.layers):
        new = replacements.get(index)
        rec_start, entries_end, rec_end = layer.record
        old_blocks = {cid: data[start:start + length]
                      for (cid, _len), (start, length) in zip(layer.channels, layer.spans)}
        if new is None:
            records += data[rec_start:rec_end]
            for (_cid, _len), (start, length) in zip(layer.channels, layer.spans):
                channels += data[start:start + length]
            continue
        if isinstance(new, tuple):
            rect, rgba = new
        else:
            rect, rgba = (layer.top, layer.left, layer.bottom, layer.right), new
        ids = [cid for cid, _len in layer.channels]
        if -1 not in ids:
            ids.insert(0, -1)
        entries = bytearray()
        for cid in ids:
            if cid in planes:
                block = struct.pack(">H", 0) + np.ascontiguousarray(rgba[:, :, planes[cid]]).tobytes()
            else:
                block = old_blocks[cid]
            entries += struct.pack(">hI", cid, len(block))
            channels += block
        records += struct.pack(">iiiiH", *rect, len(ids)) + entries + data[entries_end:rec_end]
    layer_info = data[pos["count"]:pos["records"]] + records + channels
    if len(layer_info) % 2:
        layer_info += b"\0"
    section = struct.pack(">I", len(layer_info)) + layer_info + data[pos["li_end"]:pos["lm_end"]]
    out = data[:pos["lm_len"]] + struct.pack(">I", len(section)) + section + data[pos["lm_end"]:]
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(out)
    os.replace(tmp, path)


def refresh_layers(path, names, old_ref, new_ref):
    """In layers named in `names`, swap pixels that still match `old_ref` for `new_ref` (8-bit canvases).

    Pixels the user changed on those layers, their alpha, and every other layer stay as they are.
    Returns how many layers changed.
    """
    doc = read_psd(path, decode_filter=lambda l: l.name in names)
    replacements = {}
    for index, layer in enumerate(doc.layers):
        if layer.name not in names or 0 not in layer.pixels:
            continue
        px = layer.pixels
        alpha = px.get(-1, np.ones((layer.height, layer.width), np.float32))
        cur = (np.dstack([px[0], px.get(1, px[0]), px.get(2, px[0]), alpha]) * 255.0 + 0.5).astype(np.uint8)
        y0, x0 = max(layer.top, 0), max(layer.left, 0)
        y1, x1 = min(layer.bottom, doc.height), min(layer.right, doc.width)
        if y1 <= y0 or x1 <= x0:
            continue
        sy, sx = slice(y0 - layer.top, y1 - layer.top), slice(x0 - layer.left, x1 - layer.left)
        region = cur[sy, sx]
        old = old_ref[y0:y1, x0:x1, :3].astype(np.int16)
        same = np.abs(region[:, :, :3].astype(np.int16) - old).max(axis=2) <= 1
        if not same.any():
            continue
        region[same, :3] = new_ref[y0:y1, x0:x1, :3][same]
        replacements[index] = cur
    if replacements:
        replace_layer_pixels(path, doc, replacements)
    return len(replacements)


def refresh_layer_map(path, pairs):
    """For layers named in `pairs` (name -> (old, new) 8-bit RGBA canvases), swap pixels that still
    equal `old` in all four channels for `new`. Pixels painted since stay. Returns layers changed."""
    doc = read_psd(path, decode_filter=lambda l: l.name in pairs)
    replacements = {}
    for index, layer in enumerate(doc.layers):
        if layer.section or layer.name not in pairs or 0 not in layer.pixels:
            continue
        old_ref, new_ref = pairs[layer.name]
        px = layer.pixels
        alpha = px.get(-1, np.ones((layer.height, layer.width), np.float32))
        cur = (np.dstack([px[0], px.get(1, px[0]), px.get(2, px[0]), alpha]) * 255.0 + 0.5).astype(np.uint8)
        y0, x0 = max(layer.top, 0), max(layer.left, 0)
        y1, x1 = min(layer.bottom, doc.height), min(layer.right, doc.width)
        if y1 <= y0 or x1 <= x0:
            continue
        region = cur[y0 - layer.top:y1 - layer.top, x0 - layer.left:x1 - layer.left]
        old = old_ref[y0:y1, x0:x1].astype(np.int16)
        same = np.abs(region.astype(np.int16) - old).max(axis=2) <= 1
        same &= np.abs(new_ref[y0:y1, x0:x1].astype(np.int16) - old).max(axis=2) > 0
        if not same.any():
            continue
        region[same] = new_ref[y0:y1, x0:x1][same]
        replacements[index] = cur
    if replacements:
        replace_layer_pixels(path, doc, replacements)
    return len(replacements)


def effective_visibility(doc):
    """Map layer index -> visible, taking hidden groups into account."""
    tree = layer_tree(doc)
    return {i: tree.get(i, (False, ()))[0] for i in range(len(doc.layers))}


def flatten_layers(doc, include, opacity=True):
    """Composite the visible layers accepted by `include` with normal blending.

    Group folder masks (and, with `opacity`, layer and folder opacity) apply to every layer inside them.
    Returns (premultiplied H x W x 4 float32, list of unsupported blend modes seen).
    """
    out = np.zeros((doc.height, doc.width, 4), np.float32)
    odd_modes = []
    masks = {}
    for index, (visible, groups) in sorted(layer_tree(doc).items()):
        layer = doc.layers[index]
        if not visible or not include(layer):
            continue
        src = _layer_rgba(doc, layer, opacity)
        if src is None:
            continue
        if layer.blend not in ("norm", "pass") and layer.blend not in odd_modes:
            odd_modes.append(layer.blend)
        for g in groups:
            if g not in masks:
                masks[g] = _group_mask(doc, doc.layers[g])
            factor = doc.layers[g].opacity / 255.0 if opacity else 1.0
            if masks[g] is not None:
                src *= (masks[g] * factor)[:, :, None]
            elif factor < 1.0:
                src *= factor
        out *= (1.0 - src[:, :, 3:4])
        out += src
    return out, odd_modes
