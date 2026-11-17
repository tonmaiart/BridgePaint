# SPDX-License-Identifier: GPL-3.0-or-later
"""Loose parts of a mesh and what a capture isolates: the faces shown in Edit Mode."""

import bmesh
import numpy as np

from .render import BridgeError, _loose_parts, _part_names

ALL, VISIBLE = "ALL", "VISIBLE"
PART, SELECTION = "PART", "SELECTION"  # isolation kinds of captures made by earlier versions

_cache = {}


def part_info(obj):
    """(per-vertex part label array, part names) of the object's own mesh, cached by topology."""
    mesh = obj.data
    key = (obj.name, mesh.name, len(mesh.vertices), len(mesh.edges), len(obj.vertex_groups))
    info = _cache.get(key)
    if info is None:
        edge_verts = np.empty(len(mesh.edges) * 2, np.int32)
        mesh.edges.foreach_get("vertices", edge_verts)
        labels, count = _loose_parts(len(mesh.vertices), edge_verts)
        info = (labels, _part_names(mesh, obj, labels, count))
        if len(_cache) > 32:
            _cache.clear()
        _cache[key] = info
    return info


def visible_faces(obj):
    """Bool per face, True where shown; None outside Edit Mode or when no face is hidden."""
    if obj is None or obj.type != "MESH" or obj.mode != "EDIT":
        return None
    bm = bmesh.from_edit_mesh(obj.data)
    hidden = np.fromiter((f.hide for f in bm.faces), bool, len(bm.faces))
    if not hidden.any():
        return None
    return ~hidden


def visible_part_names(obj, shown):
    """Names of the parts that have at least one shown face."""
    labels, names = part_info(obj)
    bm = bmesh.from_edit_mesh(obj.data)
    found = set()
    for face, on in zip(bm.faces, shown):
        index = face.verts[0].index
        if on and index < len(labels):
            found.add(int(labels[index]))
    return [names[i] for i in sorted(found)]


def isolation(obj):
    """Face mask a new capture keeps, its name tag, and the description stored with the capture.

    Returns (None, "", {"kind": ALL}) when nothing is hidden.
    """
    shown = visible_faces(obj)
    if shown is None:
        return None, "", {"kind": ALL}
    if not shown.any():
        raise BridgeError("Every face is hidden; unhide some (Alt+H) to capture them")
    names = visible_part_names(obj, shown)
    return shown, (names[0] if len(names) == 1 else "Isolated"), {"kind": VISIBLE}
