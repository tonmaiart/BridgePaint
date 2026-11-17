# SPDX-License-Identifier: GPL-3.0-or-later
"""Paint Bridge: capture the viewport to a layered PSD, paint it elsewhere, merge it back."""

import math
import os
import shutil
import subprocess
import sys

import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty,
                       FloatVectorProperty, IntProperty, PointerProperty, StringProperty)

if "bridge" in locals():
    # Reload Scripts re-runs only this file; reload the submodules too, dependencies first.
    import importlib
    for _module in (render, psd, main, parts, mirror, bridge):  # noqa: F821
        importlib.reload(_module)

from . import bridge, main, mirror, parts, psd, render
from .psd import PsdError
from .render import BridgeError


def reveal_in_folder(path):
    """Open the file's folder in the system file browser with the file selected."""
    if sys.platform == "win32":
        subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-R", path])
    else:
        subprocess.Popen(["xdg-open", os.path.dirname(path)])


def _prefs(context):
    addon = context.preferences.addons.get(__package__)
    return addon.preferences if addon else None


def open_in_paint_app(path, context):
    prefs = _prefs(context)
    exe = (prefs.paint_app if prefs else "") or context.preferences.filepaths.image_editor
    if exe:
        exe = bpy.path.abspath(exe)
        if not os.path.isfile(exe):
            raise BridgeError(f"Paint app not found: {exe}")
        subprocess.Popen([exe, path])
    elif sys.platform == "win32":
        try:
            os.startfile(path)
        except OSError:
            raise BridgeError("No app is set to open .psd files. Set Paint App in Paint Bridge > Options") from None
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def _capture_files(item):
    return [bridge.capture_paths(item.directory, item.name)["psd"]] + bridge.sub_psds(item.directory, item.name)


def _seen(item):
    try:
        return float(item.mtime_text)
    except ValueError:
        return 0.0


def _mark_seen(item, value=None):
    item.mtime_text = repr(_psd_mtime(item) if value is None else value)


def _psd_mtime(item):
    try:
        return max(os.path.getmtime(f) for f in _capture_files(item))
    except (OSError, ValueError):
        return 0.0


def _redraw():
    wm = bpy.context.window_manager
    for window in wm.windows if wm else ():
        for area in window.screen.areas:
            if area.type in {"VIEW_3D", "IMAGE_EDITOR"}:
                area.tag_redraw()


_syncing = False
_picking = False


def _view3d_space(context):
    space = context.space_data
    if space is not None and space.type == "VIEW_3D":
        return space
    screen = context.screen
    for area in screen.areas if screen else ():
        if area.type == "VIEW_3D":
            return area.spaces.active
    return None


def _on_select(self, context):
    if _syncing or not (0 <= self.active_index < len(self.captures)):
        return
    item = self.captures[self.active_index]
    space = _view3d_space(context)
    if space is None:
        return
    try:
        meta, _paths = bridge.load_meta(item.directory, item.name)
        bridge.snap_view(space, context.scene, meta)
    except (BridgeError, OSError, ValueError, KeyError) as exc:
        print(f"Paint Bridge: could not snap to {item.name}: {exc}")


def _on_project_dir(self, context):
    sync_captures(context.scene)


def sync_captures(scene, select=None):
    """Make the list mirror the PSD captures in the project folder. Returns True when it changed."""
    global _syncing
    s = scene.paint_bridge
    try:
        directory = bridge.project_directory(s, create=False)
    except BridgeError:
        directory = ""
    stems = bridge.list_captures(directory) if directory else []
    norm = os.path.normcase(directory)
    unchanged = [i.name for i in s.captures] == stems and \
        all(os.path.normcase(os.path.normpath(i.directory)) == norm for i in s.captures)
    if unchanged and select is None:
        for item in s.captures:
            merged = bridge.is_merged(item.directory, item.name)
            if item.merged != merged:
                item.merged = merged
        return False

    active = s.captures[s.active_index].name if 0 <= s.active_index < len(s.captures) else ""
    active = select or active
    previous = {i.name: i.mtime_text for i in s.captures}
    _syncing = True
    try:
        s.captures.clear()
        for stem in stems:
            item = s.captures.add()
            item.name = stem
            item.directory = directory
            item.merged = bridge.is_merged(directory, stem)
            if previous.get(stem):
                item.mtime_text = previous[stem]
            else:
                _mark_seen(item)
        if active in stems:
            s.active_index = stems.index(active)
        else:
            s.active_index = min(s.active_index, len(stems) - 1)
    finally:
        _syncing = False
    return True


# ---------------------------------------------------------------- properties

class PBRIDGE_Capture(bpy.types.PropertyGroup):
    directory: StringProperty(subtype="DIR_PATH")
    merged: BoolProperty()
    mtime_text: StringProperty(description="PSD modified time at the last merge, as text for full precision")


class PBRIDGE_Settings(bpy.types.PropertyGroup):
    project_dir: StringProperty(
        name="Project Folder", subtype="DIR_PATH", default="//paint_bridge/", update=_on_project_dir,
        description="Folder that holds the exported PSD files and their capture data")
    image_override: PointerProperty(
        type=bpy.types.Image, name="Texture", poll=lambda _self, image: image.type == "IMAGE",
        description="Base color texture to paint. Leave empty to use the image found in the material")
    capture_scale: FloatProperty(
        name="Resolution Scale", default=1.0, min=0.25, max=4.0,
        description="PSD size relative to the viewport. Camera view uses the render resolution")
    split_parts: BoolProperty(
        name="Layer per Part", default=True,
        description="Put each loose part of the mesh on its own PSD layer, masked to that part")
    wireframe_layer: BoolProperty(
        name="Wireframe Layer", default=True,
        description="Add the visible mesh edges as the top PSD layer. It is never merged back")
    fade_start: FloatProperty(
        name="Fade Start", subtype="ANGLE", default=math.radians(60.0), min=0.0, max=math.radians(89.0),
        description="Surfaces turned further than this from the view start to receive less paint")
    fade_end: FloatProperty(
        name="Fade End", subtype="ANGLE", default=math.radians(85.0), min=math.radians(1.0), max=math.radians(90.0),
        description="Surfaces turned further than this from the view receive no paint")
    save_master: BoolProperty(
        name="Save Texture After Merge", default=True,
        description="Write the texture file (or repack it) after every merge")
    auto_merge: BoolProperty(
        name="Auto Merge on Save", default=False,
        description="Merge a capture automatically whenever its PSD is saved in the external editor")
    refresh_others: BoolProperty(
        name="Refresh Other Captures", default=True,
        description="After every merge, redraw the Base layers of the other PSDs from the updated texture. "
                    "Their paint layers and any edits stay as they are")
    auto_mirror: BoolProperty(
        name="Auto Mirror on Merge", default=False,
        description="Also paint every merge onto the mirrored side of the mesh, across the Mirror axis")
    seam_bleed: IntProperty(
        name="Seam Fill", default=16, min=0, max=128, subtype="PIXEL",
        description="After every merge, mirror and revert, fill this many texels around every UV island "
                    "with the nearest island color, so seams never show the gutter. 0 turns it off")
    mirror_axis: EnumProperty(
        name="Axis", default="X",
        items=(("X", "X", "Mirror across the object's local X axis"),
               ("Y", "Y", "Mirror across the object's local Y axis"),
               ("Z", "Z", "Mirror across the object's local Z axis")))
    mirror_direction: EnumProperty(
        name="Direction", default="POS",
        items=(("POS", "+ to \u2212", "Copy the positive side onto the negative side"),
               ("NEG", "\u2212 to +", "Copy the negative side onto the positive side")))
    captures: CollectionProperty(type=PBRIDGE_Capture)
    pick_mode: StringProperty(default="OBJECT")
    active_index: IntProperty(update=_on_select)


def _settings(context):
    return context.scene.paint_bridge


def _item(context, index):
    s = _settings(context)
    i = s.active_index if index < 0 else index
    if 0 <= i < len(s.captures):
        return s.captures[i]
    return None


# ---------------------------------------------------------------- operators

class PBRIDGE_OT_new_texture(bpy.types.Operator):
    bl_idname = "paint_bridge.new_texture"
    bl_label = "Create New Texture"
    bl_description = "Create a new blank base color PNG in the textures folder next to the .blend and use it in place of the missing file"
    bl_options = {"REGISTER", "UNDO"}

    width: IntProperty(name="Width", default=2048, min=16, max=16384)
    height: IntProperty(name="Height", default=2048, min=16, max=16384)
    color: FloatVectorProperty(name="Color", subtype="COLOR_GAMMA", size=3, min=0.0, max=1.0,
                               default=(0.8, 0.8, 0.8))

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        try:
            image, _uv, _idx = bridge.resolve_source(context.active_object, _settings(context).image_override)
            path = bridge.create_texture_file(image, self.width, self.height, self.color)
        except (BridgeError, OSError, RuntimeError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Created {path}")
        _redraw()
        return {"FINISHED"}


class PBRIDGE_OT_capture(bpy.types.Operator):
    bl_idname = "paint_bridge.capture"
    bl_label = "Capture View to PSD"
    bl_description = "Render the active object's base color from the current view and export it as a layered PSD"
    bl_options = {"REGISTER"}

    name: StringProperty(name="Name", description="File name of the new PSD", options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return context.area is not None and context.area.type == "VIEW_3D" and obj is not None and obj.type == "MESH"

    def invoke(self, context, event):
        try:
            self.name = bridge.default_name(context, _settings(context))
        except (BridgeError, OSError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return context.window_manager.invoke_props_dialog(self, title="New Capture", confirm_text="Capture")

    def draw(self, context):
        self.layout.prop(self, "name")

    def execute(self, context):
        s = _settings(context)
        try:
            stem, directory = bridge.capture(context, s, name=self.name or None)
        except (BridgeError, OSError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        meta, paths = bridge.load_meta(directory, stem)
        sync_captures(context.scene, select=stem)
        self.report({"INFO"}, f"Exported {meta['width']}x{meta['height']} PSD: {paths['psd']}")
        return {"FINISHED"}


class PBRIDGE_OT_recapture(bpy.types.Operator):
    bl_idname = "paint_bridge.recapture"
    bl_label = "Recapture"
    bl_description = ("Capture the current view again and overwrite the selected PSD. "
                      "Paint in that PSD is lost unless it was merged")
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return context.area is not None and context.area.type == "VIEW_3D" and _item(context, -1) is not None

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(
            self, event, title="Overwrite PSD?",
            message=f"Replace {_item(context, -1).name}.psd with the current view. Unmerged paint in it is lost.",
            confirm_text="Recapture")

    def execute(self, context):
        s = _settings(context)
        item = _item(context, -1)
        try:
            stem, _directory = bridge.capture(context, s, stem=item.name)
        except (BridgeError, OSError, KeyError, ValueError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        item.merged = False
        _mark_seen(item)
        self.report({"INFO"}, f"Recaptured {stem}.psd; reload it in your paint app")
        return {"FINISHED"}


def run_merge(item, settings):
    result = bridge.merge(item.directory, item.name, settings)
    item.merged = True
    _mark_seen(item)
    result["refreshed"] = 0
    if settings.refresh_others:
        result["refreshed"], warnings = refresh_captures(settings, result["image"], skip=item.name)
        result["warnings"] += warnings
    return result


class PBRIDGE_OT_merge(bpy.types.Operator):
    bl_idname = "paint_bridge.merge"
    bl_label = "Merge"
    bl_description = "Project the painted PSD layers back onto the texture. Merging again replaces this capture's previous merge"
    bl_options = {"REGISTER"}

    index: IntProperty(default=-1, options={"SKIP_SAVE"})

    def execute(self, context):
        item = _item(context, self.index)
        if item is None:
            return {"CANCELLED"}
        try:
            result = run_merge(item, _settings(context))
        except (BridgeError, PsdError, OSError, KeyError, ValueError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        for warning in result["warnings"]:
            self.report({"WARNING"}, warning)
        msg = f"Merged {result['texels']} texels into '{result['image'].name}'"
        if result["refreshed"]:
            msg += f", refreshed {result['refreshed']} other capture(s)"
        if result["saved"]:
            msg += f", {result['saved']}"
        elif _settings(context).save_master:
            msg += " (image has no file; save it manually)"
        self.report({"INFO"}, msg)
        _redraw()
        return {"FINISHED"}


class PBRIDGE_OT_revert(bpy.types.Operator):
    bl_idname = "paint_bridge.revert"
    bl_label = "Revert Merge"
    bl_description = "Restore the texels changed by this capture's last merge"
    bl_options = {"REGISTER"}

    index: IntProperty(default=-1, options={"SKIP_SAVE"})

    def execute(self, context):
        item = _item(context, self.index)
        if item is None:
            return {"CANCELLED"}
        try:
            bridge.revert(item.directory, item.name, _settings(context))
        except (BridgeError, OSError, KeyError, ValueError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        item.merged = False
        _redraw()
        return {"FINISHED"}


class PBRIDGE_OT_open(bpy.types.Operator):
    bl_idname = "paint_bridge.open"
    bl_label = "Open Folder"
    bl_description = "Open the folder that holds this capture's PSD"

    index: IntProperty(default=-1, options={"SKIP_SAVE"})

    def execute(self, context):
        item = _item(context, self.index)
        if item is None:
            return {"CANCELLED"}
        path = bridge.capture_paths(item.directory, item.name)["psd"]
        if not os.path.exists(path):
            self.report({"ERROR"}, f"File not found: {path}")
            return {"CANCELLED"}
        try:
            reveal_in_folder(path)
        except OSError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class PBRIDGE_OT_open_file(bpy.types.Operator):
    bl_idname = "paint_bridge.open_file"
    bl_label = "Edit Capture"
    bl_description = ("Recapture the selected capture from the current view and the latest texture, then open "
                      "it in your paint app. Paint not merged yet is merged first. In Edit Mode with faces "
                      "hidden, a PSD holding only the folders of the parts still shown is opened instead")

    def execute(self, context):
        item = _item(context, -1)
        if item is None:
            return {"CANCELLED"}
        s = _settings(context)
        path = bridge.capture_paths(item.directory, item.name)["psd"]
        try:
            notes = []
            if bridge.has_unmerged(item.directory, item.name):
                result = run_merge(item, s)
                notes.append(f"merged {result['texels']} texels first")
            bridge.capture(context, s, stem=item.name)
            item.merged = False
            _mark_seen(item)
            obj = context.active_object
            shown = parts.visible_faces(obj)
            if shown is not None and obj.name == bridge.load_meta(item.directory, item.name)[0]["object"]:
                names = parts.visible_part_names(obj, shown)
                if not names:
                    raise BridgeError("Every face is hidden; unhide some (Alt+H)")
                path = bridge.part_psd(item.directory, item.name, names)
            open_in_paint_app(path, context)
        except (BridgeError, PsdError, OSError, KeyError, ValueError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Recaptured {item.name}" + (f" ({', '.join(notes)})" if notes else ""))
        _redraw()
        return {"FINISHED"}


class PBRIDGE_OT_edit_main(bpy.types.Operator):
    bl_idname = "paint_bridge.edit_main"
    bl_label = "Edit Main"
    bl_description = ("Open the main PSD, the texture's layers in UV space, in your paint app. Add, rename or "
                      "reorder layers there; saving it updates the texture and the captures")

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH"

    def execute(self, context):
        s = _settings(context)
        try:
            image, _uv, _idx = bridge.resolve_source(context.active_object, s.image_override)
            path = bridge.ensure_main(s, image, context.active_object)
            open_in_paint_app(path, context)
        except (BridgeError, PsdError, OSError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


def refresh_captures(settings, image, skip=""):
    """Redraw the layers of every capture of `image` (except `skip`). Returns (count, warnings)."""
    count, warnings = 0, []
    for other in settings.captures:
        if other.name == skip:
            continue
        try:
            if bridge.refresh_capture(other.directory, other.name, image):
                count += 1
        except (BridgeError, PsdError, OSError, KeyError, ValueError) as exc:
            warnings.append(f"Could not refresh {other.name}: {exc}")
        _mark_seen(other)
    return count, warnings


class PBRIDGE_OT_mirror(bpy.types.Operator):
    bl_idname = "paint_bridge.mirror"
    bl_label = "Mirror Texture"
    bl_description = ("Copy the texture from one side of the mesh to the other, across a local axis of its "
                      "rest shape. A backup of the texture is saved first")
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH"

    def execute(self, context):
        s = _settings(context)
        try:
            image, uv_name, indices = bridge.resolve_source(context.active_object, s.image_override)
            base = bridge.read_image(image)
            backup = bridge.backup_image(image, s)
            axis, from_positive = "XYZ".index(s.mirror_axis), s.mirror_direction == "POS"
            main_file = main.main_path(bridge.project_directory(s), image)
            if os.path.exists(main_file):
                shutil.copy2(main_file, os.path.join(os.path.dirname(backup), os.path.basename(backup)
                                                     .rsplit(".", 1)[0] + "_main.psd"))
                count = bridge.mirror_main(main_file, context.active_object, image, uv_name, indices,
                                           axis, from_positive, s)
                saved = None
                refresh_captures(s, image)
            else:
                result, count = mirror.mirror(context.active_object, base, uv_name, indices,
                                              axis, from_positive, s.seam_bleed)
                bridge.write_image(image, result)
                saved = bridge.save_master(image) if s.save_master else None
        except (BridgeError, OSError, RuntimeError, ValueError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        msg = f"Mirrored {count} texels of '{image.name}'. Backup: {backup}"
        if saved:
            msg += f"; {saved}"
        self.report({"INFO"}, msg)
        _redraw()
        return {"FINISHED"}


class PBRIDGE_OT_fill_seams(bpy.types.Operator):
    bl_idname = "paint_bridge.fill_seams"
    bl_label = "Fill Seams"
    bl_description = "Fill the gutter around every UV island of the texture with the nearest island color"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH"

    def execute(self, context):
        s = _settings(context)
        try:
            image, _uv, _idx = bridge.resolve_source(context.active_object, s.image_override)
            base = bridge.read_image(image)
            filled = bridge.fill_seams(context.active_object, image, base, max(s.seam_bleed, 1))
            bridge.write_image(image, filled)
            saved = bridge.save_master(image) if s.save_master else None
        except (BridgeError, OSError, RuntimeError, ValueError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Filled seams of '{image.name}'" + (f"; {saved}" if saved else ""))
        _redraw()
        return {"FINISHED"}


class PBRIDGE_OT_delete(bpy.types.Operator):
    bl_idname = "paint_bridge.delete"
    bl_label = "Delete Capture"
    bl_description = ("Delete the selected capture's PSD and its capture data from the project folder "
                      "(to the Recycle Bin on Windows)")
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return _item(context, -1) is not None

    def execute(self, context):
        item = _item(context, -1)
        name = item.name
        try:
            count = bridge.delete_capture(item.directory, name)
        except OSError as exc:
            self.report({"ERROR"}, f"Could not delete {name}: {exc}; close it in other apps and try again")
            return {"CANCELLED"}
        sync_captures(context.scene)
        where = "; it is in the Recycle Bin" if sys.platform == "win32" else ""
        self.report({"INFO"}, f"Deleted {name} ({count} files){where}")
        _redraw()
        return {"FINISHED"}


# ---------------------------------------------------------------- auto merge

_pending = {}


def _watch():
    try:
        scenes = bpy.data.scenes
    except AttributeError:
        return 2.0
    for scene in scenes:
        s = getattr(scene, "paint_bridge", None)
        if s is None:
            continue
        if sync_captures(scene):
            _redraw()
        try:
            for image in bridge.changed_mains(s):
                refresh_captures(s, image)
                print(f"Paint Bridge: applied the main PSD of '{image.name}'")
                _redraw()
        except Exception as exc:  # noqa: BLE001 - a failed main update must not stop the watcher
            print(f"Paint Bridge: could not apply a main PSD: {exc}")
        if not s.auto_merge:
            continue
        for item in s.captures:
            mtime = _psd_mtime(item)
            if not mtime or mtime <= _seen(item) + 1e-3:
                continue
            key = (item.directory, item.name)
            try:
                size = sum(os.path.getsize(f) for f in _capture_files(item))
            except OSError:
                continue
            if _pending.get(key) != (mtime, size):
                _pending[key] = (mtime, size)
                continue
            _pending.pop(key, None)
            try:
                result = run_merge(item, s)
                print(f"Paint Bridge: merged {item.name} ({result['texels']} texels)")
            except Exception as exc:  # noqa: BLE001 - a failed auto merge must not stop the watcher
                _mark_seen(item, mtime)
                print(f"Paint Bridge: auto merge of {item.name} failed: {exc}")
            _redraw()
    return 1.0


# ---------------------------------------------------------------- modes

def _view3d_region_at(window, x, y):
    """The 3D View main region under the mouse, or (None, None) over any other region."""
    for area in window.screen.areas:
        if area.type != "VIEW_3D" or not (area.x <= x < area.x + area.width and area.y <= y < area.y + area.height):
            continue
        main = None
        for region in area.regions:
            if region.x <= x < region.x + region.width and region.y <= y < region.y + region.height:
                if region.type == "WINDOW":
                    main = region
                else:
                    return None, None
        return (area, main) if main else (None, None)
    return None, None


class PBRIDGE_OT_pick_part(bpy.types.Operator):
    bl_idname = "paint_bridge.pick_part"
    bl_label = "Pick Part"
    bl_description = "Click a part to select it whole; Shift adds, Ctrl removes, empty space clears"

    def invoke(self, context, event):
        global _picking
        if _picking:
            return {"CANCELLED"}
        _picking = True
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        global _picking
        obj = context.active_object
        s = _settings(context)
        if s.pick_mode != "PART" or obj is None or obj.type != "MESH" or obj.mode != "EDIT":
            _picking = False
            return {"FINISHED", "PASS_THROUGH"}
        if event.type != "LEFTMOUSE" or event.value != "PRESS" or event.alt:
            return {"PASS_THROUGH"}
        area, region = _view3d_region_at(context.window, event.mouse_x, event.mouse_y)
        if region is None:
            return {"PASS_THROUGH"}
        with context.temp_override(area=area, region=region):
            if not (event.shift or event.ctrl):
                bpy.ops.mesh.select_all(action="DESELECT")
            bpy.ops.mesh.select_linked_pick("INVOKE_DEFAULT", deselect=event.ctrl, delimit=set())
        _redraw()
        return {"RUNNING_MODAL"}


_MODE_TIPS = {
    "OBJECT": "Object Mode",
    "PART": ("Select Part: click a part to select all of it (Shift adds, Ctrl removes). "
             "Faces hidden with H are left out of Capture and Open File"),
    "VERTEX": "Select Vertex: Edit Mode vertex select. Faces hidden with H are left out of Capture and Open File",
}


class PBRIDGE_OT_set_mode(bpy.types.Operator):
    bl_idname = "paint_bridge.set_mode"
    bl_label = "Switch Mode"
    bl_options = {"REGISTER"}

    mode: EnumProperty(items=(("OBJECT", "Object", ""), ("PART", "Select Part", ""), ("VERTEX", "Select Vertex", "")))

    @classmethod
    def description(cls, context, properties):
        return _MODE_TIPS[properties.mode]

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH"

    def execute(self, context):
        s = _settings(context)
        obj = context.active_object
        if self.mode == "OBJECT":
            if obj.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
            s.pick_mode = "OBJECT"
            return {"FINISHED"}
        if obj.mode != "EDIT":
            bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_mode(type="VERT")
        s.pick_mode = self.mode
        if self.mode == "PART":
            bpy.ops.paint_bridge.pick_part("INVOKE_DEFAULT")
        _redraw()
        return {"FINISHED"}


def _active_mode(context):
    obj = context.active_object
    if obj is None or obj.mode != "EDIT":
        return "OBJECT"
    return "PART" if _picking and _settings(context).pick_mode == "PART" else "VERTEX"


# ---------------------------------------------------------------- UI

class PBRIDGE_UL_captures(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_prop, index=0):
        row = layout.row(align=True)
        row.label(text=item.name, icon="CHECKMARK" if item.merged else "IMAGE_RGB")


class PBRIDGE_PT_main(bpy.types.Panel):
    bl_label = "Paint Bridge"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Paint Bridge"

    def draw_header_preset(self, context):
        op = self.layout.operator("wm.call_panel", text="", icon="PREFERENCES", emboss=False)
        op.name = "PBRIDGE_PT_options"
        op.keep_open = True

    def draw(self, context):
        layout = self.layout
        s = _settings(context)
        obj = context.active_object

        col = layout.column()
        try:
            image, uv_name, _indices = bridge.resolve_source(obj, s.image_override)
            col.label(text=f"Base Color: {image.name}", icon="IMAGE_DATA")
            col.label(text=f"UV: {uv_name or 'active render map'}", icon="UV")
            if bridge.image_missing(image):
                box = col.box()
                box.alert = True
                box.label(text="Texture file is missing", icon="ERROR")
                box.operator("paint_bridge.new_texture", icon="FILE_NEW")
        except BridgeError as exc:
            col.label(text=str(exc), icon="ERROR")
        col.prop(s, "image_override")
        col.prop(s, "project_dir", text="")

        space = context.space_data
        camera_view = space is not None and space.type == "VIEW_3D" and \
            space.region_3d.view_perspective == "CAMERA" and context.scene.camera is not None
        row = layout.row()
        row.enabled = not camera_view
        row.prop(s, "capture_scale")
        if camera_view:
            layout.label(text="Camera view: uses render resolution", icon="CAMERA_DATA")

        layout.separator()
        row = layout.row()
        row.template_list("PBRIDGE_UL_captures", "", s, "captures", s, "active_index", rows=4)
        side = row.column(align=True)
        side.operator("paint_bridge.capture", text="", icon="ADD")
        side.operator("paint_bridge.delete", text="", icon="REMOVE")
        side.separator()
        side.operator("paint_bridge.recapture", text="", icon="VIEW_CAMERA")
        side.operator("paint_bridge.open", text="", icon="FILE_FOLDER")

        current = _active_mode(context)
        row = layout.row(align=True)
        row.scale_y = 1.2
        for mode, icon in (("OBJECT", "OBJECT_DATAMODE"), ("PART", "UV_ISLANDSEL"), ("VERTEX", "VERTEXSEL")):
            row.operator("paint_bridge.set_mode", text="", icon=icon, depress=current == mode).mode = mode

        has_item = 0 <= s.active_index < len(s.captures)
        grid = layout.column()
        cells = grid.row(align=True)
        cells.scale_y = 1.2
        cell = cells.row(align=True)
        cell.enabled = has_item
        cell.operator("paint_bridge.open_file")
        cells.separator(factor=0.25)
        cells.operator("paint_bridge.edit_main")
        cells = grid.row(align=True)
        cells.scale_y = 1.2
        cell = cells.row(align=True)
        cell.enabled = has_item
        cell.operator("paint_bridge.merge")
        undo = cell.row(align=True)
        undo.enabled = has_item and s.captures[s.active_index].merged
        undo.operator("paint_bridge.revert", text="", icon="LOOP_BACK")
        cells.separator(factor=0.25)
        cells.operator("wm.call_panel", text="Mirror...").name = "PBRIDGE_PT_mirror_popup"
        cells = grid.row(align=True)
        cells.scale_y = 1.2
        cells.operator("wm.call_panel", text="Fill Seams...").name = "PBRIDGE_PT_fill_seams_popup"
        cells.separator(factor=0.25)
        cells.label(text="")


class PBRIDGE_PT_mirror_popup(bpy.types.Panel):
    """Mirror the texture across a local axis of the mesh"""
    bl_label = "Mirror"
    bl_space_type = "VIEW_3D"
    bl_region_type = "HEADER"
    bl_ui_units_x = 11

    def draw(self, context):
        layout = self.layout
        s = _settings(context)
        layout.label(text="Mirror Texture")
        row = layout.row(align=True)
        row.prop(s, "mirror_axis", expand=True)
        row.prop(s, "mirror_direction", text="")
        layout.operator("paint_bridge.mirror", icon="MOD_MIRROR")
        layout.prop(s, "auto_mirror")


class PBRIDGE_PT_fill_seams_popup(bpy.types.Panel):
    """Fill the gutter around UV islands with the nearest island color"""
    bl_label = "Fill Seams"
    bl_space_type = "VIEW_3D"
    bl_region_type = "HEADER"
    bl_ui_units_x = 11

    def draw(self, context):
        layout = self.layout
        s = _settings(context)
        layout.label(text="Fill Seams")
        layout.prop(s, "seam_bleed")
        layout.operator("paint_bridge.fill_seams", icon="UV")


class PBRIDGE_PT_options(bpy.types.Panel):
    """Paint Bridge options"""
    bl_label = "Options"
    bl_space_type = "VIEW_3D"
    bl_region_type = "HEADER"
    bl_ui_units_x = 14

    def draw(self, context):
        layout = self.layout
        layout.label(text="Options")
        layout.use_property_split = True
        s = _settings(context)
        layout.prop(s, "split_parts")
        layout.prop(s, "wireframe_layer")
        layout.prop(s, "auto_merge")
        layout.prop(s, "refresh_others")
        layout.prop(s, "save_master")
        prefs = _prefs(context)
        if prefs is not None:
            layout.prop(prefs, "paint_app")
        layout.prop(s, "fade_start")
        layout.prop(s, "fade_end")


class PBRIDGE_Preferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    paint_app: StringProperty(
        name="Paint App", subtype="FILE_PATH",
        description="Program that Open File uses for PSD captures. Empty uses the Image Editor set in "
                    "Preferences > File Paths, then the system default app")

    def draw(self, context):
        self.layout.prop(self, "paint_app")


classes = (
    PBRIDGE_Preferences,
    PBRIDGE_Capture,
    PBRIDGE_Settings,
    PBRIDGE_OT_new_texture,
    PBRIDGE_OT_capture,
    PBRIDGE_OT_recapture,
    PBRIDGE_OT_merge,
    PBRIDGE_OT_revert,
    PBRIDGE_OT_open,
    PBRIDGE_OT_open_file,
    PBRIDGE_OT_edit_main,
    PBRIDGE_OT_mirror,
    PBRIDGE_OT_fill_seams,
    PBRIDGE_OT_delete,
    PBRIDGE_OT_pick_part,
    PBRIDGE_OT_set_mode,
    PBRIDGE_UL_captures,
    PBRIDGE_PT_main,
    PBRIDGE_PT_mirror_popup,
    PBRIDGE_PT_fill_seams_popup,
    PBRIDGE_PT_options,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.paint_bridge = PointerProperty(type=PBRIDGE_Settings)
    if not bpy.app.timers.is_registered(_watch):
        bpy.app.timers.register(_watch, first_interval=2.0, persistent=True)


def unregister():
    if bpy.app.timers.is_registered(_watch):
        bpy.app.timers.unregister(_watch)
    del bpy.types.Scene.paint_bridge
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
