# Paint Bridge

Paint a mesh's base color texture in an external 2D app (any app that edits layered PSD files).
Blender does no painting: it captures views, and merges the result back.

## Files

All in the Project Folder (default `//paint_bridge/`):

- **Main PSD** (`<texture>_main.psd`): the texture's layers in UV space, at texture size. It is the
  source of the texture: the texture is the composite of its layers. It is written from the
  current texture the first time it is needed, as one `Base` layer plus a locked `UV Wireframe`
  layer on top (never part of the texture). Add, rename, reorder or paint layers in it freely.
- **Capture PSDs**, one per camera angle. Each loose part of the mesh gets a closed group folder,
  named after the vertex group that weights it most (`Part 01`, ... without vertex groups), with a
  mask cut to the part's visible outline. Inside are the main PSD's layers, drawn from that angle,
  with the same names, order and opacity. A locked `Wireframe` of the visible edges sits on top.
  With Options > Layer per Part off, the layers are not split into folders.
- Sidecar files next to each capture (`.json`, `.geo.npz`, `.ref.npz`, `.applied.npz`) hold the
  camera, the mesh as it was at capture time, the layers as captured, and the undo data for the
  last merge. Keep them with the PSD.

## Workflow

1. Select the mesh and open the **Paint Bridge** tab in the 3D Viewport sidebar.
2. Frame the view (or look through the scene camera to use the render resolution) and press **+**
   next to the capture list. Name the capture (the suggested name is kept if you just confirm).
3. **Edit Capture** recaptures the selected capture from the viewport's current angle and the
   latest main PSD, then opens it in your paint app. (Clicking a capture turns the viewport to its
   angle, so if you do not move the view it stays the same.) If it still holds paint that was not merged,
   that paint is merged first, so nothing is lost. Paint on its existing layers; do not crop or
   resize the canvas.
4. Save, then press **Merge** (or turn on **Auto Merge on Save**). Each layer's changes are
   projected onto the same layer of the main PSD, and the texture is rebuilt from it. Erasing on a
   capture layer erases the main layer there.

**Edit Main** opens the main PSD. Saving it updates the texture, and the other captures, within a
couple of seconds. Layers added there appear in captures made (or recaptured) afterwards.

Merging the same capture again replaces its previous merge, so you can save and merge as often as
you like. **Revert** removes the capture's last merge from the main PSD.

After every merge, **Refresh Other Captures** (Options, on by default) redraws the layers of every
other capture of the same texture from the main PSD, so each angle shows the latest texture. Only
pixels that still match what was captured are replaced; anything painted there since stays. The
file is edited in place, so layers this add-on does not read (adjustment layers and the like) survive.

The capture list always shows the capture PSDs in the Project Folder; files added, renamed or
deleted there show up within a second. Clicking a capture turns the viewport to its angle. The
buttons next to the list:
- **+** captures the current view to a new PSD.
- **−** deletes the selected capture right away (to the Recycle Bin on Windows).
- **Recapture** (camera icon) overwrites the selected capture with the current view, after asking;
  unmerged paint in it is lost.
- The folder button shows the capture in the file browser.

PSDs open in the Paint App set in Options (gear icon at the top right of the panel), else the
Image Editor in Preferences > File Paths, else the system default app.

**Isolating parts** follows what the viewport shows in Edit Mode: faces hidden with H (or
Shift+H) are left out; selection does not matter. In Object Mode nothing is left out.
- **+** then captures only the shown faces; hidden parts neither show nor block them. Recapture
  keeps each capture's own isolation, even from Object Mode.
- **Edit Capture** opens a PSD holding only the folders of the parts still shown
  (`<capture>@<part>.psd`, written next to the capture the first time). Merge also takes the paint
  from these per-part PSDs; deleting the capture deletes them too.

The mode buttons switch between Object Mode, **Select Part** (Edit Mode; click a part to select all
of it, Shift adds, Ctrl removes, empty space clears; then H hides it) and **Select Vertex** (Edit
Mode vertex select).

**Mirror...** opens a popup with **Mirror Texture**, which copies every layer of the main PSD from
one side of the mesh to the other across the object's local X, Y or Z axis, using the mesh's rest
shape (pose and modifiers are ignored). It works with any UV layout, including halves with separate
UV islands. The texture and the main PSD are saved to `<Project Folder>/backup/` first.
With **Auto Mirror on Merge**, each merge is also copied to the mirrored side. Only what was
painted is copied, and where both sides were painted each side keeps its own paint.

**Seam Fill** (in the **Fill Seams...** popup, default 16 texels) runs whenever the texture is
rebuilt: the gutter around every UV island is filled with the nearest island color, so texture
filtering never pulls old gutter color in along seams. **Fill Seams** does the same on demand.

## Notes

- Paint lands only on surfaces visible from the captured angle. It fades on surfaces turned away
  from the view (Options > Fade Start / Fade End).
- Faces that use another material than the texture's appear as holes in capture layers; they
  still hide what is behind them.
- The main PSD is 8-bit, so a float (EXR) texture is rebuilt at 8-bit precision.
- Main and capture PSDs must stay 8-bit for in-place updates. Blend modes other than Normal are
  treated as Normal when merging.
- Supported PSD: RGB, 8 or 16-bit, raw/RLE/ZIP channels. PSB and 32-bit files are not.
- UDIM tiles are not supported; UVs outside 0-1 are ignored.
- Captures made by earlier versions (`Base`/`Paint` folders or a `Blender Reference` layer) still
  merge straight into the texture; recapture them to use the main PSD.

## Test

```
blender --factory-startup --python tests/e2e_test.py -- <output_dir>
```

Needs a window (GPU); results are written to `<output_dir>/results.txt`.
# BridgePaint
