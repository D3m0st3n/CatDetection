# Design

This document describes how the project should look, feel, and behave from a user's perspective. It covers UX, naming, visual output, and interface decisions.

---

## Naming

| Entity | Label used in code and UI |
|--------|--------------------------|
| Cat 1  | `aioli` (class 0) |
| Cat 2  | `mayo` (class 1) |

- Display names (in the GUI and notebook outputs) use the accented form: **Aïoli** and **Mayo**.
- Internal identifiers (YOLO class names in `data.yaml`, file names, variable names) use plain ASCII: `aioli`, `mayo`.

---

## Annotation Notebook (`notebooks/01_annotate.ipynb`)

**Status**: Implemented and annotation complete (255/255 images annotated; Aïoli: 173 boxes, Mayo: 174 boxes).

**UX goal**: make annotation as frictionless as possible so the user can get through ~255 photos in a few sessions.

**Implementation**: The notebook is a thin wrapper around `src/annotate.py`, which provides two classes:
- `AnnotationSession` — state management, navigation, and label persistence.
- `AnnotationWidget` — interactive matplotlib (`%matplotlib widget` via `ipympl`) canvas for click-and-drag rectangle drawing, with `ipywidgets` controls.

**Workflow**:
- Opens with a progress summary: "X of Y images annotated — Z remaining."
- Displays one image at a time on a matplotlib canvas (12×9 figure, toolbar hidden).
- The user draws bounding boxes by click-and-drag on the canvas; the selected cat label is read from a dropdown (`Aïoli` / `Mayo`).
- Multiple boxes per image are supported (for photos with both cats). A pending box list is shown below the controls.
- Navigation: **Previous** / **Next** / **Skip** buttons. Skipped images are tracked and counted in the status bar.
- Action buttons: **Confirm** (save + advance), **Delete Last** (remove most recent pending box), **Mark Empty** (save an empty label file for images with no cats).
- Completed label files are saved immediately on confirmation — progress is never lost if the session ends or the kernel restarts.
- Previously saved labels are shown as dashed boxes when revisiting an image; pending (unsaved) boxes are shown as solid.
- EXIF rotation is handled automatically (`PIL.ImageOps.exif_transpose`) for WhatsApp-origin photos.
- Minimum box size of 5px rejects accidental clicks.
- A final summary cell shows the class distribution (how many appearances of each cat).
- On restart, the notebook resumes at the first unannotated image.

**Known limitation**: The `%matplotlib widget` backend does not work in VS Code's notebook editor (see design decision #20). The annotation notebook must be run in browser-based Jupyter: `conda activate catdetection && python -m notebook notebooks/01_annotate.ipynb`.

---

## Training Results Notebook (`notebooks/02_training_results.ipynb`)

**UX goal**: give a clear, honest picture of how well the model learned — not just good results, but failures too.

- Section 1: Loss and mAP curves over epochs (clean matplotlib plots, both stages shown).
- Section 2: YOLOv8 auto-generated plots (confusion matrix, PR curve, F1 curve) displayed inline.
- Section 3: Sample predictions from the test set — images displayed in a grid with overlaid bounding boxes and confidence scores.
- Section 4: Failure gallery — images where the model made an error, categorised:
  - Identity swap (Aïoli detected as Mayo or vice versa)
  - Missed detection (a cat present but not detected)
  - False positive (no cat present but one detected)

---

## Inference App (`app/gui.py`)

**Framework**: Gradio (browser-based local app).

**Layout**:
- Left column: image upload area (drag-and-drop or file picker).
- Right column: the same image with bounding boxes and labels overlaid.
- Below the image: a plain-English detection summary.

**Detection summary messages**:
| Detections | Message |
|------------|---------|
| 0 | "Neither Aïoli nor Mayo was detected." |
| Aïoli only | "Aïoli detected (confidence: X%)." |
| Mayo only | "Mayo detected (confidence: X%)." |
| Both | "Both Aïoli and Mayo detected." |

- Bounding box colours are distinct per cat (e.g. one colour for Aïoli, another for Mayo) and consistent across all uses (notebook, GUI, CLI output images).
- Confidence scores are shown on the bounding box label: `Aïoli 94%`.
- A "Save result" button exports the annotated image to disk.
- A confidence threshold slider (default: 0.5) lets the user tune sensitivity without restarting.

---

## CLI (`cli.py`)

**Design principle**: every command should be self-explanatory with `--help` and produce clear, human-readable output.

- All subcommands follow the pattern: `python cli.py <command> [options]`
- Progress is shown for long-running steps (annotation export, preprocessing, training) using `tqdm` progress bars.
- `--verbose` flag available on all commands to switch from `INFO` to `DEBUG` logging.
- Errors print a short human-readable message to `stderr` and exit with a non-zero code.
- No silent failures — if a label file is missing, a warning is logged and the image is skipped (not silently dropped).

**Colour coding** (if the terminal supports it):
- Green for success milestones.
- Yellow for warnings (e.g. class imbalance detected, image skipped).
- Red for errors.

---

## Bounding Box Visual Style

Applies consistently across the annotation notebook, training results notebook, CLI output images, and Gradio app.

| Cat | Box colour | Label background |
|-----|-----------|-----------------|
| Aïoli | `#FF6B35` (orange) | Same |
| Mayo | `#4ECDC4` (teal) | Same |

- Box line thickness: 2px.
- Label font: white text on coloured background, same font size as image resolution allows.
- These colours are defined as constants in `src/utils.py` (`AIOLI_COLOR`, `MAYO_COLOR`, `CLASS_COLORS`, `BOX_THICKNESS`) and used everywhere — never hardcoded inline.
- In the annotation notebook, saved boxes use dashed outlines (alpha 0.5) and pending boxes use solid outlines (alpha 1.0) to visually distinguish them.
