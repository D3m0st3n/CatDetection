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

---

## Technical Specifications

An exhaustive inventory of every tool used in the project, with the reason each was chosen.

### Language & Runtime

| Tool | Version | Role | Why |
|------|---------|------|-----|
| **Python** | 3.11 (min 3.10) | Project language | 3.10 is the minimum required by `ultralytics`; 3.11 offers measurable performance improvements. |
| **Conda** | any | Environment management | Provides an isolated, reproducible environment. Easier than `venv` for managing CUDA-dependent packages like PyTorch. |

---

### ML & Training

| Tool | Role | Why |
|------|------|-----|
| **Ultralytics YOLOv8** (`ultralytics`) | Object detection model and training API | Industry-standard real-time single-stage detector. Fully Python-native and pip-installable — no Darknet/C++ toolchain. YOLOv8s (`yolov8s.pt`) chosen as the base: a good balance of accuracy and speed for a 2-class fine-tune on a modest dataset. |
| **PyTorch** (`torch`, `torchvision`) | Deep learning backend | Required by `ultralytics`. Installed separately before `ultralytics` to guarantee CUDA support is not silently overridden by a CPU-only wheel. |
| **CUDA** (via PyTorch CUDA wheel) | GPU acceleration | Reduces each training stage from hours (CPU) to minutes. Used via `device=0` in YOLOv8 training calls. |
| **Albumentations** (`albumentations`) | Offline data augmentation | Bbox-aware augmentation: applies geometric and photometric transforms to both image and bounding box coordinates simultaneously. Generates ~9 augmented copies per training image (8–10× expansion), which is critical given the ~255 source photos. Chosen over `imgaug` for its speed and actively maintained API. |
| **scikit-learn** (`sklearn.model_selection.train_test_split`) | Stratified train/val/test splitting | Single function call handles stratified 80/10/10 splitting by image category (single-cat, both-cats, no-cat), ensuring each split has a representative class mix. |
| **PyYAML** (`yaml`) | Dataset configuration | Generates `data/data.yaml` — the configuration file YOLOv8 requires to locate images and class names. |

---

### Image Processing

| Tool | Role | Why |
|------|------|-----|
| **OpenCV** (`opencv-python` / `cv2`) | Image I/O, resizing, bbox drawing, colour operations | Core image manipulation throughout `src/utils.py`, `src/preprocess.py`, `src/infer.py`, and `app/gui.py`. Handles BGR array operations, rectangle and text drawing for bounding box overlays, and colour space conversion (RGB↔BGR). |
| **Pillow** (`Pillow` / `PIL`) | Image loading with EXIF correction | `PIL.ImageOps.exif_transpose` is used everywhere images are loaded (annotation notebook, preprocessing, GUI) to silently correct the orientation of WhatsApp-origin portrait photos before any processing occurs. OpenCV does not handle EXIF rotation. |
| **NumPy** (`numpy`) | Array operations | Underpins all image data manipulation as ndarray. Used directly in `src/utils.py`, `src/annotate.py`, and `app/gui.py`. |

---

### Annotation UI

| Tool | Role | Why |
|------|------|-----|
| **Jupyter Notebook** (`notebook`) | Annotation runtime | Browser-based Jupyter is required for the `%matplotlib widget` interactive backend. VS Code's notebook editor is explicitly incompatible with `ipympl` (see design decision #20). |
| **ipympl** | Interactive matplotlib backend | Enables click-and-drag interaction on a matplotlib canvas inside Jupyter. The `%matplotlib widget` magic activates it. Essential for the bounding box drawing workflow. |
| **matplotlib** (`matplotlib`) | Annotation canvas and results visualisation | Displays images one at a time for annotation; renders bounding boxes (saved as dashed, pending as solid). Also used in `notebooks/02_training_results.ipynb` for loss/mAP curves and sample prediction grids. |
| **ipywidgets** (`ipywidgets`) | Annotation controls | Provides the dropdown (cat selector), buttons (Confirm, Delete Last, Mark Empty, Skip, Previous, Next), and status label that make up the annotation UI around the matplotlib canvas. |

---

### Inference App

| Tool | Role | Why |
|------|------|-----|
| **Gradio** (`gradio`) | Browser-based inference GUI | Industry standard for local ML demo apps (used widely on Hugging Face). `pip install gradio` is the only setup needed — it serves a full browser UI with file upload, image display, sliders, and buttons. Chosen over Streamlit for its simpler image-in/image-out interaction model. |

---

### CLI & Progress

| Tool | Role | Why |
|------|------|-----|
| **argparse** (stdlib) | CLI entry point | Standard library — no extra dependency. Every subcommand gets a `--help` description automatically. Used in `app/gui.py` for `--weights` and `--port` flags; will be the backbone of `cli.py` (Phase 6). |
| **tqdm** (`tqdm`) | Progress bars | Wraps iterators in preprocessing loops to show per-image progress during the augmentation and split steps. Zero-config and compatible with both notebook and terminal contexts. |
| **logging** (stdlib) | Structured logging | Used throughout `src/` instead of bare `print()`. A shared `setup_logging()` helper in `src/utils.py` configures a consistent `[LEVEL] timestamp — message` format. Four levels used consistently: `DEBUG` (per-step detail), `INFO` (milestones), `WARNING` (recoverable issues), `ERROR` (failures). |

---

### Code Quality

| Tool | Role | Why |
|------|------|-----|
| **black** | Code formatter | Opinionated, zero-config formatter. Eliminates style debates. Line length: 88 (black default). Configured in `pyproject.toml`. |
| **ruff** | Linter + import sorter | Replaces `flake8` + `isort` in a single fast tool. Rules enabled: `E` (pycodestyle errors), `F` (pyflakes unused imports/variables), `I` (isort import ordering). Configured in `pyproject.toml`. |
| **pytest** | Unit testing | Industry-standard Python test framework. All tests in `tests/` use synthetic images and fixtures — no dependency on the real photo dataset. Test paths configured in `pyproject.toml` so `pytest` with no arguments runs the full suite. |

---

### Configuration

| File | Format | Role |
|------|--------|------|
| `pyproject.toml` | TOML | Centralises tool configuration for `black`, `ruff`, and `pytest` in one file. No separate `.flake8`, `setup.cfg`, or `pytest.ini` needed. |
| `data/data.yaml` | YAML | YOLOv8 dataset manifest: paths to train/val/test image directories, number of classes (`nc: 2`), and class names (`['aioli', 'mayo']`). Generated by `src/preprocess.py` and committed to git (it is configuration, not data). |
| `requirements.txt` | pip format | Runtime dependencies. Intentionally excludes dev tools. |
| `requirements-dev.txt` | pip format | Dev-only tools (`black`, `ruff`, `pytest`). Kept separate so a user who only wants to run inference does not need to install the full toolchain. |

---

### Version Control & Hosting

| Tool | Role | Why |
|------|------|-----|
| **git** | Version control | Standard. `.gitignore` excludes all large binary/generated artefacts (`data/raw/`, `data/images/`, `data/labels/`, `runs/`, `*.pt`, `outputs/`, `.ipynb_checkpoints/`) — only source code and configuration are tracked. |
| **GitHub** | Remote hosting | Hosts the repository at `github.com/D3m0st3n/CatDetection`. |
| **Conventional Commits** | Commit message convention | Structured prefix (`feat:`, `fix:`, `docs:`, `chore:`, etc.) makes the git log machine-readable and clarifies intent at a glance. |
