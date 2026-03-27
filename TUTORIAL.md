# CatDetection — User Tutorial

This tutorial walks you through the full CatDetection pipeline from scratch: setting up your environment, annotating photos, training the model, evaluating results, and running the inference app. It is written for someone coming to the project fresh.

---

## What This Project Does

CatDetection is a fully local machine learning pipeline that detects and identifies two specific cats — **Aïoli** and **Mayo** — by name in photos. It fine-tunes a pretrained [YOLOv8s](https://docs.ultralytics.com/) model using transfer learning on a small personal photo dataset, then serves inference through a browser-based Gradio GUI and (once complete) a CLI.

The pipeline covers the full ML lifecycle:
1. Interactive photo annotation
2. Image preprocessing and data augmentation
3. Two-stage model fine-tuning
4. Evaluation and failure analysis
5. Local inference via a web GUI

**Current status**: Phases 0–5 are complete. The CLI (Phase 6) is the next planned milestone.

---

## Prerequisites

Before starting, make sure you have the following installed on your machine:

- **Python 3.11** (minimum 3.10)
- **Conda** (Anaconda or Miniconda)
- **Git**
- An **NVIDIA GPU** with CUDA support (strongly recommended for training; CPU training is possible but slow)

Check your CUDA version by running `nvidia-smi`. You will need this when installing PyTorch.

---

## 1. Environment Setup

Clone the repository and create a dedicated conda environment:

```bash
git clone <repo-url>
cd CatDetection
conda create -n catdetection python=3.11 -y
conda activate catdetection
```

**Important — install PyTorch with CUDA first**, before installing other requirements. The `ultralytics` package will otherwise pull in a CPU-only version of PyTorch that cannot be overridden cleanly:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Replace `cu128` with your CUDA version if needed (e.g. `cu118` for CUDA 11.8). The CUDA version shown by `nvidia-smi` is the *driver* version; the toolkit version (used for the wheel URL) may be slightly lower.

Then install the project dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt   # optional — only needed for testing and linting
```

Verify GPU support is working:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

This should print `True` followed by your GPU name. If it prints `False`, reinstall PyTorch with the correct CUDA wheel URL and try again.

---

## 2. Project Structure

```
CatDetection/
├── data/                  # Dataset artefacts (raw photos, labels, processed splits)
├── notebooks/             # Jupyter workflows: annotation and training results
├── src/                   # Core Python library (used by notebooks, app, and CLI)
├── app/                   # Gradio inference GUI
├── tests/                 # pytest unit tests for src/
├── runs/                  # YOLOv8 training outputs — weights, metrics (gitignored)
├── outputs/               # Saved inference results from the GUI
├── run_training.py        # Convenience script to kick off training
├── pyproject.toml         # Tool config: black, ruff, pytest
├── requirements.txt       # Runtime dependencies
└── requirements-dev.txt   # Dev-only dependencies
```

The `src/` library is the heart of the project. The notebooks, the Gradio app, and the CLI are all thin wrappers around it. Each folder also contains an `AGENTS.md` file with a short description of that folder's purpose and conventions.

---

## 3. Adapting the Project to Your Own Cats

The two class labels are `aioli` (class 0) and `mayo` (class 1). These are defined in `data/data.yaml` and as constants in `src/utils.py`. To use different subjects, you would need to:

1. Replace the class names in `data/data.yaml` (`names: ['aioli', 'mayo']`).
2. Update the display names and colour constants in `src/utils.py` (`AIOLI_COLOR`, `MAYO_COLOR`, `CLASS_COLORS`).
3. Re-annotate with your own photos (Phase 1 below).

Everything else in the pipeline is generic and works without modification.

---

## 4. Phase 1 — Annotating Photos

**Skip this phase if you already have label files in `data/labels/`.**

Place your raw photos in `data/raw/`. Then open the annotation notebook:

```bash
conda activate catdetection
python -m notebook notebooks/01_annotate.ipynb
```

> **Important**: This notebook uses the `%matplotlib widget` interactive backend (`ipympl`), which does **not** work in VS Code's built-in notebook editor. Always run it in browser-based Jupyter as shown above.

### How the annotation tool works

- The notebook opens with a progress summary: "X of Y images annotated — Z remaining."
- Each image is displayed one at a time. Click and drag on the canvas to draw a bounding box around a cat.
- Select the correct cat name from the dropdown (`Aïoli` / `Mayo`) before drawing each box.
- An image with both cats gets two boxes — draw them one at a time.
- Use the control buttons:
  - **Confirm** — saves the current boxes and advances to the next image.
  - **Delete Last** — removes the most recent pending box.
  - **Mark Empty** — saves an empty label file for images that contain no cats.
  - **Skip** — moves to the next image without saving (the image will remain in the "remaining" count).
  - **Previous / Next** — navigate freely between images.
- EXIF orientation is handled automatically, so portrait-mode phone photos display correctly.
- Progress is saved immediately on confirmation — restarting the kernel or closing the browser loses nothing.

### What gets produced

Each annotated image produces a `.txt` label file in `data/labels/` in YOLO format:

```
<class_id> <x_center> <y_center> <width> <height>
```

All values are normalised to `[0, 1]` relative to image dimensions. Class 0 = Aïoli, class 1 = Mayo. An image with no cats gets an empty (but present) label file.

---

## 5. Phase 2 — Preprocessing & Augmentation

Once annotation is complete, run the preprocessing pipeline to resize images, apply offline augmentation, and generate the train/val/test splits:

```python
from pathlib import Path
from src.preprocess import run_preprocessing

run_preprocessing(Path('data/raw'), Path('data/labels'), Path('data'))
```

This does the following:
- Resizes all images to 640×640 (YOLOv8's default input size).
- Applies bbox-aware offline augmentation using `albumentations`, generating ~9 augmented copies per training image (target: 8–10× dataset expansion).
- Stratifies the 80/10/10 train/val/test split by image category (single-cat, both-cats, no-cat) to ensure each split gets a representative mix.
- Writes output to `data/images/{train,val,test}/` and `data/labels/{train,val,test}/`.
- Generates `data/data.yaml` for YOLOv8.

**Note**: Augmentation is only applied to the training split. Validation and test sets use unaugmented resized originals, so metrics reflect real-world generalisation.

The transforms applied offline are: horizontal flip, rotation (±15°), brightness/contrast jitter, Gaussian blur, random crop and resize, and HSV colour shift. YOLOv8 adds further online augmentation (mosaic, HSV jitter, scale/translate) at training time automatically.

---

## 6. Phase 3 — Training

**Prerequisite**: `data/images/train/`, `data/images/val/`, and `data/data.yaml` must all exist (Phase 2 complete).

Run both training stages with default hyperparameters:

```python
from src.train import run_training
from src.utils import setup_logging
import logging

setup_logging(logging.INFO)
result = run_training()

print(result.best_weights)   # Path to best.pt — use this for inference
print(result.results_csv)    # Path to per-epoch metrics CSV
```

Or use the convenience script at the project root:

```bash
python run_training.py
```

### How training works

Training uses a two-stage strategy:

**Stage 1 — Frozen backbone** (20 epochs, lr=0.01): The YOLOv8 backbone (pre-trained on millions of images) is frozen. Only the detection head is trained. This warms up the head without disrupting the backbone's general feature representations.

**Stage 2 — Full fine-tune** (50 epochs, lr=0.001): All layers are unfrozen and the whole network is trained at a lower learning rate. This refines the backbone to focus on cat-specific features.

YOLOv8 automatically saves training outputs to `runs/detect/train/` (and `train2`, `train3`, etc. on subsequent runs). `run_training()` resolves the latest run directory automatically.

### Custom hyperparameters

```python
result = run_training(
    stage1_epochs=30,
    stage1_freeze=10,
    stage1_lr0=0.01,
    stage2_epochs=80,
    stage2_lr0=0.0005,
    device="cpu",    # Use "cpu" if no GPU is available (much slower)
    imgsz=640,
)
```

### Running individual stages

```python
from src.train import run_stage1, run_stage2

stage1_dir = run_stage1(epochs=20)
stage2_dir = run_stage2(stage1_dir=stage1_dir, epochs=50)
```

### What training produces

The `run_training()` function returns a `TrainingResult` dataclass with:

| Field | Description |
|-------|-------------|
| `stage1_dir` | Path to the Stage 1 run directory |
| `stage2_dir` | Path to the Stage 2 run directory |
| `best_weights` | Path to `best.pt` from Stage 2 — **use this for inference** |
| `last_weights` | Path to `last.pt` from Stage 2 |
| `results_csv` | Path to `results.csv` with per-epoch loss and mAP |

Training time on a modern NVIDIA GPU (e.g. RTX 5070) is typically a few minutes per stage.

---

## 7. Phase 4 — Evaluating Results

After training, explore model performance in the training results notebook:

```bash
python -m notebook notebooks/02_training_results.ipynb
```

The notebook shows:
- **Loss and mAP curves** over both training stages (plotted from `results.csv`).
- **YOLOv8 auto-generated plots**: confusion matrix, precision-recall curve, F1 curve.
- **Sample predictions** from the test set with overlaid bounding boxes and confidence scores.
- **Failure gallery**: a categorised view of errors — identity swaps (Aïoli detected as Mayo or vice versa), missed detections, and false positives.

You can also run evaluation programmatically:

```python
from src.evaluate import run_evaluation
from pathlib import Path

run_evaluation(
    weights=Path('runs/detect/train2/weights/best.pt'),
    data_yaml=Path('data/data.yaml'),
)
```

### Known model characteristics (current training run)

The current model was trained on 255 source photos expanded to 2091 images via augmentation. Validation mAP@0.5 peaked at ~0.758 during Stage 1 and ~0.748 during Stage 2 — close to the ≥0.75 project target. The main weakness is **missed detections** (the model is conservative at the default 0.5 confidence threshold). If you see too many misses, try:

- Lowering the confidence threshold in the GUI (using the slider).
- Retraining with more epochs or a different augmentation multiplier.
- Adding more source photos, particularly both-cats images.

---

## 8. Phase 5 — Running the Inference App

The Gradio inference GUI lets you upload any photo and see the model's predictions with bounding boxes overlaid.

```bash
conda activate catdetection
python app/gui.py
```

This opens a local browser tab. The interface has:
- **Left column**: drag-and-drop or file picker to upload an image.
- **Right column**: the same image with bounding boxes and cat name labels overlaid.
- **Detection summary** below: plain-English description of what was found:
  - *"Neither Aïoli nor Mayo was detected."*
  - *"Aïoli detected (confidence: X%)."*
  - *"Mayo detected (confidence: X%)."*
  - *"Both Aïoli and Mayo detected."*
- **Confidence threshold slider** (default: 0.5): drag left to detect more (at the cost of more false positives), right to require higher confidence.
- **Save result button**: exports the annotated image to `outputs/`.

The app automatically loads the best weights from the latest training run in `runs/`. If you have retrained and want to use a specific checkpoint, edit `app/gui.py` to point to the desired `best.pt` path.

### Bounding box colour coding

| Cat | Colour |
|-----|--------|
| Aïoli | Orange `#FF6B35` |
| Mayo | Teal `#4ECDC4` |

These colours are consistent across the annotation notebook, training results notebook, and the GUI.

---

## 9. Running Tests

The project includes a full pytest suite for the `src/` library. Tests use synthetic images and never depend on the actual photo dataset.

```bash
conda activate catdetection
pytest tests/
```

Test files and what they cover:

- `tests/test_utils.py` — bbox conversion, YOLO label I/O, drawing utilities (14 tests)
- `tests/test_preprocess.py` — augmentation transforms, train/val/test split logic (21 tests)
- `tests/test_train.py` — two-stage training orchestration, run directory resolution (20 tests)
- `tests/test_infer.py` — model loading, inference output parsing, prediction drawing (13 tests)

The Gradio app and notebooks are not unit-tested (they depend on interactive widget infrastructure that is impractical to unit-test).

---

## 10. Code Standards (for Contributors)

All code in `src/`, `app/`, and `cli.py` follows these conventions:

- **Style**: PEP 8, enforced by `black` (formatter) and `ruff` (linter + import sorter). Line length: 88.
- **Type hints**: All public functions use PEP 484 annotations.
- **Docstrings**: Google-style on all public functions and classes.
- **Logging**: Use the `logging` module — no bare `print()` in library code. Call `setup_logging()` at entry points.
- **Constants**: No magic numbers. All thresholds are named constants in `src/utils.py`.

Run the formatter and linter before committing:

```bash
black src/ app/ tests/
ruff check src/ app/ tests/
```

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`, `chore:`, `test:`, `refactor:`.

---

## 11. Common Gotchas

**`torch.cuda.is_available()` returns `False`**
PyTorch was installed without CUDA support. Reinstall it:
```bash
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**Annotation notebook fails with a `WidgetManager.loadClass` error**
You are running the notebook in VS Code. The `%matplotlib widget` backend is not supported there. Use browser Jupyter instead:
```bash
python -m notebook notebooks/01_annotate.ipynb
```

**`ModuleNotFoundError: No module named 'src'` when running `app/gui.py`**
This is expected if you try to import from a script in a subdirectory. `app/gui.py` already handles this with a `sys.path.insert` at the top of the file. If you create new scripts inside subdirectories that import from `src`, add the same line:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

**Images appear rotated in the annotation notebook**
This should not happen — EXIF orientation is handled automatically via `PIL.ImageOps.exif_transpose`. If you see rotated images, ensure you are running the latest version of `src/annotate.py`.

**Training starts a new `train3`, `train4`... folder every run**
This is YOLOv8's default behaviour. `run_training()` always resolves the latest run directory automatically. If you want to evaluate or use a specific run, locate the `best.pt` inside that run's `weights/` subfolder and pass it explicitly.

---

## 12. What's Next (Phase 6 — CLI)

The CLI (`cli.py`) is currently planned but not yet implemented. Once complete, it will expose all pipeline steps as subcommands:

| Command | Description |
|---------|-------------|
| `python cli.py annotate` | Launch the annotation notebook |
| `python cli.py preprocess` | Run preprocessing and augmentation |
| `python cli.py train` | Run both training stages |
| `python cli.py evaluate` | Evaluate on the test set |
| `python cli.py infer --image photo.jpg` | Run inference on a single image |
| `python cli.py app` | Launch the Gradio GUI |

All subcommands will support `--help` and a `--verbose` flag for debug logging.

---

## Key Files Reference

| File / Folder | Purpose |
|---------------|---------|
| `ROADMAP.md` | Goals, state-of-the-art context, and phase-by-phase milestones |
| `CONTRIBUTING.md` | Architecture, code standards, data formats, git conventions |
| `DESIGN.md` | UX, visual style, naming, and interface behaviour |
| `design_decisions.md` | Numbered log of every decision made and why |
| `src/annotate.py` | Annotation session and widget logic |
| `src/preprocess.py` | Resize, augment, split, generate `data.yaml` |
| `src/train.py` | Two-stage YOLOv8 fine-tuning |
| `src/evaluate.py` | Metrics parsing, IoU matching, error categorisation |
| `src/infer.py` | Model loading and inference |
| `src/utils.py` | Shared constants, logging, bbox utilities |
| `app/gui.py` | Gradio inference app |
| `notebooks/01_annotate.ipynb` | Interactive annotation tool |
| `notebooks/02_training_results.ipynb` | Training curves and failure analysis |
| `data/data.yaml` | YOLOv8 dataset config (committed to git) |
| `runs/` | Training outputs — weights, metrics (gitignored) |
| `outputs/` | Saved GUI inference results |
