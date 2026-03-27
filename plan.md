# Cat Detection Project — Planning Document

> **Status: superseded.** This document was the working plan produced during the initial brainstorm. Its content has been split into three dedicated files:
> - **`ROADMAP.md`** — goals, state-of-the-art context, and phase milestones
> - **`DESIGN.md`** — UX, visual style, naming, and interface behaviour
> - **`CONTRIBUTING.md`** — architecture, code standards, data formats, and git conventions
>
> `plan.md` is retained as a historical reference. For active development, use the files above.

---

A machine learning pipeline to annotate photos of 2 cats (**Aïoli** and **Mayo**), fine-tune a pretrained YOLO model to identify each cat by name, and serve inference through a local GUI — all runnable via CLI as well.

The dataset consists entirely of existing photos (no video), taken in various lighting and environmental conditions. Since the photo library may be limited in size, **data augmentation is a core part of the pipeline** to ensure the model generalises well. The model must handle multi-instance detection: it can return 0, 1, or 2 detections per image, each labelled with the correct cat's name.

**Inspired by**: [DennisFaucher/ChickenDetection](https://github.com/DennisFaucher/ChickenDetection), which uses a similar workflow to recognize individual chickens using YOLO and transfer learning from pretrained weights.

---

## Goals

- **Annotation**: Label images of each cat using an interactive Jupyter notebook interface, producing YOLO-format label files.
- **Training**: Fine-tune a pretrained YOLOv8 model (transfer learning) to identify each individual cat.
- **Inference**: Submit new images through a local GUI and visualize bounding boxes + cat name labels.
- **CLI**: Every major step (annotation export, training, inference) is also invokable from the command line.

---

## Why Fine-Tune Instead of Training from Scratch

The ChickenDetection project demonstrates that training a custom YOLO model from scratch requires significant compute (AWS GPU instance) and hundreds of images per class. Fine-tuning a pretrained YOLOv8 model instead:

- Requires far fewer labeled images per cat (30–100 is often sufficient)
- Trains much faster — even on CPU or a modest GPU
- Leverages features already learned from millions of images (shapes, edges, textures)
- Produces better accuracy with less data

The approach: **freeze the backbone**, train only the detection head first, then **unfreeze all layers** for a lower-learning-rate fine-tuning pass.

---

## Model Choice: YOLOv8 (Ultralytics)

| Property | Detail |
|----------|--------|
| Framework | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (pure Python, pip-installable) |
| Pretrained weights | `yolov8s.pt` (small) — good balance of accuracy and speed for a 2-class fine-tune on GPU |
| Annotation format | YOLO `.txt` label files (one per image) with a `data.yaml` config |
| Training API | Python API + CLI (`yolo train ...`) |
| Inference API | Python API + CLI (`yolo predict ...`) |

YOLOv8 is the modern, fully Python-native successor to the Darknet-based YOLOv3 used in the ChickenDetection project.

---

## Project Structure

```
CatDetection/
├── data/
│   ├── raw/                  # Original photos (gitignored)
│   ├── labels/               # YOLO .txt label files (one per image, gitignored)
│   ├── images/               # Resized + split images (gitignored)
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── data.yaml             # YOLOv8 dataset config (class names, paths)
│
├── notebooks/
│   ├── 01_annotate.ipynb     # Interactive bounding box annotation tool
│   └── 02_training_results.ipynb  # Training curves, metrics, and sample predictions
│
├── src/
│   ├── annotate.py           # Annotation helper (image display + bbox widget logic)
│   ├── preprocess.py         # Image resizing, augmentation, splitting, data.yaml generation
│   ├── train.py              # Fine-tuning script (frozen backbone → full fine-tune)
│   ├── evaluate.py           # Load results, compute mAP, print per-class metrics
│   ├── infer.py              # Inference engine (wraps YOLOv8 predict)
│   └── utils.py              # Shared helpers (bbox drawing, label I/O, logging setup)
│
├── app/
│   └── gui.py                # Gradio inference app
│
├── tests/
│   ├── test_preprocess.py    # Unit tests for augmentation and splitting logic
│   ├── test_infer.py         # Unit tests for inference output parsing
│   └── test_utils.py         # Unit tests for shared utilities
│
├── runs/                     # YOLOv8 training outputs (gitignored)
│
├── cli.py                    # Unified CLI entry point (argparse)
├── pyproject.toml            # Tool config: black, ruff, pytest
├── requirements.txt          # Runtime dependencies
├── requirements-dev.txt      # Dev-only dependencies (black, ruff, pytest)
└── plan.md
```

---

## Phase 1 — Data Collection & Annotation

**Goal**: Produce YOLO-format labeled images for each cat.

**Inspired by ChickenDetection**: that project annotated frames extracted from video footage. Here we work directly from an existing photo library taken under various conditions (different rooms, lighting, angles, distances).

- Organize raw photos under `data/raw/` (no need to pre-sort by cat — images may contain Aïoli, Mayo, or both)
- Use `notebooks/01_annotate.ipynb` to draw bounding boxes interactively:
  - Display each image with `ipywidgets` canvas or `matplotlib`
  - Draw **one bounding box per visible cat** in the image and assign the correct cat name label
  - Images with both cats present get **two bounding box entries** in their label file
  - Save in **YOLO format**: one `.txt` file per image, one row per detection: `class x_center y_center width height` (normalized 0–1)
  - Example label file for a photo with both cats:
    ```
    0 0.35 0.52 0.18 0.30   # Aïoli
    1 0.72 0.48 0.20 0.28   # Mayo
    ```
- The annotation notebook should track progress (annotated vs. remaining) to make it easy to resume sessions
- CLI equivalent: `python cli.py annotate --image-dir data/raw --output data/labels`

**Key libraries**: `ipywidgets`, `matplotlib`, `Pillow`, `opencv-python`

**Dataset notes**:
- Expected dataset: ~100 photos total, split across Aïoli, Mayo, and both — roughly 50 appearances per cat
- This is a lean but workable dataset; the augmentation strategy in Phase 2 is calibrated for this size
- Images where both Aïoli and Mayo appear together are especially valuable — they teach the model to distinguish them side by side
- Class imbalance: if one cat appears far more often, note this and consider oversampling the rarer class during preprocessing

---

## Phase 2 — Data Preprocessing

**Goal**: Organize images and labels into the YOLOv8 expected directory structure and generate `data.yaml`.

- Resize all images to 640×640 (YOLOv8 default input)
- Split into train / val / test (e.g. 80/10/10) while keeping image/label pairs together
  - Stratify the split by image type (single-cat vs. both-cats) so each split contains a representative mix
- Write `data/data.yaml`:
  ```yaml
  path: ./data
  train: images/train
  val: images/val
  test: images/test
  nc: 2
  names: ['aioli', 'mayo']
  ```
- **Data augmentation** is essential given the potentially limited photo library. Two complementary layers:

  **Offline augmentation** (run once, expands the dataset on disk before training):
  - Horizontal flip
  - Random rotation (±15°)
  - Brightness and contrast jitter
  - Gaussian blur / sharpening
  - Random crop and zoom
  - Colour temperature shift (mimics different indoor lighting conditions)
  - All augmentations must be applied to both the image **and** its bounding box coordinates
  - Target: expand the dataset to **8–10× its original size** before training (critical given ~100 source photos)
  - Implemented in `src/preprocess.py` using `albumentations`

  **Online augmentation** (applied on-the-fly by YOLOv8 during training):
  - Mosaic (combines 4 images — helps the model see both cats in varied arrangements)
  - HSV jitter, scale, translate
  - These are enabled by default in YOLOv8 and require no extra code

**Key libraries**: `ultralytics`, `albumentations`, `Pillow`, `scikit-learn`, `PyYAML`

---

## Phase 3 — Fine-Tuning (Transfer Learning)

**Goal**: Adapt a pretrained YOLOv8 model to detect individual cats.

**Two-stage fine-tuning** (inspired by the ChickenDetection frozen-then-full approach):

**Stage 1 — Frozen backbone (head-only training)**:
```python
from ultralytics import YOLO
model = YOLO('yolov8s.pt')
model.train(data='data/data.yaml', epochs=20, freeze=10, lr0=0.01, device=0)
```

**Stage 2 — Full fine-tuning**:
```python
model = YOLO('runs/detect/train/weights/last.pt')
model.train(data='data/data.yaml', epochs=50, lr0=0.001, device=0)
```

With an NVIDIA GPU and ~800–1000 augmented images, each stage should complete in a few minutes.

**Training outputs** (auto-saved by YOLOv8 to `runs/`):
- `best.pt` and `last.pt` checkpoints
- `results.csv` — per-epoch loss and mAP
- Confusion matrix, PR curve, F1 curve plots

**CLI**: `python cli.py train --data data/data.yaml --epochs 50`
or directly: `yolo train model=yolov8s.pt data=data/data.yaml epochs=50`

**Key libraries**: `ultralytics`, `torch`, `torchvision`

---

## Phase 4 — Training Results Visualization

**Goal**: Understand and communicate model performance.

Use `notebooks/02_training_results.ipynb` to:
- Load `runs/detect/train/results.csv` and plot loss + mAP curves
- Display the auto-generated confusion matrix and PR curve images
- Run the model on test set images and display predictions with bounding boxes
- Highlight failure cases, in particular:
  - One cat detected as the other (identity swap)
  - A cat missed entirely when both are present in the frame
  - False positives (background detected as a cat)

YOLOv8 already saves most plots automatically — the notebook loads and presents them cleanly.

---

## Phase 5 — Inference App (GUI)

**Goal**: A user-friendly local interface to run the model on new images.

- Load `runs/detect/train/weights/best.pt`
- Accept image file via picker
- Run `model.predict(image)` and render all detected bounding boxes with cat name labels and confidence scores
  - The GUI should handle 0, 1, or 2 detections gracefully (e.g. "Neither cat detected", "Only [cat1] detected", "Both cats detected")
- Option to save the annotated output image

**GUI**: **Gradio** — the industry standard for ML inference demos (used widely on Hugging Face). Runs locally in your browser, zero extra setup beyond `pip install gradio`.

**CLI equivalent**: `python cli.py infer --image photo.jpg --output result.jpg`
or directly: `yolo predict model=best.pt source=photo.jpg`

---

## Phase 6 — CLI Wrapper

**Goal**: Make every pipeline step scriptable.

`cli.py` exposes subcommands:

| Command | Description |
|--------|-------------|
| `annotate` | Launch annotation notebook or export annotations |
| `preprocess` | Resize, split dataset, generate data.yaml |
| `train` | Fine-tune YOLOv8 (stage 1 then stage 2) |
| `evaluate` | Run evaluation on test set, print mAP |
| `infer` | Run inference on one or more images |
| `app` | Launch the GUI inference app |

---

## Environment Setup

- **Python version**: 3.11 (minimum 3.10 — required by `ultralytics`)
- **Virtual environment**: use `python -m venv .venv` and activate before installing dependencies
- **Tool configuration**: `pyproject.toml` at the project root centralises settings for `black`, `ruff`, and `pytest`:
  ```toml
  [tool.black]
  line-length = 88

  [tool.ruff]
  line-length = 88
  select = ["E", "F", "I"]   # pycodestyle errors, pyflakes, isort

  [tool.pytest.ini_options]
  testpaths = ["tests"]
  ```
- **Dependency separation**: runtime deps in `requirements.txt`, dev-only tools (`black`, `ruff`, `pytest`) in `requirements-dev.txt`

---

## Code Standards & Conventions

### Python (`src/`, `app/`, `cli.py`)
- **Style**: [PEP 8](https://peps.python.org/pep-0008/) throughout — enforced with `black` (auto-formatter) and `ruff` (linter)
- **Type hints**: All function signatures use PEP 484 type annotations (`def predict(image: Path) -> list[dict]:`)
- **Docstrings**: [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) on all public functions and classes
- **Imports**: grouped and sorted per PEP 8 (stdlib → third-party → local), enforced by `ruff`
- **Naming**: `snake_case` for functions and variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants
- **No magic numbers**: all thresholds (confidence, IoU, augmentation probabilities, etc.) defined as named constants or config values

### Logging (`src/`, `cli.py`)
- Use Python's built-in `logging` module — no bare `print()` statements in library code
- A shared `setup_logging()` helper in `utils.py` configures a consistent format: `[LEVEL] timestamp — message`
- Log levels used consistently: `DEBUG` for per-step detail, `INFO` for milestones (e.g. "Epoch 10/50 complete"), `WARNING` for recoverable issues (e.g. missing label file), `ERROR` for failures
- The CLI passes a `--verbose` flag to switch between `INFO` (default) and `DEBUG`

### Testing (`tests/`)
- **Framework**: `pytest`
- Unit tests cover all pure functions in `src/` — especially preprocessing transforms, label file I/O, and bounding box utilities
- Tests must not depend on the actual photo dataset (use small synthetic images generated in the test itself)
- Run with: `pytest tests/`
- Tests are not required for notebooks or the Gradio app

### Jupyter Notebooks (`notebooks/`)
- Cell order must be linear and re-runnable top-to-bottom without errors
- One logical step per cell; cells over ~20 lines should be refactored into a helper in `src/`
- Markdown cells before each section explaining what it does and why
- No hardcoded paths — use `pathlib.Path` relative to the project root

### Configuration (`data/data.yaml`, training params)
- **YAML** for all dataset and training configuration
- Hyperparameters are not hardcoded in scripts — they are read from config files or CLI arguments

### CLI (`cli.py`)
- Built with `argparse`; every subcommand has a `--help` description
- Non-zero exit codes on failure; human-readable error messages

### Version control
- `.gitignore` excludes `runs/`, `data/raw/`, `data/images/`, `data/labels/`, `__pycache__/`, `*.pt` weights
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `chore:`, etc.)

### Additional dev dependencies
```
black        # Code formatter
ruff         # Fast linter (replaces flake8 + isort)
```

---

## Requirements

**`requirements.txt`** (runtime):
```
ultralytics       # YOLOv8 (includes torch dependency)
opencv-python     # Image I/O and preprocessing
Pillow            # Image manipulation
albumentations    # Offline data augmentation (bbox-aware transforms)
matplotlib        # Visualization
ipywidgets        # Annotation UI in Jupyter
scikit-learn      # Train/val/test split
PyYAML            # data.yaml generation
gradio            # GUI inference app
tqdm              # Progress bars in CLI and preprocessing scripts
```

**`requirements-dev.txt`** (development only):
```
black             # Code formatter
ruff              # Linter and import sorter
pytest            # Unit testing
```

Install: `pip install -r requirements.txt` and `pip install -r requirements-dev.txt`

---

## Open Questions / Decisions

- [x] How many cats? → **2 cats**, `nc: 2` in `data.yaml`
- [x] What are the cats' names? → **Aïoli** (class 0) and **Mayo** (class 1)
- [x] Video source available? → **No**, photos only
- [x] Data augmentation needed? → **Yes**, offline (`albumentations`) + online (YOLOv8 built-in)
- [x] GUI choice: **Gradio** (browser-based, industry standard for ML demos)
- [x] Hardware: **NVIDIA GPU** available — training uses `device=0` (CUDA)
- [x] Photo library size: **~100 photos** total → offline augmentation targets 8–10× expansion
