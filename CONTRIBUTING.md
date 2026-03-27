# Contributing

Technical reference for developers. Covers environment setup, architecture, code standards, data formats, and git conventions.

---

## Environment Setup

**Requirements**: Python 3.11 (minimum 3.10), an NVIDIA GPU with CUDA, `git`.

```bash
git clone <repo-url>
cd CatDetection
conda create -n catdetection python=3.11 -y
conda activate catdetection
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

**Important**: PyTorch must be installed with CUDA support *before* `requirements.txt`, because `pip install ultralytics` pulls in CPU-only PyTorch by default. The `--index-url` flag selects the CUDA 12.8 build. Replace `cu128` with your CUDA toolkit version if different (check with `nvidia-smi`).

**Runtime dependencies** (`requirements.txt`): `ultralytics`, `opencv-python`, `Pillow`, `albumentations`, `matplotlib`, `ipywidgets`, `ipympl`, `notebook`, `scikit-learn`, `PyYAML`, `gradio`, `tqdm`

**Dev dependencies** (`requirements-dev.txt`): `black`, `ruff`, `pytest`

**Tool configuration** lives in `pyproject.toml`:

```toml
[tool.black]
line-length = 88

[tool.ruff]
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I"]   # pycodestyle, pyflakes, isort

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Running notebooks**: The annotation notebook uses `%matplotlib widget` (ipympl) which does **not** work in VS Code's notebook editor. Use browser-based Jupyter instead:

```bash
conda activate catdetection
python -m notebook notebooks/01_annotate.ipynb
```

Run the formatter and linter before committing:

```bash
black src/ app/ cli.py tests/
ruff check src/ app/ cli.py tests/
```

---

## Architecture

```
CatDetection/
├── data/               # All dataset artefacts (mostly gitignored — see .gitignore)
├── notebooks/          # Interactive Jupyter workflows (annotation + results)
├── src/                # Core Python library — imported by notebooks, app, and CLI
├── app/                # Gradio inference app
├── tests/              # pytest unit tests for src/
├── runs/               # YOLOv8 training outputs (gitignored)
└── cli.py              # Unified CLI entry point (argparse)
```

### `src/` module responsibilities

| File | Responsibility |
|------|---------------|
| `annotate.py` | Image display and bounding box widget logic for the annotation notebook |
| `preprocess.py` | Resize, offline augmentation, train/val/test split, `data.yaml` generation |
| `train.py` | Two-stage YOLOv8 fine-tuning (frozen backbone → full fine-tune) |
| `evaluate.py` | Load `results.csv`, compute per-class metrics, print mAP summary |
| `infer.py` | Load a checkpoint and run inference on one or more images |
| `utils.py` | Shared helpers: bbox drawing, YOLO label I/O, `setup_logging()`, colour constants |

The notebooks and the Gradio app are thin wrappers — all logic lives in `src/`.

---

## Data Formats

### YOLO label files (`data/labels/`)

One `.txt` file per image, same base name as the image. Each row is one bounding box:

```
<class_id> <x_center> <y_center> <width> <height>
```

All values normalised to [0, 1] relative to image dimensions. Class IDs: `0` = Aïoli, `1` = Mayo.

Example for a photo containing both cats:
```
0 0.35 0.52 0.18 0.30
1 0.72 0.48 0.20 0.28
```

Images with no cats visible have an empty label file (not absent — absent means unlabelled).

### Dataset config (`data/data.yaml`)

```yaml
path: ./data
train: images/train
val: images/val
test: images/test
nc: 2
names: ['aioli', 'mayo']
```

---

## Augmentation Strategy

Offline augmentation is implemented in `src/preprocess.py` using `albumentations`. It runs once and writes augmented images + label files to disk before training. Target expansion: **8–10×** the original dataset size.

All transforms are **bbox-aware** — bounding box coordinates are transformed alongside the image.

| Transform | Parameters | Purpose |
|-----------|-----------|---------|
| `HorizontalFlip` | p=0.5 | Basic symmetry |
| `Rotate` | limit=15°, p=0.7 | Varied shooting angles |
| `RandomBrightnessContrast` | p=0.8 | Indoor lighting variation |
| `GaussianBlur` | blur_limit=3, p=0.3 | Camera focus variation |
| `RandomCropAndResize` | p=0.5 | Varied framing / zoom |
| `HueSaturationValue` | p=0.4 | Colour temperature / white balance |

Online augmentation (mosaic, HSV jitter, scale/translate) is handled by YOLOv8 at training time and requires no additional code.

---

## Training

Two-stage fine-tuning using `src/train.py`:

**Stage 1 — frozen backbone** (warms up the detection head):
```python
model = YOLO('yolov8s.pt')
model.train(data='data/data.yaml', epochs=20, freeze=10, lr0=0.01, device=0)
```

**Stage 2 — full fine-tune** (refines the whole network):
```python
model = YOLO('runs/detect/train/weights/last.pt')
model.train(data='data/data.yaml', epochs=50, lr0=0.001, device=0)
```

YOLOv8 auto-increments the output folder (`train`, `train2`, etc.) on each run. `src/train.py` resolves the latest run directory automatically rather than hardcoding the path.

All hyperparameters (epochs, learning rates, freeze depth) are passed via CLI arguments or a YAML config — never hardcoded in the script.

### Running training

**Prerequisites**: preprocessing must be complete (`data/images/train/`, `data/images/val/`, and `data/data.yaml` must exist). PyTorch must be installed with CUDA support if training on GPU (see [GPU setup](#gpu-setup) below).

**Default run** (both stages with default hyperparameters):

```python
from src.train import run_training
from src.utils import setup_logging
import logging

setup_logging(logging.INFO)
result = run_training()

print(result.best_weights)   # Path to the best checkpoint
print(result.results_csv)    # Path to per-epoch metrics
```

**Custom hyperparameters**:

```python
result = run_training(
    stage1_epochs=30,       # default: 20
    stage1_freeze=10,       # default: 10 (backbone layers to freeze)
    stage1_lr0=0.01,        # default: 0.01
    stage2_epochs=80,       # default: 50
    stage2_lr0=0.0005,      # default: 0.001
    device="cpu",           # default: 0 (first GPU)
    imgsz=640,              # default: 640
)
```

**Running a single stage** (useful for experimentation):

```python
from src.train import run_stage1, run_stage2

stage1_dir = run_stage1(epochs=20)                        # frozen backbone
stage2_dir = run_stage2(stage1_dir=stage1_dir, epochs=50) # full fine-tune
```

**Return value**: `run_training()` returns a `TrainingResult` dataclass with:

| Field | Description |
|-------|-------------|
| `stage1_dir` | Path to the Stage 1 run directory |
| `stage2_dir` | Path to the Stage 2 run directory |
| `best_weights` | Path to `best.pt` from Stage 2 (use this for inference) |
| `last_weights` | Path to `last.pt` from Stage 2 |
| `results_csv` | Path to `results.csv` with per-epoch loss and mAP |

### GPU setup

Training requires PyTorch with CUDA support. If you followed the environment setup above, PyTorch with CUDA is already installed. Verify with:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If `torch.cuda.is_available()` returns `False`, reinstall PyTorch for your CUDA version:

```bash
conda activate catdetection
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Replace `cu128` with your CUDA toolkit version (check the "CUDA Version" shown by `nvidia-smi`). To train on CPU instead (slower), pass `device="cpu"` to `run_training()`.

---

## Code Standards

### Style
- **PEP 8** enforced by `black` (formatter) and `ruff` (linter + import sorter).
- Line length: 88 characters.
- Imports: stdlib → third-party → local, one blank line between groups.

### Type hints & docstrings
- All public functions and classes have PEP 484 type annotations and Google-style docstrings.

```python
def load_labels(label_path: Path) -> list[dict]:
    """Load a YOLO label file into a list of bounding box dicts.

    Args:
        label_path: Path to the .txt label file.

    Returns:
        List of dicts with keys: class_id, x_center, y_center, width, height.

    Raises:
        FileNotFoundError: If label_path does not exist.
    """
```

### Logging
- Use `logging` module throughout — no bare `print()` in library code.
- Call `utils.setup_logging()` at the entry point (CLI or notebook top cell).
- Log levels: `DEBUG` (per-step detail), `INFO` (milestones), `WARNING` (recoverable issues), `ERROR` (failures).

### Constants
- No magic numbers. All thresholds and visual constants are named:

```python
# src/utils.py
CONFIDENCE_THRESHOLD: float = 0.5
IOU_THRESHOLD: float = 0.45
AIOLI_COLOR: tuple[int, int, int] = (255, 107, 53)   # #FF6B35 orange
MAYO_COLOR: tuple[int, int, int] = (78, 205, 196)    # #4ECDC4 teal
```

---

## Testing

```bash
pytest tests/
```

- Tests live in `tests/` and mirror the structure of `src/`.
- Each test file covers one `src/` module: `test_preprocess.py`, `test_infer.py`, `test_utils.py`.
- Tests never depend on the real photo dataset — synthetic 640×640 images are generated in fixtures.
- The Gradio app and notebooks are not unit-tested.

---

## Keeping Documentation Current

The project docs are a first-class deliverable — they are the primary way future agents and developers get up to speed. Treat them with the same discipline as code.

### What to update and when

| Event | Files to update |
|-------|----------------|
| A phase milestone is completed | `ROADMAP.md` — tick the checklist item |
| A new architectural decision is made | `CONTRIBUTING.md` (relevant section) + `design_decisions.md` (new numbered entry) |
| The UX or visual style changes | `DESIGN.md` |
| A new folder or module is added | Add or update the folder's `AGENTS.md` |
| `plan.md` content becomes outdated | Leave `plan.md` as-is (it's a historical record); update the relevant active doc instead |
| Dependencies change | `CONTRIBUTING.md` (Environment Setup) + `requirements.txt` / `requirements-dev.txt` |

### Rule of thumb

If you make a change that would surprise a future agent reading only the docs, the docs need updating. Commit documentation changes in the same commit as the code change that prompted them, using the `docs:` prefix.

### What never needs updating

`plan.md` is a fully frozen historical document — never edit it.

`design_decisions.md` has two layers: the **Decision Log entries** (numbered 1, 2, 3…) are append-only — never rewrite an existing entry, only add new ones. The surrounding sections (intro, "How to Use", "References") may be updated if the project structure or workflow changes significantly.

### Writing `design_decisions.md` entries

This project serves a dual purpose: it is both a working ML pipeline and a record of how to build a project effectively with AI agents. Each decision log entry should reflect both dimensions.

**Required structure for each entry:**

- **Trigger**: What prompted the decision (a phase milestone, a user question, a technical constraint, an agent suggestion).
- **Decision**: What was decided.
- **Reason**: Why — the technical rationale.
- **Exchange summary**: A brief recap of the human–agent interaction that led to the decision. What did the user ask or challenge? What did the agent propose? Where did the conversation pivot? This captures the *process*, not just the outcome — it's what makes the log useful as a learning resource for agent-assisted development.

The exchange summary should be honest and concise (2–4 sentences). If the agent got something wrong and the user corrected it, say so. If the user asked a clarifying question that reshaped the approach, note it. The goal is to help future readers (human or agent) understand *how* good decisions get made in a collaborative human–agent workflow, not just *what* was decided.

---

## Git Conventions

**Commit messages** follow [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | Use for |
|--------|---------|
| `feat:` | New functionality |
| `fix:` | Bug fix |
| `chore:` | Tooling, config, deps |
| `docs:` | Documentation only |
| `test:` | Adding or fixing tests |
| `refactor:` | Code change with no behaviour change |

**`.gitignore`** excludes:
```
runs/
data/raw/
data/images/
data/labels/
*.pt
__pycache__/
.venv/
*.egg-info/
```

The `data/data.yaml` file **is** committed — it's config, not data.
