# AGENTS.md — `src/`

## Purpose

The core Python library. All pipeline logic lives here. Notebooks, the Gradio app, and the CLI import from `src/` — they do not duplicate logic.

## Modules

| File | Responsibility | Status |
|------|---------------|--------|
| `__init__.py` | Empty — makes `src` an importable package. | Implemented |
| `utils.py` | Shared utilities: `setup_logging()`, bbox drawing with per-cat colours, YOLO label file I/O (`load_labels`, `save_labels`), bbox coordinate conversion (`normalize_bbox`, `denormalize_bbox`), colour constants (`AIOLI_COLOR`, `MAYO_COLOR`). | Implemented |
| `annotate.py` | `AnnotationSession` (state management, navigation, label persistence) and `AnnotationWidget` (interactive matplotlib/ipywidgets bounding box drawing UI) for `notebooks/01_annotate.ipynb`. Also provides `get_class_distribution()`. | Implemented |
| `preprocess.py` | Resize images to 640×640, run offline `albumentations` augmentation (bbox-aware, 9 augmented copies per train image), perform stratified 80/10/10 train/val/test split, write `data/data.yaml`. Orchestrated by `run_preprocessing()`. | Implemented |
| `train.py` | Two-stage YOLOv8 fine-tuning: Stage 1 (frozen backbone, 20 epochs) followed by Stage 2 (full fine-tune, 50 epochs). Resolves the latest `runs/` directory automatically. Returns `TrainingResult` dataclass. | Implemented |
| `evaluate.py` | Load `results.csv` from the latest training run, compute per-class precision/recall/mAP, print a summary table. | Planned (Phase 4) |
| `infer.py` | Load a `best.pt` checkpoint and run inference on a single image or a directory of images. Returns structured results (bounding boxes, class names, confidence scores). | Planned (Phase 5) |

## Standards

All code in this directory must follow the standards in `CONTRIBUTING.md`:
- PEP 8, enforced by `black` and `ruff`
- Type hints on all public functions
- Google-style docstrings on all public functions and classes
- `logging` module only — no bare `print()` statements
- No magic numbers — use named constants from `utils.py`
