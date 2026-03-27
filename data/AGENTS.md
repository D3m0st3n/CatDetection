# AGENTS.md — `data/`

## Purpose

Holds all dataset artefacts: source photos, YOLO label files, processed images, and the dataset config.

## Subdirectories

| Path | Contents | Gitignored? |
|------|----------|-------------|
| `raw/` | Original photos, unmodified | Yes |
| `labels/` | YOLO `.txt` label files, one per source image (255 files) | Yes |
| `labels/train/` | Label files for training split (including augmented copies) | Yes |
| `labels/val/` | Label files for validation split | Yes |
| `labels/test/` | Label files for test split | Yes |
| `images/train/` | Augmented + resized 640×640 training images (2040 files) | Yes |
| `images/val/` | Resized 640×640 validation images (25 files) | Yes |
| `images/test/` | Resized 640×640 test images (26 files) | Yes |
| `data.yaml` | YOLOv8 dataset config — class names and split paths | **No** |

## Why Most of This Is Gitignored

Raw photos and generated images are large binary files that don't belong in version control. The label files are also excluded to avoid accidentally committing personally identifiable pet photos. The only committed file here is `data.yaml` — it's config, not data.

## Generating This Directory

Run preprocessing from Python or the CLI after completing annotation. It reads from `raw/` and `labels/` and populates `images/` and `labels/{split}/` with the full augmented, split dataset:

```python
from pathlib import Path
from src.preprocess import run_preprocessing
run_preprocessing(Path('data/raw'), Path('data/labels'), Path('data'))
```

**Current state**: Preprocessing complete — 2091 total images (8.2× expansion from 255 originals).
