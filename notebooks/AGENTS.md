# AGENTS.md — `notebooks/`

## Purpose

Interactive Jupyter workflows for the two human-facing steps of the pipeline: annotation and results review. All reusable logic is imported from `src/` — notebooks are thin orchestration layers, not where the code lives.

## Notebooks

### `01_annotate.ipynb` — Bounding Box Annotation
Walk through every image in `data/raw/`, draw bounding boxes, assign cat labels (Aïoli / Mayo), and save YOLO-format label files to `data/labels/`. Includes a progress tracker and supports resuming across sessions.

### `02_training_results.ipynb` — Training Results
Load the outputs from `runs/` and visualise training performance: loss and mAP curves, confusion matrix, PR curve, sample predictions on the test set, and a failure gallery (identity swaps, missed detections, false positives).

## Conventions

- Cells must run top-to-bottom without errors on a clean kernel restart.
- No hardcoded paths — use `pathlib.Path` relative to the project root.
- Cells longer than ~20 lines should be refactored into a helper function in `src/`.
- Each section begins with a Markdown cell explaining what the section does and why.

## Status

| Notebook | Status |
|----------|--------|
| `01_annotate.ipynb` | Implemented |
| `02_training_results.ipynb` | Planned (Phase 4) |

## Running Notebooks

**Important**: The annotation notebook uses `%matplotlib widget` (ipympl) which does **not** work in VS Code's notebook editor. Run notebooks in browser-based Jupyter:

```bash
conda activate catdetection
jupyter notebook notebooks/
```

Ensure the conda environment is active so that `src/` and all dependencies are on the Python path.
