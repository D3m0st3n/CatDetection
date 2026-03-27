# AGENTS.md — `tests/`

## Purpose

pytest unit tests for the pure functions in `src/`. Ensures shared utilities, preprocessing, and inference parsing behave correctly and keep working as the code evolves.

## Files

| File | Tests | Status |
|------|-------|--------|
| `conftest.py` | pytest configuration — adds the project root to `sys.path` so `src` is importable | Implemented |
| `test_utils.py` | 14 tests covering `normalize_bbox` / `denormalize_bbox` round-trips and edge cases, `save_labels` / `load_labels` round-trip + empty/missing file handling, `draw_bboxes` shape and immutability | Implemented (all passing) |
| `test_preprocess.py` | 21 tests: image pair loading, classification, stratified splitting, resize, augmentation pipeline (output shapes, bbox format, empty labels, large bbox survival), write_split with/without augmentation, `data.yaml` generation, end-to-end integration | Implemented (all passing) |
| `test_train.py` | 20 tests: `find_latest_run_dir` filesystem resolution, Stage 1/Stage 2 hyperparameter forwarding (mocked YOLO), orchestration ordering, error handling (missing data.yaml, missing weights, failed training) | Implemented (all passing) |
| `test_infer.py` | Inference result parsing (correct number of detections, label names, confidence score ranges) | Planned (Phase 5) |

**Note**: `test_annotate.py` is intentionally absent — see design decision #19. The annotation widget is tightly coupled to Jupyter/matplotlib and tested manually via the notebook.

## Conventions

- Tests never depend on the real photo dataset. Synthetic images are generated in fixtures using `numpy`.
- Each test function tests one behaviour and has a descriptive name.
- Tests for the Gradio app and notebooks are not required.

## Running Tests

```bash
conda activate catdetection
pytest tests/
# with verbose output:
pytest tests/ -v
# for a single file:
pytest tests/test_utils.py
```

All tests must pass before merging any change to `src/`.
