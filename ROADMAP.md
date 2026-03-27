# Roadmap

## Project Goal

Build a fully local ML pipeline that detects and identifies two cats — **Aïoli** and **Mayo** — by name in photos. The pipeline covers the full lifecycle: annotation, preprocessing, fine-tuning, evaluation, and inference via a GUI and CLI.

---

## State of the Art & Approach

### Why YOLO?

YOLO (You Only Look Once) is the industry standard for real-time single-stage object detection. It processes the entire image in one forward pass, predicting bounding boxes and class probabilities simultaneously. YOLOv8 (Ultralytics, 2023) is the current generation: fully Python-native, pip-installable, and actively maintained.

### Why Fine-Tune Instead of Training from Scratch?

Training a custom detection model from scratch requires hundreds of images per class and significant GPU compute (the reference [ChickenDetection](https://github.com/DennisFaucher/ChickenDetection) project required an AWS GPU instance). Fine-tuning a pretrained model instead:

- Works well with 50–100 images per class
- Trains in minutes on a local GPU
- Leverages features already learned from millions of images (edges, shapes, textures)
- Produces better accuracy with less data

**Strategy**: two-stage fine-tuning — freeze the backbone and train only the detection head first, then unfreeze all layers for a lower-learning-rate full pass.

### Why Data Augmentation is Critical

The source dataset is ~100 photos total (~50 appearances per cat). This is lean. Offline augmentation via `albumentations` expands the dataset 8–10× before training, applying bbox-aware transforms that simulate varied real-world conditions (lighting, angle, distance). YOLOv8's built-in online augmentation (mosaic, HSV jitter) adds further variety at training time.

---

## Milestones

### Phase 0 — Project Setup
- [x] Scaffold directory structure with `AGENTS.md` and `.gitkeep` files
- [x] Create `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`
- [x] Set up conda environment `catdetection` (Python 3.11)
- [x] Implement `src/utils.py` — constants, logging, YOLO label I/O, bbox conversion, bbox drawing
- [x] Implement `src/annotate.py` — `AnnotationSession`, `AnnotationWidget`, `get_class_distribution()`
- [x] Implement `notebooks/01_annotate.ipynb` — interactive annotation tool
- [x] Implement `tests/test_utils.py` — 14 unit tests for utils (all passing)

### Phase 1 — Annotation ✅
- [x] Organise all source photos under `data/raw/` (255 photos)
- [x] Annotate every photo in `notebooks/01_annotate.ipynb` with bounding boxes for Aïoli and Mayo
- [x] Produce YOLO-format `.txt` label files in `data/labels/` (255 label files)
- [x] Confirm ~50+ bounding box appearances per cat (Aïoli: 173, Mayo: 174)

### Phase 2 — Preprocessing & Augmentation ✅
- [x] Implement `src/preprocess.py` — full preprocessing pipeline
- [x] Implement `tests/test_preprocess.py` — 21 unit tests (all passing)
- [x] Resize all images to 640×640
- [x] Run offline augmentation to reach 8–10× dataset size (achieved 8.2×: 255 → 2091 images)
- [x] Stratified train/val/test split (80/10/10) preserving single-cat vs. both-cats ratio (204/25/26 originals; 2040/25/26 after augmentation)
- [x] Generate `data/data.yaml`

### Phase 3 — Fine-Tuning ✅
- [x] Implement `src/train.py` — two-stage fine-tuning with `TrainingResult` dataclass
- [x] Implement `tests/test_train.py` — 20 unit tests (all passing)
- [x] Stage 1: frozen backbone, head-only training (20 epochs, lr=0.01)
- [x] Stage 2: full fine-tune from Stage 1 checkpoint (50 epochs, lr=0.001)
- [x] Run training and confirm `runs/` contains `best.pt`, `results.csv`, and evaluation plots

### Phase 4 — Evaluation & Results
- [x] Implement `src/infer.py` — model loading, inference, prediction drawing (13 unit tests)
- [x] Implement `src/evaluate.py` — metrics parsing, IoU matching, error categorisation (17 unit tests)
- [x] Review training curves and mAP in `notebooks/02_training_results.ipynb`
- [x] Inspect failure cases: identity swaps, missed detections, false positives
- [x] Decide whether to retrain with more augmentation or additional photos (decided to proceed to Phase 5 without retraining)
- [ ] Perform retraining (varying hyperparameters) after completing other Phases

### Phase 5 — Inference App ✅
- [x] Implement `app/gui.py` with Gradio
- [x] Image upload → bounding box overlay → detection summary (0, 1, or 2 cats)
- [x] Save-output option working

### Phase 6 — CLI
- [ ] All subcommands implemented and documented (`--help`)
- [ ] End-to-end pipeline runnable with: `annotate → preprocess → train → evaluate → infer`

---

## Success Criteria

A successful project run produces a model that:
- Correctly identifies Aïoli and Mayo individually when each appears alone
- Correctly identifies both cats (with distinct boxes and labels) when they appear together
- Has a validation mAP@0.5 ≥ 0.75
- Runs inference on a new photo in under 2 seconds on the local GPU
