# Design Decisions & Brainstorm Log

A record of the planning session that produced the project documentation — what was decided, why, and in what order. Paste this file into a future chat to give an agent full context without replaying the whole conversation.

**Active documents** (use these for development):
- `ROADMAP.md` — goals, state of the art, phase milestones
- `DESIGN.md` — UX, visual style, naming, interface behaviour
- `CONTRIBUTING.md` — architecture, code standards, data formats, git conventions
- `AGENTS.md` files — per-folder context for agents navigating the repo

**Historical reference** (do not edit):
- `plan.md` — original working plan, superseded by the three documents above

---

## What This Project Is

A local, Python-only ML pipeline to detect and identify **two cats (Aïoli and Mayo)** by name in photos, using a fine-tuned YOLOv8 model served through a Gradio GUI and a CLI.

---

## Decision Log

### 1. Fine-tuning over training from scratch
**Trigger**: The [ChickenDetection](https://github.com/DennisFaucher/ChickenDetection) reference project was reviewed.
**Decision**: Use a pretrained `yolov8s.pt` model and fine-tune it with a two-stage strategy (frozen backbone first, then full fine-tune) rather than training from scratch.
**Reason**: Training from scratch required an AWS GPU and hundreds of images per class in the reference project. Fine-tuning works well with 50–100 images per class, trains in minutes on a local GPU, and produces better accuracy with less data.
**Exchange summary**: Not available — this decision predates the exchange summary guideline.

### 2. Model: YOLOv8s (Ultralytics)
**Decision**: `yolov8s.pt` (small variant).
**Reason**: Modern, fully Python-native, pip-installable. The small variant gives better accuracy than nano with a dataset this size, and the available NVIDIA GPU handles it easily.
**Exchange summary**: Not available — this decision predates the exchange summary guideline.

### 3. Dataset: photos only, ~100 total, two cats
**Constraints established**:
- No video source — existing photo library only.
- ~100 photos total, split across Aïoli, Mayo, and both-cats images (~50 appearances per cat).
- Some photos contain both cats simultaneously → multi-instance detection required (0, 1, or 2 boxes per image).
**Exchange summary**: Not available — this decision predates the exchange summary guideline.

### 4. Data augmentation is a first-class concern
**Trigger**: ~100 photos is lean for fine-tuning.
**Decision**: Two-layer augmentation strategy:
- **Offline** (via `albumentations`): expand dataset 8–10× on disk before training. Transforms: flip, rotation, brightness/contrast jitter, blur, crop/zoom, colour temperature shift. All bbox-aware.
- **Online** (via YOLOv8 built-ins): mosaic, HSV jitter, scale/translate at training time.
**Exchange summary**: Not available — this decision predates the exchange summary guideline.

### 5. Annotation format: YOLO `.txt` + `data.yaml`
**Decision**: One `.txt` label file per image with one row per visible cat: `class x_center y_center width height` (normalised 0–1). Dataset config in `data.yaml`.
**Reason**: Native YOLOv8 format — no conversion step needed.
**Exchange summary**: Not available — this decision predates the exchange summary guideline.

### 6. Annotation UI: Jupyter notebook with `ipywidgets`
**Decision**: Interactive bounding box drawing in `notebooks/01_annotate.ipynb`.
**Reason**: Keeps everything local and Python-native; no external annotation tool dependency.
**Exchange summary**: Not available — this decision predates the exchange summary guideline.

### 7. Inference GUI: Gradio
**Decision**: Gradio browser-based local app.
**Reason**: Industry standard for ML inference demos (used on Hugging Face). Minimal boilerplate, polished output, runs entirely locally.
**Exchange summary**: Not available — this decision predates the exchange summary guideline.

### 8. Code standards: PEP 8 + tooling
**Decision**: PEP 8 enforced by `black` + `ruff`, Google-style docstrings, PEP 484 type hints, `logging` module (no bare `print()`), `pytest` for unit tests on `src/` utilities.
**Reason**: Explicit request from the user; also best practice for a maintainable project.
**Exchange summary**: Not available — this decision predates the exchange summary guideline.

### 9. Environment: Python 3.11, `venv`, `pyproject.toml`
**Decision**: Pin to Python 3.11, use `venv`, centralise tool config (`black`, `ruff`, `pytest`) in `pyproject.toml`. Split runtime (`requirements.txt`) and dev (`requirements-dev.txt`) dependencies.
**Reason**: Reproducibility and standard project hygiene.
**Exchange summary**: Not available — this decision predates the exchange summary guideline.

### 10. Project scaffolding: file structure + AGENTS.md files
**Decision**: Create all project directories up front (with `.gitkeep` for gitignored empty dirs), and add an `AGENTS.md` file to every folder describing its purpose, contents, and conventions.
**Reason**: A future agent (or human developer) dropped into any folder can immediately understand what belongs there and what to do. `.gitkeep` files ensure git tracks the folder structure even before any real data exists.
**Exchange summary**: Not available — this decision predates the exchange summary guideline.

### 11. Splitting `plan.md` into three dedicated documents
**Decision**: Split the monolithic `plan.md` into `ROADMAP.md` (goals + milestones), `DESIGN.md` (UX + visual style), and `CONTRIBUTING.md` (technical/architectural reference). `plan.md` is retained as a read-only historical record.
**Reason**: A single large document becomes hard to navigate and hard to hand to a focused agent. Separate documents let you say "read CONTRIBUTING.md before writing any code" or "check ROADMAP.md for current milestone status" without the agent wading through unrelated content. The split also mirrors a pattern used in well-maintained open source projects (CONTRIBUTING, ROADMAP are standard filenames that tools and agents recognise).
**Exchange summary**: Not available — this decision predates the exchange summary guideline.

### 12. Conda environment over venv
**Trigger**: User preference at the start of Phase 1 implementation.
**Decision**: Use a conda environment named `catdetection` (Python 3.11) instead of `python -m venv`.
**Reason**: User requested a conda-managed environment for the project. Updated `CONTRIBUTING.md` accordingly.
**Exchange summary**: Not available — this decision predates the exchange summary guideline.

### 13. `ipympl` added to runtime dependencies
**Trigger**: The annotation widget needs `%matplotlib widget` which requires the `ipympl` backend.
**Decision**: Add `ipympl` to `requirements.txt`.
**Reason**: Without `ipympl`, the `%matplotlib widget` magic fails. This package bridges matplotlib's interactive backend with Jupyter's ipywidgets infrastructure. It was not in the original requirements list but is essential for the annotation notebook.
**Exchange summary**: Not available — this decision predates the exchange summary guideline.

### 14. Matplotlib interactive backend for annotation drawing
**Trigger**: Phase 1 implementation — choosing how to draw bounding boxes in the annotation notebook.
**Decision**: Use matplotlib's `%matplotlib widget` backend with mouse event handlers (`button_press_event`, `motion_notify_event`, `button_release_event`) for click-and-drag rectangle drawing. ipywidgets provide the surrounding controls (buttons, dropdown, status).
**Reason**: `ipycanvas` would add an extra dependency not in the project spec and has known compatibility issues on Windows with JupyterLab. A form-based coordinate entry approach would be too tedious for 255 images. The matplotlib event system is reliable, requires only `ipympl` (already needed), and maps axes coordinates directly to image pixel coordinates via `imshow`.
**Exchange summary**: Not available — this decision predates the exchange summary guideline.

### 15. Annotation state persistence via label file existence
**Trigger**: Phase 1 implementation — how to track which images have been annotated and support resuming across sessions.
**Decision**: Use the presence/absence of a `.txt` label file in `data/labels/` as the sole source of truth for annotation status. An existing file (even if empty) means "annotated"; a missing file means "not yet annotated". No separate progress database or JSON file.
**Reason**: Simple, robust, and inherently resumable. The file system IS the state — restarting the kernel or closing the notebook loses nothing. The `AnnotationSession` constructor scans the labels directory on init and jumps to the first unannotated image automatically.
**Exchange summary**: Not available — this decision predates the exchange summary guideline.

### 16. EXIF transpose on image load
**Trigger**: Phase 1 implementation — the raw photos have WhatsApp-origin filenames (`IMG-*-WA*.jpg`) which often carry EXIF orientation tags.
**Decision**: Call `PIL.ImageOps.exif_transpose()` on every image load in the annotation widget.
**Reason**: Without this, some images display rotated 90° or 180° in the notebook, and any bounding boxes drawn would have incorrect coordinates relative to the actual image content. This is a one-line fix applied at load time in `AnnotationWidget._load_image_rgb()`.
**Exchange summary**: Not available — this decision predates the exchange summary guideline.

### 17. Ruff lint config uses `[tool.ruff.lint]` section
**Trigger**: Phase 1 verification — ruff 0.15+ deprecated top-level `select` under `[tool.ruff]`.
**Decision**: Move `select = ["E", "F", "I"]` to `[tool.ruff.lint]` in `pyproject.toml`.
**Reason**: Ruff emitted a deprecation warning. The new `[tool.ruff.lint]` section is the supported location for lint rule selection in current ruff versions.
**Exchange summary**: Not available — this decision predates the exchange summary guideline.

### 18. `conftest.py` for test imports
**Trigger**: Phase 1 verification — `pytest tests/test_utils.py` failed with `ModuleNotFoundError: No module named 'src'`.
**Decision**: Add `tests/conftest.py` that inserts the project root into `sys.path`.
**Reason**: Since `src` is not an installed package (no `pip install -e .`), pytest cannot find it. The conftest approach is lightweight and avoids the need for a full `[project]` section in `pyproject.toml` at this stage. The notebook uses the same `sys.path.insert` pattern.
**Exchange summary**: Not available — this decision predates the exchange summary guideline.

### 19. Unit tests scoped to `src/utils.py` only for Phase 1
**Trigger**: Phase 1 planning — deciding what to test.
**Decision**: Write unit tests only for the pure functions in `src/utils.py` (bbox conversion, label I/O, draw function). Do not test `AnnotationSession`, `AnnotationWidget`, or the notebook.
**Reason**: CONTRIBUTING.md exempts notebooks and the GUI from testing. `AnnotationWidget` is tightly coupled to Jupyter's display system and matplotlib mouse events, making it impractical to unit test without a widget testing framework. `AnnotationSession` is mostly I/O coordination whose core logic (label read/write, bbox conversion) is already covered through `utils.py` tests. If `AnnotationSession` grows complex logic in future phases, it can warrant its own tests then.
**Exchange summary**: Not available — this decision predates the exchange summary guideline.

### 20. Annotation notebook must run in browser Jupyter, not VS Code
**Trigger**: Phase 1 testing — the annotation cell failed in VS Code with a `WidgetManager.loadClass` error. The ipympl (`%matplotlib widget`) frontend widget model cannot be loaded by VS Code's ipywidgets renderer.
**Decision**: Run the annotation notebook in browser-based Jupyter (`jupyter notebook`) rather than VS Code's built-in notebook editor.
**Reason**: `%matplotlib widget` relies on ipympl's JavaScript frontend, which works in browser Jupyter but not in VS Code's custom widget manager. Rewriting the widget to avoid ipympl would require adding `ipycanvas` or a custom HTML/JS approach — significant extra complexity for no functional gain. Browser Jupyter is the supported environment for interactive matplotlib widgets.
**Exchange summary**: Not available — this decision predates the exchange summary guideline.

### 21. `notebook` added to runtime dependencies
**Trigger**: Phase 1 — running the annotation notebook in browser-based Jupyter required the `notebook` package, which was not installed in the conda environment.
**Decision**: Add `notebook` to `requirements.txt`.
**Reason**: The `jupyter notebook` CLI command (and `python -m notebook`) is provided by the `notebook` package. Since the annotation notebook must run in browser Jupyter (not VS Code — see decision #20), this is a required runtime dependency.
**Exchange summary**: Not available — this decision predates the exchange summary guideline.

### 22. Phase 1 annotation complete — dataset statistics
**Trigger**: All 255 raw photos annotated in `notebooks/01_annotate.ipynb`.
**Decision**: Phase 1 is complete. Final dataset: 255 images, 173 Aïoli bounding boxes, 174 Mayo bounding boxes. Both cats are well above the 50+ appearance threshold and nearly perfectly balanced.
**Observations**:
- The class balance is excellent (173 vs 174) — no oversampling needed during preprocessing.
- The total bounding box count (347) across 255 images means many images contain exactly one cat, with a good number of both-cats images mixed in.
- This dataset size is lean but workable; the 8–10× offline augmentation in Phase 2 will expand it to ~2000–2500 training images.
**Exchange summary**: Not available — this decision predates the exchange summary guideline.

### 23. Augmentation strategy: 9 copies per train image
**Trigger**: Phase 2 implementation — choosing the augmentation multiplier.
**Decision**: Generate 9 augmented copies per training image (original + 9 = 10× per image). Combined with the 80/10/10 split, this produces 2040 training images from 204 originals (8.2× overall expansion from 255 source images).
**Reason**: The ROADMAP targets 8–10× expansion. With 204 images in the train split, 9 augmented copies yields 2040 train images — within the target range without excessive disk usage or diminishing returns.
**Exchange summary**: Not available — this decision predates the exchange summary guideline.

### 24. Stratification by image category (single / both / empty)
**Trigger**: Phase 2 implementation — designing the train/val/test split.
**Decision**: Classify images into three categories (`single`, `both`, `empty`) and use `sklearn.model_selection.train_test_split` with `stratify=categories` to preserve the ratio across splits. A fallback to unstratified splitting is used when a category has too few members in the temp set for the second split.
**Reason**: Both-cats images are especially valuable for teaching the model to distinguish Aïoli and Mayo side-by-side. Without stratification, random chance could concentrate them in one split and starve the others.
**Exchange summary**: Not available — this decision predates the exchange summary guideline.

### 25. Augmentation only on train split
**Trigger**: Phase 2 implementation — deciding which splits to augment.
**Decision**: Offline augmentation is applied only to the training split. Validation and test splits contain only the resized originals.
**Reason**: Augmenting val/test would inflate metrics and hide overfitting. The purpose of val/test is to measure generalisation on real (unaugmented) images.
**Exchange summary**: Not available — this decision predates the exchange summary guideline.

### 26. Preprocessing output uses `data/labels/train|val|test/` alongside `data/images/train|val|test/`
**Trigger**: Phase 2 implementation — output directory structure.
**Decision**: Write augmented labels to `data/labels/{split}/` matching the `data/images/{split}/` structure, rather than keeping a flat `data/labels/` directory.
**Reason**: YOLOv8's default dataset loader expects label files in a `labels/` directory that mirrors the `images/` directory structure. This avoids needing a custom label-mapping configuration.
**Exchange summary**: Not available — this decision predates the exchange summary guideline.

### 27. Phase 2 preprocessing complete — dataset statistics
**Trigger**: Full preprocessing pipeline run on 255 annotated images.
**Decision**: Phase 2 is complete. Final preprocessed dataset:
- **Split**: 204 train / 25 val / 26 test (original images)
- **Category distribution**: 164 single-cat, 91 both-cats (no empty images in dataset)
- **After augmentation**: 2040 train / 25 val / 26 test = 2091 total images (8.2× expansion)
- **Image size**: all 640×640 JPEG
- **`data/data.yaml`** generated and ready for YOLOv8 training
**Exchange summary**: Not available — this decision predates the exchange summary guideline.

### 28. TrainingResult dataclass as structured return type
**Trigger**: Phase 3 implementation — choosing how to return training output paths.
**Decision**: Use a `@dataclass` (`TrainingResult`) with fields for `stage1_dir`, `stage2_dir`, `best_weights`, `last_weights`, and `results_csv` instead of returning a plain dict or just a path.
**Reason**: A dataclass provides attribute access with IDE autocomplete, is self-documenting via field names and the class docstring, and can be extended later (e.g. adding `mAP_score` after evaluation). It follows the pattern of returning structured data that `preprocess.py` established with its function return types.
**Exchange summary**: Not available — this decision predates the exchange summary guideline.

### 29. Mocking strategy for training tests
**Trigger**: Phase 3 — designing unit tests that must not trigger actual GPU training.
**Decision**: Mock `ultralytics.YOLO` at the class level and `find_latest_run_dir` separately. Use real filesystem operations (`tmp_path`) for data.yaml validation and weight file existence checks.
**Reason**: This separation keeps tests fast (no GPU), while still exercising the real filesystem validation logic. Mocking at the class level (not instance methods) ensures the test verifies which model path is passed to the constructor.
**Exchange summary**: Not available — this decision predates the exchange summary guideline.

### 30. `find_latest_run_dir` as a public function
**Trigger**: Phase 3 — deciding how to resolve YOLOv8's auto-incremented run directories.
**Decision**: Make `find_latest_run_dir(project: Path) -> Path` a public function rather than an internal helper.
**Reason**: The function contains the trickiest logic in the module (parsing the `train`, `train2`, `train3` naming convention). Making it public allows thorough unit testing with pure filesystem operations (no YOLO mock needed) and enables reuse in Phase 4 (evaluation) and Phase 5 (inference) where the latest run directory also needs resolving.
**Exchange summary**: Not available — this decision predates the exchange summary guideline.

### 31. PyTorch CUDA must be installed before ultralytics
**Trigger**: Phase 3 — running `python run_training.py` failed with CUDA not recognising the GPU. Investigation revealed `torch.cuda.is_available()` returned `False` because pip had installed CPU-only PyTorch.
**Decision**: Install PyTorch with the CUDA index URL (`--index-url https://download.pytorch.org/whl/cu128`) *before* `pip install -r requirements.txt`. Added a comment to `requirements.txt` and updated the install instructions in `CONTRIBUTING.md`.
**Reason**: PyPI's default `torch` package is CPU-only. The `ultralytics` package depends on `torch` and pulls it in automatically — but from the default index, which gives the CPU build. Installing the CUDA build first means `ultralytics` finds it already satisfied and doesn't overwrite it. This is a well-known PyTorch packaging quirk. The RTX 5070 with CUDA 13.1 driver is compatible with the `cu128` wheels (CUDA toolkit 12.8).
**Exchange summary**: The user reported that training failed with a CUDA error on a new machine. The agent installed dependencies, confirmed `torch.cuda.is_available()` was `False` with the default pip install, then reinstalled PyTorch from the `cu128` index URL which resolved it. The user asked for the requirements and docs to be updated accordingly so the issue wouldn't recur.

### 32. Inference and evaluation as separate modules
**Trigger**: Phase 4 implementation — deciding where inference-vs-evaluation logic belongs.
**Decision**: Create `src/infer.py` (model loading, prediction, drawing with confidence) and `src/evaluate.py` (metrics parsing, IoU matching, error categorisation) as separate modules rather than a single combined module.
**Reason**: `infer.py` is reusable by the Gradio app (Phase 5) and CLI (Phase 6), while `evaluate.py` contains evaluation-specific logic (GT comparison, error classification) that those consumers don't need. This follows the single-responsibility pattern already established in the codebase.
**Exchange summary**: The agent proposed splitting responsibilities during planning. The user approved the plan without modification. The split proved clean — `evaluate.py` imports `Detection` and `PredictionResult` from `infer.py`, while `infer.py` has no dependency on `evaluate.py`.

### 33. Phase 4 evaluation results — below target, high missed-detection rate
**Trigger**: Phase 4 evaluation of the trained model on the 26-image test set.
**Decision**: Training results are documented but the mAP50 target of 0.75 is not met on the test set. The model's main weakness is missed detections (19 out of 35 GT boxes), not false positives. Per-class: Aïoli recall 23.5%, Mayo recall 44.4%.
**Observations**:
- Validation mAP50 during training peaked at 0.758 (Stage 1 epoch 9) and 0.748 (Stage 2 epoch 45) — close to target.
- But test set evaluation at confidence threshold 0.5 shows only 18 predictions for 35 GT boxes — the model is too conservative.
- 4 identity swaps indicate the model sometimes confuses the two cats.
- Lowering the confidence threshold or retraining with more epochs/augmentation may help.
**Exchange summary**: The user noted that precision of 0.67 "does not sound great" and asked to proceed with Phase 4 evaluation. The agent built the evaluation infrastructure and ran it, revealing that the main problem is missed detections rather than low precision — the model is under-predicting rather than making wrong predictions.

### 35. `sys.path` injection for scripts in subdirectories
**Trigger**: Running `python app/gui.py` raised `ModuleNotFoundError: No module named 'src'` because Python adds the script's own directory (`app/`) to `sys.path`, not the project root.
**Decision**: Add `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` at the top of `app/gui.py`, before any `src.*` imports.
**Reason**: All `src.*` imports resolve relative to the project root. Scripts at the root (e.g. `run_training.py`) work naturally because Python adds the script's directory to `sys.path`. Scripts in subdirectories must explicitly add the root. Using `Path(__file__).resolve().parent.parent` is robust — it works regardless of the current working directory or how the script is invoked.
**How to apply**: Any future script placed inside a subdirectory (e.g. `app/`, `scripts/`) that imports from `src` must include this same `sys.path.insert` line. Scripts at the project root do not need it.
**Exchange summary**: The user reported the import error after the initial Phase 5 implementation. The fix was a one-line `sys.path.insert` addition at the top of `gui.py`. The same pattern should be applied to any future subdirectory scripts.

### 34. EXIF-safe image loading in the GUI (PIL over cv2)
**Trigger**: Implementing `app/gui.py` — needed to load the uploaded image for display and bounding box drawing.
**Decision**: Load images using `PIL.ImageOps.exif_transpose(Image.open(path))` rather than `cv2.imread`, then convert to BGR for drawing.
**Reason**: `cv2.imread` ignores EXIF orientation metadata, so photos taken in portrait mode on a phone (common for cat photos) would appear rotated in the output. YOLOv8 applies EXIF correction internally during inference, so not applying the same correction to the display image would misalign drawn bounding boxes with the visible subject. Using PIL + `exif_transpose` keeps the display image orientation consistent with what YOLO processed.
**Exchange summary**: The agent noted the mismatch risk during planning — `draw_predictions` operates on a numpy array the GUI loads separately from the path YOLO uses for inference. The user approved the PIL approach in the plan review. This is the same EXIF handling pattern already used in `src/annotate.py`.

---

## The Planning Process We Used

This is an example of **iterative requirement elicitation with an agent** — a useful pattern to know when starting any non-trivial project.

### How it worked, step by step

1. **Start with a stub** — the user had a minimal `plan.md` (3 bullet points). The agent read it, identified what was underspecified, and asked clarifying questions before writing anything.

2. **Ask before acting** — rather than guessing at preferences, the agent grouped open questions and presented them as multiple-choice (detection goal, image source, model strategy, app format). This front-loaded the decisions that would have the biggest structural impact.

3. **Incorporate external references** — the user pointed to a reference project (ChickenDetection). The agent researched it and extracted the transferable decisions (two-stage fine-tuning, YOLO format, annotation workflow) while modernising the stack.

4. **Refine in layers** — each round of user input unlocked a specific set of updates:
   - Cat names → updated labels, `data.yaml`, GUI copy
   - Photos-only + lean dataset → removed video workflow, elevated augmentation to a core concern
   - PEP 8 request → added full standards section, tooling, project structure additions
   - "Anything missing?" → agent self-audited and found: environment setup, testing, logging, two inconsistencies in code snippets
   - "Create the file structure" → scaffolded all directories, split `plan.md` into `ROADMAP.md` / `DESIGN.md` / `CONTRIBUTING.md`, added `AGENTS.md` per folder

5. **Close all open questions explicitly** — the plan tracks a checklist of open decisions; nothing was left ambiguous by the end.

### What made this work well

- **Constraints given upfront** (local only, Python only) narrowed the solution space early.
- **Concrete examples** (ChickenDetection) gave the agent a real artifact to reason from rather than a vague instruction.
- **The user pushed back** ("actually, fine-tune instead of scratch") — agents improve plans faster when challenged.
- **Asking "is anything missing?" at the end** is a reliable way to catch gaps a human might not think to raise.

---

## How to Use This in a Future Chat

Paste this file plus the relevant active document(s) at the start of the conversation:

> "Here is the design decisions log for my cat detection project, along with [ROADMAP.md / CONTRIBUTING.md / DESIGN.md]. I'd like to [start implementing Phase 1 / revisit the augmentation strategy / ...]."

The agent will have full context on what was decided and why, without needing to re-derive it.

For implementation work, also include the `AGENTS.md` of the specific folder being worked on — it gives the agent immediate orientation about what belongs there.

---

## References

### On working with AI agents for project planning
- [Anthropic prompt engineering guide](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview) — covers how to write clear prompts, use examples, and structure requests for best results with Claude.
- [Anthropic docs: Claude Code & agents](https://docs.claude.com) — general documentation on Claude's agentic capabilities.

### On the tools and libraries chosen
- [Ultralytics YOLOv8 docs](https://docs.ultralytics.com/) — training, fine-tuning, inference, and CLI reference.
- [Albumentations docs](https://albumentations.ai/docs/) — bbox-aware image augmentation library; the [getting started with detection](https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/) page is directly relevant.
- [Gradio docs](https://www.gradio.app/docs/) — building local ML demo UIs.
- [PEP 8 style guide](https://peps.python.org/pep-0008/)
- [Google Python style guide (docstrings)](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- [Conventional Commits](https://www.conventionalcommits.org/) — commit message format used in this project.
