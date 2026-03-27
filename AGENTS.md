# AGENTS.md — Project Root

## What This Project Is

A fully local ML pipeline to detect and identify two cats — **Aïoli** and **Mayo** — by name in photos. It fine-tunes a pretrained YOLOv8s model using transfer learning and serves results through a Gradio GUI and a CLI.

## Key Documents

| File | Purpose |
|------|---------|
| `README.md` | Project overview, demo, model performance, and AI-agent development methodology |
| `TUTORIAL.md` | Step-by-step guide for new users — environment setup through inference |
| `ROADMAP.md` | Goals, state-of-the-art context, and phase-by-phase milestones |
| `DESIGN.md` | UX, visual style, naming, interface behaviour, and exhaustive technical specifications |
| `CONTRIBUTING.md` | Architecture, code standards, data formats, and git conventions |
| `plan.md` | Original planning document — retained for historical reference |
| `design_decisions.md` | Log of every decision made during planning and why (decisions 1–38) |

## Project Structure at a Glance

```
data/               Raw photos, YOLO labels, and processed train/val/test splits
notebooks/          Jupyter workflows: annotation and training results
src/                Core Python library (imported by notebooks, app, and CLI)
app/                Gradio inference GUI
tests/              pytest unit tests for src/
assets/             Static assets committed to the repo (e.g. demo detection image for README)
runs/               YOLOv8 training outputs — weights, metrics, plots (gitignored)
outputs/            Annotated images saved by the Gradio GUI (gitignored)
cli.py              Unified CLI entry point (planned)
pyproject.toml      Tool config: black, ruff, pytest
requirements.txt    Runtime dependencies
requirements-dev.txt  Dev-only dependencies
```

## Current Status

**Phase 5 (Inference App) complete** — `app/gui.py` implements the Gradio interface: image upload, per-cat bounding box overlay with confidence scores, plain-English detection summary, confidence threshold slider, and save-result to `outputs/`. Run with `python app/gui.py` or `python cli.py app`. Next: Phase 6 (CLI).

## Where to Start

- **New to this project** → read `TUTORIAL.md` end-to-end before anything else
- **Next step: CLI** → implement `cli.py` subcommands (Phase 6 in `ROADMAP.md`)
- **Reviewing annotations** → `notebooks/01_annotate.ipynb` (run in browser Jupyter, not VS Code)
- **Re-running preprocessing** → `from src.preprocess import run_preprocessing; run_preprocessing(Path('data/raw'), Path('data/labels'), Path('data'))`
- **Running the full pipeline** → `python cli.py --help`
- **Understanding a design decision** → `design_decisions.md`
- **Adding code** → read `CONTRIBUTING.md` first
- **Checking project status** → `ROADMAP.md` milestones
