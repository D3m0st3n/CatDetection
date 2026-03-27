"""Two-stage YOLOv8 fine-tuning for the CatDetection pipeline.

Stage 1 freezes the backbone and trains only the detection head.
Stage 2 unfreezes all layers for a lower-learning-rate full fine-tune.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — default hyperparameters
# ---------------------------------------------------------------------------

DEFAULT_BASE_MODEL: str = "yolov8s.pt"
DEFAULT_DATA_YAML: str = "data/data.yaml"
DEFAULT_PROJECT: str = "runs/detect"

# Stage 1 — frozen backbone
STAGE1_EPOCHS: int = 20
STAGE1_FREEZE: int = 10
STAGE1_LR0: float = 0.01

# Stage 2 — full fine-tune
STAGE2_EPOCHS: int = 50
STAGE2_LR0: float = 0.001

DEFAULT_DEVICE: int | str = 0  # GPU 0; can be "cpu"
DEFAULT_IMGSZ: int = 640
DEFAULT_WORKERS: int = 0  # 0 avoids Windows multiprocessing spawn issues

# Pattern for YOLOv8 auto-incremented run directories.
_TRAIN_DIR_PATTERN: re.Pattern[str] = re.compile(r"^train(\d*)$")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainingResult:
    """Paths and metadata from a completed training run.

    Attributes:
        stage1_dir: Path to the Stage 1 run directory.
        stage2_dir: Path to the Stage 2 run directory.
        best_weights: Path to ``best.pt`` from Stage 2.
        last_weights: Path to ``last.pt`` from Stage 2.
        results_csv: Path to ``results.csv`` from Stage 2.
    """

    stage1_dir: Path
    stage2_dir: Path
    best_weights: Path
    last_weights: Path
    results_csv: Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_latest_run_dir(project: Path) -> Path:
    """Find the most recent YOLOv8 training run directory.

    YOLOv8 names output directories ``train``, ``train2``, ``train3``, etc.
    This function returns the one with the highest suffix number.

    Args:
        project: Parent directory containing run folders
            (e.g. ``runs/detect``).

    Returns:
        Path to the latest ``train*`` directory.

    Raises:
        FileNotFoundError: If *project* does not exist or contains no
            ``train*`` directories.
    """
    if not project.is_dir():
        raise FileNotFoundError(f"Project directory not found: {project}")

    run_dirs: list[tuple[int, Path]] = []
    for entry in project.iterdir():
        if not entry.is_dir():
            continue
        match = _TRAIN_DIR_PATTERN.match(entry.name)
        if match:
            suffix = int(match.group(1)) if match.group(1) else 0
            run_dirs.append((suffix, entry))

    if not run_dirs:
        raise FileNotFoundError(f"No train* directories found in {project}")

    run_dirs.sort(key=lambda t: t[0])
    return run_dirs[-1][1]


def _validate_data_yaml(data_yaml: Path) -> None:
    """Raise FileNotFoundError if the data YAML does not exist.

    Args:
        data_yaml: Path to the dataset YAML configuration.

    Raises:
        FileNotFoundError: If *data_yaml* does not exist.
    """
    if not data_yaml.exists():
        raise FileNotFoundError(f"Data YAML not found: {data_yaml}")


# ---------------------------------------------------------------------------
# Stage 1 — frozen backbone
# ---------------------------------------------------------------------------


def run_stage1(
    data_yaml: Path = Path(DEFAULT_DATA_YAML),
    base_model: str = DEFAULT_BASE_MODEL,
    epochs: int = STAGE1_EPOCHS,
    freeze: int = STAGE1_FREEZE,
    lr0: float = STAGE1_LR0,
    device: int | str = DEFAULT_DEVICE,
    imgsz: int = DEFAULT_IMGSZ,
    project: str = DEFAULT_PROJECT,
    workers: int = DEFAULT_WORKERS,
    amp: bool = False,
) -> Path:
    """Run Stage 1: frozen-backbone training.

    Loads a pretrained YOLOv8 model and trains with the first *freeze*
    layers frozen, warming up the detection head on the custom dataset.

    Args:
        data_yaml: Path to the dataset YAML configuration.
        base_model: Pretrained model name or path (e.g. ``"yolov8s.pt"``).
        epochs: Number of training epochs.
        freeze: Number of backbone layers to freeze.
        lr0: Initial learning rate.
        device: Training device (GPU index or ``"cpu"``).
        imgsz: Input image size.
        project: Parent directory for YOLOv8 output.
        workers: Number of dataloader workers (0 for Windows compatibility).
        amp: Enable Automatic Mixed Precision (disable on GPUs that
            fail AMP checks, e.g. MX550).

    Returns:
        Path to the Stage 1 run directory (resolved via
        ``find_latest_run_dir``).

    Raises:
        FileNotFoundError: If *data_yaml* does not exist.
        RuntimeError: If training fails or no weights are produced.
    """
    _validate_data_yaml(data_yaml)

    project_path = Path(project).resolve()

    logger.info(
        "Stage 1 — frozen backbone: model=%s, epochs=%d, freeze=%d, lr0=%s",
        base_model,
        epochs,
        freeze,
        lr0,
    )

    model = YOLO(base_model)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        freeze=freeze,
        lr0=lr0,
        device=device,
        imgsz=imgsz,
        project=str(project_path),
        workers=workers,
        amp=amp,
    )

    run_dir = find_latest_run_dir(project_path)
    checkpoint = run_dir / "weights" / "last.pt"
    if not checkpoint.exists():
        raise RuntimeError(f"Stage 1 did not produce weights: {checkpoint}")

    logger.info("Stage 1 complete — run directory: %s", run_dir)
    return run_dir


# ---------------------------------------------------------------------------
# Stage 2 — full fine-tune
# ---------------------------------------------------------------------------


def run_stage2(
    stage1_dir: Path,
    data_yaml: Path = Path(DEFAULT_DATA_YAML),
    epochs: int = STAGE2_EPOCHS,
    lr0: float = STAGE2_LR0,
    device: int | str = DEFAULT_DEVICE,
    imgsz: int = DEFAULT_IMGSZ,
    project: str = DEFAULT_PROJECT,
    workers: int = DEFAULT_WORKERS,
    amp: bool = False,
) -> Path:
    """Run Stage 2: full fine-tuning from a Stage 1 checkpoint.

    Loads ``last.pt`` from *stage1_dir* and trains all layers with a
    lower learning rate.

    Args:
        stage1_dir: Path to the Stage 1 run directory.
        data_yaml: Path to the dataset YAML configuration.
        epochs: Number of training epochs.
        lr0: Initial learning rate.
        device: Training device (GPU index or ``"cpu"``).
        imgsz: Input image size.
        project: Parent directory for YOLOv8 output.
        workers: Number of dataloader workers (0 for Windows compatibility).
        amp: Enable Automatic Mixed Precision (disable on GPUs that
            fail AMP checks, e.g. MX550).

    Returns:
        Path to the Stage 2 run directory.

    Raises:
        FileNotFoundError: If *data_yaml* or the Stage 1 checkpoint
            does not exist.
        RuntimeError: If training fails or no weights are produced.
    """
    checkpoint = stage1_dir / "weights" / "last.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {checkpoint}")
    _validate_data_yaml(data_yaml)

    project_path = Path(project).resolve()

    logger.info(
        "Stage 2 — full fine-tune: checkpoint=%s, epochs=%d, lr0=%s",
        checkpoint,
        epochs,
        lr0,
    )

    model = YOLO(str(checkpoint))
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        lr0=lr0,
        device=device,
        imgsz=imgsz,
        project=str(project_path),
        workers=workers,
        amp=amp,
    )

    run_dir = find_latest_run_dir(project_path)
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"Stage 2 did not produce best weights: {best}")

    logger.info("Stage 2 complete — run directory: %s", run_dir)
    return run_dir


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_training(
    data_yaml: Path = Path(DEFAULT_DATA_YAML),
    base_model: str = DEFAULT_BASE_MODEL,
    stage1_epochs: int = STAGE1_EPOCHS,
    stage1_freeze: int = STAGE1_FREEZE,
    stage1_lr0: float = STAGE1_LR0,
    stage2_epochs: int = STAGE2_EPOCHS,
    stage2_lr0: float = STAGE2_LR0,
    device: int | str = DEFAULT_DEVICE,
    imgsz: int = DEFAULT_IMGSZ,
    project: str = DEFAULT_PROJECT,
    workers: int = DEFAULT_WORKERS,
    amp: bool = False,
) -> TrainingResult:
    """Run the full two-stage fine-tuning pipeline.

    Stage 1 freezes the backbone and trains the detection head.
    Stage 2 unfreezes all layers and fine-tunes with a lower learning
    rate.  Returns a ``TrainingResult`` with paths to all outputs.

    Args:
        data_yaml: Path to the dataset YAML configuration.
        base_model: Pretrained model name or path.
        stage1_epochs: Epochs for Stage 1.
        stage1_freeze: Layers to freeze in Stage 1.
        stage1_lr0: Learning rate for Stage 1.
        stage2_epochs: Epochs for Stage 2.
        stage2_lr0: Learning rate for Stage 2.
        device: Training device.
        imgsz: Input image size.
        project: Parent directory for YOLOv8 output.
        workers: Number of dataloader workers (0 for Windows compatibility).
        amp: Enable Automatic Mixed Precision.

    Returns:
        A ``TrainingResult`` dataclass with paths to weights, results,
        and run directories.

    Raises:
        FileNotFoundError: If *data_yaml* does not exist.
        RuntimeError: If either training stage fails.
    """
    logger.info("Starting two-stage fine-tuning pipeline")

    stage1_dir = run_stage1(
        data_yaml=data_yaml,
        base_model=base_model,
        epochs=stage1_epochs,
        freeze=stage1_freeze,
        lr0=stage1_lr0,
        device=device,
        imgsz=imgsz,
        project=project,
        workers=workers,
        amp=amp,
    )

    stage2_dir = run_stage2(
        stage1_dir=stage1_dir,
        data_yaml=data_yaml,
        epochs=stage2_epochs,
        lr0=stage2_lr0,
        device=device,
        imgsz=imgsz,
        project=project,
        workers=workers,
        amp=amp,
    )

    result = TrainingResult(
        stage1_dir=stage1_dir,
        stage2_dir=stage2_dir,
        best_weights=stage2_dir / "weights" / "best.pt",
        last_weights=stage2_dir / "weights" / "last.pt",
        results_csv=stage2_dir / "results.csv",
    )

    logger.info("Training complete — best weights: %s", result.best_weights)
    return result
