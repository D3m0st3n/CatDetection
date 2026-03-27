"""Shared utilities for the CatDetection pipeline.

Provides constants (colours, thresholds), YOLO label I/O, bounding box
coordinate conversion, bbox drawing, and logging setup.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD: float = 0.5
IOU_THRESHOLD: float = 0.45

AIOLI_COLOR: tuple[int, int, int] = (255, 107, 53)  # #FF6B35 orange
MAYO_COLOR: tuple[int, int, int] = (78, 205, 196)  # #4ECDC4 teal
BOX_THICKNESS: int = 2

CLASS_NAMES: dict[int, str] = {0: "Aïoli", 1: "Mayo"}
CLASS_COLORS: dict[int, tuple[int, int, int]] = {0: AIOLI_COLOR, 1: MAYO_COLOR}

IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with a consistent format.

    Args:
        level: Logging level (e.g. ``logging.INFO``, ``logging.DEBUG``).
    """
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(asctime)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# YOLO label I/O
# ---------------------------------------------------------------------------


def load_labels(label_path: Path) -> list[dict]:
    """Load a YOLO label file into a list of bounding box dicts.

    Args:
        label_path: Path to the ``.txt`` label file.

    Returns:
        List of dicts with keys: ``class_id``, ``x_center``, ``y_center``,
        ``width``, ``height``.  Returns an empty list when the file exists
        but is empty (meaning "no cats visible").

    Raises:
        FileNotFoundError: If *label_path* does not exist.
    """
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    labels: list[dict] = []
    text = label_path.read_text().strip()
    if not text:
        return labels

    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        labels.append(
            {
                "class_id": int(parts[0]),
                "x_center": float(parts[1]),
                "y_center": float(parts[2]),
                "width": float(parts[3]),
                "height": float(parts[4]),
            }
        )
    return labels


def save_labels(label_path: Path, labels: list[dict]) -> None:
    """Write a list of bounding box dicts to a YOLO ``.txt`` label file.

    Creates the parent directory if it does not exist.  An empty *labels*
    list produces an empty file (meaning "no cats visible").

    Args:
        label_path: Destination path for the label file.
        labels: List of dicts with keys ``class_id``, ``x_center``,
            ``y_center``, ``width``, ``height``.
    """
    label_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for lb in labels:
        lines.append(
            f"{lb['class_id']} "
            f"{lb['x_center']:.6f} "
            f"{lb['y_center']:.6f} "
            f"{lb['width']:.6f} "
            f"{lb['height']:.6f}"
        )
    label_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Bounding box coordinate conversion
# ---------------------------------------------------------------------------


def normalize_bbox(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:
    """Convert pixel corner coordinates to YOLO normalised centre format.

    Args:
        x1: Left edge in pixels.
        y1: Top edge in pixels.
        x2: Right edge in pixels.
        y2: Bottom edge in pixels.
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        ``(x_center, y_center, width, height)`` normalised to [0, 1].
    """
    # Ensure x1 < x2 and y1 < y2
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    x_center = (x1 + x2) / 2.0 / img_w
    y_center = (y1 + y2) / 2.0 / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return (x_center, y_center, width, height)


def denormalize_bbox(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    """Convert YOLO normalised centre format to pixel corner coordinates.

    Args:
        x_center: Normalised centre x.
        y_center: Normalised centre y.
        width: Normalised box width.
        height: Normalised box height.
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        ``(x1, y1, x2, y2)`` in pixels.
    """
    half_w = width * img_w / 2.0
    half_h = height * img_h / 2.0
    cx = x_center * img_w
    cy = y_center * img_h
    return (
        int(round(cx - half_w)),
        int(round(cy - half_h)),
        int(round(cx + half_w)),
        int(round(cy + half_h)),
    )


# ---------------------------------------------------------------------------
# Bounding box drawing
# ---------------------------------------------------------------------------


def draw_bboxes(
    image: np.ndarray,
    labels: list[dict],
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """Draw coloured bounding boxes and labels on an image copy.

    Args:
        image: BGR numpy array (OpenCV format).
        labels: List of label dicts (``class_id``, ``x_center``, …).
        img_w: Image width in pixels (used for denormalisation).
        img_h: Image height in pixels (used for denormalisation).

    Returns:
        A new BGR image with boxes and labels drawn.
    """
    out = image.copy()
    for lb in labels:
        cls = lb["class_id"]
        x1, y1, x2, y2 = denormalize_bbox(
            lb["x_center"], lb["y_center"], lb["width"], lb["height"], img_w, img_h
        )
        color = CLASS_COLORS.get(cls, (200, 200, 200))
        # OpenCV uses BGR — the constants are already in RGB order for
        # display but we keep them as-is since draw_bboxes is only used with
        # matplotlib (RGB) or will be converted as needed.
        cv2.rectangle(out, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        label_text = CLASS_NAMES.get(cls, f"class {cls}")
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        # Label background
        cv2.rectangle(
            out,
            (x1, y1 - th - baseline - 4),
            (x1 + tw + 4, y1),
            color,
            cv2.FILLED,
        )
        # White text
        cv2.putText(
            out,
            label_text,
            (x1 + 2, y1 - baseline - 2),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return out
