"""Inference engine for the CatDetection pipeline.

Loads a trained YOLOv8 checkpoint and runs inference on one or more
images, returning structured prediction results.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.utils import (
    BOX_THICKNESS,
    CLASS_COLORS,
    CLASS_NAMES,
    CONFIDENCE_THRESHOLD,
    IMAGE_EXTENSIONS,
    IOU_THRESHOLD,
    denormalize_bbox,
    normalize_bbox,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CONFIDENCE: float = CONFIDENCE_THRESHOLD
DEFAULT_IOU: float = IOU_THRESHOLD

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Detection:
    """A single detected object.

    Attributes:
        class_id: Integer class label (0 = Aïoli, 1 = Mayo).
        x_center: Normalised centre x coordinate.
        y_center: Normalised centre y coordinate.
        width: Normalised box width.
        height: Normalised box height.
        confidence: Detection confidence score in [0, 1].
    """

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float


@dataclass
class PredictionResult:
    """Predictions for a single image.

    Attributes:
        image_path: Path to the source image.
        detections: List of Detection objects.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
    """

    image_path: Path
    detections: list[Detection]
    image_width: int
    image_height: int


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(weights: Path) -> YOLO:
    """Load a YOLOv8 model from a checkpoint file.

    Args:
        weights: Path to the ``.pt`` weights file.

    Returns:
        A loaded YOLO model ready for inference.

    Raises:
        FileNotFoundError: If *weights* does not exist.
    """
    if not weights.exists():
        raise FileNotFoundError(f"Weights file not found: {weights}")

    logger.info("Loading model from %s", weights)
    return YOLO(str(weights))


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict_image(
    model: YOLO,
    image_path: Path,
    confidence: float = DEFAULT_CONFIDENCE,
    iou: float = DEFAULT_IOU,
) -> PredictionResult:
    """Run inference on a single image.

    Args:
        model: A loaded YOLO model.
        image_path: Path to the image file.
        confidence: Minimum confidence threshold.
        iou: IoU threshold for NMS.

    Returns:
        A ``PredictionResult`` with all detections above the confidence
        threshold.

    Raises:
        FileNotFoundError: If *image_path* does not exist.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = model.predict(str(image_path), conf=confidence, iou=iou, verbose=False)
    result = results[0]

    img_h, img_w = result.orig_shape
    detections: list[Detection] = []

    if result.boxes is not None and len(result.boxes) > 0:
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        for xyxy, cls, conf in zip(boxes_xyxy, classes, confidences):
            x1, y1, x2, y2 = xyxy
            xc, yc, w, h = normalize_bbox(
                int(round(x1)), int(round(y1)),
                int(round(x2)), int(round(y2)),
                img_w, img_h,
            )
            detections.append(
                Detection(
                    class_id=int(cls),
                    x_center=xc,
                    y_center=yc,
                    width=w,
                    height=h,
                    confidence=float(conf),
                )
            )

    logger.debug(
        "%s: %d detection(s)", image_path.name, len(detections)
    )
    return PredictionResult(
        image_path=image_path,
        detections=detections,
        image_width=img_w,
        image_height=img_h,
    )


def predict_batch(
    model: YOLO,
    image_dir: Path,
    confidence: float = DEFAULT_CONFIDENCE,
    iou: float = DEFAULT_IOU,
) -> list[PredictionResult]:
    """Run inference on all images in a directory.

    Args:
        model: A loaded YOLO model.
        image_dir: Directory containing image files.
        confidence: Minimum confidence threshold.
        iou: IoU threshold for NMS.

    Returns:
        List of ``PredictionResult``, one per image, sorted by filename.

    Raises:
        FileNotFoundError: If *image_dir* does not exist.
        ValueError: If no images are found in the directory.
    """
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    logger.info("Running inference on %d images in %s", len(image_paths), image_dir)

    results: list[PredictionResult] = []
    for path in image_paths:
        results.append(predict_image(model, path, confidence, iou))

    return results


# ---------------------------------------------------------------------------
# Conversion and drawing
# ---------------------------------------------------------------------------


def detections_to_labels(detections: list[Detection]) -> list[dict]:
    """Convert Detection objects to the label-dict format used by draw_bboxes.

    Args:
        detections: List of ``Detection`` objects.

    Returns:
        List of dicts with keys ``class_id``, ``x_center``, ``y_center``,
        ``width``, ``height``.
    """
    return [
        {
            "class_id": d.class_id,
            "x_center": d.x_center,
            "y_center": d.y_center,
            "width": d.width,
            "height": d.height,
        }
        for d in detections
    ]


def draw_predictions(
    image: np.ndarray,
    detections: list[Detection],
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """Draw predicted bounding boxes with confidence scores on an image.

    Like :func:`~src.utils.draw_bboxes` but appends confidence percentages
    to labels (e.g. ``Aïoli 87%``).

    Args:
        image: BGR numpy array (OpenCV format).
        detections: List of ``Detection`` objects.
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        A new BGR image with boxes, labels, and confidence scores drawn.
    """
    out = image.copy()
    for det in detections:
        cls = det.class_id
        x1, y1, x2, y2 = denormalize_bbox(
            det.x_center, det.y_center, det.width, det.height, img_w, img_h
        )
        color = CLASS_COLORS.get(cls, (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        name = CLASS_NAMES.get(cls, f"class {cls}")
        label_text = f"{name} {det.confidence:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(
            label_text, font, font_scale, thickness
        )
        cv2.rectangle(
            out,
            (x1, y1 - th - baseline - 4),
            (x1 + tw + 4, y1),
            color,
            cv2.FILLED,
        )
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
