"""Evaluation utilities for the CatDetection pipeline.

Loads YOLOv8 training metrics from ``results.csv``, compares model
predictions against ground truth labels, and categorises errors
(identity swaps, missed detections, false positives).
"""

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

from src.infer import Detection, PredictionResult
from src.utils import CLASS_NAMES, load_labels

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IOU_MATCH_THRESHOLD: float = 0.5

# ---------------------------------------------------------------------------
# Training metrics dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EpochMetrics:
    """Metrics from a single training epoch.

    Attributes:
        epoch: Epoch number (1-based).
        train_box_loss: Training box regression loss.
        train_cls_loss: Training classification loss.
        train_dfl_loss: Training distribution focal loss.
        precision: Validation precision.
        recall: Validation recall.
        mAP50: Validation mAP at IoU 0.5.
        mAP50_95: Validation mAP at IoU 0.5:0.95.
        val_box_loss: Validation box loss.
        val_cls_loss: Validation classification loss.
        val_dfl_loss: Validation distribution focal loss.
    """

    epoch: int
    train_box_loss: float
    train_cls_loss: float
    train_dfl_loss: float
    precision: float
    recall: float
    mAP50: float
    mAP50_95: float
    val_box_loss: float
    val_cls_loss: float
    val_dfl_loss: float


@dataclass
class TrainingMetrics:
    """Aggregated training metrics from a ``results.csv`` file.

    Attributes:
        epochs: Per-epoch metrics.
        best_mAP50: Highest mAP50 across all epochs.
        best_mAP50_epoch: Epoch that achieved ``best_mAP50``.
        final_precision: Precision at the last epoch.
        final_recall: Recall at the last epoch.
        final_mAP50: mAP50 at the last epoch.
        final_mAP50_95: mAP50-95 at the last epoch.
    """

    epochs: list[EpochMetrics]
    best_mAP50: float
    best_mAP50_epoch: int
    final_precision: float
    final_recall: float
    final_mAP50: float
    final_mAP50_95: float


# ---------------------------------------------------------------------------
# Evaluation dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MatchResult:
    """Result of matching predictions to ground truth for one image.

    Attributes:
        image_path: Path to the source image.
        true_positives: List of (gt_box, matched_detection) pairs where
            the class ID agrees.
        missed_detections: Ground truth boxes with no matching prediction.
        false_positives: Predictions with no matching ground truth box.
        identity_swaps: Pairs where IoU matched but the class ID differs.
    """

    image_path: Path
    true_positives: list[tuple[dict, Detection]] = field(default_factory=list)
    missed_detections: list[dict] = field(default_factory=list)
    false_positives: list[Detection] = field(default_factory=list)
    identity_swaps: list[tuple[dict, Detection]] = field(default_factory=list)


@dataclass
class EvaluationSummary:
    """Overall evaluation across all test images.

    Attributes:
        total_images: Number of images evaluated.
        total_gt_boxes: Total ground truth bounding boxes.
        total_predictions: Total predictions made.
        total_true_positives: Correctly matched detections.
        total_missed_detections: Ground truth boxes not detected.
        total_false_positives: Predictions with no ground truth match.
        total_identity_swaps: Detections that matched a box but got the
            class wrong.
        per_class_precision: Precision per class ID.
        per_class_recall: Recall per class ID.
        match_results: Per-image match results.
    """

    total_images: int
    total_gt_boxes: int
    total_predictions: int
    total_true_positives: int
    total_missed_detections: int
    total_false_positives: int
    total_identity_swaps: int
    per_class_precision: dict[int, float]
    per_class_recall: dict[int, float]
    match_results: list[MatchResult]


# ---------------------------------------------------------------------------
# Training metrics loading
# ---------------------------------------------------------------------------


def load_training_metrics(results_csv: Path) -> TrainingMetrics:
    """Load and parse a YOLOv8 ``results.csv`` file.

    Args:
        results_csv: Path to the CSV file.

    Returns:
        A ``TrainingMetrics`` dataclass with per-epoch and summary data.

    Raises:
        FileNotFoundError: If *results_csv* does not exist.
    """
    if not results_csv.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_csv}")

    epochs: list[EpochMetrics] = []
    with open(results_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Strip whitespace from keys (YOLOv8 sometimes pads them)
            row = {k.strip(): v.strip() for k, v in row.items()}
            epochs.append(
                EpochMetrics(
                    epoch=int(row["epoch"]),
                    train_box_loss=float(row["train/box_loss"]),
                    train_cls_loss=float(row["train/cls_loss"]),
                    train_dfl_loss=float(row["train/dfl_loss"]),
                    precision=float(row["metrics/precision(B)"]),
                    recall=float(row["metrics/recall(B)"]),
                    mAP50=float(row["metrics/mAP50(B)"]),
                    mAP50_95=float(row["metrics/mAP50-95(B)"]),
                    val_box_loss=float(row["val/box_loss"]),
                    val_cls_loss=float(row["val/cls_loss"]),
                    val_dfl_loss=float(row["val/dfl_loss"]),
                )
            )

    best_epoch = max(epochs, key=lambda e: e.mAP50)
    final = epochs[-1]

    logger.info(
        "Loaded %d epochs — best mAP50 %.3f at epoch %d, final mAP50 %.3f",
        len(epochs),
        best_epoch.mAP50,
        best_epoch.epoch,
        final.mAP50,
    )

    return TrainingMetrics(
        epochs=epochs,
        best_mAP50=best_epoch.mAP50,
        best_mAP50_epoch=best_epoch.epoch,
        final_precision=final.precision,
        final_recall=final.recall,
        final_mAP50=final.mAP50,
        final_mAP50_95=final.mAP50_95,
    )


# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------


def compute_iou(box_a: dict, box_b: dict | Detection) -> float:
    """Compute Intersection over Union between two YOLO-format boxes.

    Both boxes use normalised centre format (``x_center``, ``y_center``,
    ``width``, ``height``).

    Args:
        box_a: First box (dict with ``x_center``, ``y_center``, ``width``,
            ``height`` keys).
        box_b: Second box (dict or ``Detection`` with same attributes).

    Returns:
        IoU value in [0, 1].
    """
    # Extract coordinates — handle both dict and Detection
    if isinstance(box_b, Detection):
        bx, by, bw, bh = box_b.x_center, box_b.y_center, box_b.width, box_b.height
    else:
        bx, by, bw, bh = (
            box_b["x_center"],
            box_b["y_center"],
            box_b["width"],
            box_b["height"],
        )

    ax, ay, aw, ah = (
        box_a["x_center"],
        box_a["y_center"],
        box_a["width"],
        box_a["height"],
    )

    # Convert to corner format
    a_x1 = ax - aw / 2
    a_y1 = ay - ah / 2
    a_x2 = ax + aw / 2
    a_y2 = ay + ah / 2

    b_x1 = bx - bw / 2
    b_y1 = by - bh / 2
    b_x2 = bx + bw / 2
    b_y2 = by + bh / 2

    # Intersection
    inter_x1 = max(a_x1, b_x1)
    inter_y1 = max(a_y1, b_y1)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Union
    a_area = aw * ah
    b_area = bw * bh
    union_area = a_area + b_area - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


# ---------------------------------------------------------------------------
# Prediction matching
# ---------------------------------------------------------------------------


def match_predictions(
    ground_truth: list[dict],
    predictions: list[Detection],
    iou_threshold: float = IOU_MATCH_THRESHOLD,
) -> tuple[
    list[tuple[dict, Detection]],
    list[dict],
    list[Detection],
    list[tuple[dict, Detection]],
]:
    """Match predicted detections to ground truth boxes for one image.

    Uses greedy matching: for each GT box, the prediction with the
    highest IoU above *iou_threshold* is selected. Each prediction can
    match at most one GT box.

    Args:
        ground_truth: List of GT label dicts from
            :func:`~src.utils.load_labels`.
        predictions: List of ``Detection`` objects.
        iou_threshold: Minimum IoU to consider a match.

    Returns:
        A 4-tuple of:
        - **true_positives**: ``(gt_box, prediction)`` pairs with
          matching class ID.
        - **missed_detections**: GT boxes with no matching prediction.
        - **false_positives**: Predictions with no matching GT box.
        - **identity_swaps**: ``(gt_box, prediction)`` pairs where IoU
          matched but class ID differs.
    """
    true_positives: list[tuple[dict, Detection]] = []
    identity_swaps: list[tuple[dict, Detection]] = []
    matched_pred_indices: set[int] = set()

    for gt in ground_truth:
        best_iou = 0.0
        best_idx = -1
        for idx, pred in enumerate(predictions):
            if idx in matched_pred_indices:
                continue
            iou = compute_iou(gt, pred)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_iou >= iou_threshold and best_idx >= 0:
            matched_pred_indices.add(best_idx)
            pred = predictions[best_idx]
            if gt["class_id"] == pred.class_id:
                true_positives.append((gt, pred))
            else:
                identity_swaps.append((gt, pred))

    missed_detections = [
        gt
        for i, gt in enumerate(ground_truth)
        if not any(gt is tp[0] for tp in true_positives)
        and not any(gt is sw[0] for sw in identity_swaps)
    ]

    false_positives = [
        pred
        for idx, pred in enumerate(predictions)
        if idx not in matched_pred_indices
    ]

    return true_positives, missed_detections, false_positives, identity_swaps


# ---------------------------------------------------------------------------
# Full test set evaluation
# ---------------------------------------------------------------------------


def evaluate_test_set(
    predictions: list[PredictionResult],
    labels_dir: Path,
) -> EvaluationSummary:
    """Evaluate model predictions against ground truth for all test images.

    Args:
        predictions: List of ``PredictionResult`` from
            :func:`~src.infer.predict_batch`.
        labels_dir: Directory containing YOLO ``.txt`` ground truth labels.

    Returns:
        An ``EvaluationSummary`` with per-image match results and
        aggregate statistics.
    """
    match_results: list[MatchResult] = []
    total_gt = 0
    total_preds = 0
    total_tp = 0
    total_missed = 0
    total_fp = 0
    total_swaps = 0

    # Per-class counters for precision/recall
    class_tp: dict[int, int] = {}
    class_gt_count: dict[int, int] = {}
    class_pred_count: dict[int, int] = {}

    for pred_result in predictions:
        label_path = labels_dir / (pred_result.image_path.stem + ".txt")
        if label_path.exists():
            gt_labels = load_labels(label_path)
        else:
            gt_labels = []
            logger.warning("No label file for %s", pred_result.image_path.name)

        tp, missed, fp, swaps = match_predictions(gt_labels, pred_result.detections)

        match_results.append(
            MatchResult(
                image_path=pred_result.image_path,
                true_positives=tp,
                missed_detections=missed,
                false_positives=fp,
                identity_swaps=swaps,
            )
        )

        total_gt += len(gt_labels)
        total_preds += len(pred_result.detections)
        total_tp += len(tp)
        total_missed += len(missed)
        total_fp += len(fp)
        total_swaps += len(swaps)

        # Per-class stats
        for gt_box in gt_labels:
            cls = gt_box["class_id"]
            class_gt_count[cls] = class_gt_count.get(cls, 0) + 1

        for det in pred_result.detections:
            class_pred_count[det.class_id] = (
                class_pred_count.get(det.class_id, 0) + 1
            )

        for gt_box, det in tp:
            cls = gt_box["class_id"]
            class_tp[cls] = class_tp.get(cls, 0) + 1

    # Compute per-class precision and recall
    all_classes = set(class_gt_count.keys()) | set(class_pred_count.keys())
    per_class_precision: dict[int, float] = {}
    per_class_recall: dict[int, float] = {}

    for cls in sorted(all_classes):
        tp_count = class_tp.get(cls, 0)
        pred_count = class_pred_count.get(cls, 0)
        gt_count = class_gt_count.get(cls, 0)
        per_class_precision[cls] = tp_count / pred_count if pred_count > 0 else 0.0
        per_class_recall[cls] = tp_count / gt_count if gt_count > 0 else 0.0

    return EvaluationSummary(
        total_images=len(predictions),
        total_gt_boxes=total_gt,
        total_predictions=total_preds,
        total_true_positives=total_tp,
        total_missed_detections=total_missed,
        total_false_positives=total_fp,
        total_identity_swaps=total_swaps,
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        match_results=match_results,
    )


def print_evaluation_summary(summary: EvaluationSummary) -> None:
    """Log a formatted evaluation summary at INFO level.

    Args:
        summary: An ``EvaluationSummary`` from
            :func:`evaluate_test_set`.
    """
    logger.info("=== Evaluation Summary ===")
    logger.info("Images: %d", summary.total_images)
    logger.info("Ground truth boxes: %d", summary.total_gt_boxes)
    logger.info("Predictions: %d", summary.total_predictions)
    logger.info(
        "True positives: %d | Missed: %d | False positives: %d | Identity swaps: %d",
        summary.total_true_positives,
        summary.total_missed_detections,
        summary.total_false_positives,
        summary.total_identity_swaps,
    )
    for cls in sorted(summary.per_class_precision.keys()):
        name = CLASS_NAMES.get(cls, f"class {cls}")
        logger.info(
            "%s — precision: %.3f, recall: %.3f",
            name,
            summary.per_class_precision[cls],
            summary.per_class_recall[cls],
        )
