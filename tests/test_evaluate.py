"""Unit tests for src.evaluate."""

from pathlib import Path

import pytest

from src.evaluate import (
    EvaluationSummary,
    MatchResult,
    TrainingMetrics,
    compute_iou,
    evaluate_test_set,
    load_training_metrics,
    match_predictions,
)
from src.infer import Detection, PredictionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RESULTS_CSV_HEADER = (
    "epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,"
    "metrics/precision(B),metrics/recall(B),metrics/mAP50(B),"
    "metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss,"
    "lr/pg0,lr/pg1,lr/pg2"
)

RESULTS_CSV_ROW_1 = "1,30.0,1.5,2.0,1.8,0.60,0.55,0.60,0.30,1.7,2.4,1.9,0.001,0.001,0.001"
RESULTS_CSV_ROW_2 = "2,60.0,1.3,1.5,1.6,0.70,0.65,0.75,0.40,1.5,1.8,1.7,0.001,0.001,0.001"
RESULTS_CSV_ROW_3 = "3,90.0,1.1,1.2,1.4,0.65,0.60,0.70,0.35,1.4,1.6,1.5,0.001,0.001,0.001"


def _make_box(
    class_id: int, x: float, y: float, w: float, h: float
) -> dict:
    """Create a ground truth label dict."""
    return {
        "class_id": class_id,
        "x_center": x,
        "y_center": y,
        "width": w,
        "height": h,
    }


def _make_det(
    class_id: int, x: float, y: float, w: float, h: float, conf: float = 0.9
) -> Detection:
    """Create a Detection object."""
    return Detection(
        class_id=class_id,
        x_center=x,
        y_center=y,
        width=w,
        height=h,
        confidence=conf,
    )


# ---------------------------------------------------------------------------
# TestComputeIou
# ---------------------------------------------------------------------------


class TestComputeIou:
    """Tests for compute_iou."""

    def test_perfect_overlap(self) -> None:
        """Identical boxes have IoU of 1.0."""
        box = _make_box(0, 0.5, 0.5, 0.2, 0.2)
        det = _make_det(0, 0.5, 0.5, 0.2, 0.2)
        assert compute_iou(box, det) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        """Non-overlapping boxes have IoU of 0.0."""
        box = _make_box(0, 0.1, 0.1, 0.1, 0.1)
        det = _make_det(0, 0.9, 0.9, 0.1, 0.1)
        assert compute_iou(box, det) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        """Known partial overlap has the expected IoU."""
        # Box A: centre (0.5, 0.5), size 0.4x0.4 → corners (0.3, 0.3) to (0.7, 0.7)
        # Box B: centre (0.6, 0.6), size 0.4x0.4 → corners (0.4, 0.4) to (0.8, 0.8)
        # Intersection: (0.4, 0.4) to (0.7, 0.7) = 0.3 × 0.3 = 0.09
        # Union: 0.16 + 0.16 - 0.09 = 0.23
        # IoU = 0.09 / 0.23 ≈ 0.3913
        box = _make_box(0, 0.5, 0.5, 0.4, 0.4)
        det = _make_det(0, 0.6, 0.6, 0.4, 0.4)
        assert compute_iou(box, det) == pytest.approx(0.09 / 0.23, abs=0.001)

    def test_dict_vs_dict(self) -> None:
        """Both arguments as dicts works."""
        box_a = _make_box(0, 0.5, 0.5, 0.2, 0.2)
        box_b = _make_box(1, 0.5, 0.5, 0.2, 0.2)
        assert compute_iou(box_a, box_b) == pytest.approx(1.0)

    def test_zero_area_box(self) -> None:
        """Zero-area box returns 0.0 IoU."""
        box = _make_box(0, 0.5, 0.5, 0.0, 0.0)
        det = _make_det(0, 0.5, 0.5, 0.2, 0.2)
        assert compute_iou(box, det) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestMatchPredictions
# ---------------------------------------------------------------------------


class TestMatchPredictions:
    """Tests for match_predictions."""

    def test_all_correct(self) -> None:
        """All predictions match their GT boxes correctly."""
        gt = [_make_box(0, 0.3, 0.3, 0.2, 0.2), _make_box(1, 0.7, 0.7, 0.2, 0.2)]
        preds = [_make_det(0, 0.3, 0.3, 0.2, 0.2), _make_det(1, 0.7, 0.7, 0.2, 0.2)]

        tp, missed, fp, swaps = match_predictions(gt, preds)

        assert len(tp) == 2
        assert len(missed) == 0
        assert len(fp) == 0
        assert len(swaps) == 0

    def test_missed_detection(self) -> None:
        """GT box with no matching prediction is a missed detection."""
        gt = [_make_box(0, 0.5, 0.5, 0.2, 0.2)]
        preds: list[Detection] = []

        tp, missed, fp, swaps = match_predictions(gt, preds)

        assert len(tp) == 0
        assert len(missed) == 1
        assert len(fp) == 0
        assert len(swaps) == 0

    def test_false_positive(self) -> None:
        """Prediction with no GT match is a false positive."""
        gt: list[dict] = []
        preds = [_make_det(0, 0.5, 0.5, 0.2, 0.2)]

        tp, missed, fp, swaps = match_predictions(gt, preds)

        assert len(tp) == 0
        assert len(missed) == 0
        assert len(fp) == 1
        assert len(swaps) == 0

    def test_identity_swap(self) -> None:
        """Prediction matching GT box but with wrong class is an identity swap."""
        gt = [_make_box(0, 0.5, 0.5, 0.2, 0.2)]
        preds = [_make_det(1, 0.5, 0.5, 0.2, 0.2)]  # Wrong class

        tp, missed, fp, swaps = match_predictions(gt, preds)

        assert len(tp) == 0
        assert len(missed) == 0
        assert len(fp) == 0
        assert len(swaps) == 1

    def test_empty_both(self) -> None:
        """Empty GT and empty predictions produces all-empty results."""
        tp, missed, fp, swaps = match_predictions([], [])

        assert len(tp) == 0
        assert len(missed) == 0
        assert len(fp) == 0
        assert len(swaps) == 0

    def test_mixed_scenario(self) -> None:
        """Mix of TP, missed, FP, and identity swap."""
        gt = [
            _make_box(0, 0.2, 0.2, 0.15, 0.15),  # Will be matched correctly
            _make_box(1, 0.5, 0.5, 0.15, 0.15),  # Will be identity swapped
            _make_box(0, 0.8, 0.8, 0.15, 0.15),  # Will be missed
        ]
        preds = [
            _make_det(0, 0.2, 0.2, 0.15, 0.15),  # TP
            _make_det(0, 0.5, 0.5, 0.15, 0.15),  # Swap (pred class 0 vs GT class 1)
            _make_det(1, 0.1, 0.1, 0.05, 0.05),  # FP (no GT match)
        ]

        tp, missed, fp, swaps = match_predictions(gt, preds)

        assert len(tp) == 1
        assert len(missed) == 1
        assert len(fp) == 1
        assert len(swaps) == 1

    def test_low_iou_not_matched(self) -> None:
        """Prediction far from GT is not matched even if same class."""
        gt = [_make_box(0, 0.1, 0.1, 0.1, 0.1)]
        preds = [_make_det(0, 0.9, 0.9, 0.1, 0.1)]

        tp, missed, fp, swaps = match_predictions(gt, preds)

        assert len(tp) == 0
        assert len(missed) == 1
        assert len(fp) == 1
        assert len(swaps) == 0


# ---------------------------------------------------------------------------
# TestLoadTrainingMetrics
# ---------------------------------------------------------------------------


class TestLoadTrainingMetrics:
    """Tests for load_training_metrics."""

    def test_loads_csv(self, tmp_path: Path) -> None:
        """Correctly parses a synthetic results.csv."""
        csv_path = tmp_path / "results.csv"
        csv_path.write_text(
            f"{RESULTS_CSV_HEADER}\n"
            f"{RESULTS_CSV_ROW_1}\n"
            f"{RESULTS_CSV_ROW_2}\n"
            f"{RESULTS_CSV_ROW_3}\n"
        )

        metrics = load_training_metrics(csv_path)

        assert len(metrics.epochs) == 3
        assert metrics.best_mAP50 == pytest.approx(0.75)
        assert metrics.best_mAP50_epoch == 2
        assert metrics.final_mAP50 == pytest.approx(0.70)
        assert metrics.final_precision == pytest.approx(0.65)
        assert metrics.final_recall == pytest.approx(0.60)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """FileNotFoundError when CSV does not exist."""
        with pytest.raises(FileNotFoundError, match="Results CSV not found"):
            load_training_metrics(tmp_path / "nope.csv")

    def test_single_epoch(self, tmp_path: Path) -> None:
        """Works with a single epoch."""
        csv_path = tmp_path / "results.csv"
        csv_path.write_text(f"{RESULTS_CSV_HEADER}\n{RESULTS_CSV_ROW_1}\n")

        metrics = load_training_metrics(csv_path)

        assert len(metrics.epochs) == 1
        assert metrics.best_mAP50 == pytest.approx(0.60)
        assert metrics.final_mAP50 == pytest.approx(0.60)


# ---------------------------------------------------------------------------
# TestEvaluateTestSet
# ---------------------------------------------------------------------------


class TestEvaluateTestSet:
    """Tests for evaluate_test_set."""

    def test_basic_evaluation(self, tmp_path: Path) -> None:
        """Aggregate stats are computed correctly."""
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()

        # Image 1: one GT box, one correct prediction
        (labels_dir / "img1.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        pred1 = PredictionResult(
            image_path=Path("img1.jpg"),
            detections=[_make_det(0, 0.5, 0.5, 0.2, 0.2)],
            image_width=640,
            image_height=640,
        )

        # Image 2: one GT box, one missed (no predictions)
        (labels_dir / "img2.txt").write_text("1 0.3 0.3 0.15 0.15\n")
        pred2 = PredictionResult(
            image_path=Path("img2.jpg"),
            detections=[],
            image_width=640,
            image_height=640,
        )

        summary = evaluate_test_set([pred1, pred2], labels_dir)

        assert summary.total_images == 2
        assert summary.total_gt_boxes == 2
        assert summary.total_true_positives == 1
        assert summary.total_missed_detections == 1
        assert summary.total_false_positives == 0
        assert summary.total_identity_swaps == 0
        assert summary.per_class_precision[0] == pytest.approx(1.0)
        assert summary.per_class_recall[0] == pytest.approx(1.0)
        assert summary.per_class_recall[1] == pytest.approx(0.0)

    def test_missing_label_file(self, tmp_path: Path) -> None:
        """Missing label file logs warning and treats as no GT."""
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()

        pred = PredictionResult(
            image_path=Path("unknown.jpg"),
            detections=[_make_det(0, 0.5, 0.5, 0.2, 0.2)],
            image_width=640,
            image_height=640,
        )

        summary = evaluate_test_set([pred], labels_dir)

        assert summary.total_gt_boxes == 0
        assert summary.total_false_positives == 1
