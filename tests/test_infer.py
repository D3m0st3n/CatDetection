"""Unit tests for src.infer."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.infer import (
    Detection,
    PredictionResult,
    detections_to_labels,
    draw_predictions,
    load_model,
    predict_batch,
    predict_image,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_result(
    boxes_xyxy: list[list[float]],
    classes: list[int],
    confidences: list[float],
    orig_shape: tuple[int, int] = (640, 640),
) -> MagicMock:
    """Create a fake ultralytics Results object.

    Args:
        boxes_xyxy: List of [x1, y1, x2, y2] pixel coordinates.
        classes: List of integer class IDs.
        confidences: List of confidence scores.
        orig_shape: (height, width) of the original image.

    Returns:
        A MagicMock mimicking ultralytics Results.
    """
    result = MagicMock()
    result.orig_shape = orig_shape

    if boxes_xyxy:
        xyxy_tensor = MagicMock()
        xyxy_tensor.cpu.return_value.numpy.return_value = np.array(
            boxes_xyxy, dtype=np.float32
        )
        cls_tensor = MagicMock()
        cls_tensor.cpu.return_value.numpy.return_value = np.array(
            classes, dtype=np.float32
        )
        conf_tensor = MagicMock()
        conf_tensor.cpu.return_value.numpy.return_value = np.array(
            confidences, dtype=np.float32
        )
        result.boxes = MagicMock()
        result.boxes.xyxy = xyxy_tensor
        result.boxes.cls = cls_tensor
        result.boxes.conf = conf_tensor
        result.boxes.__len__ = lambda self: len(boxes_xyxy)
    else:
        result.boxes = MagicMock()
        result.boxes.__len__ = lambda self: 0

    return result


def _create_synthetic_image(path: Path, size: int = 640) -> None:
    """Write a minimal JPEG image to disk.

    Args:
        path: Destination file path.
        size: Image dimension (square).
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


# ---------------------------------------------------------------------------
# TestLoadModel
# ---------------------------------------------------------------------------


class TestLoadModel:
    """Tests for load_model."""

    def test_missing_weights_raises(self, tmp_path: Path) -> None:
        """FileNotFoundError when weights do not exist."""
        with pytest.raises(FileNotFoundError, match="Weights file not found"):
            load_model(tmp_path / "nonexistent.pt")

    @patch("src.infer.YOLO")
    def test_loads_model(self, mock_yolo: MagicMock, tmp_path: Path) -> None:
        """YOLO is constructed with the weights path."""
        weights = tmp_path / "best.pt"
        weights.write_text("fake")

        load_model(weights)

        mock_yolo.assert_called_once_with(str(weights))


# ---------------------------------------------------------------------------
# TestPredictImage
# ---------------------------------------------------------------------------


class TestPredictImage:
    """Tests for predict_image."""

    def test_missing_image_raises(self, tmp_path: Path) -> None:
        """FileNotFoundError when image does not exist."""
        model = MagicMock()
        with pytest.raises(FileNotFoundError, match="Image not found"):
            predict_image(model, tmp_path / "noimage.jpg")

    def test_returns_detections(self, tmp_path: Path) -> None:
        """Detections are correctly parsed from model output."""
        img_path = tmp_path / "test.jpg"
        _create_synthetic_image(img_path)

        fake_result = _make_fake_result(
            boxes_xyxy=[[100.0, 200.0, 300.0, 400.0]],
            classes=[0],
            confidences=[0.95],
            orig_shape=(640, 640),
        )
        model = MagicMock()
        model.predict.return_value = [fake_result]

        result = predict_image(model, img_path)

        assert isinstance(result, PredictionResult)
        assert result.image_path == img_path
        assert result.image_width == 640
        assert result.image_height == 640
        assert len(result.detections) == 1

        det = result.detections[0]
        assert det.class_id == 0
        assert det.confidence == pytest.approx(0.95, abs=0.01)
        # Centre of [100, 200, 300, 400] in 640×640: (200/640, 300/640)
        assert det.x_center == pytest.approx(200 / 640, abs=0.02)
        assert det.y_center == pytest.approx(300 / 640, abs=0.02)

    def test_no_detections(self, tmp_path: Path) -> None:
        """Empty result when model finds nothing."""
        img_path = tmp_path / "empty.jpg"
        _create_synthetic_image(img_path)

        fake_result = _make_fake_result(
            boxes_xyxy=[], classes=[], confidences=[]
        )
        model = MagicMock()
        model.predict.return_value = [fake_result]

        result = predict_image(model, img_path)

        assert len(result.detections) == 0

    def test_multiple_detections(self, tmp_path: Path) -> None:
        """Multiple boxes are all returned."""
        img_path = tmp_path / "both.jpg"
        _create_synthetic_image(img_path)

        fake_result = _make_fake_result(
            boxes_xyxy=[[50, 50, 200, 200], [400, 400, 600, 600]],
            classes=[0, 1],
            confidences=[0.9, 0.8],
        )
        model = MagicMock()
        model.predict.return_value = [fake_result]

        result = predict_image(model, img_path)

        assert len(result.detections) == 2
        assert result.detections[0].class_id == 0
        assert result.detections[1].class_id == 1


# ---------------------------------------------------------------------------
# TestPredictBatch
# ---------------------------------------------------------------------------


class TestPredictBatch:
    """Tests for predict_batch."""

    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        """FileNotFoundError when directory does not exist."""
        model = MagicMock()
        with pytest.raises(FileNotFoundError, match="Image directory not found"):
            predict_batch(model, tmp_path / "nonexistent")

    def test_empty_dir_raises(self, tmp_path: Path) -> None:
        """ValueError when directory contains no images."""
        model = MagicMock()
        with pytest.raises(ValueError, match="No images found"):
            predict_batch(model, tmp_path)

    @patch("src.infer.predict_image")
    def test_processes_all_images_sorted(
        self, mock_predict: MagicMock, tmp_path: Path
    ) -> None:
        """All images are processed in sorted filename order."""
        for name in ["c.jpg", "a.jpg", "b.png"]:
            _create_synthetic_image(tmp_path / name)
        # Non-image file should be skipped
        (tmp_path / "readme.txt").write_text("not an image")

        mock_predict.side_effect = lambda model, path, *a, **kw: PredictionResult(
            image_path=path, detections=[], image_width=640, image_height=640
        )

        model = MagicMock()
        results = predict_batch(model, tmp_path)

        assert len(results) == 3
        assert results[0].image_path.name == "a.jpg"
        assert results[1].image_path.name == "b.png"
        assert results[2].image_path.name == "c.jpg"


# ---------------------------------------------------------------------------
# TestDetectionsToLabels
# ---------------------------------------------------------------------------


class TestDetectionsToLabels:
    """Tests for detections_to_labels."""

    def test_conversion(self) -> None:
        """Detection objects are converted to label dicts."""
        dets = [
            Detection(
                class_id=0,
                x_center=0.5,
                y_center=0.5,
                width=0.2,
                height=0.3,
                confidence=0.9,
            ),
            Detection(
                class_id=1,
                x_center=0.7,
                y_center=0.3,
                width=0.1,
                height=0.1,
                confidence=0.8,
            ),
        ]
        labels = detections_to_labels(dets)

        assert len(labels) == 2
        assert labels[0]["class_id"] == 0
        assert labels[0]["x_center"] == 0.5
        assert "confidence" not in labels[0]
        assert labels[1]["class_id"] == 1

    def test_empty_list(self) -> None:
        """Empty input produces empty output."""
        assert detections_to_labels([]) == []


# ---------------------------------------------------------------------------
# TestDrawPredictions
# ---------------------------------------------------------------------------


class TestDrawPredictions:
    """Tests for draw_predictions."""

    def test_draws_on_image(self) -> None:
        """Output image is modified and has the same shape."""
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        dets = [
            Detection(
                class_id=0,
                x_center=0.5,
                y_center=0.5,
                width=0.2,
                height=0.3,
                confidence=0.92,
            )
        ]

        result = draw_predictions(img, dets, 640, 640)

        assert result.shape == img.shape
        assert not np.array_equal(result, img)

    def test_empty_detections(self) -> None:
        """No detections returns an unchanged copy."""
        img = np.zeros((640, 640, 3), dtype=np.uint8)

        result = draw_predictions(img, [], 640, 640)

        assert np.array_equal(result, img)
        assert result is not img  # must be a copy
