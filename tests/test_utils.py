"""Unit tests for src.utils."""

import numpy as np
import pytest

from src.utils import (
    denormalize_bbox,
    draw_bboxes,
    load_labels,
    normalize_bbox,
    save_labels,
)

# ---------------------------------------------------------------------------
# normalize_bbox / denormalize_bbox
# ---------------------------------------------------------------------------


class TestNormalizeBbox:
    """Tests for normalize_bbox."""

    def test_centre_box(self) -> None:
        """A box centred in a 100x100 image."""
        xc, yc, w, h = normalize_bbox(25, 25, 75, 75, 100, 100)
        assert xc == pytest.approx(0.5)
        assert yc == pytest.approx(0.5)
        assert w == pytest.approx(0.5)
        assert h == pytest.approx(0.5)

    def test_full_image_box(self) -> None:
        """A box covering the entire image."""
        xc, yc, w, h = normalize_bbox(0, 0, 640, 480, 640, 480)
        assert xc == pytest.approx(0.5)
        assert yc == pytest.approx(0.5)
        assert w == pytest.approx(1.0)
        assert h == pytest.approx(1.0)

    def test_swapped_corners(self) -> None:
        """Coordinates given in reverse order are handled correctly."""
        xc, yc, w, h = normalize_bbox(75, 75, 25, 25, 100, 100)
        assert xc == pytest.approx(0.5)
        assert yc == pytest.approx(0.5)
        assert w == pytest.approx(0.5)
        assert h == pytest.approx(0.5)

    def test_top_left_corner_box(self) -> None:
        """A small box in the top-left corner."""
        xc, yc, w, h = normalize_bbox(0, 0, 10, 20, 100, 200)
        assert xc == pytest.approx(0.05)
        assert yc == pytest.approx(0.05)
        assert w == pytest.approx(0.1)
        assert h == pytest.approx(0.1)


class TestDenormalizeBbox:
    """Tests for denormalize_bbox."""

    def test_centre_box(self) -> None:
        """Round-trip from normalised centre values."""
        x1, y1, x2, y2 = denormalize_bbox(0.5, 0.5, 0.5, 0.5, 100, 100)
        assert x1 == 25
        assert y1 == 25
        assert x2 == 75
        assert y2 == 75

    def test_full_image_box(self) -> None:
        x1, y1, x2, y2 = denormalize_bbox(0.5, 0.5, 1.0, 1.0, 640, 480)
        assert x1 == 0
        assert y1 == 0
        assert x2 == 640
        assert y2 == 480


class TestRoundTrip:
    """Normalize then denormalize should return the original corners."""

    def test_round_trip(self) -> None:
        orig = (30, 40, 80, 120)
        img_w, img_h = 200, 200
        xc, yc, w, h = normalize_bbox(*orig, img_w, img_h)
        x1, y1, x2, y2 = denormalize_bbox(xc, yc, w, h, img_w, img_h)
        assert (x1, y1, x2, y2) == orig


# ---------------------------------------------------------------------------
# save_labels / load_labels
# ---------------------------------------------------------------------------


class TestLabelIO:
    """Tests for save_labels and load_labels."""

    def test_roundtrip(self, tmp_path: object) -> None:
        """Save then load should return equivalent data."""
        labels = [
            {
                "class_id": 0,
                "x_center": 0.35,
                "y_center": 0.52,
                "width": 0.18,
                "height": 0.30,
            },
            {
                "class_id": 1,
                "x_center": 0.72,
                "y_center": 0.48,
                "width": 0.20,
                "height": 0.28,
            },
        ]
        label_path = tmp_path / "test.txt"
        save_labels(label_path, labels)
        loaded = load_labels(label_path)

        assert len(loaded) == 2
        for orig, got in zip(labels, loaded):
            assert got["class_id"] == orig["class_id"]
            assert got["x_center"] == pytest.approx(orig["x_center"], abs=1e-5)
            assert got["y_center"] == pytest.approx(orig["y_center"], abs=1e-5)
            assert got["width"] == pytest.approx(orig["width"], abs=1e-5)
            assert got["height"] == pytest.approx(orig["height"], abs=1e-5)

    def test_empty_file(self, tmp_path: object) -> None:
        """An empty label list produces an empty file that loads back empty."""
        label_path = tmp_path / "empty.txt"
        save_labels(label_path, [])
        loaded = load_labels(label_path)
        assert loaded == []

    def test_missing_file(self, tmp_path: object) -> None:
        """Loading a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_labels(tmp_path / "nonexistent.txt")

    def test_creates_parent_dirs(self, tmp_path: object) -> None:
        """save_labels creates parent directories if needed."""
        label_path = tmp_path / "nested" / "dir" / "label.txt"
        save_labels(
            label_path,
            [
                {
                    "class_id": 0,
                    "x_center": 0.5,
                    "y_center": 0.5,
                    "width": 0.1,
                    "height": 0.1,
                }
            ],
        )
        assert label_path.exists()


# ---------------------------------------------------------------------------
# draw_bboxes
# ---------------------------------------------------------------------------


class TestDrawBboxes:
    """Tests for draw_bboxes."""

    def test_returns_correct_shape(self) -> None:
        """Output image should have the same shape as input."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        labels = [
            {
                "class_id": 0,
                "x_center": 0.5,
                "y_center": 0.5,
                "width": 0.3,
                "height": 0.3,
            }
        ]
        result = draw_bboxes(img, labels, 640, 480)
        assert result.shape == img.shape

    def test_does_not_mutate_input(self) -> None:
        """The original image should not be modified."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        original = img.copy()
        labels = [
            {
                "class_id": 1,
                "x_center": 0.5,
                "y_center": 0.5,
                "width": 0.5,
                "height": 0.5,
            }
        ]
        draw_bboxes(img, labels, 100, 100)
        np.testing.assert_array_equal(img, original)

    def test_empty_labels(self) -> None:
        """An empty label list should return an unchanged copy."""
        img = np.ones((50, 50, 3), dtype=np.uint8) * 128
        result = draw_bboxes(img, [], 50, 50)
        np.testing.assert_array_equal(result, img)
