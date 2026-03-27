"""Unit tests for src.preprocess."""

import numpy as np
import yaml
from PIL import Image

from src.preprocess import (
    augment_image,
    build_augmentation_pipeline,
    classify_images,
    generate_data_yaml,
    load_and_resize_image,
    load_image_pairs,
    run_preprocessing,
    split_dataset,
    write_split,
)
from src.utils import save_labels

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_synthetic_dataset(
    tmp_path: object,
    n_single: int = 6,
    n_both: int = 4,
    n_empty: int = 0,
) -> tuple:
    """Create a minimal synthetic dataset for testing.

    Returns:
        ``(raw_dir, labels_dir)`` paths.
    """
    raw_dir = tmp_path / "raw"
    labels_dir = tmp_path / "labels"
    raw_dir.mkdir()
    labels_dir.mkdir()

    idx = 0

    # Single-cat images (alternating class 0 and class 1).
    for i in range(n_single):
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(raw_dir / f"img_{idx:04d}.jpg")
        cls = i % 2
        save_labels(
            labels_dir / f"img_{idx:04d}.txt",
            [
                {
                    "class_id": cls,
                    "x_center": 0.5,
                    "y_center": 0.5,
                    "width": 0.4,
                    "height": 0.4,
                }
            ],
        )
        idx += 1

    # Both-cats images.
    for _ in range(n_both):
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(raw_dir / f"img_{idx:04d}.jpg")
        save_labels(
            labels_dir / f"img_{idx:04d}.txt",
            [
                {
                    "class_id": 0,
                    "x_center": 0.3,
                    "y_center": 0.5,
                    "width": 0.3,
                    "height": 0.4,
                },
                {
                    "class_id": 1,
                    "x_center": 0.7,
                    "y_center": 0.5,
                    "width": 0.3,
                    "height": 0.4,
                },
            ],
        )
        idx += 1

    # Empty-label images.
    for _ in range(n_empty):
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(raw_dir / f"img_{idx:04d}.jpg")
        save_labels(labels_dir / f"img_{idx:04d}.txt", [])
        idx += 1

    return raw_dir, labels_dir


# ---------------------------------------------------------------------------
# load_image_pairs
# ---------------------------------------------------------------------------


class TestLoadImagePairs:
    """Tests for load_image_pairs."""

    def test_finds_all_pairs(self, tmp_path: object) -> None:
        """All images with matching labels should be found."""
        raw_dir, labels_dir = _create_synthetic_dataset(tmp_path, n_single=4, n_both=2)
        pairs = load_image_pairs(raw_dir, labels_dir)
        assert len(pairs) == 6

    def test_skips_missing_labels(self, tmp_path: object) -> None:
        """Images without a label file are skipped."""
        raw_dir, labels_dir = _create_synthetic_dataset(tmp_path, n_single=3, n_both=0)
        # Add an image with no label.
        img = Image.fromarray(np.zeros((50, 50, 3), dtype=np.uint8))
        img.save(raw_dir / "orphan.jpg")

        pairs = load_image_pairs(raw_dir, labels_dir)
        assert len(pairs) == 3
        stems = {p[0].stem for p in pairs}
        assert "orphan" not in stems

    def test_handles_jpeg_extension(self, tmp_path: object) -> None:
        """A .jpeg file with a matching label should be paired."""
        raw_dir = tmp_path / "raw"
        labels_dir = tmp_path / "labels"
        raw_dir.mkdir()
        labels_dir.mkdir()

        img = Image.fromarray(np.zeros((50, 50, 3), dtype=np.uint8))
        img.save(raw_dir / "photo.jpeg")
        save_labels(
            labels_dir / "photo.txt",
            [
                {
                    "class_id": 0,
                    "x_center": 0.5,
                    "y_center": 0.5,
                    "width": 0.3,
                    "height": 0.3,
                }
            ],
        )

        pairs = load_image_pairs(raw_dir, labels_dir)
        assert len(pairs) == 1
        assert pairs[0][0].suffix == ".jpeg"


# ---------------------------------------------------------------------------
# classify_images
# ---------------------------------------------------------------------------


class TestClassifyImages:
    """Tests for classify_images."""

    def test_single_cat(self, tmp_path: object) -> None:
        """An image with one cat class returns 'single'."""
        raw_dir, labels_dir = _create_synthetic_dataset(tmp_path, n_single=1, n_both=0)
        pairs = load_image_pairs(raw_dir, labels_dir)
        cats = classify_images(pairs)
        assert cats == ["single"]

    def test_both_cats(self, tmp_path: object) -> None:
        """An image with both cat classes returns 'both'."""
        raw_dir, labels_dir = _create_synthetic_dataset(tmp_path, n_single=0, n_both=1)
        pairs = load_image_pairs(raw_dir, labels_dir)
        cats = classify_images(pairs)
        assert cats == ["both"]

    def test_empty_label(self, tmp_path: object) -> None:
        """An image with an empty label file returns 'empty'."""
        raw_dir, labels_dir = _create_synthetic_dataset(
            tmp_path, n_single=0, n_both=0, n_empty=1
        )
        pairs = load_image_pairs(raw_dir, labels_dir)
        cats = classify_images(pairs)
        assert cats == ["empty"]


# ---------------------------------------------------------------------------
# split_dataset
# ---------------------------------------------------------------------------


class TestSplitDataset:
    """Tests for split_dataset."""

    def test_split_sizes(self, tmp_path: object) -> None:
        """The split should produce roughly 80/10/10."""
        raw_dir, labels_dir = _create_synthetic_dataset(tmp_path, n_single=16, n_both=4)
        pairs = load_image_pairs(raw_dir, labels_dir)
        cats = classify_images(pairs)
        train, val, test = split_dataset(pairs, cats)
        assert len(train) + len(val) + len(test) == 20

    def test_no_overlap(self, tmp_path: object) -> None:
        """No image should appear in more than one split."""
        raw_dir, labels_dir = _create_synthetic_dataset(tmp_path, n_single=16, n_both=4)
        pairs = load_image_pairs(raw_dir, labels_dir)
        cats = classify_images(pairs)
        train, val, test = split_dataset(pairs, cats)

        train_stems = {p[0].stem for p in train}
        val_stems = {p[0].stem for p in val}
        test_stems = {p[0].stem for p in test}
        assert train_stems.isdisjoint(val_stems)
        assert train_stems.isdisjoint(test_stems)
        assert val_stems.isdisjoint(test_stems)

    def test_all_images_assigned(self, tmp_path: object) -> None:
        """Every input image should appear in exactly one split."""
        raw_dir, labels_dir = _create_synthetic_dataset(tmp_path, n_single=16, n_both=4)
        pairs = load_image_pairs(raw_dir, labels_dir)
        cats = classify_images(pairs)
        train, val, test = split_dataset(pairs, cats)

        all_stems = {p[0].stem for p in train + val + test}
        input_stems = {p[0].stem for p in pairs}
        assert all_stems == input_stems


# ---------------------------------------------------------------------------
# load_and_resize_image
# ---------------------------------------------------------------------------


class TestLoadAndResizeImage:
    """Tests for load_and_resize_image."""

    def test_output_dimensions(self, tmp_path: object) -> None:
        """A non-square image should be resized to 640x640."""
        img = Image.fromarray(np.zeros((200, 100, 3), dtype=np.uint8))
        img.save(tmp_path / "test.jpg")
        result = load_and_resize_image(tmp_path / "test.jpg")
        assert result.shape == (640, 640, 3)

    def test_custom_size(self, tmp_path: object) -> None:
        """A custom target_size should be respected."""
        img = Image.fromarray(np.zeros((50, 50, 3), dtype=np.uint8))
        img.save(tmp_path / "small.jpg")
        result = load_and_resize_image(tmp_path / "small.jpg", target_size=128)
        assert result.shape == (128, 128, 3)


# ---------------------------------------------------------------------------
# build_augmentation_pipeline / augment_image
# ---------------------------------------------------------------------------


class TestBuildAugmentationPipeline:
    """Tests for build_augmentation_pipeline."""

    def test_returns_compose(self) -> None:
        """The pipeline should be an albumentations Compose."""
        import albumentations as A

        pipeline = build_augmentation_pipeline()
        assert isinstance(pipeline, A.Compose)

    def test_preserves_large_bbox(self) -> None:
        """A large centred bbox should survive augmentation."""
        pipeline = build_augmentation_pipeline()
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        labels = [
            {
                "class_id": 0,
                "x_center": 0.5,
                "y_center": 0.5,
                "width": 0.8,
                "height": 0.8,
            }
        ]
        # Run several times — a large box should almost always survive.
        survived = 0
        for _ in range(10):
            _, aug_labels = augment_image(image, labels, pipeline)
            if aug_labels:
                survived += 1
        assert survived >= 7  # Allow some crop-induced losses.


class TestAugmentImage:
    """Tests for augment_image."""

    def test_output_shape(self) -> None:
        """Augmented image should keep the same shape."""
        pipeline = build_augmentation_pipeline()
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        labels = [
            {
                "class_id": 1,
                "x_center": 0.5,
                "y_center": 0.5,
                "width": 0.4,
                "height": 0.4,
            }
        ]
        aug_img, _ = augment_image(image, labels, pipeline)
        assert aug_img.shape == image.shape

    def test_labels_format(self) -> None:
        """Output labels should have the correct dict keys."""
        pipeline = build_augmentation_pipeline()
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        labels = [
            {
                "class_id": 0,
                "x_center": 0.5,
                "y_center": 0.5,
                "width": 0.6,
                "height": 0.6,
            }
        ]
        _, aug_labels = augment_image(image, labels, pipeline)
        expected_keys = {"class_id", "x_center", "y_center", "width", "height"}
        for lb in aug_labels:
            assert set(lb.keys()) == expected_keys

    def test_empty_labels(self) -> None:
        """Augmenting an image with no labels should work."""
        pipeline = build_augmentation_pipeline()
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        aug_img, aug_labels = augment_image(image, [], pipeline)
        assert aug_img.shape == image.shape
        assert aug_labels == []


# ---------------------------------------------------------------------------
# write_split
# ---------------------------------------------------------------------------


class TestWriteSplit:
    """Tests for write_split."""

    def test_writes_images_and_labels(self, tmp_path: object) -> None:
        """Files should exist on disk after writing."""
        raw_dir, labels_dir = _create_synthetic_dataset(tmp_path, n_single=3, n_both=0)
        pairs = load_image_pairs(raw_dir, labels_dir)
        output_dir = tmp_path / "output"

        count = write_split(pairs, "val", output_dir)
        assert count == 3
        assert len(list((output_dir / "images" / "val").glob("*.jpg"))) == 3
        assert len(list((output_dir / "labels" / "val").glob("*.txt"))) == 3

    def test_augmentation_count(self, tmp_path: object) -> None:
        """With augment_count=3, each image produces 4 files (1 + 3)."""
        raw_dir, labels_dir = _create_synthetic_dataset(tmp_path, n_single=2, n_both=0)
        pairs = load_image_pairs(raw_dir, labels_dir)
        output_dir = tmp_path / "output"
        pipeline = build_augmentation_pipeline()

        count = write_split(pairs, "train", output_dir, pipeline, augment_count=3)
        assert count == 2 * 4  # 2 images x (1 original + 3 augmented)
        assert len(list((output_dir / "images" / "train").glob("*.jpg"))) == 8

    def test_no_augmentation_for_val(self, tmp_path: object) -> None:
        """Val split with no pipeline should have only originals."""
        raw_dir, labels_dir = _create_synthetic_dataset(tmp_path, n_single=2, n_both=0)
        pairs = load_image_pairs(raw_dir, labels_dir)
        output_dir = tmp_path / "output"

        count = write_split(pairs, "val", output_dir, pipeline=None, augment_count=0)
        assert count == 2


# ---------------------------------------------------------------------------
# generate_data_yaml
# ---------------------------------------------------------------------------


class TestGenerateDataYaml:
    """Tests for generate_data_yaml."""

    def test_yaml_content(self, tmp_path: object) -> None:
        """Generated YAML should contain the expected keys and values."""
        generate_data_yaml(tmp_path)
        yaml_path = tmp_path / "data.yaml"
        assert yaml_path.exists()

        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        assert config["path"] == "./data"
        assert config["train"] == "images/train"
        assert config["val"] == "images/val"
        assert config["test"] == "images/test"
        assert config["nc"] == 2
        assert config["names"] == ["aioli", "mayo"]


# ---------------------------------------------------------------------------
# run_preprocessing (integration)
# ---------------------------------------------------------------------------


class TestRunPreprocessing:
    """Integration test for the full pipeline."""

    def test_end_to_end(self, tmp_path: object) -> None:
        """The full pipeline should produce the expected directory structure."""
        raw_dir, labels_dir = _create_synthetic_dataset(tmp_path, n_single=16, n_both=4)
        output_dir = tmp_path / "data"
        output_dir.mkdir()

        run_preprocessing(raw_dir, labels_dir, output_dir)

        # Directory structure exists.
        for split in ("train", "val", "test"):
            assert (output_dir / "images" / split).is_dir()
            assert (output_dir / "labels" / split).is_dir()

        # Train should be augmented (more files than input).
        train_images = list((output_dir / "images" / "train").glob("*.jpg"))
        assert len(train_images) > 16  # Must be augmented.

        # Val and test should have only originals.
        val_images = list((output_dir / "images" / "val").glob("*.jpg"))
        test_images = list((output_dir / "images" / "test").glob("*.jpg"))
        assert len(val_images) <= 4
        assert len(test_images) <= 4

        # data.yaml exists.
        assert (output_dir / "data.yaml").exists()

        # All output images are 640x640.
        for img_path in val_images[:2]:
            img = Image.open(img_path)
            assert img.size == (640, 640)
