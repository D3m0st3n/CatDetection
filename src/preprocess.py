"""Preprocessing and augmentation for the CatDetection pipeline.

Resizes images to 640x640, performs offline augmentation via albumentations,
splits into stratified train/val/test sets, and generates ``data/data.yaml``.
"""

import logging
import random
import shutil
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import yaml
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.utils import IMAGE_EXTENSIONS, load_labels, save_labels

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SIZE: int = 640
AUGMENTATIONS_PER_IMAGE: int = 9  # original + 9 augmented = 10x on train set
TRAIN_RATIO: float = 0.8
VAL_RATIO: float = 0.1
TEST_RATIO: float = 0.1
RANDOM_SEED: int = 42

# Augmentation transform probabilities (from CONTRIBUTING.md).
_HORIZONTAL_FLIP_P: float = 0.5
_ROTATE_LIMIT: int = 15
_ROTATE_P: float = 0.7
_BRIGHTNESS_CONTRAST_P: float = 0.8
_GAUSSIAN_BLUR_LIMIT: int = 3
_GAUSSIAN_BLUR_P: float = 0.3
_RANDOM_CROP_SCALE: tuple[float, float] = (0.7, 1.0)
_RANDOM_CROP_P: float = 0.5
_HUE_SATURATION_P: float = 0.4
_MIN_VISIBILITY: float = 0.3


# ---------------------------------------------------------------------------
# Image pair loading
# ---------------------------------------------------------------------------


def load_image_pairs(
    raw_dir: Path,
    labels_dir: Path,
) -> list[tuple[Path, Path]]:
    """Scan for images with matching label files.

    Args:
        raw_dir: Directory containing source images.
        labels_dir: Directory containing YOLO ``.txt`` label files.

    Returns:
        Sorted list of ``(image_path, label_path)`` tuples.  Images without
        a matching label file are skipped with a warning.
    """
    pairs: list[tuple[Path, Path]] = []
    for img_path in sorted(raw_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            logger.warning("No label file for %s — skipping", img_path.name)
            continue
        pairs.append((img_path, label_path))

    logger.info("Found %d image/label pairs", len(pairs))
    return pairs


# ---------------------------------------------------------------------------
# Image classification for stratification
# ---------------------------------------------------------------------------


def classify_images(pairs: list[tuple[Path, Path]]) -> list[str]:
    """Classify each image for stratified splitting.

    Args:
        pairs: List of ``(image_path, label_path)`` tuples.

    Returns:
        List of category strings: ``"single"`` (one cat), ``"both"`` (two
        cats), or ``"empty"`` (no cats).
    """
    categories: list[str] = []
    for _, label_path in pairs:
        labels = load_labels(label_path)
        if not labels:
            categories.append("empty")
        else:
            class_ids = {lb["class_id"] for lb in labels}
            categories.append("both" if len(class_ids) > 1 else "single")
    return categories


# ---------------------------------------------------------------------------
# Stratified splitting
# ---------------------------------------------------------------------------


def split_dataset(
    pairs: list[tuple[Path, Path]],
    categories: list[str],
) -> tuple[
    list[tuple[Path, Path]],
    list[tuple[Path, Path]],
    list[tuple[Path, Path]],
]:
    """Perform a stratified 80/10/10 train/val/test split.

    Args:
        pairs: List of ``(image_path, label_path)`` tuples.
        categories: Per-image category strings (used for stratification).

    Returns:
        ``(train_pairs, val_pairs, test_pairs)`` tuples.
    """
    train_pairs, temp_pairs, train_cats, temp_cats = train_test_split(
        pairs,
        categories,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED,
        stratify=categories,
    )

    # For the second split, stratification may fail if a category has only
    # one member in the temp set.  Fall back to unstratified in that case.
    from collections import Counter

    cat_counts = Counter(temp_cats)
    can_stratify = all(c >= 2 for c in cat_counts.values())

    val_pairs, test_pairs, _, _ = train_test_split(
        temp_pairs,
        temp_cats,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED,
        stratify=temp_cats if can_stratify else None,
    )

    logger.info(
        "Split: %d train / %d val / %d test",
        len(train_pairs),
        len(val_pairs),
        len(test_pairs),
    )
    return train_pairs, val_pairs, test_pairs


# ---------------------------------------------------------------------------
# Image loading and resizing
# ---------------------------------------------------------------------------


def load_and_resize_image(
    image_path: Path,
    target_size: int = TARGET_SIZE,
) -> np.ndarray:
    """Load an image, fix EXIF rotation, and resize to a square.

    Args:
        image_path: Path to the source image.
        target_size: Output dimension (both width and height).

    Returns:
        RGB numpy array of shape ``(target_size, target_size, 3)``.
    """
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img = img.resize((target_size, target_size), Image.LANCZOS)
    return np.asarray(img)


# ---------------------------------------------------------------------------
# Augmentation pipeline
# ---------------------------------------------------------------------------


def build_augmentation_pipeline() -> A.Compose:
    """Build the offline augmentation pipeline.

    Returns:
        An ``albumentations.Compose`` configured with bbox-aware transforms
        matching the specification in ``CONTRIBUTING.md``.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=_HORIZONTAL_FLIP_P),
            A.Rotate(
                limit=_ROTATE_LIMIT,
                p=_ROTATE_P,
                border_mode=cv2.BORDER_REFLECT_101,
            ),
            A.RandomBrightnessContrast(p=_BRIGHTNESS_CONTRAST_P),
            A.GaussianBlur(blur_limit=_GAUSSIAN_BLUR_LIMIT, p=_GAUSSIAN_BLUR_P),
            A.RandomResizedCrop(
                size=(TARGET_SIZE, TARGET_SIZE),
                scale=_RANDOM_CROP_SCALE,
                p=_RANDOM_CROP_P,
            ),
            A.HueSaturationValue(p=_HUE_SATURATION_P),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_ids"],
            min_visibility=_MIN_VISIBILITY,
        ),
    )


def augment_image(
    image: np.ndarray,
    labels: list[dict],
    pipeline: A.Compose,
) -> tuple[np.ndarray, list[dict]]:
    """Apply augmentation to a single image and its labels.

    Args:
        image: RGB numpy array.
        labels: Label dicts with keys ``class_id``, ``x_center``, etc.
        pipeline: An ``albumentations.Compose`` with bbox parameters.

    Returns:
        ``(augmented_image, augmented_labels)`` tuple.
    """
    if not labels:
        # No bboxes — apply transforms without bbox params.
        transformed = pipeline(image=image, bboxes=[], class_ids=[])
        return transformed["image"], []

    bboxes = [
        (lb["x_center"], lb["y_center"], lb["width"], lb["height"]) for lb in labels
    ]
    class_ids = [lb["class_id"] for lb in labels]

    transformed = pipeline(image=image, bboxes=bboxes, class_ids=class_ids)

    aug_labels: list[dict] = []
    for bbox, cid in zip(transformed["bboxes"], transformed["class_ids"]):
        aug_labels.append(
            {
                "class_id": cid,
                "x_center": bbox[0],
                "y_center": bbox[1],
                "width": bbox[2],
                "height": bbox[3],
            }
        )
    return transformed["image"], aug_labels


# ---------------------------------------------------------------------------
# Image saving
# ---------------------------------------------------------------------------


def save_image(image: np.ndarray, output_path: Path) -> None:
    """Save an RGB numpy array as a JPEG file.

    Args:
        image: RGB numpy array.
        output_path: Destination path (will be created if parents missing).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), bgr)


# ---------------------------------------------------------------------------
# Split writing
# ---------------------------------------------------------------------------


def write_split(
    pairs: list[tuple[Path, Path]],
    split_name: str,
    output_dir: Path,
    pipeline: A.Compose | None = None,
    augment_count: int = 0,
) -> int:
    """Write a dataset split to disk (images + labels).

    Args:
        pairs: List of ``(image_path, label_path)`` tuples.
        split_name: Split identifier (``"train"``, ``"val"``, or ``"test"``).
        output_dir: Root output directory (e.g. ``data/``).
        pipeline: Augmentation pipeline (only used for the train split).
        augment_count: Number of augmented copies per image (0 = no
            augmentation).

    Returns:
        Total number of images written.
    """
    images_dir = output_dir / "images" / split_name
    labels_dir = output_dir / "labels" / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    total_written = 0
    desc = f"Writing {split_name}"

    for img_path, lbl_path in tqdm(pairs, desc=desc, unit="img"):
        stem = img_path.stem
        image = load_and_resize_image(img_path)
        labels = load_labels(lbl_path)

        # Save the resized original.
        save_image(image, images_dir / f"{stem}.jpg")
        save_labels(labels_dir / f"{stem}.txt", labels)
        total_written += 1

        # Generate augmented copies (train split only).
        if pipeline is not None and augment_count > 0:
            for i in range(augment_count):
                aug_img, aug_labels = augment_image(image, labels, pipeline)
                save_image(aug_img, images_dir / f"{stem}_aug{i}.jpg")
                save_labels(labels_dir / f"{stem}_aug{i}.txt", aug_labels)
                total_written += 1

    logger.info("%s: %d images written", split_name, total_written)
    return total_written


# ---------------------------------------------------------------------------
# data.yaml generation
# ---------------------------------------------------------------------------


def generate_data_yaml(output_dir: Path) -> None:
    """Write the YOLOv8 dataset configuration file.

    Args:
        output_dir: Root data directory (the file is written to
            ``output_dir / "data.yaml"``).
    """
    config = {
        "path": "./data",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 2,
        "names": ["aioli", "mayo"],
    }
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))
    logger.info("Wrote dataset config to %s", yaml_path)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_preprocessing(
    raw_dir: Path,
    labels_dir: Path,
    output_dir: Path,
) -> None:
    """Run the full preprocessing pipeline.

    Steps: load pairs, classify, stratified split, resize + augment train
    set, write all splits, generate ``data.yaml``.

    Args:
        raw_dir: Directory containing source images.
        labels_dir: Directory containing YOLO label files.
        output_dir: Root output directory (e.g. ``data/``).
    """
    # Seed for reproducibility.
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Load and classify.
    pairs = load_image_pairs(raw_dir, labels_dir)
    categories = classify_images(pairs)

    category_counts = {cat: categories.count(cat) for cat in set(categories)}
    logger.info("Category distribution: %s", category_counts)

    # Split.
    train_pairs, val_pairs, test_pairs = split_dataset(pairs, categories)

    # Clear existing output directories.
    for split in ("train", "val", "test"):
        for subdir in ("images", "labels"):
            split_dir = output_dir / subdir / split
            if split_dir.exists():
                logger.info("Clearing existing %s", split_dir)
                shutil.rmtree(split_dir)

    # Build augmentation pipeline.
    pipeline = build_augmentation_pipeline()

    # Write splits.
    train_count = write_split(
        train_pairs, "train", output_dir, pipeline, AUGMENTATIONS_PER_IMAGE
    )
    val_count = write_split(val_pairs, "val", output_dir)
    test_count = write_split(test_pairs, "test", output_dir)

    total = train_count + val_count + test_count
    original = len(pairs)
    logger.info(
        "Preprocessing complete: %d total images (%.1fx expansion from %d originals)",
        total,
        total / original if original else 0,
        original,
    )

    # Generate data.yaml.
    generate_data_yaml(output_dir)
