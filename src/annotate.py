"""Annotation engine for the CatDetection pipeline.

Provides :class:`AnnotationSession` (state management and label persistence)
and :class:`AnnotationWidget` (interactive matplotlib/ipywidgets drawing UI).
"""

import logging
from pathlib import Path

import ipywidgets as widgets
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display as ipy_display
from PIL import Image, ImageOps

from src.utils import (
    CLASS_COLORS,
    CLASS_NAMES,
    IMAGE_EXTENSIONS,
    denormalize_bbox,
    load_labels,
    normalize_bbox,
    save_labels,
)

logger = logging.getLogger(__name__)

# Minimum rectangle size in pixels to reject accidental clicks.
_MIN_BOX_PX: int = 5


# ---------------------------------------------------------------------------
# AnnotationSession
# ---------------------------------------------------------------------------


class AnnotationSession:
    """Manages annotation state: image list, navigation, and label I/O.

    Args:
        raw_dir: Directory containing source images.
        labels_dir: Directory where YOLO ``.txt`` label files are stored.
    """

    def __init__(self, raw_dir: Path, labels_dir: Path) -> None:
        self.raw_dir = raw_dir
        self.labels_dir = labels_dir
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        self.image_paths: list[Path] = sorted(
            p for p in raw_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.image_paths:
            logger.warning("No images found in %s", raw_dir)

        self.skipped: set[int] = set()

        # Start at the first unannotated image.
        first_unannotated = self._first_unannotated_index()
        self.current_index: int = first_unannotated if first_unannotated >= 0 else 0

    # -- Properties ----------------------------------------------------------

    @property
    def total_images(self) -> int:
        """Total number of images in the raw directory."""
        return len(self.image_paths)

    @property
    def annotated_count(self) -> int:
        """Number of images that have a corresponding label file."""
        return sum(1 for i in range(self.total_images) if self.is_annotated(i))

    @property
    def remaining_count(self) -> int:
        """Number of images still lacking a label file."""
        return self.total_images - self.annotated_count

    @property
    def current_image_path(self) -> Path:
        """Path to the image at the current index."""
        return self.image_paths[self.current_index]

    # -- Helpers -------------------------------------------------------------

    def _label_path_for(self, index: int) -> Path:
        """Return the expected label file path for the image at *index*."""
        stem = self.image_paths[index].stem
        return self.labels_dir / f"{stem}.txt"

    def is_annotated(self, index: int) -> bool:
        """Check whether a label file exists for the image at *index*."""
        return self._label_path_for(index).exists()

    def _first_unannotated_index(self) -> int:
        """Return the index of the first unannotated image, or -1."""
        for i in range(self.total_images):
            if not self.is_annotated(i):
                return i
        return -1

    def get_unannotated_indices(self) -> list[int]:
        """Return indices of all images without a label file."""
        return [i for i in range(self.total_images) if not self.is_annotated(i)]

    # -- Navigation ----------------------------------------------------------

    def go_next(self) -> None:
        """Advance to the next image (clamped to the last index)."""
        if self.current_index < self.total_images - 1:
            self.current_index += 1

    def go_previous(self) -> None:
        """Go back to the previous image (clamped to 0)."""
        if self.current_index > 0:
            self.current_index -= 1

    def skip(self) -> None:
        """Mark the current image as skipped and advance."""
        self.skipped.add(self.current_index)
        logger.info(
            "Skipped image %d: %s", self.current_index, self.current_image_path.name
        )
        self.go_next()

    def go_to(self, index: int) -> None:
        """Jump to a specific image index."""
        self.current_index = max(0, min(index, self.total_images - 1))

    # -- Label I/O -----------------------------------------------------------

    def load_current_labels(self) -> list[dict]:
        """Load saved labels for the current image, or return an empty list."""
        lp = self._label_path_for(self.current_index)
        if not lp.exists():
            return []
        return load_labels(lp)

    def save_current_labels(self, labels: list[dict]) -> None:
        """Persist labels for the current image to disk."""
        lp = self._label_path_for(self.current_index)
        save_labels(lp, labels)
        logger.info(
            "Saved %d label(s) for %s",
            len(labels),
            self.current_image_path.name,
        )


# ---------------------------------------------------------------------------
# AnnotationWidget
# ---------------------------------------------------------------------------


class AnnotationWidget:
    """Interactive bounding box annotation widget for Jupyter notebooks.

    Uses matplotlib's event system for click-and-drag rectangle drawing and
    ipywidgets for navigation controls.

    Args:
        session: An :class:`AnnotationSession` instance.
        figsize: Figure size for the matplotlib canvas.
    """

    def __init__(
        self,
        session: AnnotationSession,
        figsize: tuple[int, int] = (12, 9),
    ) -> None:
        self.session = session
        self.figsize = figsize

        # Pending (unsaved) boxes for the current image.
        self._pending_boxes: list[dict] = []

        # Drawing state.
        self._drawing: bool = False
        self._start_xy: tuple[float, float] | None = None
        self._temp_rect: patches.Rectangle | None = None

        # Build UI elements.
        self._build_widgets()
        self._build_figure()

    # -- Widget construction -------------------------------------------------

    def _build_widgets(self) -> None:
        """Create ipywidgets controls."""
        self._class_dropdown = widgets.Dropdown(
            options=[(name, cid) for cid, name in CLASS_NAMES.items()],
            value=0,
            description="Cat:",
        )

        btn_style = {"button_width": "100px"}

        self._btn_prev = widgets.Button(description="◀ Previous", **btn_style)
        self._btn_next = widgets.Button(description="Next ▶", **btn_style)
        self._btn_skip = widgets.Button(description="Skip", **btn_style)
        self._btn_confirm = widgets.Button(
            description="Confirm",
            button_style="success",
            **btn_style,
        )
        self._btn_delete = widgets.Button(
            description="Delete Last",
            button_style="danger",
            **btn_style,
        )
        self._btn_empty = widgets.Button(
            description="Mark Empty",
            button_style="warning",
            **btn_style,
        )

        self._btn_prev.on_click(self._on_previous)
        self._btn_next.on_click(self._on_next)
        self._btn_skip.on_click(self._on_skip)
        self._btn_confirm.on_click(self._on_confirm)
        self._btn_delete.on_click(self._on_delete_last)
        self._btn_empty.on_click(self._on_mark_empty)

        self._progress_html = widgets.HTML()
        self._bbox_list_html = widgets.HTML()

    def _build_figure(self) -> None:
        """Create the matplotlib figure and connect mouse events."""
        self._fig, self._ax = plt.subplots(1, 1, figsize=self.figsize)
        self._fig.canvas.toolbar_visible = False
        self._fig.canvas.header_visible = False
        self._fig.canvas.footer_visible = False

        self._fig.canvas.mpl_connect("button_press_event", self._on_press)
        self._fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._fig.canvas.mpl_connect("button_release_event", self._on_release)

    # -- Display -------------------------------------------------------------

    def display(self) -> None:
        """Render the full annotation interface in the notebook."""
        nav_row = widgets.HBox(
            [
                self._btn_prev,
                self._btn_next,
                self._btn_skip,
            ]
        )
        action_row = widgets.HBox(
            [
                self._class_dropdown,
                self._btn_confirm,
                self._btn_delete,
                self._btn_empty,
            ]
        )
        controls = widgets.VBox(
            [
                self._progress_html,
                nav_row,
                action_row,
                self._bbox_list_html,
            ]
        )

        self._refresh_display()
        ipy_display(controls)

    # -- Image loading -------------------------------------------------------

    def _load_image_rgb(self) -> np.ndarray:
        """Load the current image as an RGB numpy array with EXIF fix."""
        img = Image.open(self.session.current_image_path)
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        return np.asarray(img)

    # -- Refresh -------------------------------------------------------------

    def _refresh_display(self) -> None:
        """Redraw the image, overlays, and status widgets."""
        self._ax.clear()
        self._ax.set_axis_off()

        img_rgb = self._load_image_rgb()
        img_h, img_w = img_rgb.shape[:2]
        self._img_w = img_w
        self._img_h = img_h

        self._ax.imshow(img_rgb)

        # Draw already-saved labels (dashed style).
        saved = self.session.load_current_labels()
        for lb in saved:
            self._draw_rect(lb, linestyle="--", alpha=0.5)

        # Draw pending boxes (solid style).
        for lb in self._pending_boxes:
            self._draw_rect(lb, linestyle="-", alpha=1.0)

        self._ax.set_title(self.session.current_image_path.name, fontsize=10)
        self._fig.canvas.draw_idle()

        # Update HTML widgets.
        idx = self.session.current_index
        total = self.session.total_images
        annotated = self.session.annotated_count
        remaining = self.session.remaining_count
        status = "✓ annotated" if self.session.is_annotated(idx) else "not annotated"
        self._progress_html.value = (
            f"<b>Image {idx + 1} / {total}</b> — {status} "
            f"| Annotated: {annotated} | Remaining: {remaining} "
            f"| Skipped: {len(self.session.skipped)}"
        )

        # Bbox list.
        lines: list[str] = []
        for i, lb in enumerate(self._pending_boxes):
            name = CLASS_NAMES.get(lb["class_id"], "?")
            xc = lb["x_center"]
            yc = lb["y_center"]
            lines.append(f"{i + 1}. <b>{name}</b>" f" — centre ({xc:.3f}, {yc:.3f})")
        if lines:
            self._bbox_list_html.value = "<br>".join(lines)
        else:
            self._bbox_list_html.value = (
                "<i>No pending boxes. Draw on the image or click Mark Empty.</i>"
            )

    def _draw_rect(
        self,
        label: dict,
        linestyle: str = "-",
        alpha: float = 1.0,
    ) -> None:
        """Draw a single bounding box rectangle on the axes."""
        x1, y1, x2, y2 = denormalize_bbox(
            label["x_center"],
            label["y_center"],
            label["width"],
            label["height"],
            self._img_w,
            self._img_h,
        )
        color_rgb = CLASS_COLORS.get(label["class_id"], (200, 200, 200))
        mpl_color = tuple(c / 255.0 for c in color_rgb)
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor=mpl_color,
            facecolor="none",
            linestyle=linestyle,
            alpha=alpha,
        )
        self._ax.add_patch(rect)

        name = CLASS_NAMES.get(label["class_id"], "?")
        self._ax.text(
            x1,
            y1 - 4,
            name,
            color="white",
            fontsize=9,
            fontweight="bold",
            bbox={
                "facecolor": mpl_color,
                "alpha": alpha,
                "pad": 2,
                "edgecolor": "none",
            },
        )

    # -- Mouse events --------------------------------------------------------

    def _on_press(self, event: object) -> None:
        """Handle mouse button press: start drawing a rectangle."""
        if event.inaxes != self._ax or event.xdata is None:
            return
        self._drawing = True
        self._start_xy = (event.xdata, event.ydata)

    def _on_motion(self, event: object) -> None:
        """Handle mouse motion: update the temporary rectangle."""
        if not self._drawing or event.xdata is None:
            return

        sx, sy = self._start_xy  # type: ignore[misc]
        w = event.xdata - sx
        h = event.ydata - sy

        if self._temp_rect is not None:
            self._temp_rect.remove()

        cls = self._class_dropdown.value
        color_rgb = CLASS_COLORS.get(cls, (200, 200, 200))
        mpl_color = tuple(c / 255.0 for c in color_rgb)

        self._temp_rect = patches.Rectangle(
            (sx, sy),
            w,
            h,
            linewidth=2,
            edgecolor=mpl_color,
            facecolor=(*mpl_color, 0.15),
            linestyle="--",
        )
        self._ax.add_patch(self._temp_rect)
        self._fig.canvas.draw_idle()

    def _on_release(self, event: object) -> None:
        """Handle mouse button release: finalise the drawn rectangle."""
        if not self._drawing or event.xdata is None:
            self._drawing = False
            return
        self._drawing = False

        sx, sy = self._start_xy  # type: ignore[misc]
        ex, ey = event.xdata, event.ydata

        # Remove the temporary rectangle.
        if self._temp_rect is not None:
            self._temp_rect.remove()
            self._temp_rect = None

        # Reject tiny accidental clicks.
        if abs(ex - sx) < _MIN_BOX_PX or abs(ey - sy) < _MIN_BOX_PX:
            self._fig.canvas.draw_idle()
            return

        x_center, y_center, w, h = normalize_bbox(
            int(round(sx)),
            int(round(sy)),
            int(round(ex)),
            int(round(ey)),
            self._img_w,
            self._img_h,
        )

        cls = self._class_dropdown.value
        box = {
            "class_id": cls,
            "x_center": x_center,
            "y_center": y_center,
            "width": w,
            "height": h,
        }
        self._pending_boxes.append(box)
        logger.debug("Added pending box: %s", box)

        self._refresh_display()

    # -- Button handlers -----------------------------------------------------

    def _on_confirm(self, _button: object) -> None:
        """Save pending boxes and advance to the next image."""
        saved = self.session.load_current_labels()
        merged = saved + self._pending_boxes
        self.session.save_current_labels(merged)
        self._pending_boxes = []
        self.session.go_next()
        self._refresh_display()

    def _on_next(self, _button: object) -> None:
        """Advance without saving (clears pending boxes)."""
        if self._pending_boxes:
            logger.warning(
                "Discarding %d unsaved box(es) for %s",
                len(self._pending_boxes),
                self.session.current_image_path.name,
            )
        self._pending_boxes = []
        self.session.go_next()
        self._refresh_display()

    def _on_previous(self, _button: object) -> None:
        """Go back to the previous image, loading its saved labels."""
        self._pending_boxes = []
        self.session.go_previous()
        self._pending_boxes = self.session.load_current_labels()
        self._refresh_display()

    def _on_skip(self, _button: object) -> None:
        """Skip the current image and advance."""
        self._pending_boxes = []
        self.session.skip()
        self._refresh_display()

    def _on_delete_last(self, _button: object) -> None:
        """Remove the last pending box."""
        if self._pending_boxes:
            removed = self._pending_boxes.pop()
            logger.debug("Removed pending box: %s", removed)
        self._refresh_display()

    def _on_mark_empty(self, _button: object) -> None:
        """Save an empty label file (no cats visible) and advance."""
        self._pending_boxes = []
        self.session.save_current_labels([])
        self.session.go_next()
        self._refresh_display()


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def get_class_distribution(labels_dir: Path) -> dict[str, int]:
    """Count class appearances across all label files in a directory.

    Args:
        labels_dir: Directory containing YOLO ``.txt`` label files.

    Returns:
        Dict mapping display names (``"Aïoli"``, ``"Mayo"``) to counts.
    """
    counts: dict[int, int] = {cid: 0 for cid in CLASS_NAMES}
    for lp in sorted(labels_dir.glob("*.txt")):
        for lb in load_labels(lp):
            cid = lb["class_id"]
            if cid in counts:
                counts[cid] += 1
    return {CLASS_NAMES[cid]: n for cid, n in counts.items()}
