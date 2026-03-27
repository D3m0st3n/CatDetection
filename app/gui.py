"""Gradio inference app for the CatDetection pipeline.

Loads a trained YOLOv8 checkpoint and serves a browser-based local GUI
for running the model on new photos. Displays bounding boxes, confidence
scores, and a plain-English detection summary.

Usage::

    python app/gui.py                          # auto-detect weights
    python app/gui.py --weights runs/.../best.pt
    python app/gui.py --port 8080
"""

import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path when running as `python app/gui.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO

from src.infer import Detection, draw_predictions, load_model, predict_image
from src.train import find_latest_run_dir
from src.utils import CLASS_NAMES, CONFIDENCE_THRESHOLD, setup_logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_default_weights(project: Path = Path("runs/detect")) -> Path:
    """Find ``best.pt`` from the most recent training run.

    Args:
        project: Parent directory containing YOLOv8 run folders
            (e.g. ``runs/detect``).

    Returns:
        Path to ``best.pt`` inside the latest ``train*`` subdirectory.

    Raises:
        FileNotFoundError: If no training run exists or ``best.pt`` is absent.
    """
    latest_dir = find_latest_run_dir(project)
    weights = latest_dir / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"No best.pt found in {latest_dir}")
    logger.info("Using weights: %s", weights)
    return weights


def build_summary(detections: list[Detection]) -> str:
    """Return a plain-English description of what was detected.

    Args:
        detections: List of ``Detection`` objects from inference.

    Returns:
        Human-readable detection summary string.
    """
    detected_names = {CLASS_NAMES[d.class_id]: d.confidence for d in detections}

    if not detected_names:
        return "Neither Aïoli nor Mayo was detected."

    has_aioli = "Aïoli" in detected_names
    has_mayo = "Mayo" in detected_names

    if has_aioli and has_mayo:
        return "Both Aïoli and Mayo detected."
    if has_aioli:
        conf = detected_names["Aïoli"]
        return f"Aïoli detected (confidence: {conf:.0%})."
    # Mayo only
    conf = detected_names["Mayo"]
    return f"Mayo detected (confidence: {conf:.0%})."


# ---------------------------------------------------------------------------
# Gradio interface builder
# ---------------------------------------------------------------------------


def build_demo(model: YOLO) -> gr.Blocks:
    """Construct and return the Gradio Blocks interface.

    The model is captured in the closure so it is loaded once at startup
    and reused for every request.

    Args:
        model: A loaded YOLO model ready for inference.

    Returns:
        A configured ``gr.Blocks`` instance (not yet launched).
    """

    def run_inference(
        image_path: str | None,
        conf_threshold: float,
    ) -> tuple[np.ndarray | None, str, np.ndarray | None, str | None]:
        """Run inference and return display image, summary, and save state.

        Args:
            image_path: Filepath of the uploaded image (from Gradio).
            conf_threshold: Confidence threshold for detections.

        Returns:
            Tuple of:
            - RGB numpy array for display (or None if no image)
            - Detection summary string
            - BGR numpy array for saving (or None if no image)
            - Original image path string (or None if no image)
        """
        if image_path is None:
            return None, "", None, None

        # Load with EXIF correction so displayed orientation matches YOLO's.
        pil_img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        img_rgb = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]

        result = predict_image(model, Path(image_path), confidence=conf_threshold)

        annotated_bgr = draw_predictions(img_bgr, result.detections, w, h)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        summary = build_summary(result.detections)

        return annotated_rgb, summary, annotated_bgr, image_path

    def save_result(
        annotated_bgr: np.ndarray | None,
        original_path: str | None,
    ) -> str:
        """Save the annotated image to the ``outputs/`` directory.

        Args:
            annotated_bgr: BGR numpy array of the annotated image.
            original_path: Path of the original uploaded image (used for
                deriving the output filename).

        Returns:
            Status message indicating where the file was saved, or an
            error message if no result is available.
        """
        if annotated_bgr is None:
            return "No result to save — upload an image first."

        outputs_dir = Path("outputs")
        outputs_dir.mkdir(exist_ok=True)

        stem = Path(original_path).stem if original_path else "result"
        out_path = outputs_dir / f"{stem}_detected.jpg"
        cv2.imwrite(str(out_path), annotated_bgr)
        logger.info("Saved annotated image to %s", out_path)
        return f"Saved to: {out_path}"

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    with gr.Blocks(title="Cat Detection") as demo:
        gr.Markdown("# Cat Detection\nIdentify **Aïoli** and **Mayo** in your photos.")

        annotated_state = gr.State(None)   # BGR array for save
        input_path_state = gr.State(None)  # original filepath for filename

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Upload photo", type="filepath")
                conf_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=CONFIDENCE_THRESHOLD,
                    step=0.05,
                    label="Confidence threshold",
                )
            with gr.Column():
                output_image = gr.Image(label="Detection result")

        summary_text = gr.Textbox(label="Detection summary", interactive=False)

        with gr.Row():
            save_btn = gr.Button("Save result")
            save_status = gr.Textbox(
                label="",
                interactive=False,
                show_label=False,
                placeholder="Click 'Save result' to export the annotated image.",
            )

        # ------------------------------------------------------------------
        # Event wiring
        # ------------------------------------------------------------------

        _infer_outputs = [output_image, summary_text, annotated_state, input_path_state]

        input_image.change(
            fn=run_inference,
            inputs=[input_image, conf_slider],
            outputs=_infer_outputs,
        )
        conf_slider.change(
            fn=run_inference,
            inputs=[input_image, conf_slider],
            outputs=_infer_outputs,
        )
        save_btn.click(
            fn=save_result,
            inputs=[annotated_state, input_path_state],
            outputs=[save_status],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cat Detection — Gradio GUI")
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to model weights .pt file (default: auto-detect from runs/)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to serve the app on (default: 7860)",
    )
    args = parser.parse_args()

    setup_logging()
    weights_path = args.weights if args.weights is not None else find_default_weights()
    model = load_model(weights_path)
    build_demo(model).launch(server_port=args.port)
