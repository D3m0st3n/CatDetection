# AGENTS.md — `app/`

## Purpose

The Gradio inference app. Provides a browser-based local GUI for running the trained model on new photos.

## File

### `gui.py`
Launches a Gradio interface that accepts an image upload, runs `src/infer.py`, and displays the result with bounding boxes overlaid and a plain-English detection summary.

**Behaviour by detection count**:
- 0 detections → "Neither Aïoli nor Mayo was detected."
- Aïoli only → "Aïoli detected (confidence: X%)."
- Mayo only → "Mayo detected (confidence: X%)."
- Both → "Both Aïoli and Mayo detected."

**UI features**:
- Drag-and-drop or file picker image upload.
- Confidence threshold slider (default: 0.5).
- Save-result button to export the annotated image to disk.
- Bounding boxes use consistent per-cat colours: orange for Aïoli (`#FF6B35`), teal for Mayo (`#4ECDC4`).

## Running the App

```bash
python cli.py app
# or directly:
python app/gui.py

# optional flags:
python app/gui.py --weights runs/detect/train2/weights/best.pt  # custom checkpoint
python app/gui.py --port 8080                                    # custom port (default: 7860)
```

The app runs locally and opens in your default browser. It does not require an internet connection.

**Note**: `gui.py` uses `sys.path.insert` to add the project root to `sys.path` so that `src.*` imports work when the script is run directly from any working directory. See decision #35 in `design_decisions.md`.

## Note

The Gradio app is not unit-tested. Manual testing is sufficient — launch the app, upload a photo, and verify the output visually.
