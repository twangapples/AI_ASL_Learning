"""
ASL web app: serves the UI and a /predict API for hand landmark → letter inference.
Inference runs in-process in a single thread to avoid macOS multiprocessing mutex crashes.
"""
# Must be set before any protobuf/TF/MediaPipe import to avoid macOS "mutex lock failed" crash
import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("GLOG_minloglevel", "2")          # suppress MediaPipe C++ INFO/WARNING logs

import asyncio
import base64
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException  # pyright: ignore[reportMissingImports]
from fastapi.responses import HTMLResponse, FileResponse  # pyright: ignore[reportMissingImports]
from fastapi.staticfiles import StaticFiles  # pyright: ignore[reportMissingImports]
from pydantic import BaseModel  # pyright: ignore[reportMissingImports]

REPO_ROOT = Path(__file__).resolve().parent
MODEL_PATH = REPO_ROOT / "model" / "keypoint_classifier" / "keypoint_classifier.tflite"
LABEL_PATH = REPO_ROOT / "model" / "keypoint_classifier" / "keypoint_classifier_label.csv"

app = FastAPI(title="ASL Recognition")

# In-process inference: single thread to avoid blocking and to avoid macOS spawn mutex crash
_inference_executor = ThreadPoolExecutor(max_workers=1)
_interpreter = None
_input_details = None
_output_details = None
_labels = None
_models_lock = Lock()


def _load_model():
    """Load TFLite model and labels. Call with _models_lock held. Runs in inference thread."""
    global _interpreter, _input_details, _output_details, _labels
    try:
        from ai_edge_litert.interpreter import Interpreter  # pyright: ignore[reportMissingImports]
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter  # pyright: ignore[reportMissingImports]
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter
    import numpy as np
    _interpreter = Interpreter(model_path=str(MODEL_PATH))
    _interpreter.allocate_tensors()
    _input_details = _interpreter.get_input_details()
    _output_details = _interpreter.get_output_details()
    with open(LABEL_PATH, encoding="utf-8-sig") as f:
        _labels = [row[0] for row in csv.reader(f)]
    return np


def _run_inference_sync(image_base64: str) -> dict:
    """
    Run MediaPipe + TFLite in the main process (in the single inference thread).
    Returns {"letter", "confidence", "message"} or {"error": "..."}.
    """
    try:
        return _run_inference_sync_impl(image_base64)
    except Exception as e:
        return {"letter": None, "confidence": 0.0, "message": None, "error": str(e)}


def _run_inference_sync_impl(image_base64: str) -> dict:
    import cv2
    import numpy as np
    import mediapipe as mp
    global _interpreter, _input_details, _output_details, _labels
    with _models_lock:
        if _interpreter is None:
            np = _load_model()
        else:
            import numpy as np

    try:
        raw = base64.b64decode(image_base64)
        buf = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            return {"letter": None, "confidence": 0.0, "message": "Invalid image data"}
    except Exception as e:
        return {"letter": None, "confidence": 0.0, "message": str(e)}

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False

    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        results = hands.process(frame_rgb)

    if not results.multi_hand_landmarks:
        return {"letter": None, "confidence": 0.0, "message": "No hand detected"}

    hand_landmarks = results.multi_hand_landmarks[0]
    image_width, image_height = frame.shape[1], frame.shape[0]
    landmark_list = []
    for lm in hand_landmarks.landmark:
        x = min(int(lm.x * image_width), image_width - 1)
        y = min(int(lm.y * image_height), image_height - 1)
        landmark_list.append([x, y])

    temp = [list(p) for p in landmark_list]
    base_x, base_y = temp[0]
    for point in temp:
        point[0] -= base_x
        point[1] -= base_y
    flattened = np.array(temp, dtype=np.float32).flatten()
    max_value = np.max(np.abs(flattened))
    if max_value == 0:
        processed = flattened.tolist()
    else:
        processed = (flattened / max_value).tolist()
    if not processed:
        return {"letter": None, "confidence": 0.0, "message": "Could not process landmarks"}

    input_tensor = np.array([processed], dtype=np.float32)
    with _models_lock:
        _interpreter.set_tensor(_input_details[0]["index"], input_tensor)
        _interpreter.invoke()
        preds = _interpreter.get_tensor(_output_details[0]["index"])[0]
    top_idx = int(np.argmax(preds))
    confidence = float(np.max(preds))
    letter = _labels[top_idx]
    return {"letter": letter, "confidence": confidence, "message": None}


class PredictRequest(BaseModel):
    image_base64: str


@app.post("/predict")
async def predict(req: PredictRequest):
    """Run MediaPipe + TFLite in a single thread (in-process) to avoid macOS spawn crash."""
    import logging
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            _inference_executor,
            _run_inference_sync,
            req.image_base64,
        )
    except Exception as e:
        logging.exception("Predict failed")
        raise HTTPException(status_code=500, detail=str(e))
    if result.get("error"):
        logging.error("Predict error: %s", result["error"])
        raise HTTPException(status_code=500, detail=result["error"])
    if result.get("message") == "Invalid image data":
        raise HTTPException(status_code=400, detail="Invalid image data")
    return {k: v for k, v in result.items() if k in ("letter", "confidence", "message")}


# Serve static assets if present
static_dir = REPO_ROOT / "static"
if static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    html = (REPO_ROOT / "static" / "index.html")
    if html.is_file():
        return FileResponse(html)
    return HTMLResponse(
        "<!DOCTYPE html><html><body><h1>ASL Recognition</h1><p>Add static/index.html for the web UI.</p></body></html>"
    )
