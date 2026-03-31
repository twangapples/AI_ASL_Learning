# AI ASL Learning — Real-Time ASL Recognizer

Real-time American Sign Language (ASL) hand gesture recognition using MediaPipe hand landmarks and a custom TFLite/Keras classifier. Supports two modes: a **local Python webcam app** and a **browser-based web app** (no install required).

---

## How It Works

1. **MediaPipe Hands** detects 21 hand landmarks from your webcam feed.
2. The 42 normalized (x, y) coordinates are fed into a lightweight neural network.
3. The model outputs one of 26 ASL letters (A–Z) with a confidence score.
4. Letters are "typed" after you hold a sign steady for ~8 frames.

**Model architecture:** `Input(42) → BatchNorm → Dense(128, Mish) → Dense(64, Mish) → Dense(32, Mish) → Dense(26, Softmax)`

---

## Option 1: Web App (No Install)

**[Try it live → aisignlanguagetutor.netlify.app](https://aisignlanguagetutor.netlify.app/)**

Or open `web/index.html` directly in your browser — no server needed.

```
open web/index.html
```

- Uses the exported `web/model/weights.bin` + `weights.json` for inference in pure JavaScript.
- Hand landmark detection runs via MediaPipe's browser SDK.

**Controls:**

| Key | Action |
|-----|--------|
| `S` | Add space |
| `B` | Backspace |
| `C` | Clear all text |

Buttons are also available on-screen.

---

## Option 2: Local Python Webcam App

### Prerequisites

- **Python 3.12** (TensorFlow/MediaPipe/OpenCV are not reliable on 3.13.)
- A webcam

### Setup

```bash
python3.12 -m venv my_venv
source my_venv/bin/activate        # Windows: my_venv\Scripts\activate
pip install -r requirements.txt
```

You also need the TFLite model and label CSV. Update the paths in `live_asl_webcam.py`:

```python
MODEL_PATH = Path("path/to/keypoint_classifier.tflite")
LABEL_PATH = Path("path/to/keypoint_classifier_label.csv")
```

### Run

```bash
python live_asl_webcam.py
```

A window opens showing your webcam feed with hand landmarks drawn. The predicted letter and confidence appear at the top; typed text accumulates at the bottom.

**Controls:**

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `C` | Clear typed text |
| `B` | Backspace |

---

## Exporting Model Weights for the Web App

If you retrain the Keras model, re-export weights for the browser:

```bash
pip install h5py numpy tensorflow
python convert_model.py
```

This writes `web/model/weights.bin` and `web/model/weights.json`.

---

## Project Structure

```
├── live_asl_webcam.py     # Python real-time webcam app
├── convert_model.py       # Export Keras weights → web/model/
├── requirements.txt       # Python dependencies
├── model/
│   └── keypoint_classifier/
│       ├── keypoint_classifier.keras
│       ├── keypoint_classifier.tflite
│       └── keypoint_classifier_label.csv
└── web/
    ├── index.html
    ├── app.js
    ├── style.css
    └── model/
        ├── weights.bin
        └── weights.json
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `mediapipe` | Hand landmark detection |
| `tensorflow` | TFLite inference + model export |
| `opencv-python` | Webcam capture & display |
| `numpy` | Landmark preprocessing |
