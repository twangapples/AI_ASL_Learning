# AI_ASL_Learning
Deep learning model for recognizing American Sign Language gestures using a CNN.

This project uses a Convolutional Neural Network (CNN) to recognize American Sign Language (ASL) hand gestures from 28×28 grayscale images. The model was trained in Google Colab using TensorFlow/Keras, saved as a .keras file, and integrated into a local Python environment for real-time predictions and image visualization.

## Requirements

- **Python 3.12** (TensorFlow/MediaPipe/OpenCV are not reliable on 3.13.)

Create a venv with 3.12 and install deps:

```bash
python3.12 -m venv my_venv
source my_venv/bin/activate   # Windows: my_venv\Scripts\activate
pip install -r requirements.txt
```

## Local run

**Desktop (OpenCV window):**
```bash
python live_asl_webcam.py
```

**Web app (browser):**
```bash
uvicorn app:app --reload --port 8000 --reload-exclude 'my_venv'
```
Then open http://localhost:8000 and allow camera access.  
*(Excluding `my_venv` prevents the reloader from watching the venv and triggering TensorFlow mutex crashes. If you still see crashes, run without `--reload`.)*

## Deploy

- **Railway / Render:** Connect your GitHub repo. Use the Procfile; set start command to `uvicorn app:app --host 0.0.0.0 --port $PORT` if needed.
- **Docker:** `docker build -t asl-app . && docker run -p 8000:8000 asl-app`
