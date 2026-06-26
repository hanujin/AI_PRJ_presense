"""
PreSense inference server (FastAPI).

Endpoints
─────────
  GET  /health  → liveness + whether a trained checkpoint is loaded
  POST /infer   → multipart "frame" (JPEG/PNG) → { stressScore, confidence }

Run
───
  pip install -r requirements.txt
  uvicorn app:app --host 0.0.0.0 --port 8000

Point the frontend at it with:
  NEXT_PUBLIC_SIGNAL_SOURCE=live
  NEXT_PUBLIC_INFERENCE_URL=http://localhost:8000
"""

import io

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from inference import get_model, predict

app = FastAPI(title="PreSense Inference Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _warmup() -> None:
    # Build the model (and load the checkpoint) once at startup so the first
    # request is not penalized by ResNet/Wav2Vec2 download + init.
    get_model()


@app.get("/health")
def health() -> dict:
    from inference import _checkpoint_loaded
    return {"status": "ok", "checkpointLoaded": _checkpoint_loaded}


@app.post("/infer")
async def infer(frame: UploadFile = File(...)) -> dict:
    data = await frame.read()
    img = Image.open(io.BytesIO(data))
    return predict(img)
