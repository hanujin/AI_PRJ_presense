"""FastAPI service for PreSense multimodal stress inference."""

from __future__ import annotations

import io

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from inference import get_engine, health, predict

app = FastAPI(title="PreSense Multimodal Inference Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def warmup() -> None:
    try:
        get_engine()
    except Exception as error:
        # Keep /health available with the initialization diagnostic.
        print(f"[inference] startup failed: {error}")


@app.get("/health")
def get_health() -> dict:
    status = health()
    return {"status": "ok" if status["checkpointLoaded"] else "degraded", **status}


@app.post("/infer")
async def infer(
    frames: list[UploadFile] = File(...),
    audio: UploadFile = File(...),
    task: str = Form("Speaking"),
) -> dict:
    if len(frames) != 16:
        raise HTTPException(422, "Exactly 16 JPEG/PNG frames are required.")

    decoded_frames: list[np.ndarray] = []
    for frame in frames:
        payload = await frame.read()
        image = np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"))
        decoded_frames.append(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    audio_samples = np.frombuffer(await audio.read(), dtype=np.float32)
    if audio_samples.size == 0:
        raise HTTPException(422, "Audio samples are required.")
    # The browser sends 16 kHz Float32 PCM. Pad short warm-up windows for the model.
    target_samples = 160_000
    if audio_samples.size < target_samples:
        audio_samples = np.pad(audio_samples, (target_samples - audio_samples.size, 0))
    else:
        audio_samples = audio_samples[-target_samples:]

    try:
        return predict(np.stack(decoded_frames), audio_samples, task)
    except Exception as error:
        raise HTTPException(500, str(error)) from error
