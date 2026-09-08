"""Inference adapter for the current PreSense video-and-audio model."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
# Keep the provider key on the server.  This file is ignored by Git and must
# never be exposed to the browser or prefixed with NEXT_PUBLIC_.
load_dotenv(REPO_ROOT / "server" / ".env")
for directory in (REPO_ROOT, REPO_ROOT / "agent"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from model_wrapper import StressInferenceEngine  # noqa: E402

_engine: StressInferenceEngine | None = None
_load_error = ""
_lock = threading.Lock()
_predict_lock = threading.Lock()


def get_engine(task_name: str = "Speaking") -> StressInferenceEngine:
    """Lazily load the Student-CBM model and its feature extractors."""
    global _engine, _load_error
    if _engine is not None:
        return _engine

    with _lock:
        if _engine is None:
            try:
                engine = StressInferenceEngine(task_name=task_name)
                if engine.model is None:
                    raise RuntimeError("Student-CBM weights could not be loaded.")
                _engine = engine
                _load_error = ""
            except Exception as error:
                _load_error = str(error)
                raise
    return _engine


def health() -> dict:
    return {
        "checkpointLoaded": _engine is not None and _engine.model is not None,
        "error": _load_error or None,
        "model": "student_kd + CBM residual",
    }


def predict(video_frames: np.ndarray, audio_samples: np.ndarray, task_name: str = "Speaking") -> dict:
    """Analyze a 16-frame BGR sequence and 16 kHz float32 audio buffer."""
    if video_frames.shape[0] != 16:
        raise ValueError("Exactly 16 video frames are required.")

    engine = get_engine(task_name)
    # Share model weights, not session histories. Protect task selection and
    # stateful feature extractors from concurrent requests.
    with _predict_lock:
        engine.set_task(task_name)
        stress, gate_weights, dominant, audio_features, video_features, concepts = engine.predict(
            video_frames, audio_samples, record_session=False,
        )

    return {
        "stressScore": float(stress),
        "confidence": float(max(stress, 1 - stress)),
        "checkpointLoaded": True,
        "gateWeights": gate_weights,
        "dominantModality": dominant,
        "concepts": concepts,
        "audioFeatures": audio_features,
        "videoFeatures": video_features,
    }
