"""
PreSense real-time inference (Progress Report §1.2.2 / §1.3.5).

Loads the End-to-End Student model defined in models/models_e2e.py and turns a
webcam frame (+ optional audio) into a normalized stress score and confidence.

The encoders use pretrained ImageNet / Wav2Vec2 weights, so the forward pass is
real. The classifier head is only meaningful once a trained checkpoint is loaded
via the PRESENSE_CHECKPOINT env var — until then output is effectively untrained.
"""

from __future__ import annotations

import os
import sys
import threading

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Make the repository root importable so `models` resolves.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)
from models.models_e2e import E2EStudentModel  # noqa: E402

# Verified to match E2EStudentModel keys (video_enc + audio_enc + classifier).
DEFAULT_CHECKPOINT = os.path.join(REPO_ROOT, "checkpoints", "student_kd", "student_kd_best.pt")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
NUM_FRAMES = 16          # sliding buffer length (Progress Report §1.2.1)
AUDIO_SAMPLES = 16000    # 1 s @ 16 kHz placeholder when no audio is streamed

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_checkpoint_loaded = False
_lock = threading.Lock()


def get_model() -> E2EStudentModel:
    """Lazily build the Student model and load a checkpoint if one is provided."""
    global _model, _checkpoint_loaded
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
        model = E2EStudentModel(num_classes=2)
        ckpt = os.environ.get("PRESENSE_CHECKPOINT")
        if not ckpt and os.path.exists(DEFAULT_CHECKPOINT):
            ckpt = DEFAULT_CHECKPOINT
        if ckpt and os.path.exists(ckpt):
            state = torch.load(ckpt, map_location=_device)
            # Accept raw state_dict, {"model": ...}, or {"model_state_dict": ...}
            # (the repo's train/load_checkpoint uses the latter, see models/models.py).
            if isinstance(state, dict):
                state = state.get("model_state_dict", state.get("model", state))
            missing, unexpected = model.load_state_dict(state, strict=False)
            _checkpoint_loaded = True
            print(f"[inference] checkpoint loaded: {ckpt} "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")
        else:
            print("[inference] WARNING: no checkpoint — classifier head is UNTRAINED. "
                  "Set PRESENSE_CHECKPOINT to a trained .pt to get real predictions.")
        model.to(_device).eval()
        _model = model
        return model


def _preprocess_frame(img: Image.Image) -> torch.Tensor:
    """PIL image -> (1, T, 3, 224, 224) ImageNet-normalized tensor."""
    img = img.convert("RGB").resize((224, 224))
    arr = np.asarray(img, dtype=np.float32) / 255.0          # (H, W, 3)
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    t = torch.from_numpy(arr).permute(2, 0, 1)               # (3, 224, 224)
    # The frozen ResNet averages over T, so replicating one frame is a valid
    # single-shot approximation of the 16-frame buffer.
    frames = t.unsqueeze(0).repeat(NUM_FRAMES, 1, 1, 1).unsqueeze(0)
    return frames.float()


@torch.no_grad()
def predict(img: Image.Image, audio: np.ndarray | None = None) -> dict:
    """Run one forward pass and return stress score + confidence."""
    model = get_model()
    frames = _preprocess_frame(img).to(_device)

    if audio is None:
        # Zero-audio fallback (Progress Report uses a has_audio mask for silent samples).
        audio_t = torch.zeros(1, AUDIO_SAMPLES, device=_device)
    else:
        audio_t = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).to(_device)

    logits, _, _ = model(frames, audio_t)
    probs = F.softmax(logits, dim=1)[0]
    stress = float(probs[1].item())            # P(stress)
    confidence = float(probs.max().item())     # max class probability (§2.3)
    return {
        "stressScore": stress,
        "confidence": confidence,
        "checkpointLoaded": _checkpoint_loaded,
    }
