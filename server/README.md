# PreSense Inference Server

Wraps the End-to-End **Student** model from [`models/models_e2e.py`](../models/models_e2e.py)
and exposes it to the frontend as a real-time stress/confidence endpoint
(Progress Report §1.2.2, §1.3.5, §3.3).

## What it does

- Loads `E2EStudentModel` (ResNet18 + Wav2Vec2 encoders + classifier head).
- Receives a webcam frame from the frontend, runs a real forward pass, and
  returns `{ stressScore, confidence }`.
- Audio is sent as a zero waveform for now (valid silent-sample fallback); the
  video path is fully real.

## ⚠️ About accuracy

The ResNet18 / Wav2Vec2 encoders use **pretrained** weights, so features are
real — but the **classifier head is untrained** until you provide a checkpoint.
Without one, `stressScore` is essentially arbitrary. This is expected: the repo
ships model *code*, not trained weights (`checkpoints/` is gitignored).

The server auto-loads `checkpoints/student_kd/student_kd_best.pt` if present (its
key structure — `video_enc` + `audio_enc` + `classifier`, no physio/LSTM — is
verified to match `E2EStudentModel`). Override with another file via:

```bash
export PRESENSE_CHECKPOINT=/path/to/student.pt   # Windows: $env:PRESENSE_CHECKPOINT="..."
```

Note: `hybrid_teacher_best.pt` is the Teacher (has physio + LSTM) — do **not**
use it here; it won't match the Student architecture.

The loader accepts either a raw `state_dict` or `{"model": state_dict}` and uses
`strict=False`, so partial checkpoints still load.

## Run

```bash
cd server
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

First start downloads ResNet18 (~45 MB) and Wav2Vec2-base (~360 MB).

## Connect the frontend

Create `frontend/.env.local`:

```
NEXT_PUBLIC_SIGNAL_SOURCE=live
NEXT_PUBLIC_INFERENCE_URL=http://localhost:8000
```

Restart `next dev`. Leave `NEXT_PUBLIC_SIGNAL_SOURCE` unset (or `simulated`) to
keep the built-in demo signal.

## Endpoints

| Method | Path      | Body                  | Returns                                   |
| ------ | --------- | --------------------- | ----------------------------------------- |
| GET    | `/health` | —                     | `{ status, checkpointLoaded }`            |
| POST   | `/infer`  | multipart `frame` img | `{ stressScore, confidence, checkpointLoaded }` |
