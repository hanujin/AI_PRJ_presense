// Live inference signal — model integration boundary (Progress Report §1.3.5).
//
// The UI depends ONLY on the LiveInferenceFrame shape below, never on the model
// internals. Today the values come from a local simulator; once the backend
// Student model is available it will stream the same shape over WebSocket and
// only this module needs to change (Progress Report §2.3, §3.3).

export type LiveInferenceFrame = {
  /** Normalized stress level, 0..1 (stress-class probability). */
  stressScore: number;
  /** Max class probability of the binary softmax output, 0.5..1 (Progress Report §2.3). */
  confidence: number;
  timestampMs: number;
};

export type SignalSource = "simulated" | "live";

// "live" routes frames to the Python inference server (server/); anything else
// uses the built-in simulator. Controlled by env so no code change is needed.
export const SIGNAL_SOURCE: SignalSource =
  process.env.NEXT_PUBLIC_SIGNAL_SOURCE === "live" ? "live" : "simulated";

export const INFERENCE_URL =
  process.env.NEXT_PUBLIC_INFERENCE_URL ?? "http://localhost:8000";

// POST a single webcam frame to the inference server and get a real model frame.
export async function inferFrame(imageBlob: Blob): Promise<LiveInferenceFrame> {
  const form = new FormData();
  form.append("frame", imageBlob, "frame.jpg");
  const res = await fetch(`${INFERENCE_URL}/infer`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`Inference failed: ${res.status}`);
  const data = await res.json();
  return {
    stressScore: data.stressScore,
    confidence: data.confidence,
    timestampMs: Date.now(),
  };
}

export type LiveSignalController = {
  /** Produce the next inference frame. */
  next: () => LiveInferenceFrame;
  /** Reset internal state at the start of a session. */
  reset: () => void;
};

const START_STRESS = 0.38;

// Random-walk simulator standing in for the trained Student model output.
// This is the ONLY place that fabricates numbers; replace with the real
// backend stream when integrating the model.
export function createSimulatedSignal(): LiveSignalController {
  let stress = START_STRESS;
  return {
    reset() {
      stress = START_STRESS;
    },
    next() {
      stress = Math.max(0.16, Math.min(0.92, stress + (Math.random() - 0.48) * 0.04));
      // Confidence ≈ distance from the 0.5 decision boundary, mapped into 0.6..0.97.
      const confidence = Math.min(0.97, 0.6 + Math.abs(stress - 0.5) * 0.8);
      return { stressScore: stress, confidence, timestampMs: Date.now() };
    },
  };
}
