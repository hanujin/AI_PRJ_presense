"use client";

import { ChangeEvent, useEffect, useRef, useState } from "react";
import {
  BrainCircuit,
  Camera,
  ChevronLeft,
  ChevronRight,
  CircleStop,
  FileText,
  Gauge,
  MessageSquareText,
  Mic,
  Play,
  Sparkles,
  Upload,
  Users,
  Waves,
  X,
} from "lucide-react";
import { useSupabaseAuth } from "@/components/supabase-provider";
import { agentStarters, initialTimeline, postSessionHighlights, practiceScenes, samplePresentation } from "@/lib/dashboard-data";
import { createSimulatedSignal, inferMultimodalFrame, SIGNAL_SOURCE, type LiveInferenceFrame, type LiveSignalController } from "@/lib/live-signal";
import { useLang } from "@/lib/i18n";
import { savePracticeSession } from "@/lib/supabase/data";
import { StressChart } from "@/components/stress-chart";

type SessionState = "idle" | "starting" | "live" | "ended" | "error";
type ChatMessage = { role: "agent" | "user"; text: string };
type ReportSummary = {
  startBaseline: number;
  peakValue: number;
  peakBin: number; // 0-based
  peakPhase: string; // Open/Build/Core/Close
  mostStressfulPhase: string;
  averageStress: number;
  confidence: number; // 0..1, 0 when none
  noticeable: boolean;
  deckName: string;
};
type SceneId = (typeof practiceScenes)[number]["id"];
type DeckFile = {
  name: string;
  type: "pdf" | "ppt" | "pptx" | "presentation" | "unsupported";
  url: string;
  source: "uploaded" | "sample";
};

const SAMPLE_DECK: DeckFile = {
  name: samplePresentation.name,
  type: "presentation",
  url: "",
  source: "sample",
};

function formatSeconds(s: number) {
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}

function getDeckType(fileName: string): DeckFile["type"] {
  const lower = fileName.toLowerCase();
  if (lower.endsWith(".pdf")) return "pdf";
  if (lower.endsWith(".ppt")) return "ppt";
  if (lower.endsWith(".pptx")) return "pptx";
  return "unsupported";
}

function buildAgentReply(input: string, t: (k: string) => string, peakText: string) {
  const s = input.toLowerCase();
  if (s.includes("stress") || s.includes("tense") || input.includes("스트레스") || input.includes("긴장")) {
    return t("agent.stressPre") + peakText + t("agent.stressPost");
  }
  if (s.includes("calm") || s.includes("confidence") || input.includes("차분") || input.includes("자신")) {
    return t("agent.calm");
  }
  if (s.includes("improve") || s.includes("next") || input.includes("개선") || input.includes("다음")) {
    return t("report.nextActionText");
  }
  return t("agent.default");
}

const PHASES = ["Open", "Build", "Core", "Close"] as const;

// The 12-bin timeline is 4 phases × 3 bins (Progress Report §1.3.3, §2.3).
function summarizeTimeline(bins: number[]) {
  const phaseAvgs = PHASES.map((_, p) => {
    const slice = bins.slice(p * 3, p * 3 + 3);
    return slice.reduce((s, v) => s + v, 0) / Math.max(slice.length, 1);
  });
  const peakValue = Math.max(...bins);
  const peakBin = bins.indexOf(peakValue);
  const peakPhase = PHASES[Math.min(PHASES.length - 1, Math.floor(peakBin / 3))];
  const mostStressfulPhase = PHASES[phaseAvgs.indexOf(Math.max(...phaseAvgs))];
  return { peakValue, peakBin, peakPhase, mostStressfulPhase };
}

function buildSessionTimeline(history: number[]) {
  if (history.length === 0) return initialTimeline;
  return Array.from({ length: 12 }, (_, bin) => {
    const start = Math.floor((bin / 12) * history.length);
    const end = Math.max(start + 1, Math.floor(((bin + 1) / 12) * history.length));
    const values = history.slice(start, end);
    return Math.round((values.reduce((sum, value) => sum + value, 0) / values.length) * 100);
  });
}

function PresentationDeck({
  index,
  onPrev,
  onNext,
}: {
  index: number;
  onPrev: () => void;
  onNext: () => void;
}) {
  const { t } = useLang();
  const slides = samplePresentation.slides;
  const slide = slides[index] ?? slides[0];
  const atStart = index <= 0;
  const atEnd = index >= slides.length - 1;

  return (
    <div className="slide-deck">
      <div className="slide-hint">{t("slide.hint")}</div>
      <div className="slide" data-kind={slide.kind ?? "content"}>
        {slide.kind === "title" ? (
          <div className="slide-title-block">
            <span className="slide-eyebrow">{slide.eyebrow}</span>
            <h2>{slide.title}</h2>
            <p>{slide.subtitle}</p>
          </div>
        ) : (
          <>
            <h3 className="slide-heading">{slide.title}</h3>
            <ul className="slide-bullets">
              {slide.bullets.map((b) => (
                <li key={b}>{b}</li>
              ))}
            </ul>
          </>
        )}
      </div>
      <div className="slide-controls">
        <button type="button" onClick={onPrev} disabled={atStart} aria-label="Previous slide">
          <ChevronLeft size={18} />
        </button>
        <span className="slide-counter">{index + 1} / {slides.length}</span>
        <button type="button" onClick={onNext} disabled={atEnd} aria-label="Next slide">
          <ChevronRight size={18} />
        </button>
      </div>
    </div>
  );
}

function DeckPreview({
  deck,
  slideIndex,
  onPrev,
  onNext,
}: {
  deck: DeckFile | null;
  slideIndex: number;
  onPrev: () => void;
  onNext: () => void;
}) {
  const { t } = useLang();
  if (!deck) {
    return (
      <div className="deck-empty">
        <FileText size={42} />
        <strong>{t("deck.addFile")}</strong>
        <span>{t("deck.addFileDesc")}</span>
      </div>
    );
  }

  if (deck.type === "presentation") {
    return <PresentationDeck index={slideIndex} onPrev={onPrev} onNext={onNext} />;
  }

  if (deck.type !== "pdf") {
    return (
      <div className="deck-empty">
        <FileText size={42} />
        <strong>{deck.name}</strong>
        <span>{t("deck.pptDesc")}</span>
      </div>
    );
  }

  return (
    <object className="deck-frame" data={deck.url} type="application/pdf">
      <iframe className="deck-frame" src={deck.url} title={deck.name} />
    </object>
  );
}

export function DashboardShell() {
  const { hasEnv, user } = useSupabaseAuth();
  const { lang, t } = useLang();
  const mainVideoRef = useRef<HTMLVideoElement | null>(null);
  const pipVideoRef = useRef<HTMLVideoElement | null>(null);
  const preparationVideoRef = useRef<HTMLVideoElement | null>(null);
  const preparationDialogRef = useRef<HTMLDialogElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const audioProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const audioChunksRef = useRef<Float32Array[]>([]);
  const audioSampleRateRef = useRef(48_000);
  const frameBufferRef = useRef<Blob[]>([]);
  const collectionTickRef = useRef<number | null>(null);
  const collectionBusyRef = useRef(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const sessionVideoRef = useRef<HTMLVideoElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const tickRef = useRef<number | null>(null);
  const deckObjectUrlRef = useRef<string | null>(null);
  const stressHistoryRef = useRef<number[]>([]);
  const signalRef = useRef<LiveSignalController | null>(null);
  const inferInFlightRef = useRef(false);
  const saveCompleteDialogRef = useRef<HTMLDialogElement | null>(null);
  const savingRef = useRef(false);
  const analysisGenerationRef = useRef(0);
  const inferenceAbortRef = useRef<AbortController | null>(null);
  const recordingActiveRef = useRef(false);
  const preparationFrameRef = useRef<LiveInferenceFrame | null>(null);
  const startBaselineRef = useRef(0);

  const [sessionState, setSessionState] = useState<SessionState>("idle");
  const [errorMessage, setErrorMessage] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const [audioLevel, setAudioLevel] = useState(0);
  const [stressScore, setStressScore] = useState(0.38);
  const [confidence, setConfidence] = useState(0);
  const [modelConnected, setModelConnected] = useState(false);
  const [liveStatus, setLiveStatus] = useState<"idle" | "collecting" | "camera" | "audio" | "analyzing" | "ok" | "error">("idle");
  const [collectedFrames, setCollectedFrames] = useState(0);
  const [audioRunning, setAudioRunning] = useState(false);
  const [preparationComplete, setPreparationComplete] = useState(false);
  const [preparationPhase, setPreparationPhase] = useState<"collecting" | "analyzing" | "ready" | "error">("collecting");
  const [audioSeconds, setAudioSeconds] = useState(0);
  const [voiceDetected, setVoiceDetected] = useState(false);
  const [timeline, setTimeline] = useState(initialTimeline);
  const [report, setReport] = useState<ReportSummary | null>(null);
  const [chatInput, setChatInput] = useState("");
  const [selectedScene, setSelectedScene] = useState<SceneId>("camera");
  const [deck, setDeck] = useState<DeckFile | null>(SAMPLE_DECK);
  const [slideIndex, setSlideIndex] = useState(0);
  const [cameraReady, setCameraReady] = useState(false);
  const [saveStatus, setSaveStatus] = useState("");
  const [sessionTitle, setSessionTitle] = useState("");
  const [saved, setSaved] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  const [saveVideo, setSaveVideo] = useState(true);
  const [recordedVideo, setRecordedVideo] = useState<Blob | null>(null);
  const [recordedVideoUrl, setRecordedVideoUrl] = useState("");
  const [recordingFinalizing, setRecordingFinalizing] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    { role: "agent", text: t("report.chatIntro") },
  ]);

  useEffect(() => {
    if (!preparationComplete) preparationDialogRef.current?.showModal();
  }, [preparationComplete]);

  useEffect(() => {
    if (sessionState === "live" || sessionState === "starting" || preparationPhase === "error") return;
    if (collectedFrames < 16 || audioSeconds < 10 || !voiceDetected) return;
    void requestLiveFrame(true);
    const timer = window.setInterval(() => void requestLiveFrame(true), 1000);
    return () => window.clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionState, preparationPhase, collectedFrames, audioSeconds, voiceDetected]);

  useEffect(() => {
    let cancelled = false;
    void startPreview(() => cancelled);
    return () => {
      cancelled = true;
      cleanupMedia();
      cleanupDeckUrl();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (streamRef.current) attachStreamToVideos(streamRef.current);
  }, [selectedScene]);

  // Arrow-key navigation for the sample presentation while the deck is on stage.
  useEffect(() => {
    const deckVisible =
      (selectedScene === "slides" || selectedScene === "classroom-slides") &&
      deck?.type === "presentation";
    if (!deckVisible) return;

    function onKey(event: KeyboardEvent) {
      const tag = (event.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      if (event.key === "ArrowRight") {
        event.preventDefault();
        goToNextSlide();
      } else if (event.key === "ArrowLeft") {
        event.preventDefault();
        goToPrevSlide();
      }
    }

    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedScene, deck]);

  function goToNextSlide() {
    setSlideIndex((i) => Math.min(samplePresentation.slides.length - 1, i + 1));
  }

  function goToPrevSlide() {
    setSlideIndex((i) => Math.max(0, i - 1));
  }

  function cleanupDeckUrl() {
    if (deckObjectUrlRef.current) {
      URL.revokeObjectURL(deckObjectUrlRef.current);
      deckObjectUrlRef.current = null;
    }
  }

  function attachStreamToVideos(stream: MediaStream) {
    [mainVideoRef.current, pipVideoRef.current, preparationVideoRef.current].forEach((video) => {
      if (!video) return;
      if (video.srcObject !== stream) video.srcObject = stream;
      void video.play().catch(() => undefined);
    });
  }

  function startSessionRecording(stream: MediaStream) {
    if (typeof MediaRecorder === "undefined") return;
    const chunks: Blob[] = [];
    const mimeType = ["video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm", "video/mp4"].find((type) => MediaRecorder.isTypeSupported(type));
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunks.push(event.data);
    };
    recorder.onstop = () => {
      const video = new Blob(chunks, { type: recorder.mimeType || "video/webm" });
      chunks.length = 0;
      setRecordedVideo(video);
      setRecordedVideoUrl(URL.createObjectURL(video));
      setRecordingFinalizing(false);
    };
    mediaRecorderRef.current = recorder;
    recorder.start(1000);
  }

  function stopSessionRecording() {
    const recorder = mediaRecorderRef.current;
    if (recorder?.state === "recording") {
      setRecordingFinalizing(true);
      recorder.stop();
    }
    mediaRecorderRef.current = null;
  }

  function stopAnalysis() {
    recordingActiveRef.current = false;
    preparationFrameRef.current = null;
    analysisGenerationRef.current += 1;
    inferenceAbortRef.current?.abort();
    inferenceAbortRef.current = null;
    inferInFlightRef.current = false;
    audioChunksRef.current = [];
    frameBufferRef.current = [];
    setCollectedFrames(0);
    setAudioSeconds(0);
    setVoiceDetected(false);
    if (collectionTickRef.current !== null) {
      window.clearInterval(collectionTickRef.current);
      collectionTickRef.current = null;
    }
    collectionBusyRef.current = false;
    if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null; }
    if (tickRef.current) { window.clearInterval(tickRef.current); tickRef.current = null; }
    if (audioContextRef.current) {
      audioContextRef.current.onstatechange = null;
      if (audioProcessorRef.current) audioProcessorRef.current.onaudioprocess = null;
      void audioContextRef.current.close();
      audioContextRef.current = null;
    }
    setAudioRunning(false);
    analyserRef.current = null;
    audioProcessorRef.current = null;
    setAudioLevel(0);
  }

  function cleanupMedia() {
    stopSessionRecording();
    stopAnalysis();
    if (streamRef.current) { streamRef.current.getTracks().forEach((t) => t.stop()); streamRef.current = null; }
    [mainVideoRef.current, pipVideoRef.current, preparationVideoRef.current].forEach((video) => {
      if (video) video.srcObject = null;
    });
    setCameraReady(false);
  }

  // Acquire the camera while idle so the user can check themselves before starting.
  async function startPreview(cancelled: () => boolean = () => false) {
    if (streamRef.current) {
      attachStreamToVideos(streamRef.current);
      startCollection(streamRef.current);
      void audioContextRef.current?.resume().catch(() => setAudioRunning(false));
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: true });
      if (cancelled()) { stream.getTracks().forEach((track) => track.stop()); return; }
      streamRef.current = stream;
      setCameraReady(true);
      setErrorMessage("");
      attachStreamToVideos(stream);
      startCollection(stream);
    } catch {
      if (cancelled()) return;
      setCameraReady(false);
      setErrorMessage("Camera permission was denied. Allow access to preview yourself before starting.");
    }
  }

  function startAudioMeter(stream: MediaStream) {
    if (audioContextRef.current) return;
    const ctx = new window.AudioContext();
    ctx.onstatechange = () => setAudioRunning(ctx.state === "running");
    setAudioRunning(ctx.state === "running");
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 512;
    const source = ctx.createMediaStreamSource(stream);
    source.connect(analyser);
    const processor = ctx.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0);
      const rms = Math.sqrt(input.reduce((sum, value) => sum + value * value, 0) / input.length);
      if (rms > 0.01) setVoiceDetected(true);
      audioChunksRef.current.push(new Float32Array(input));
      let samples = audioChunksRef.current.reduce((total, chunk) => total + chunk.length, 0);
      setAudioSeconds(Math.min(10, Math.floor(samples / audioSampleRateRef.current)));
      const limit = audioSampleRateRef.current * 10;
      // Keep at least a full window; crop the extra partial chunk when sending.
      while (audioChunksRef.current.length > 1 && samples - audioChunksRef.current[0].length >= limit) {
        samples -= audioChunksRef.current.shift()?.length ?? 0;
      }
    };
    source.connect(processor);
    processor.connect(ctx.destination);
    const data = new Uint8Array(analyser.frequencyBinCount);
    audioContextRef.current = ctx;
    analyserRef.current = analyser;
    audioProcessorRef.current = processor;
    audioSampleRateRef.current = ctx.sampleRate;

    const sample = () => {
      analyser.getByteFrequencyData(data);
      const avg = data.reduce((s, v) => s + v, 0) / data.length;
      setAudioLevel(Math.min(1, avg / 120));
      rafRef.current = requestAnimationFrame(sample);
    };
    sample();
  }

  // Collect 16 frames over the same 10-second window as the audio.
  function startCollection(stream: MediaStream) {
    startAudioMeter(stream);
    if (collectionTickRef.current !== null) return;
    const generation = analysisGenerationRef.current;
    collectionTickRef.current = window.setInterval(async () => {
      if (collectionBusyRef.current) return;
      collectionBusyRef.current = true;
      try {
        const blob = await captureFrameBlob();
        if (generation !== analysisGenerationRef.current || !blob) return;
        frameBufferRef.current.push(blob);
        if (frameBufferRef.current.length > 16) frameBufferRef.current.shift();
        setCollectedFrames(frameBufferRef.current.length);
      } catch {
        if (generation === analysisGenerationRef.current) setLiveStatus("camera");
      } finally {
        if (generation === analysisGenerationRef.current) collectionBusyRef.current = false;
      }
    }, 10_000 / 16);
  }

  function handleDeckUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    cleanupDeckUrl();
    const url = URL.createObjectURL(file);
    deckObjectUrlRef.current = url;
    setDeck({
      name: file.name,
      type: getDeckType(file.name),
      url,
      source: "uploaded",
    });
    if (selectedScene === "camera" || selectedScene === "audience" || selectedScene === "interview") {
      setSelectedScene("slides");
    }
    event.target.value = "";
  }

  function handleUseSampleDeck() {
    cleanupDeckUrl();
    setDeck(SAMPLE_DECK);
    setSlideIndex(0);
    if (selectedScene === "camera" || selectedScene === "audience" || selectedScene === "interview") {
      setSelectedScene("slides");
    }
  }

  // Apply one inference frame to the live UI state (shared by both sources).
  function applyFrame(frame: LiveInferenceFrame) {
    stressHistoryRef.current.push(frame.stressScore);
    setStressScore(frame.stressScore);
    setConfidence(frame.confidence);
  }

  // Grab whichever video element currently holds the live webcam stream.
  function getActiveWebcamVideo(): HTMLVideoElement | null {
    for (const v of [mainVideoRef.current, pipVideoRef.current, preparationVideoRef.current]) {
      if (v && v.srcObject && v.videoWidth > 0) return v;
    }
    return null;
  }

  function captureFrameBlob(): Promise<Blob | null> {
    const video = getActiveWebcamVideo();
    if (!video) return Promise.resolve(null);
    const canvas = document.createElement("canvas");
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext("2d");
    if (!ctx) return Promise.resolve(null);
    ctx.drawImage(video, 0, 0, 224, 224);
    return new Promise((resolve) => canvas.toBlob((b) => resolve(b), "image/jpeg", 0.8));
  }

  function getRecentAudioPcm() {
    const chunks = audioChunksRef.current;
    const inputLength = chunks.reduce((total, chunk) => total + chunk.length, 0);
    if (inputLength === 0) return new Float32Array(0);
    const input = new Float32Array(inputLength);
    let offset = 0;
    chunks.forEach((chunk) => { input.set(chunk, offset); offset += chunk.length; });
    const ratio = audioSampleRateRef.current / 16_000;
    const output = new Float32Array(Math.min(160_000, Math.floor(input.length / ratio)));
    const start = Math.max(0, input.length - Math.ceil(output.length * ratio));
    for (let index = 0; index < output.length; index += 1) {
      output[index] = input[Math.min(start + Math.floor(index * ratio), input.length - 1)];
    }
    return output;
  }

  // Send the latest 16 webcam frames and 10-second audio buffer to the model.
  async function requestLiveFrame(preparation = false) {
    if (preparation && recordingActiveRef.current) return;
    if (inferInFlightRef.current) return;
    const generation = analysisGenerationRef.current;
    inferInFlightRef.current = true;
    let timeout: ReturnType<typeof setTimeout> | undefined;
    try {
      if (frameBufferRef.current.length < 16) { setLiveStatus("collecting"); return; }
      const audio = getRecentAudioPcm();
      if (audio.length === 0) { setLiveStatus("audio"); return; }
      if (audio.length < 160_000) { setLiveStatus("collecting"); return; }
      if (preparation && !preparationFrameRef.current) setPreparationPhase("analyzing");
      if (preparation && SIGNAL_SOURCE === "simulated") {
        preparationFrameRef.current = createSimulatedSignal().next();
        setPreparationPhase("ready");
        return;
      }
      const controller = new AbortController();
      inferenceAbortRef.current = controller;
      timeout = setTimeout(() => controller.abort(), 90_000);
      setLiveStatus("analyzing");
      const frame = await inferMultimodalFrame([...frameBufferRef.current], audio, "Speaking", controller.signal);
      if (generation !== analysisGenerationRef.current) return;
      if (preparation && recordingActiveRef.current) return;
      if (preparation) {
        preparationFrameRef.current = frame;
        setPreparationPhase("ready");
      }
      else applyFrame(frame);
      setModelConnected(true);
      setLiveStatus("ok");
    } catch {
      if (generation === analysisGenerationRef.current) {
        if (preparation && recordingActiveRef.current) return;
        setModelConnected(false);
        setLiveStatus("error");
        if (preparation) {
          preparationFrameRef.current = null;
          setPreparationPhase("error");
        }
      }
    } finally {
      if (timeout !== undefined) clearTimeout(timeout);
      if (generation === analysisGenerationRef.current) {
        inferInFlightRef.current = false;
        inferenceAbortRef.current = null;
      }
    }
  }

  async function handleStartSession() {
    if (!preparationComplete || preparationPhase !== "ready" || !preparationFrameRef.current || recordingActiveRef.current || saving || recordingFinalizing) return;
    const baseline = preparationFrameRef.current;
    if (Date.now() - baseline.timestampMs > 30_000 || !streamRef.current?.getTracks().every((track) => track.readyState === "live") || audioContextRef.current?.state !== "running") {
      preparationFrameRef.current = null;
      setPreparationPhase("collecting");
      void startPreview();
      return;
    }
    recordingActiveRef.current = true;
    saveCompleteDialogRef.current?.close();
    // Keep only this session's warm-up buffer; the previous session was cleared on stop.
    try {
      setSessionState("starting");
      setErrorMessage("");
      setReport(null);
      setElapsed(0);
      setTimeline(initialTimeline);
      setStressScore(baseline.stressScore);
      setConfidence(baseline.confidence);
      startBaselineRef.current = baseline.stressScore;
      // Keep the last verified connection status, not the preparation score.
      stressHistoryRef.current = [baseline.stressScore];
      signalRef.current = createSimulatedSignal();
      signalRef.current.reset();
      setLiveStatus("ok");
      setSaveStatus("");
      setSaved(false);
      setSaveDialogOpen(false);
      setSaveVideo(true);
      setRecordedVideo(null);
      if (recordedVideoUrl) {
        URL.revokeObjectURL(recordedVideoUrl);
        setRecordedVideoUrl("");
      }
      setChatMessages([{ role: "agent", text: t("report.chatWaiting") }]);

      let stream = streamRef.current;
      if (!stream) {
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: true });
        streamRef.current = stream;
      }
      setCameraReady(true);
      attachStreamToVideos(stream);

      startCollection(stream);
      await audioContextRef.current?.resume();
      startSessionRecording(stream);
      setSessionState("live");
      if (SIGNAL_SOURCE === "live") void requestLiveFrame();

      // Check once a second; in-flight requests are never duplicated.
      // Frames come through the integration boundary (lib/live-signal.ts):
      // either the local simulator or the real Python inference server.
      const startedAt = performance.now();
      tickRef.current = window.setInterval(() => {
        setElapsed(Math.floor((performance.now() - startedAt) / 1000));
        if (SIGNAL_SOURCE === "live") {
          void requestLiveFrame();
        } else {
          const frame = signalRef.current?.next();
          if (frame) applyFrame(frame);
        }
      }, 1000);
    } catch {
      cleanupMedia();
      setPreparationPhase("collecting");
      setSessionState("error");
      setErrorMessage(t("err.permission"));
    }
  }

  // Localized "peak pressure" phrase for the report / chat.
  function peakText(r: ReportSummary | null) {
    if (!r) return t("agent.stressFallback");
    return lang === "ko"
      ? `${r.peakPhase} 단계 부근 ${r.peakValue}%`
      : `${r.peakValue}% around the ${r.peakPhase} phase`;
  }

  function handleStopSession() {
    if (!recordingActiveRef.current) return;
    stopSessionRecording();
    stopAnalysis();
    setSessionState("ended");
    setPreparationPhase("collecting");
    setModelConnected(false);
    // New generation, new buffers, no previous session responses or baseline.
    void startPreview();

    if (stressHistoryRef.current.length === 0) {
      setTimeline([]);
      setReport(null);
      setSaveDialogOpen(false);
      setErrorMessage(lang === "ko" ? "실제 연습의 분석 결과가 도착하기 전에 종료됐습니다. 준비 단계 결과는 기록에 포함하지 않습니다. 조금 더 길게 연습해 주세요." : "Practice ended before an analysis result arrived. Preparation results are not saved. Please practice a little longer.");
      return;
    }

    const sessionTimeline = buildSessionTimeline(stressHistoryRef.current);
    const avg = sessionTimeline.reduce((s, v) => s + v, 0) / sessionTimeline.length / 100;
    const { peakValue, peakBin, peakPhase, mostStressfulPhase } = summarizeTimeline(sessionTimeline);
    setTimeline(sessionTimeline);

    const nextReport: ReportSummary = {
      startBaseline: startBaselineRef.current,
      peakValue,
      peakBin,
      peakPhase,
      mostStressfulPhase,
      averageStress: avg,
      confidence,
      noticeable: peakValue > 70,
      deckName: deck?.name ?? "",
    };

    setReport(nextReport);
    setSessionTitle(`${t(`scene.${selectedScene}.label`, activeScene.label)} ${t("save.defaultSuffix")}`);
    setSaved(false);
    setSaveStatus("");
    setSaveDialogOpen(true);
    setChatMessages([{ role: "agent", text: t("report.chatSummaryPrefix") + t("report.nextActionText") }]);
  }

  async function handleSaveSession() {
    if (!report || saved || savingRef.current) return;
    if (saveVideo && recordingFinalizing) return;
    if (saveVideo && (!recordedVideo || recordedVideo.size === 0)) {
      setSaveStatus(lang === "ko" ? "저장할 영상이 없습니다. 영상을 다시 녹화하거나 영상 저장 선택을 해제해 주세요." : "No recording is available. Record again or uncheck video saving.");
      return;
    }
    const title = sessionTitle.trim() || `${activeScene.label} Practice`;

    if (!hasEnv || !user) {
      setSaveStatus(lang === "ko" ? "기록을 저장하려면 로그인해 주세요." : "Please sign in to save your session.");
      return;
    }

    // Stored in English for stable, language-independent dashboard data.
    const englishResult = report.noticeable
      ? `Noticeable ${report.mostStressfulPhase.toLowerCase()}-section pressure`
      : "Mostly stable delivery";
    const englishDiagnosis = `Starting baseline: ${Math.round(report.startBaseline * 100)}% (includes this session's pre-start preparation; first timeline sample). Peak pressure reached ${report.peakValue}% around the ${report.peakPhase} phase. Deck: ${report.deckName || "No deck used"}.`;
    const englishNextAction = "Practice slower transitions between your main explanation blocks and leave a short pause after each key term.";

    savingRef.current = true;
    setSaving(true);
    setSaveStatus("");
    try {
      const result = await savePracticeSession({
        title,
        durationSeconds: elapsed,
        averageStress: report.averageStress,
        result: englishResult,
        diagnosis: englishDiagnosis,
        nextAction: englishNextAction,
        sceneLabel: activeScene.label,
        timeline,
        video: saveVideo ? recordedVideo : null,
      });
      setSaved(true);
      setSaveStatus(result?.warning ?? t("status.saved"));
      setSaveDialogOpen(false);
      saveCompleteDialogRef.current?.showModal();
    } catch (error) {
      setSaveStatus(error instanceof Error ? error.message : t("status.saveFailed"));
    } finally {
      savingRef.current = false;
      setSaving(false);
    }
  }

  function handleSendMessage() {
    if (!chatInput.trim() || sessionState !== "ended") return;
    const userText = chatInput.trim();
    const reply = buildAgentReply(userText, t, peakText(report));
    setChatMessages((prev) => [...prev, { role: "user", text: userText }, { role: "agent", text: reply }]);
    setChatInput("");
  }

  function seekToTimelineBin(bin: number) {
    const video = sessionVideoRef.current;
    if (!video || !Number.isFinite(video.duration)) return;
    video.currentTime = ((bin + 0.5) / 12) * video.duration;
    void video.play().catch(() => undefined);
  }

  function renderStageVisual() {
    if (selectedScene === "camera") {
      return <video ref={mainVideoRef} autoPlay muted playsInline />;
    }

    if (selectedScene === "slides") {
      return (
        <div className="deck-stage">
          <DeckPreview deck={deck} slideIndex={slideIndex} onPrev={goToPrevSlide} onNext={goToNextSlide} />
        </div>
      );
    }

    if (selectedScene === "classroom-slides") {
      return (
        <div className="classroom-deck-stage">
          <img src="/scenes/classroom-audience.png" alt="" />
          <div className="classroom-deck-overlay">
            <DeckPreview deck={deck} slideIndex={slideIndex} onPrev={goToPrevSlide} onNext={goToNextSlide} />
          </div>
        </div>
      );
    }

    if (selectedScene === "audience") {
      return (
        <div className="scene-image-stage">
          <img src="/scenes/classroom-audience.png" alt="" />
          <div className="scene-vignette" />
          <div className="scene-caption">
            <Users size={18} />
            <span>{t("scene.audienceCaption")}</span>
          </div>
        </div>
      );
    }

    return (
      <div className="interview-stage">
        <div className="interview-panel-card">
          <div className="interview-avatar">DR</div>
          <strong>{t("interview.dr.name")}</strong>
          <span>{t("interview.dr.q")}</span>
        </div>
        <div className="interview-panel-card">
          <div className="interview-avatar">PM</div>
          <strong>{t("interview.pm.name")}</strong>
          <span>{t("interview.pm.q")}</span>
        </div>
        <div className="interview-panel-card">
          <div className="interview-avatar">AI</div>
          <strong>{t("interview.ai.name")}</strong>
          <span>{t("interview.ai.q")}</span>
        </div>
      </div>
    );
  }

  const audioPercent = Math.round(audioLevel * 100);
  const activeScene = practiceScenes.find((s) => s.id === selectedScene) ?? practiceScenes[0];
  const shouldShowPip = selectedScene !== "camera";

  const stateLabel =
    sessionState === "live" ? t("state.live") :
    sessionState === "ended" ? t("state.ended") :
    sessionState === "starting" ? t("state.starting") :
    sessionState === "error" ? t("state.error") :
    t("state.ready");

  const stateChipClass =
    sessionState === "live" ? "chip-live" :
    sessionState === "ended" ? "chip-ended" :
    sessionState === "error" ? "chip-error" :
    "chip-idle";

  const activeSceneLabel = t(`scene.${activeScene.id}.label`, activeScene.label);
  const activeSceneDesc = t(`scene.${activeScene.id}.desc`, activeScene.description);

  // Localized report display values (report holds structured data).
  const reportAvgText = report ? `${report.averageStress.toFixed(2)} / 1.00` : "";
  const reportPeakPointText = report ? `${report.peakValue}% · ${t("report.binLabel")} ${report.peakBin + 1} (${report.peakPhase})` : "";
  const reportPhaseText = report ? `${report.mostStressfulPhase} ${t("phase.suffix")}` : "";
  const reportStateText = report
    ? report.noticeable
      ? lang === "ko"
        ? `${report.mostStressfulPhase} 구간의 압박이 두드러집니다`
        : `Noticeable ${report.mostStressfulPhase.toLowerCase()}-section pressure`
      : t("report.stateStable")
    : "";
  const reportConfidenceText = report && report.confidence > 0 ? `${Math.round(report.confidence * 100)}%` : "—";

  return (
    <main className="page-content">
      <div className="page-top">
        <div>
          <h1 className="page-title">{t("practice.title")}</h1>
          <p className="page-subtitle">{t("practice.subtitle")}</p>
        </div>
        <div className="page-actions">
          <button
            type="button"
            className="btn btn-outline"
            onClick={handleStopSession}
            disabled={sessionState !== "live"}
          >
            <CircleStop size={14} />
            {t("btn.endSession")}
          </button>
          <button
            type="button"
            className="btn btn-primary"
            onClick={handleStartSession}
            disabled={!preparationComplete || preparationPhase !== "ready" || sessionState === "starting" || sessionState === "live" || saving || recordingFinalizing}
          >
            <Play size={14} />
            {t("btn.startSession")}
          </button>
        </div>
      </div>

      {preparationComplete && sessionState !== "live" && sessionState !== "starting" && (
        <div className="practice-setup-card" role="status">
          <div>
            {preparationPhase === "ready"
              ? (lang === "ko" ? "분석 준비 완료 · 시작하면 현재 준비값부터 바로 기록합니다." : "Analysis ready · Start recording with the current baseline.")
              : preparationPhase === "error"
              ? (lang === "ko" ? "분석 준비에 실패했습니다. 서버를 확인한 뒤 다시 시도해 주세요." : "Preparation failed. Check the server and retry.")
              : (lang === "ko" ? "새 세션 준비 중 · 카메라를 보며 짧게 말해 주세요. 새 영상·음성 수집 및 모델 분석이 완료되면 시작할 수 있습니다." : "Preparing a new session · Look at the camera and speak briefly. Start becomes available after fresh data collection and analysis.")}
            <small style={{ display: "block" }}>{lang === "ko" ? "준비 중에는 카메라·마이크 분석만 진행하며 녹화하지 않습니다. 이전 세션 데이터는 사용하지 않습니다." : "Camera and microphone analysis continues without recording. Previous session data is not reused."}</small>
          </div>
          {(preparationPhase === "error" || !audioRunning) && <button className="btn btn-outline" onClick={() => { setPreparationPhase("collecting"); void startPreview(); }}>{lang === "ko" ? "다시 준비" : "Retry preparation"}</button>}
        </div>
      )}

      <div className="practice-setup-card">
        <div>
          <div className="setup-label">{t("setup.rehearsalDeck")}</div>
          <div className="setup-title">{deck ? deck.name : t("setup.noDeck")}</div>
          <div className="setup-desc">
            {deck?.type === "pdf" || deck?.type === "presentation"
              ? t("setup.ready")
              : deck
              ? t("setup.needsPdf")
              : t("setup.uploadOrSample")}
          </div>
        </div>
        <div className="setup-actions">
          <label className="btn btn-outline file-upload-btn">
            <Upload size={14} />
            {t("btn.uploadDeck")}
            <input type="file" accept=".pdf,.ppt,.pptx,application/pdf" onChange={handleDeckUpload} />
          </label>
          <button type="button" className="btn btn-outline" onClick={handleUseSampleDeck}>
            <FileText size={14} />
            {t("btn.useSample")}
          </button>
        </div>
      </div>

      <div className="session-grid" style={{ marginBottom: 14 }}>
        <div>
          <div className="video-box rehearsal-stage">
            {renderStageVisual()}

            {shouldShowPip && (
              <div className="camera-pip">
                <video ref={pipVideoRef} autoPlay muted playsInline />
                <span>{t("signal.camera")}</span>
              </div>
            )}

            <div className="video-overlay">
              <span className={`status-chip ${stateChipClass}`}>
                <span className="status-dot" />
                {stateLabel}
              </span>
              {sessionState !== "live" && <span className="live-cue">{activeSceneDesc}</span>}
              <span style={{ fontSize: 12, opacity: 0.8 }}>{formatSeconds(elapsed)}</span>
            </div>
          </div>

          <div className="scene-chips">
            {practiceScenes.map((scene) => (
              <button
                key={scene.id}
                type="button"
                className={`scene-chip${selectedScene === scene.id ? " active" : ""}`}
                onClick={() => setSelectedScene(scene.id)}
              >
                {t(`scene.${scene.id}.label`, scene.label)}
              </button>
            ))}
          </div>

          {errorMessage && <p className="error-msg" style={{ marginTop: 10 }}>{errorMessage}</p>}
          {saveStatus && <p className="success-msg" style={{ marginTop: 10 }}>{saveStatus}</p>}
        </div>

        <div className="signal-grid" style={{ alignContent: "start" }}>
          <div className="signal-card">
            <div className="signal-label">{t("signal.sessionMode")}</div>
            <div className={`signal-value ${sessionState === "live" ? "ok" : sessionState === "error" ? "alert" : "data"}`}>
              {sessionState === "live" ? t("val.live") : sessionState === "ended" ? t("val.done") : t("val.ready")}
            </div>
          </div>

          <div className="signal-card">
            <div className="signal-label">{t("signal.elapsed")}</div>
            <div className="signal-value data">{formatSeconds(elapsed)}</div>
          </div>

          <div className="signal-card">
            <div className="signal-label">
              <Mic size={11} style={{ display: "inline", marginRight: 4 }} />
              {t("signal.mic")}
            </div>
            <div className={`signal-value ${audioPercent > 8 ? "ok" : "data"}`}>{audioPercent}%</div>
          </div>

          <div className="signal-card">
            <div className="signal-label">
              <Camera size={11} style={{ display: "inline", marginRight: 4 }} />
              {t("signal.camera")}
            </div>
            <div className={`signal-value ${cameraReady ? "ok" : "data"}`}>
              {cameraReady ? (sessionState === "live" ? t("val.on") : t("val.preview")) : t("val.stby")}
            </div>
          </div>

          <div className="signal-card">
            <div className="signal-label">
              <Gauge size={11} style={{ display: "inline", marginRight: 4 }} />
              {t("signal.liveStress")}
              {sessionState === "live" && stressHistoryRef.current.length === 1 && <small style={{ display: "block" }}>{lang === "ko" ? "시작 기준값 · 이번 세션 준비 구간 포함" : "Starting baseline · Includes this session's warm-up"}</small>}
            </div>
            <div className={`signal-value ${sessionState === "live" ? (stressScore > 0.65 ? "alert" : "ok") : "data"}`}>
              {sessionState === "live" && confidence > 0 ? stressScore.toFixed(2) : "—"}
            </div>
          </div>

          {SIGNAL_SOURCE === "simulated" ? (
            <p className="signal-note signal-card-wide">
              {t("note.simulated")}
            </p>
          ) : sessionState !== "live" ? null : liveStatus === "error" || liveStatus === "audio" || liveStatus === "camera" ? (
            <p className={`signal-note signal-card-wide${liveStatus === "error" ? " signal-note-error" : ""}`}>
              {liveStatus === "error"
                ? t("note.liveError")
                : liveStatus === "camera"
                ? t("note.liveCamera")
                : t("note.liveAudio")}
            </p>
          ) : confidence === 0 ? (
            <p className="signal-note signal-card-wide" role="status">
              {modelConnected && <>{t("note.liveVerified")}<br /></>}
              {liveStatus === "analyzing" ? t("note.liveAnalyzing") : t("note.liveCollecting")}
            </p>
          ) : modelConnected ? (
            <p className="signal-note signal-card-wide" role="status" style={{ color: "var(--accent-ok)" }}>
              {t("note.liveOk")}
            </p>
          ) : null}
        </div>
      </div>

      {sessionState === "ended" && report && (
        <>
          <div className="dash-grid" style={{ marginBottom: 14 }}>
            <div className="card">
              <div className="panel-head">
                <div className="panel-head-left">
                  <span className="panel-title">{t("report.timeline")}</span>
                  <span className="panel-subtitle">{t("report.timelineSub")}</span>
                </div>
                <Waves size={16} color="var(--text-secondary)" />
              </div>
              <StressChart series={timeline} />
              <p className="signal-note">{lang === "ko" ? `첫 지점은 이번 세션의 시작 전 준비 기준값(${Math.round(report.startBaseline * 100)}%)이며, 요약 통계에도 포함됩니다.` : `The first point is this session's pre-start baseline (${Math.round(report.startBaseline * 100)}%), included in summary statistics.`}</p>
              {recordedVideoUrl && (
                <div className="session-video-review">
                  <video ref={sessionVideoRef} controls playsInline src={recordedVideoUrl} />
                  <div className="video-timeline-label">
                    {lang === "ko" ? "구간을 누르면 해당 영상 위치로 이동합니다" : "Select a segment to jump to that moment"}
                  </div>
                  <div className="video-timeline" aria-label="Stress timeline linked to video">
                    {timeline.map((value, bin) => (
                      <button
                        key={bin}
                        type="button"
                        className={value >= 70 ? "high" : value >= 50 ? "mid" : "low"}
                        style={{ flex: Math.max(value, 20) }}
                        onClick={() => seekToTimelineBin(bin)}
                        title={`${lang === "ko" ? "스트레스" : "Stress"} ${value}%`}
                      />
                    ))}
                  </div>
                  <div className="video-timeline-legend">
                    <span><i className="low" />{lang === "ko" ? "좋음" : "Good"}</span>
                    <span><i className="mid" />{lang === "ko" ? "보완 필요" : "Needs attention"}</span>
                    <span><i className="high" />{lang === "ko" ? "스트레스 높음" : "High stress"}</span>
                  </div>
                </div>
              )}
            </div>

            <div className="card">
              <div className="panel-head">
                <div className="panel-head-left">
                  <span id="session-analysis-results" className="panel-title">{t("report.diagnosis")}</span>
                  <span className="panel-subtitle">{t("report.diagnosisSub")}</span>
                </div>
                <Sparkles size={16} color="var(--text-secondary)" />
              </div>
              <div className="coach-stack">
                <div className="coach-card coach-alert">
                  <h3>{t("report.overallJudgment")}</h3>
                  <p>{reportStateText}</p>
                </div>
                <div className="coach-card coach-data">
                  <h3>{t("report.peakPoint")}</h3>
                  <p>{peakText(report)}</p>
                </div>
                <div className="coach-card coach-ok">
                  <h3>{t("report.nextStep")}</h3>
                  <p>{t("report.nextActionText")}</p>
                </div>
              </div>
            </div>
          </div>

          <div className="dash-grid">
            <div className="card">
              <div className="panel-head">
                <div className="panel-head-left">
                  <span className="panel-title">{t("report.summary")}</span>
                  <span className="panel-subtitle">{t("report.summarySub")}</span>
                </div>
                <BrainCircuit size={16} color="var(--text-secondary)" />
              </div>

              <div className="report-grid">
                <div className="report-card">
                  <div className="rc-label">{t("report.overallAvg")}</div>
                  <div className="rc-value">{reportAvgText}</div>
                  <div className="rc-desc">{t("report.overallAvgDesc")}</div>
                </div>
                <div className="report-card">
                  <div className="rc-label">{t("report.peakStress")}</div>
                  <div className="rc-value" style={{ fontSize: 16, lineHeight: 1.3 }}>{reportPeakPointText}</div>
                  <div className="rc-desc">{t("report.peakStressDesc")}</div>
                </div>
                <div className="report-card">
                  <div className="rc-label">{t("report.mostStressfulPhase")}</div>
                  <div className="rc-value" style={{ fontSize: 16, lineHeight: 1.3 }}>{reportPhaseText}</div>
                  <div className="rc-desc">{t("report.mostStressfulPhaseDesc")}</div>
                </div>
                <div className="report-card">
                  <div className="rc-label">{t("report.nextStep")}</div>
                  <div className="rc-value" style={{ fontSize: 14, lineHeight: 1.4 }}>{t("report.nextActionText")}</div>
                  <div className="rc-desc">{t("report.nextStepDesc")}</div>
                </div>
              </div>

              <div className="report-meta">
                <span><strong>{t("report.confidence")}:</strong> {reportConfidenceText}</span>
                <span><strong>{t("report.deck")}:</strong> {report?.deckName || t("setup.noDeck")}</span>
                <span><strong>{t("report.scene")}:</strong> {activeSceneLabel}</span>
              </div>

              {/* Save / name this session */}
              <div className="save-session">
                <div className="save-session-label">{t("save.subtitle")}</div>
                <div className="save-session-row">
                  <input
                    className="save-session-input"
                    value={sessionTitle}
                    onChange={(e) => setSessionTitle(e.target.value)}
                    placeholder={t("save.namePlaceholder")}
                    disabled={saved}
                  />
                  <button type="button" className="btn btn-primary" onClick={handleSaveSession} disabled={saved || saving || (saveVideo && recordingFinalizing) || !sessionTitle.trim()}>
                    {saving ? (lang === "ko" ? "저장 중…" : "Saving…") : saved ? t("save.saved") : t("save.button")}
                  </button>
                </div>
              </div>

              <div className="highlight-list">
                {postSessionHighlights.map((item, i) => (
                  <div key={item.title} className="highlight-card">
                    <h3>{t(`highlight.${i}.title`, item.title)}</h3>
                    <p>{t(`highlight.${i}.detail`, item.detail)}</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="card">
              <div className="panel-head">
                <div className="panel-head-left">
                  <span className="panel-title">{t("report.aiAgent")}</span>
                  <span className="panel-subtitle">{t("report.aiAgentSub")}</span>
                </div>
                <MessageSquareText size={16} color="var(--text-secondary)" />
              </div>

              <div className="starter-pills">
                {agentStarters.map((s, i) => {
                  const label = t(`starter.${i}`, s);
                  return (
                    <button key={s} type="button" className="starter-pill" onClick={() => setChatInput(label)}>
                      {label}
                    </button>
                  );
                })}
              </div>

              <div className="chat-messages">
                {chatMessages.map((m, i) => (
                  <div key={i} className={`chat-bubble ${m.role}`}>{m.text}</div>
                ))}
              </div>

              <div className="chat-input-row">
                <input
                  className="chat-input"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter") handleSendMessage(); }}
                  placeholder={t("report.chatPlaceholder")}
                />
                <button type="button" className="btn btn-primary" onClick={handleSendMessage}>
                  {t("report.ask")}
                </button>
              </div>
            </div>
          </div>
        </>
      )}
      {saveDialogOpen && report && (
        <div className="save-modal-backdrop" role="presentation">
          <section className="save-modal" role="dialog" aria-modal="true" aria-labelledby="save-session-title">
            <button type="button" className="save-modal-close" disabled={saving} onClick={() => setSaveDialogOpen(false)} aria-label="Close">
              <X size={18} />
            </button>
            <span className="save-modal-eyebrow">{lang === "ko" ? "세션 완료" : "Session complete"}</span>
            <h2 id="save-session-title">{lang === "ko" ? "이번 세션을 저장할까요?" : "Save this session?"}</h2>
            <p>{lang === "ko" ? "영상이 없어도 분석 결과와 스트레스 타임라인은 기록에 저장됩니다." : "Your analysis and stress timeline are saved even without a video."}</p>
            <label className="save-modal-field">
              <span>{lang === "ko" ? "세션 이름" : "Session name"}</span>
              <input value={sessionTitle} onChange={(event) => setSessionTitle(event.target.value)} placeholder={t("save.namePlaceholder")} />
            </label>
            <label className="save-video-option">
              <input className="save-video-checkbox" type="checkbox" checked={saveVideo} onChange={(event) => setSaveVideo(event.target.checked)} disabled={recordingFinalizing} />
              <span>
                <strong>{lang === "ko" ? "녹화 영상도 함께 저장" : "Include recorded video"}</strong>
                <small>{recordingFinalizing ? (lang === "ko" ? "녹화 파일을 마무리하는 중입니다. 준비가 끝나면 저장할 수 있어요." : "Finalizing your recording. Saving will be available when ready.") : recordedVideo ? (lang === "ko" ? "체크하면 영상과 타임라인을 함께 저장합니다." : "When checked, the video and timeline are saved together.") : (lang === "ko" ? "녹화 영상이 없습니다. 기록만 저장하려면 선택을 해제해 주세요." : "No recording is available. Uncheck this to save analysis only.")}</small>
              </span>
            </label>
            {saveStatus && !saved && <p role="alert" className="error-msg">{saveStatus}</p>}
            <div className="save-modal-actions">
              <button type="button" className="btn btn-outline" disabled={saving} onClick={() => setSaveDialogOpen(false)}>{lang === "ko" ? "저장하지 않기" : "Don't save"}</button>
              <button type="button" className="btn btn-primary" onClick={handleSaveSession} disabled={saving || (saveVideo && recordingFinalizing) || !sessionTitle.trim()}>{saving ? (lang === "ko" ? "저장 중…" : "Saving…") : saveVideo && recordingFinalizing ? (lang === "ko" ? "영상 준비 중…" : "Preparing video…") : (lang === "ko" ? "저장하기" : "Save session")}</button>
            </div>
          </section>
        </div>
      )}
      <dialog ref={preparationDialogRef} className="save-modal preparation-dialog" aria-labelledby="preparation-title" aria-describedby="preparation-description" onCancel={(event) => event.preventDefault()}>
        <span className="save-modal-eyebrow">{lang === "ko" ? "연습 전 준비" : "Before you practice"}</span>
        <h2 id="preparation-title">{lang === "ko" ? "카메라와 마이크를 확인할게요." : "Let's check your camera and microphone."}</h2>
        <p id="preparation-description">{lang === "ko" ? "얼굴이 화면에 잘 보이도록 앉은 뒤, 아래 문장을 평소 발표하듯 자연스럽게 읽어주세요." : "Sit so your face is clearly visible, then read the sentence below in your normal presentation voice."}</p>
        <video className="preparation-video" ref={preparationVideoRef} autoPlay muted playsInline />
        <blockquote className="preparation-script">{lang === "ko" ? "안녕하세요. 지금부터 발표 연습을 시작하겠습니다. 오늘 준비한 내용을 여러분께 차근차근 설명하겠습니다." : "Hello. I am about to begin my presentation practice. I will explain the material I have prepared, step by step."}</blockquote>
        <p className="preparation-privacy">{lang === "ko" ? "읽는 동안 영상·음성을 분석합니다. 준비 완료 후에도 분석을 유지하며, 시작할 때 최신 준비 분석값 하나를 시작 기준값으로 기록합니다. 준비 중 원본 영상·음성은 저장하지 않습니다." : "Analysis continues after preparation. The latest preparation score is saved as the starting baseline when you start. Raw preparation video and audio are not saved."}</p>
        <ul className="preparation-checks">
          <li>{lang === "ko" ? "영상 수집" : "Video collection"}: {collectedFrames}/16</li>
          <li>{lang === "ko" ? "음성 수집" : "Audio collection"}: {audioSeconds}/10{lang === "ko" ? "초" : "s"}</li>
          <li>{voiceDetected ? (lang === "ko" ? "마이크 소리 감지됨" : "Microphone sound detected") : (lang === "ko" ? "마이크에 문장을 읽어주세요" : "Read the sentence into your microphone")}</li>
        </ul>
        {!audioRunning && <button autoFocus type="button" className="btn btn-outline" onClick={() => void startPreview()}>{t("note.prepareButton")}</button>}
        {errorMessage && <p role="alert" className="error-msg">{errorMessage}</p>}
        <p role="status" className="preparation-status">{preparationPhase === "ready"
          ? (lang === "ko" ? (SIGNAL_SOURCE === "live" ? "모델 응답 확인 완료! 준비 완료를 누른 뒤 시작해 주세요." : "기기 준비 완료! 현재는 데모 분석 모드입니다.") : (SIGNAL_SOURCE === "live" ? "Model response verified! Confirm readiness, then start practice." : "Devices are ready. Demo analysis mode is active."))
          : preparationPhase === "analyzing" ? (lang === "ko" ? "모델 응답을 확인하고 있어요. 잠시 기다려주세요…" : "Checking the model response. Please wait…")
          : preparationPhase === "error" ? (lang === "ko" ? "모델 확인에 실패했습니다. 서버 상태를 확인하고 다시 시도해 주세요." : "Model check failed. Check the server and retry.")
          : (lang === "ko" ? "준비 데이터를 모으고 있어요. 문장을 다 읽었다면 잠시 기다려주세요." : "Collecting preparation data. If you have finished reading, please wait.")}</p>
        <div className="save-modal-actions">
          <a className="btn btn-outline" href="/dashboard">{lang === "ko" ? "나가기" : "Leave"}</a>
          {preparationPhase === "error" && <button type="button" className="btn btn-outline" onClick={() => setPreparationPhase("collecting")}>{lang === "ko" ? "다시 확인" : "Retry"}</button>}
          <button type="button" className="btn btn-primary" disabled={preparationPhase !== "ready"} onClick={() => {
            setPreparationComplete(true);
            preparationDialogRef.current?.close();
          }}>{lang === "ko" ? "준비 완료" : "Ready"}</button>
        </div>
      </dialog>
      <dialog ref={saveCompleteDialogRef} className="save-modal save-complete-dialog" aria-labelledby="save-complete-title" aria-describedby="save-complete-description">
        <span className="save-modal-eyebrow">{lang === "ko" ? "저장 완료" : "Session saved"}</span>
        <h2 id="save-complete-title">{lang === "ko" ? "세션이 저장되었어요!" : "Your session has been saved!"}</h2>
        <p id="save-complete-description">{lang === "ko" ? "아래에서 분석 결과를 보세요!" : "See your analysis results below!"}</p>
        {saveStatus && <p role="status">{saveStatus}</p>}
        <div className="save-modal-actions">
          <button type="button" className="btn btn-primary" autoFocus onClick={() => {
            saveCompleteDialogRef.current?.close();
            document.getElementById("session-analysis-results")?.scrollIntoView({ behavior: "smooth", block: "center" });
          }}>{lang === "ko" ? "분석 결과 보기" : "View analysis results"}</button>
        </div>
      </dialog>
    </main>
  );
}
