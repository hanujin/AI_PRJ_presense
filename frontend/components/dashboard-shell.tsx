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

type SessionState = "idle" | "starting" | "live" | "ended" | "error";
type ChatMessage = { role: "agent" | "user"; text: string };
type ReportSummary = {
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

function StressChart({ series }: { series: number[] }) {
  const max = Math.max(...series, 1);
  return (
    <div className="phase-chart">
      {PHASES.map((phase, p) => (
        <div key={phase} className="phase-group">
          <div className="phase-bars">
            {series.slice(p * 3, p * 3 + 3).map((v, i) => (
              <div
                key={i}
                className="stress-bar"
                style={{ height: `${Math.round((v / max) * 100)}%` }}
                title={`${phase} · bin ${i + 1}: ${v}`}
              />
            ))}
          </div>
          <span className="phase-label">{phase}</span>
        </div>
      ))}
    </div>
  );
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
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const audioProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const audioChunksRef = useRef<Float32Array[]>([]);
  const audioSampleRateRef = useRef(48_000);
  const frameBufferRef = useRef<Blob[]>([]);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const sessionVideoRef = useRef<HTMLVideoElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const tickRef = useRef<number | null>(null);
  const deckObjectUrlRef = useRef<string | null>(null);
  const stressHistoryRef = useRef<number[]>([]);
  const signalRef = useRef<LiveSignalController | null>(null);
  const inferInFlightRef = useRef(false);
  const analysisGenerationRef = useRef(0);
  const inferenceAbortRef = useRef<AbortController | null>(null);

  const [sessionState, setSessionState] = useState<SessionState>("idle");
  const [errorMessage, setErrorMessage] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const [audioLevel, setAudioLevel] = useState(0);
  const [stressScore, setStressScore] = useState(0.38);
  const [confidence, setConfidence] = useState(0);
  const [liveStatus, setLiveStatus] = useState<"idle" | "ok" | "error">("idle");
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
  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  const [saveVideo, setSaveVideo] = useState(true);
  const [recordedVideo, setRecordedVideo] = useState<Blob | null>(null);
  const [recordedVideoUrl, setRecordedVideoUrl] = useState("");
  const [recordingFinalizing, setRecordingFinalizing] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    { role: "agent", text: t("report.chatIntro") },
  ]);

  useEffect(() => {
    void startPreview();
    return () => {
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
    [mainVideoRef.current, pipVideoRef.current].forEach((video) => {
      if (!video) return;
      if (video.srcObject !== stream) video.srcObject = stream;
      void video.play().catch(() => undefined);
    });
  }

  function startSessionRecording(stream: MediaStream) {
    if (typeof MediaRecorder === "undefined") return;
    recordedChunksRef.current = [];
    const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp9,opus")
      ? "video/webm;codecs=vp9,opus"
      : "video/webm";
    const recorder = new MediaRecorder(stream, { mimeType });
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) recordedChunksRef.current.push(event.data);
    };
    recorder.onstop = () => {
      const video = new Blob(recordedChunksRef.current, { type: recorder.mimeType || "video/webm" });
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
    analysisGenerationRef.current += 1;
    inferenceAbortRef.current?.abort();
    inferenceAbortRef.current = null;
    inferInFlightRef.current = false;
    audioChunksRef.current = [];
    frameBufferRef.current = [];
    if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null; }
    if (tickRef.current) { window.clearInterval(tickRef.current); tickRef.current = null; }
    if (audioContextRef.current) { void audioContextRef.current.close(); audioContextRef.current = null; }
    analyserRef.current = null;
    audioProcessorRef.current = null;
    setAudioLevel(0);
  }

  function cleanupMedia() {
    stopSessionRecording();
    stopAnalysis();
    if (streamRef.current) { streamRef.current.getTracks().forEach((t) => t.stop()); streamRef.current = null; }
    [mainVideoRef.current, pipVideoRef.current].forEach((video) => {
      if (video) video.srcObject = null;
    });
    setCameraReady(false);
  }

  // Acquire the camera while idle so the user can check themselves before starting.
  async function startPreview() {
    if (streamRef.current) {
      attachStreamToVideos(streamRef.current);
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: true });
      streamRef.current = stream;
      setCameraReady(true);
      attachStreamToVideos(stream);
    } catch {
      setCameraReady(false);
      setErrorMessage("Camera permission was denied. Allow access to preview yourself before starting.");
    }
  }

  function startAudioMeter(stream: MediaStream) {
    const ctx = new window.AudioContext();
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 512;
    const source = ctx.createMediaStreamSource(stream);
    source.connect(analyser);
    const processor = ctx.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0);
      audioChunksRef.current.push(new Float32Array(input));
      let samples = audioChunksRef.current.reduce((total, chunk) => total + chunk.length, 0);
      const limit = audioSampleRateRef.current * 10;
      while (samples > limit && audioChunksRef.current.length > 1) {
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
    for (const v of [mainVideoRef.current, pipVideoRef.current]) {
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
    const output = new Float32Array(Math.floor(input.length / ratio));
    for (let index = 0; index < output.length; index += 1) {
      output[index] = input[Math.min(Math.floor(index * ratio), input.length - 1)];
    }
    return output;
  }

  // Send the latest 16 webcam frames and 10-second audio buffer to the model.
  async function requestLiveFrame() {
    if (inferInFlightRef.current) return;
    const generation = analysisGenerationRef.current;
    inferInFlightRef.current = true;
    try {
      const blob = await captureFrameBlob();
      if (generation !== analysisGenerationRef.current || !blob) return;
      frameBufferRef.current.push(blob);
      if (frameBufferRef.current.length > 16) frameBufferRef.current.shift();
      if (frameBufferRef.current.length < 16) return;
      const audio = getRecentAudioPcm();
      if (audio.length === 0) return;
      const controller = new AbortController();
      inferenceAbortRef.current = controller;
      const frame = await inferMultimodalFrame(frameBufferRef.current, audio, "Speaking", controller.signal);
      if (generation !== analysisGenerationRef.current) return;
      applyFrame(frame);
      setLiveStatus("ok");
    } catch {
      if (generation === analysisGenerationRef.current) setLiveStatus("error");
    } finally {
      if (generation === analysisGenerationRef.current) {
        inferInFlightRef.current = false;
        inferenceAbortRef.current = null;
      }
    }
  }

  async function handleStartSession() {
    stopAnalysis();
    try {
      setSessionState("starting");
      setErrorMessage("");
      setReport(null);
      setElapsed(0);
      setTimeline(initialTimeline);
      setStressScore(0.38);
      setConfidence(0);
      stressHistoryRef.current = [];
      audioChunksRef.current = [];
      frameBufferRef.current = [];
      signalRef.current = createSimulatedSignal();
      signalRef.current.reset();
      setLiveStatus(SIGNAL_SOURCE === "live" ? "idle" : "ok");
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

      startAudioMeter(stream);
      startSessionRecording(stream);
      setSessionState("live");

      // Inference loop runs at 1.5s intervals (Progress Report §1.2.4).
      // Frames come through the integration boundary (lib/live-signal.ts):
      // either the local simulator or the real Python inference server.
      tickRef.current = window.setInterval(() => {
        setElapsed((c) => c + 1);
        if (SIGNAL_SOURCE === "live") {
          void requestLiveFrame();
        } else {
          const frame = signalRef.current?.next();
          if (frame) applyFrame(frame);
        }
      }, 1500);
    } catch {
      cleanupMedia();
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
    stopSessionRecording();
    stopAnalysis();
    setSessionState("ended");

    const sessionTimeline = buildSessionTimeline(stressHistoryRef.current);
    const avg = sessionTimeline.reduce((s, v) => s + v, 0) / sessionTimeline.length / 100;
    const { peakValue, peakBin, peakPhase, mostStressfulPhase } = summarizeTimeline(sessionTimeline);
    setTimeline(sessionTimeline);

    const nextReport: ReportSummary = {
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
    if (!report || saved) return;
    const title = sessionTitle.trim() || `${activeScene.label} Practice`;

    if (!hasEnv || !user) {
      setSaved(true);
      setSaveStatus(t("status.localOnly"));
      return;
    }

    // Stored in English for stable, language-independent dashboard data.
    const englishResult = report.noticeable
      ? `Noticeable ${report.mostStressfulPhase.toLowerCase()}-section pressure`
      : "Mostly stable delivery";
    const englishDiagnosis = `Peak pressure reached ${report.peakValue}% around the ${report.peakPhase} phase. Deck: ${report.deckName || "No deck used"}.`;
    const englishNextAction = "Practice slower transitions between your main explanation blocks and leave a short pause after each key term.";

    try {
      await savePracticeSession({
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
      setSaveStatus(t("status.saved"));
      setSaveDialogOpen(false);
    } catch (error) {
      setSaveStatus(error instanceof Error ? error.message : t("status.saveFailed"));
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
            disabled={sessionState === "starting" || sessionState === "live"}
          >
            <Play size={14} />
            {t("btn.startSession")}
          </button>
        </div>
      </div>

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
            </div>
            <div className={`signal-value ${sessionState === "live" ? (stressScore > 0.65 ? "alert" : "ok") : "data"}`}>
              {sessionState === "live" ? stressScore.toFixed(2) : t("val.stby")}
            </div>
          </div>

          {SIGNAL_SOURCE === "simulated" ? (
            <p className="signal-note signal-card-wide">
              {t("note.simulated")}
            </p>
          ) : (
            <p className={`signal-note signal-card-wide${liveStatus === "error" ? " signal-note-error" : ""}`}>
              {liveStatus === "error"
                ? t("note.liveError")
                : liveStatus === "ok"
                ? t("note.liveOk")
                : t("note.liveConnecting")}
            </p>
          )}
        </div>
      </div>

      {sessionState === "ended" && (
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
                  <span className="panel-title">{t("report.diagnosis")}</span>
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
                  <button type="button" className="btn btn-primary" onClick={handleSaveSession} disabled={saved || !sessionTitle.trim()}>
                    {saved ? t("save.saved") : t("save.button")}
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
            <button type="button" className="save-modal-close" onClick={() => setSaveDialogOpen(false)} aria-label="Close">
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
                <small>{recordingFinalizing ? (lang === "ko" ? "영상 준비 중... 기록은 영상 없이도 저장할 수 있습니다." : "Preparing video... You can still save the record without it.") : recordedVideo ? (lang === "ko" ? "체크하면 영상과 타임라인을 함께 저장합니다." : "When checked, the video and timeline are saved together.") : (lang === "ko" ? "이번 세션의 영상이 없어 기록과 타임라인만 저장됩니다." : "No video is available for this session; only the record and timeline will be saved.")}</small>
              </span>
            </label>
            <div className="save-modal-actions">
              <button type="button" className="btn btn-outline" onClick={() => setSaveDialogOpen(false)}>{lang === "ko" ? "저장하지 않기" : "Don't save"}</button>
              <button type="button" className="btn btn-primary" onClick={handleSaveSession} disabled={!sessionTitle.trim()}>{lang === "ko" ? "저장하기" : "Save session"}</button>
            </div>
          </section>
        </div>
      )}
    </main>
  );
}
