"use client";

import { useEffect, useRef, useState } from "react";
import {
  BrainCircuit,
  Camera,
  CircleStop,
  Gauge,
  MessageSquareText,
  Mic,
  Play,
  Sparkles,
  Video,
  Volume2,
  Waves,
} from "lucide-react";
import { useSupabaseAuth } from "@/components/supabase-provider";
import { agentStarters, initialTimeline, postSessionHighlights, practiceScenes } from "@/lib/dashboard-data";
import { savePracticeSession } from "@/lib/supabase/data";

type SessionState = "idle" | "starting" | "live" | "ended" | "error";
type ChatMessage = { role: "agent" | "user"; text: string };
type ReportSummary = {
  peak: string;
  averageStress: string;
  speakingState: string;
  nextAction: string;
  confidence: string;
};
type SceneId = (typeof practiceScenes)[number]["id"];

function formatSeconds(s: number) {
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}

function buildAgentReply(input: string, report: ReportSummary | null) {
  const t = input.toLowerCase();
  if (t.includes("stress") || t.includes("tense"))
    return `The strongest pressure point appears around ${report?.peak ?? "the later middle section"}. Focus on slowing your pace and adding a short pause before key terms.`;
  if (t.includes("calm") || t.includes("confidence"))
    return `Steadier pacing will help most. Keep sentence lengths shorter and let one point land before moving to the next.`;
  if (t.includes("improve") || t.includes("next"))
    return report?.nextAction ?? "Focus on clearer transitions and a slightly slower delivery during dense sections.";
  return "Review the high-pressure explanation segment first, then rehearse one calmer transition into your technical content.";
}

function StressChart({ series }: { series: number[] }) {
  const max = Math.max(...series, 1);
  return (
    <div>
      <div
        className="stress-chart-wrap"
        style={{ gridTemplateColumns: `repeat(${series.length}, minmax(0, 1fr))` }}
      >
        {series.map((v, i) => (
          <div key={i} className="stress-bar" style={{ height: `${Math.round((v / max) * 100)}%` }} />
        ))}
      </div>
      <div className="bar-chart-labels">
        <span>Opening</span>
        <span>Build-up</span>
        <span>Core content</span>
        <span>Conclusion</span>
      </div>
    </div>
  );
}

export function DashboardShell() {
  const { hasEnv, user } = useSupabaseAuth();
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const rafRef = useRef<number | null>(null);
  const tickRef = useRef<number | null>(null);

  const [sessionState, setSessionState] = useState<SessionState>("idle");
  const [errorMessage, setErrorMessage] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const [audioLevel, setAudioLevel] = useState(0);
  const [stressScore, setStressScore] = useState(0.38);
  const [modelConfidence, setModelConfidence] = useState(0.84);
  const [timeline, setTimeline] = useState(initialTimeline);
  const [report, setReport] = useState<ReportSummary | null>(null);
  const [chatInput, setChatInput] = useState("");
  const [selectedScene, setSelectedScene] = useState<SceneId>("camera");
  const [saveStatus, setSaveStatus] = useState("");
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    { role: "agent", text: "After the session ends, I will summarize your performance and answer follow-up questions here." },
  ]);

  useEffect(() => () => { cleanupMedia(); }, []);

  function cleanupMedia() {
    if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null; }
    if (tickRef.current) { window.clearInterval(tickRef.current); tickRef.current = null; }
    if (streamRef.current) { streamRef.current.getTracks().forEach((t) => t.stop()); streamRef.current = null; }
    if (videoRef.current) videoRef.current.srcObject = null;
    if (audioContextRef.current) { void audioContextRef.current.close(); audioContextRef.current = null; }
    analyserRef.current = null;
  }

  function startAudioMeter(stream: MediaStream) {
    const ctx = new window.AudioContext();
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 512;
    const source = ctx.createMediaStreamSource(stream);
    source.connect(analyser);
    const data = new Uint8Array(analyser.frequencyBinCount);
    audioContextRef.current = ctx;
    analyserRef.current = analyser;

    const sample = () => {
      analyser.getByteFrequencyData(data);
      const avg = data.reduce((s, v) => s + v, 0) / data.length;
      setAudioLevel(Math.min(1, avg / 120));
      rafRef.current = requestAnimationFrame(sample);
    };
    sample();
  }

  async function handleStartSession() {
    try {
      setSessionState("starting");
      setErrorMessage("");
      setReport(null);
      setElapsed(0);
      setTimeline(initialTimeline);
      setStressScore(0.38);
      setModelConfidence(0.84);
      setSaveStatus("");
      setChatMessages([{ role: "agent", text: "Waiting for the session to finish. I will help interpret the results when you end it." }]);

      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: true });
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      startAudioMeter(stream);
      setSessionState("live");

      tickRef.current = window.setInterval(() => {
        setElapsed((c) => c + 1);
        setStressScore((c) => Math.max(0.16, Math.min(0.92, c + (Math.random() - 0.48) * 0.04)));
        setModelConfidence(0.74 + Math.random() * 0.18);
      }, 1500);
    } catch {
      cleanupMedia();
      setSessionState("error");
      setErrorMessage("Camera or microphone permission was denied.");
    }
  }

  async function handleStopSession() {
    cleanupMedia();
    setSessionState("ended");

    const gen = Array.from({ length: 12 }, (_, i) => Math.min(86, 32 + i * 2 + Math.round(Math.random() * 18)));
    const avg = gen.reduce((s, v) => s + v, 0) / gen.length / 100;
    const peak = Math.max(...gen);
    setTimeline(gen);

    const nextReport: ReportSummary = {
      peak: `${peak}% around the technical middle section`,
      averageStress: `${avg.toFixed(2)} / 1.00`,
      speakingState: peak > 70 ? "Noticeable mid-session pressure" : "Mostly stable delivery",
      nextAction: "Practice slower transitions between your main explanation blocks and leave a short pause after each key term.",
      confidence: `${Math.round(modelConfidence * 100)}%`,
    };

    setReport(nextReport);
    setChatMessages([{ role: "agent", text: `Session summary ready. Biggest improvement area: ${nextReport.nextAction}` }]);

    if (!hasEnv || !user) { setSaveStatus("Supabase not connected — session saved locally only."); return; }

    try {
      await savePracticeSession({
        durationSeconds: elapsed,
        averageStress: avg,
        result: nextReport.speakingState,
        diagnosis: `Peak pressure reached ${nextReport.peak}.`,
        nextAction: nextReport.nextAction,
        sceneLabel: activeScene.label,
      });
      setSaveStatus("Session saved to your account.");
    } catch (error) {
      setSaveStatus(error instanceof Error ? error.message : "Failed to save session.");
    }
  }

  function handleSendMessage() {
    if (!chatInput.trim() || sessionState !== "ended") return;
    const userText = chatInput.trim();
    const reply = buildAgentReply(userText, report);
    setChatMessages((prev) => [...prev, { role: "user", text: userText }, { role: "agent", text: reply }]);
    setChatInput("");
  }

  const audioPercent = Math.round(audioLevel * 100);
  const confidencePercent = Math.round(modelConfidence * 100);
  const activeScene = practiceScenes.find((s) => s.id === selectedScene) ?? practiceScenes[0];

  const stateLabel =
    sessionState === "live" ? "Live — Practice in progress" :
    sessionState === "ended" ? "Session completed" :
    sessionState === "starting" ? "Starting…" :
    sessionState === "error" ? "Permission needed" :
    "Ready to start";

  const stateChipClass =
    sessionState === "live" ? "chip-live" :
    sessionState === "ended" ? "chip-ended" :
    sessionState === "error" ? "chip-error" :
    "chip-idle";

  return (
    <main className="page-content">
      <div className="page-top">
        <div>
          <h1 className="page-title">Practice Session</h1>
          <p className="page-subtitle">Quiet during practice — full analysis revealed after you finish.</p>
        </div>
        <div className="page-actions">
          <button
            type="button"
            className="btn btn-outline"
            onClick={handleStopSession}
            disabled={sessionState !== "live"}
          >
            <CircleStop size={14} />
            End Session
          </button>
          <button
            type="button"
            className="btn btn-primary"
            onClick={handleStartSession}
            disabled={sessionState === "starting" || sessionState === "live"}
          >
            <Play size={14} />
            Start Session
          </button>
        </div>
      </div>

      {/* Session layout: video + controls */}
      <div className="session-grid" style={{ marginBottom: 14 }}>
        {/* Left: video + scene chips */}
        <div>
          <div className="video-box">
            {selectedScene === "camera" ? (
              <>
                <video ref={videoRef} autoPlay muted playsInline />
                <div className="video-overlay">
                  <span className={`status-chip ${stateChipClass}`}>
                    <span className="status-dot" />
                    {stateLabel}
                  </span>
                  <span style={{ fontSize: 12, opacity: 0.8 }}>{formatSeconds(elapsed)}</span>
                </div>
              </>
            ) : (
              <div className="video-placeholder">
                <Video size={40} />
                <div style={{ fontWeight: 600, fontSize: 16 }}>{activeScene.label}</div>
                <div style={{ fontSize: 13, textAlign: "center", maxWidth: "30ch" }}>{activeScene.description}</div>
                <div className="video-overlay">
                  <span className={`status-chip ${stateChipClass}`}>
                    <span className="status-dot" />
                    {stateLabel}
                  </span>
                  <span style={{ fontSize: 12, opacity: 0.8 }}>{formatSeconds(elapsed)}</span>
                </div>
              </div>
            )}
          </div>

          <div className="scene-chips">
            {practiceScenes.map((scene) => (
              <button
                key={scene.id}
                type="button"
                className={`scene-chip${selectedScene === scene.id ? " active" : ""}`}
                onClick={() => setSelectedScene(scene.id)}
              >
                {scene.label}
              </button>
            ))}
          </div>

          {errorMessage && <p className="error-msg" style={{ marginTop: 10 }}>{errorMessage}</p>}
          {saveStatus && <p className="success-msg" style={{ marginTop: 10 }}>{saveStatus}</p>}
        </div>

        {/* Right: signal cards */}
        <div className="signal-grid" style={{ alignContent: "start" }}>
          <div className="signal-card">
            <div className="signal-label">Session Mode</div>
            <div className={`signal-value ${sessionState === "live" ? "ok" : sessionState === "error" ? "alert" : "data"}`}>
              {sessionState === "live" ? "LIVE" : sessionState === "ended" ? "DONE" : "READY"}
            </div>
          </div>

          <div className="signal-card">
            <div className="signal-label">Elapsed Time</div>
            <div className="signal-value data">{formatSeconds(elapsed)}</div>
          </div>

          <div className="signal-card">
            <div className="signal-label">
              <Mic size={11} style={{ display: "inline", marginRight: 4 }} />
              Mic Activity
            </div>
            <div className={`signal-value ${audioPercent > 8 ? "ok" : "data"}`}>{audioPercent}%</div>
          </div>

          <div className="signal-card">
            <div className="signal-label">
              <Camera size={11} style={{ display: "inline", marginRight: 4 }} />
              Camera
            </div>
            <div className={`signal-value ${sessionState === "live" || sessionState === "ended" ? "ok" : "data"}`}>
              {sessionState === "live" || sessionState === "ended" ? "ON" : "STBY"}
            </div>
          </div>

          <div className="signal-card">
            <div className="signal-label">
              <BrainCircuit size={11} style={{ display: "inline", marginRight: 4 }} />
              Model Confidence
            </div>
            <div className="signal-value ok">{sessionState === "live" ? `${confidencePercent}%` : "—"}</div>
          </div>

          <div className="signal-card">
            <div className="signal-label">
              <Gauge size={11} style={{ display: "inline", marginRight: 4 }} />
              Live Stress
            </div>
            <div className={`signal-value ${sessionState === "live" ? (stressScore > 0.65 ? "alert" : "ok") : "data"}`}>
              {sessionState === "live" ? stressScore.toFixed(2) : "—"}
            </div>
          </div>
        </div>
      </div>

      {/* Post-session analysis */}
      {sessionState === "ended" && (
        <>
          {/* Stress timeline + diagnosis */}
          <div className="dash-grid" style={{ marginBottom: 14 }}>
            <div className="card">
              <div className="panel-head">
                <div className="panel-head-left">
                  <span className="panel-title">Session Stress Timeline</span>
                  <span className="panel-subtitle">Post-session overview</span>
                </div>
                <Waves size={16} color="var(--text-secondary)" />
              </div>
              <StressChart series={timeline} />
            </div>

            <div className="card">
              <div className="panel-head">
                <div className="panel-head-left">
                  <span className="panel-title">Diagnosis</span>
                  <span className="panel-subtitle">AI analysis result</span>
                </div>
                <Sparkles size={16} color="var(--text-secondary)" />
              </div>
              <div className="coach-stack">
                <div className="coach-card coach-alert">
                  <h3>Overall Judgment</h3>
                  <p>{report?.speakingState}</p>
                </div>
                <div className="coach-card coach-data">
                  <h3>Peak Pressure Point</h3>
                  <p>{report?.peak}</p>
                </div>
                <div className="coach-card coach-ok">
                  <h3>Recommended Next Step</h3>
                  <p>{report?.nextAction}</p>
                </div>
              </div>
            </div>
          </div>

          {/* Summary + AI chat */}
          <div className="dash-grid">
            <div className="card">
              <div className="panel-head">
                <div className="panel-head-left">
                  <span className="panel-title">Session Summary</span>
                  <span className="panel-subtitle">What to improve next rehearsal</span>
                </div>
                <BrainCircuit size={16} color="var(--text-secondary)" />
              </div>

              <div className="report-grid">
                <div className="report-card">
                  <div className="rc-label">Average Stress</div>
                  <div className="rc-value">{report?.averageStress}</div>
                  <div className="rc-desc">Session-level picture without momentary noise.</div>
                </div>
                <div className="report-card">
                  <div className="rc-label">Model Confidence</div>
                  <div className="rc-value">{report?.confidence}</div>
                  <div className="rc-desc">Student model output — backend inference ready.</div>
                </div>
                <div className="report-card">
                  <div className="rc-label">Next Goal</div>
                  <div className="rc-value" style={{ fontSize: 15, lineHeight: 1.3 }}>Slow transitions</div>
                  <div className="rc-desc">{report?.nextAction}</div>
                </div>
                <div className="report-card">
                  <div className="rc-label">Scene Used</div>
                  <div className="rc-value" style={{ fontSize: 15, lineHeight: 1.3 }}>{activeScene.label}</div>
                  <div className="rc-desc">Practice environment for this session.</div>
                </div>
              </div>

              <div className="highlight-list">
                {postSessionHighlights.map((item) => (
                  <div key={item.title} className="highlight-card">
                    <h3>{item.title}</h3>
                    <p>{item.detail}</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="card">
              <div className="panel-head">
                <div className="panel-head-left">
                  <span className="panel-title">AI Agent</span>
                  <span className="panel-subtitle">Ask follow-up questions</span>
                </div>
                <MessageSquareText size={16} color="var(--text-secondary)" />
              </div>

              <div className="starter-pills">
                {agentStarters.map((s) => (
                  <button key={s} type="button" className="starter-pill" onClick={() => setChatInput(s)}>
                    {s}
                  </button>
                ))}
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
                  placeholder="Ask about your session result…"
                />
                <button type="button" className="btn btn-primary" onClick={handleSendMessage}>
                  Ask
                </button>
              </div>
            </div>
          </div>
        </>
      )}
    </main>
  );
}
