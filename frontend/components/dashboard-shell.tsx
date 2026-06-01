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
  Volume2,
  Waves,
} from "lucide-react";
import { agentStarters, initialTimeline, postSessionHighlights } from "@/lib/dashboard-data";

type SessionState = "idle" | "starting" | "live" | "ended" | "error";

type ChatMessage = {
  role: "agent" | "user";
  text: string;
};

type ReportSummary = {
  peak: string;
  averageStress: string;
  speakingState: string;
  nextAction: string;
  confidence: string;
};

function formatSeconds(seconds: number) {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

function buildAgentReply(input: string, report: ReportSummary | null) {
  const text = input.toLowerCase();

  if (text.includes("stress") || text.includes("tense")) {
    return `The strongest pressure point appears around ${report?.peak ?? "the later middle section"}. I would focus on slowing your explanation pace and adding a short pause before key terms.`;
  }

  if (text.includes("calm") || text.includes("confidence")) {
    return `Your current summary suggests that steadier pacing will help most. Keep sentence lengths shorter and let one point land before moving to the next.`;
  }

  if (text.includes("improve") || text.includes("next")) {
    return report?.nextAction ?? "For the next rehearsal, focus on clearer transitions and a slightly slower delivery during dense sections.";
  }

  return "Based on this session, I would review the high-pressure explanation segment first, then rehearse one calmer transition into your technical content.";
}

function StressChart({ series }: { series: number[] }) {
  const maxValue = Math.max(...series, 1);

  return (
    <div className="chart-shell">
      <div className="chart-grid" />
      <div className="chart-area">
        {series.map((value, index) => {
          const height = `${Math.round((value / maxValue) * 100)}%`;
          return <div key={`${value}-${index}`} className="chart-bar" style={{ height }} />;
        })}
      </div>
      <div className="chart-labels">
        <span>Opening</span>
        <span>Build-up</span>
        <span>Core content</span>
        <span>Conclusion</span>
      </div>
    </div>
  );
}

export function DashboardShell() {
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
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      role: "agent",
      text: "After the session ends, I will summarize your performance and answer follow-up questions here.",
    },
  ]);

  useEffect(() => {
    return () => {
      cleanupMedia();
    };
  }, []);

  function cleanupMedia() {
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }

    if (tickRef.current) {
      window.clearInterval(tickRef.current);
      tickRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (audioContextRef.current) {
      void audioContextRef.current.close();
      audioContextRef.current = null;
    }

    analyserRef.current = null;
  }

  function startAudioMeter(stream: MediaStream) {
    const context = new window.AudioContext();
    const analyser = context.createAnalyser();
    analyser.fftSize = 512;

    const source = context.createMediaStreamSource(stream);
    source.connect(analyser);

    const data = new Uint8Array(analyser.frequencyBinCount);
    audioContextRef.current = context;
    analyserRef.current = analyser;

    const sample = () => {
      analyser.getByteFrequencyData(data);
      const average = data.reduce((sum, value) => sum + value, 0) / data.length;
      setAudioLevel(Math.min(1, average / 120));
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
      setChatMessages([
        {
          role: "agent",
          text: "I am waiting for the session to finish. When you end it, I will help interpret the results.",
        },
      ]);

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
        audio: true,
      });

      streamRef.current = stream;
      startAudioMeter(stream);
      setSessionState("live");

      tickRef.current = window.setInterval(() => {
        setElapsed((current) => current + 1);
        setStressScore((current) => Math.max(0.16, Math.min(0.92, current + (Math.random() - 0.48) * 0.04)));
        setModelConfidence(0.74 + Math.random() * 0.18);
      }, 1500);
    } catch {
      cleanupMedia();
      setSessionState("error");
      setErrorMessage("Camera or microphone permission was denied, so the session could not start.");
    }
  }

  function handleStopSession() {
    cleanupMedia();
    setSessionState("ended");

    const generatedTimeline = Array.from({ length: 12 }, (_, index) => {
      const base = 32 + index * 2;
      const variance = Math.round(Math.random() * 18);
      return Math.min(86, base + variance);
    });

    const average = generatedTimeline.reduce((sum, value) => sum + value, 0) / generatedTimeline.length / 100;
    const peak = Math.max(...generatedTimeline);

    setTimeline(generatedTimeline);

    const nextReport = {
      peak: `${peak}% around the technical middle section`,
      averageStress: `${average.toFixed(2)} / 1.00`,
      speakingState: peak > 70 ? "Noticeable mid-session pressure" : "Mostly stable delivery",
      nextAction: "Practice slower transitions between your main explanation blocks and leave a short pause after each key term.",
      confidence: `${Math.round(modelConfidence * 100)}%`,
    };

    setReport(nextReport);
    setChatMessages([
      {
        role: "agent",
        text: `Your session summary is ready. The biggest improvement area is this: ${nextReport.nextAction}`,
      },
    ]);
  }

  function handleSendMessage() {
    if (!chatInput.trim() || sessionState !== "ended") {
      return;
    }

    const userMessage = chatInput.trim();
    const reply = buildAgentReply(userMessage, report);

    setChatMessages((prev) => [
      ...prev,
      { role: "user", text: userMessage },
      { role: "agent", text: reply },
    ]);
    setChatInput("");
  }

  const audioPercent = Math.round(audioLevel * 100);
  const confidencePercent = Math.round(modelConfidence * 100);

  return (
    <main className="page-shell">
      <section className="hero-card">
        <div className="hero-copy-block">
          <p className="eyebrow">PreSense Practice Service</p>
          <h1>Quiet during practice, helpful after the session ends</h1>
          <p className="hero-copy">
            This UI is now designed like a real service. During the presentation, it stays calm and
            minimally distracting. After the user presses `End Session`, it reveals a full diagnosis,
            a summary timeline, and an AI Agent chat area for follow-up questions.
          </p>
          <div className="hero-actions">
            <button
              type="button"
              className="primary-button"
              onClick={handleStartSession}
              disabled={sessionState === "starting" || sessionState === "live"}
            >
              <Play size={16} />
              Start Session
            </button>
            <button
              type="button"
              className="secondary-button"
              onClick={handleStopSession}
              disabled={sessionState !== "live"}
            >
              <CircleStop size={16} />
              End Session
            </button>
          </div>
          <div className="hero-badges">
            <span>
              <Camera size={16} />
              Camera active in background
            </span>
            <span>
              <Mic size={16} />
              Voice input active
            </span>
            <span>
              <BrainCircuit size={16} />
              Frontend ready for model API
            </span>
          </div>
          {errorMessage ? <p className="error-text">{errorMessage}</p> : null}
        </div>

        <div className="session-rail">
          <div className={`session-status ${sessionState}`}>
            <span className="live-dot" />
            {sessionState === "live"
              ? "Practice in progress"
              : sessionState === "ended"
                ? "Session completed"
                : sessionState === "starting"
                  ? "Starting session"
                  : sessionState === "error"
                    ? "Permission needed"
                    : "Ready to practice"}
          </div>

          <div className="video-preview calm-preview">
            <div className="preview-grid" />
            <div className="preview-center preview-poster">
              <div className="poster-badge">Placeholder Scene</div>
              <h2>Practice Focus Screen</h2>
              <p>
                Replace this area later with your own image, prompt card, or presentation practice
                visual instead of showing the user&apos;s live face on screen.
              </p>
            </div>
          </div>

          <div className="signal-strip">
            <div className="signal-pill">
              <span>Camera Permission</span>
              <strong className={sessionState === "live" || sessionState === "ended" ? "tone-ok" : "tone-data"}>
                {sessionState === "live" || sessionState === "ended" ? "Granted" : "Standby"}
              </strong>
            </div>
            <div className="signal-pill">
              <span>Microphone Activity</span>
              <strong className={audioPercent > 8 ? "tone-ok" : "tone-data"}>{audioPercent}%</strong>
            </div>
            <div className="signal-pill">
              <span>Elapsed Time</span>
              <strong className="tone-data">{formatSeconds(elapsed)}</strong>
            </div>
            <div className="signal-pill">
              <span>Model Confidence Slot</span>
              <strong className="tone-ok">{confidencePercent}%</strong>
            </div>
          </div>
        </div>
      </section>

      <section className="kpi-grid">
        <article className="panel">
          <div className="panel-heading">
            <span>Session Mode</span>
            <Gauge size={18} />
          </div>
          <div className="metric-line">
            <strong>{sessionState === "live" ? "LIVE" : sessionState === "ended" ? "DONE" : "READY"}</strong>
            <span>state</span>
          </div>
          <p className="delta delta-data">The interface stays quiet while the presenter is speaking.</p>
        </article>

        <article className="panel">
          <div className="panel-heading">
            <span>Voice Activity</span>
            <Volume2 size={18} />
          </div>
          <div className="metric-line">
            <strong>{audioPercent}</strong>
            <span>%</span>
          </div>
          <p className="delta delta-ok">Live microphone meter only, without distracting coaching updates.</p>
        </article>

        <article className="panel">
          <div className="panel-heading">
            <span>Report Status</span>
            <BrainCircuit size={18} />
          </div>
          <div className="metric-line">
            <strong>{sessionState === "ended" ? "READY" : "WAIT"}</strong>
            <span>summary</span>
          </div>
          <p className="delta delta-data">Detailed diagnosis appears only after the session is finished.</p>
        </article>
      </section>

      {sessionState === "ended" ? (
        <section className="dashboard-grid">
          <article className="panel panel-chart">
            <div className="panel-heading">
              <span>Session Timeline</span>
              <Waves size={18} />
            </div>
            <div className="panel-title-row">
              <h2>Post-session stress overview</h2>
              <span className="live-chip">Generated after finish</span>
            </div>
            <p className="section-standalone">
              Instead of moving every second, the timeline is revealed once the rehearsal ends so
              the user can review the entire presentation calmly.
            </p>
            <StressChart series={timeline} />
          </article>

          <aside className="panel panel-diagnosis">
            <div className="panel-heading">
              <span>Diagnosis Result</span>
              <Sparkles size={18} />
            </div>
            <div className="coach-stack">
              <article className="coach-card coach-alert">
                <h3>Overall Judgment</h3>
                <p>{report?.speakingState}</p>
              </article>
              <article className="coach-card coach-data">
                <h3>Peak Pressure Point</h3>
                <p>{report?.peak}</p>
              </article>
              <article className="coach-card coach-ok">
                <h3>Recommended Next Step</h3>
                <p>{report?.nextAction}</p>
              </article>
            </div>
          </aside>

          <article className="panel panel-report">
            <div className="panel-heading">
              <span>Session Summary</span>
              <BrainCircuit size={18} />
            </div>
            <div className="section-copy">
              <h2>What to improve before the next rehearsal</h2>
              <p>
                The report is intentionally shown after completion so the presenter can focus first
                on delivering the talk, then on reviewing the results.
              </p>
            </div>
            <div className="report-grid">
              <article className="report-card">
                <span>Average Stress</span>
                <strong>{report?.averageStress}</strong>
                <p>This gives a session-level picture instead of distracting you with momentary noise.</p>
              </article>
              <article className="report-card">
                <span>Model Confidence</span>
                <strong>{report?.confidence}</strong>
                <p>Reserved for the current student model output once the backend inference route is attached.</p>
              </article>
              <article className="report-card">
                <span>Next Rehearsal Goal</span>
                <strong>Slow transitions</strong>
                <p>{report?.nextAction}</p>
              </article>
            </div>
            <div className="highlight-list">
              {postSessionHighlights.map((item) => (
                <article key={item.title} className="highlight-card">
                  <h3>{item.title}</h3>
                  <p>{item.detail}</p>
                </article>
              ))}
            </div>
          </article>

          <article className="panel panel-agent">
            <div className="panel-heading">
              <span>AI Agent</span>
              <MessageSquareText size={18} />
            </div>
            <div className="section-copy">
              <h2>Ask follow-up questions about this result</h2>
              <p>
                This is frontend-only for now. The chat UI is ready, and later you can connect it to
                a real agent once you have the API key and backend route.
              </p>
            </div>
            <div className="starter-row">
              {agentStarters.map((starter) => (
                <button key={starter} type="button" className="starter-pill" onClick={() => setChatInput(starter)}>
                  {starter}
                </button>
              ))}
            </div>
            <div className="chat-shell">
              {chatMessages.map((message, index) => (
                <div key={`${message.role}-${index}`} className={`chat-bubble ${message.role}`}>
                  {message.text}
                </div>
              ))}
            </div>
            <div className="chat-input-row">
              <input
                value={chatInput}
                onChange={(event) => setChatInput(event.target.value)}
                placeholder="Ask the AI Agent about your presentation result"
              />
              <button type="button" className="primary-button" onClick={handleSendMessage}>
                Ask
              </button>
            </div>
          </article>
        </section>
      ) : null}
    </main>
  );
}
