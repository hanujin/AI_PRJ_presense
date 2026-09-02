"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Activity, BrainCircuit, CalendarDays, Clock, FolderClock, MoreHorizontal, Play, SlidersHorizontal, Sparkles, TrendingDown, TrendingUp, X } from "lucide-react";
import { useSupabaseAuth } from "@/components/supabase-provider";
import { useLang } from "@/lib/i18n";
import { loadDashboardData, type DiagnosisRecord, type PresentationRecord } from "@/lib/supabase/data";

type Translate = (key: string, fallback?: string) => string;

function localizeDiagnosis(value: string, t: Translate) {
  const fixedLabels: Record<string, string> = {
    "Latest session result": "diag.latestSessionResult",
    "Average stress": "diag.averageStress",
    "Next coaching focus": "diag.nextCoachingFocus",
    "Generated from the most recent practice session.": "diag.generatedFromSession",
    "Mostly stable delivery": "diag.stableDelivery",
    "Practice slower transitions between your main explanation blocks and leave a short pause after each key term.": "diag.nextAction",
  };
  const fixedKey = fixedLabels[value];
  if (fixedKey) return t(fixedKey);

  const phaseKey = (phase: string) => `diag.phase.${phase.toLowerCase()}`;
  const peakMatch = value.match(/^Peak pressure reached (\d+(?:\.\d+)?)% around the (Open|Build|Core|Close) phase\. Deck: (.+)\.$/);
  if (peakMatch) {
    const [, stress, phase, deck] = peakMatch;
    return t("diag.peakPressure")
      .replace("{value}", stress)
      .replace("{phase}", t(phaseKey(phase)))
      .replace("{deck}", deck === "No deck used" ? t("diag.noDeck") : deck);
  }

  const pressureMatch = value.match(/^Noticeable (Open|Build|Core|Close)-section pressure$/);
  if (pressureMatch) return t("diag.noticeablePressure").replace("{phase}", t(phaseKey(pressureMatch[1])));

  return value;
}

function StressTrend({ records }: { records: PresentationRecord[] }) {
  const { t } = useLang();
  if (records.length === 0) {
    return (
      <div className="state-empty">{t("state.noSessionsTrend")}</div>
    );
  }

  const values = records
    .slice()
    .reverse()
    .map((r) => parseFloat(r.stressAverage));

  const max = Math.max(...values, 0.01);

  return (
    <div>
      <div
        className="stress-chart-wrap"
        style={{ gridTemplateColumns: `repeat(${values.length}, minmax(0, 1fr))` }}
      >
        {values.map((v, i) => (
          <div
            key={i}
            className="stress-bar"
            style={{ height: `${Math.round((v / max) * 100)}%` }}
            title={`Session ${i + 1}: ${v.toFixed(2)}`}
          />
        ))}
      </div>
      <div className="bar-chart-labels">
        <span>{t("chart.oldest")}</span>
        <span>{t("chart.latest")}</span>
      </div>
    </div>
  );
}

function StressBar({ percent, color }: { percent: number; color: string }) {
  return (
    <div style={{ background: "var(--border)", borderRadius: 4, height: 8, overflow: "hidden" }}>
      <div style={{ height: "100%", borderRadius: 4, width: `${percent}%`, background: color, transition: "width 0.5s ease" }} />
    </div>
  );
}

function DiagnosisModal({ diagnosis, onClose }: { diagnosis: DiagnosisRecord; onClose: () => void }) {
  const { t } = useLang();

  return (
    <div className="modal-backdrop" onClick={onClose} role="presentation">
      <div className="card diagnosis-modal" role="dialog" aria-modal="true" aria-labelledby="diagnosis-modal-title" onClick={(e) => e.stopPropagation()}>
        <button type="button" className="modal-close" onClick={onClose} aria-label="Close diagnosis details">
          <X size={18} />
        </button>
        <div className="diagnosis-modal-eyebrow">{t("kpi.latestDiagnosis")}</div>
        <h2 id="diagnosis-modal-title" className="diagnosis-modal-title">{localizeDiagnosis(diagnosis.label, t)}</h2>
        {diagnosis.value && <div className="diagnosis-modal-value">{localizeDiagnosis(diagnosis.value, t)}</div>}
        <div className="diagnosis-modal-detail">{localizeDiagnosis(diagnosis.detail, t)}</div>
      </div>
    </div>
  );
}

function ReviewModal({ record, onClose, severityClass }: { record: PresentationRecord; onClose: () => void; severityClass: (r: string) => string }) {
  const { t } = useLang();
  const stressValue = parseFloat(record.stressAverage);
  const stressPercent = Math.round(stressValue * 100);
  const stressColor = stressPercent > 65 ? "#ef4444" : stressPercent > 45 ? "#f59e0b" : "#22c55e";
  const stressTier = stressPercent > 65 ? "high" : stressPercent > 45 ? "moderate" : "low";

  const simulatedBars = (() => {
    const base = stressValue;
    return [
      Math.min(1, base * 0.72),
      Math.min(1, base * 1.05),
      Math.min(1, base * 1.28),
      Math.min(1, base * 0.95),
    ];
  })();
  const barMax = Math.max(...simulatedBars, 0.01);

  return (
    <div
      style={{
        position: "fixed", inset: 0, zIndex: 50,
        background: "rgba(0,0,0,0.6)", display: "flex",
        alignItems: "center", justifyContent: "center", padding: 24,
      }}
      onClick={onClose}
    >
      <div
        className="card"
        style={{ width: "100%", maxWidth: 580, maxHeight: "92vh", overflowY: "auto", position: "relative" }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Close */}
        <button
          type="button"
          onClick={onClose}
          style={{ position: "absolute", top: 16, right: 16, background: "none", border: "none", cursor: "pointer", color: "var(--text-secondary)", display: "flex" }}
        >
          <X size={18} />
        </button>

        {/* Header */}
        <div style={{ marginBottom: 20 }}>
          <div style={{ fontSize: 11, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 4 }}>{t("modal.sessionReview")}</div>
          <div style={{ fontWeight: 700, fontSize: 20, marginBottom: 8 }}>{record.title}</div>
          <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
            <span style={{ fontSize: 12, color: "var(--text-secondary)", display: "flex", alignItems: "center", gap: 4 }}>
              <CalendarDays size={11} />{record.date}
            </span>
            <span style={{ fontSize: 12, color: "var(--text-secondary)", display: "flex", alignItems: "center", gap: 4 }}>
              <Clock size={11} />{record.duration}
            </span>
            {record.sceneLabel && (
              <span style={{ fontSize: 12, color: "var(--text-secondary)", display: "flex", alignItems: "center", gap: 4 }}>
                <Activity size={11} />{record.sceneLabel}
              </span>
            )}
            <span className={`severity-pill ${severityClass(record.result)}`}>{record.result}</span>
          </div>
        </div>

        {/* Stress metrics */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 16 }}>
          <div className="report-card">
            <div className="rc-label">{t("modal.avgStressScore")}</div>
            <div className="rc-value" style={{ color: stressColor }}>{record.stressAverage}</div>
            <div className="rc-desc" style={{ marginBottom: 8 }}>{t(`modal.outOf${stressTier === "high" ? "High" : stressTier === "moderate" ? "Moderate" : "Low"}`)}</div>
            <StressBar percent={stressPercent} color={stressColor} />
          </div>
          <div className="report-card">
            <div className="rc-label">{t("modal.stressPhaseBreakdown")}</div>
            <div style={{ display: "flex", alignItems: "flex-end", gap: 5, height: 52, marginTop: 8, marginBottom: 8 }}>
              {simulatedBars.map((v, i) => (
                <div key={i} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 3 }}>
                  <div style={{ width: "100%", background: stressColor, borderRadius: "3px 3px 0 0", opacity: 0.75 + i * 0.06, height: `${Math.round((v / barMax) * 48)}px` }} />
                </div>
              ))}
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "var(--text-secondary)" }}>
              <span>Open</span><span>Build</span><span>Core</span><span>Close</span>
            </div>
          </div>
        </div>

        {/* AI Diagnosis */}
        <div style={{ borderTop: "1px solid var(--border)", paddingTop: 16, marginBottom: 14 }}>
          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 10, display: "flex", alignItems: "center", gap: 6 }}>
            <Sparkles size={13} color="var(--accent)" /> {t("modal.aiDiagnosis")}
          </div>
          <div className="coach-stack" style={{ gap: 8 }}>
            <div className="coach-card coach-alert">
              <h3>{t("modal.overallJudgment")}</h3>
              <p>{localizeDiagnosis(record.result, t)}</p>
            </div>
            <div className="coach-card coach-data">
              <h3>{t("modal.peakPressureAnalysis")}</h3>
              <p>{record.diagnosis ? localizeDiagnosis(record.diagnosis, t) : t("modal.noDiagnosis")}</p>
            </div>
            {record.nextAction && (
              <div className="coach-card coach-ok">
                <h3>{t("modal.recommendedNextStep")}</h3>
                <p>{localizeDiagnosis(record.nextAction, t)}</p>
              </div>
            )}
          </div>
        </div>

        {/* Coaching tips based on stress level */}
        <div style={{ borderTop: "1px solid var(--border)", paddingTop: 16 }}>
          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 10, display: "flex", alignItems: "center", gap: 6 }}>
            <BrainCircuit size={13} color="var(--accent)" /> {t("modal.coachingInsight")}
          </div>
          <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.7, background: "var(--surface-2, var(--border))", borderRadius: 8, padding: "12px 14px" }}>
            {t(`modal.coach.${stressTier}`)}
          </div>
        </div>
      </div>
    </div>
  );
}

export function DashboardOverview() {
  const { hasEnv, user } = useSupabaseAuth();
  const { t } = useLang();
  const [presentationRecords, setPresentationRecords] = useState<PresentationRecord[]>([]);
  const [diagnosisHistory, setDiagnosisHistory] = useState<DiagnosisRecord[]>([]);
  const [isLoading, setIsLoading] = useState(hasEnv);
  const [loadError, setLoadError] = useState("");
  const [selectedRecord, setSelectedRecord] = useState<PresentationRecord | null>(null);
  const [selectedDiagnosis, setSelectedDiagnosis] = useState<DiagnosisRecord | null>(null);

  useEffect(() => {
    if (!hasEnv || !user) {
      setIsLoading(false);
      return;
    }

    let isMounted = true;

    void loadDashboardData()
      .then((data) => {
        if (!isMounted) return;
        setPresentationRecords(data.presentationRecords);
        setDiagnosisHistory(data.diagnosisHistory);
        setLoadError("");
      })
      .catch((error) => {
        if (!isMounted) return;
        setLoadError(error instanceof Error ? error.message : "Failed to load dashboard data.");
      })
      .finally(() => {
        if (isMounted) setIsLoading(false);
      });

    return () => { isMounted = false; };
  }, [hasEnv, user]);

  const totalSessions = presentationRecords.length;
  const latestAvg = parseFloat(presentationRecords[0]?.stressAverage ?? "0");
  const prevAvg = parseFloat(presentationRecords[1]?.stressAverage ?? "0");
  const stressDelta = totalSessions >= 2 ? latestAvg - prevAvg : null;
  const latestDiagnosis = diagnosisHistory[0] ? localizeDiagnosis(diagnosisHistory[0].detail, t) : "No sessions yet";

  // Current Focus Area (Progress Report §1.3.2): the practice context that needs
  // the most attention — derived from the highest-stress recent session's scene.
  const focusRecord = presentationRecords
    .filter((r) => r.sceneLabel)
    .slice()
    .sort((a, b) => parseFloat(b.stressAverage) - parseFloat(a.stressAverage))[0];
  const currentFocus =
    focusRecord?.sceneLabel ??
    diagnosisHistory.find((d) => d.label.toLowerCase().includes("focus"))?.value ??
    "—";

  // Sessions by Scene: aggregated from real saved sessions, not a fixed table.
  const sceneCounts = presentationRecords.reduce<Record<string, number>>((acc, r) => {
    const key = r.sceneLabel?.trim() || "Unlabeled";
    acc[key] = (acc[key] ?? 0) + 1;
    return acc;
  }, {});
  const sceneTotals = Object.entries(sceneCounts).sort((a, b) => b[1] - a[1]);
  const maxScene = Math.max(...sceneTotals.map(([, v]) => v), 1);

  function severityClass(result: string) {
    const r = result.toLowerCase();
    if (r.includes("high") || r.includes("noticeable")) return "sev-high";
    if (r.includes("moderate") || r.includes("mid")) return "sev-medium";
    return "sev-low";
  }

  return (
    <>
    {selectedRecord && (
      <ReviewModal
        record={selectedRecord}
        onClose={() => setSelectedRecord(null)}
        severityClass={severityClass}
      />
    )}
    {selectedDiagnosis && <DiagnosisModal diagnosis={selectedDiagnosis} onClose={() => setSelectedDiagnosis(null)} />}
    <main className="page-content">
      <div className="page-top">
        <div>
          <h1 className="page-title">{t("dash.title")}</h1>
          <p className="page-subtitle">{t("dash.subtitle")}</p>
        </div>
        <div className="page-actions">
          <Link href="/practice" className="btn btn-primary">
            <Play size={14} />
            {t("cta.startPractice")}
          </Link>
        </div>
      </div>

      {/* KPI Row */}
      <div className="kpi-row">
        <div className="kpi-card">
          <div className="kpi-top">
            <div className="kpi-icon"><FolderClock size={16} /></div>
            <span className={`kpi-badge neutral`}>
              {hasEnv ? t("badge.synced") : t("badge.local")}
            </span>
          </div>
          <div className="kpi-value">{totalSessions}</div>
          <div className="kpi-label">{t("kpi.totalSessions")}</div>
          <div className="kpi-sub">{hasEnv ? t("kpi.totalSessions.synced") : t("kpi.totalSessions.local")}</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-top">
            <div className="kpi-icon" style={{ background: "var(--accent-ok-light)", color: "var(--accent-ok)" }}>
              <BrainCircuit size={16} />
            </div>
            {stressDelta !== null && (
              <span className={`kpi-badge ${stressDelta < 0 ? "up" : "down"}`}>
                {stressDelta < 0 ? <TrendingDown size={10} /> : <TrendingUp size={10} />}
                {Math.abs(stressDelta).toFixed(2)}
              </span>
            )}
          </div>
          <div className="kpi-value">{latestAvg.toFixed(2)}</div>
          <div className="kpi-label">{t("kpi.latestAvgStress")}</div>
          <div className="kpi-sub">{t("kpi.scoreOutOf")}</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-top">
            <div className="kpi-icon" style={{ background: "rgba(245,158,11,0.1)", color: "var(--accent-warn)" }}>
              <Activity size={16} />
            </div>
            <span className="kpi-badge neutral">{t("badge.latest")}</span>
          </div>
          <div className="kpi-value" style={{ fontSize: 18, marginBottom: 6, lineHeight: 1.3 }}>{currentFocus}</div>
          <div className="kpi-label">{t("kpi.currentFocus")}</div>
          <div className="kpi-sub">{t("kpi.currentFocusSub")}</div>
        </div>

        <button
          type="button"
          className="kpi-card kpi-card-button"
          onClick={() => diagnosisHistory[0] && setSelectedDiagnosis(diagnosisHistory[0])}
          disabled={!diagnosisHistory[0]}
          aria-label={diagnosisHistory[0] ? `${t("kpi.latestDiagnosis")}: ${t("table.review")}` : t("kpi.latestDiagnosis")}
        >
          <div className="kpi-top">
            <div className="kpi-icon" style={{ background: "rgba(139,92,246,0.1)", color: "#7c3aed" }}>
              <Sparkles size={16} />
            </div>
          </div>
          <div className="kpi-value" style={{ fontSize: 18, marginBottom: 6, lineHeight: 1.3, overflow: "hidden", display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical" as const }}>{latestDiagnosis.slice(0, 40)}{latestDiagnosis.length > 40 ? "…" : ""}</div>
          <div className="kpi-label">{t("kpi.latestDiagnosis")}</div>
          <div className="kpi-sub">{t("kpi.latestDiagnosisSub")}</div>
        </button>
      </div>

      {/* Main grid: trend chart + scene breakdown */}
      <div className="dash-grid">
        <div className="card">
          <div className="panel-head">
            <div className="panel-head-left">
              <span className="panel-title">{t("panel.stressTrend")}</span>
              <span className="panel-subtitle">{t("panel.stressTrendSub")}</span>
            </div>
            <div className="chart-legend">
              <div className="legend-item">
                <div className="legend-dot" style={{ background: "var(--accent)" }} />
                {t("legend.stressScore")}
              </div>
            </div>
          </div>

          {isLoading ? (
            <p className="state-loading">{t("state.loadingSessions")}</p>
          ) : loadError ? (
            <p className="error-msg">{loadError}</p>
          ) : (
            <StressTrend records={presentationRecords} />
          )}
        </div>

        <div className="card">
          <div className="panel-head">
            <div className="panel-head-left">
              <span className="panel-title">{t("panel.sessionsByScene")}</span>
              <span className="panel-subtitle">{t("panel.sessionsBySceneSub")}</span>
            </div>
          </div>

          {isLoading ? (
            <p className="state-loading">{t("state.loadingSessions")}</p>
          ) : sceneTotals.length === 0 ? (
            <div className="state-empty">{t("state.noSessionsScene")}</div>
          ) : (
            sceneTotals.map(([scene, value]) => (
              <div key={scene} className="asset-row">
                <div className="asset-info">
                  <div className="asset-name" style={{ textTransform: "capitalize" }}>{scene}</div>
                  <div className="asset-bar-track">
                    <div className="asset-bar-fill" style={{ width: `${Math.round((value / maxScene) * 100)}%` }} />
                  </div>
                </div>
                <div className="asset-value">{value} {value === 1 ? t("unit.session") : t("unit.sessions")}</div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Recent sessions table */}
      <div className="card" style={{ padding: 0 }}>
        <div style={{ padding: "18px 20px 14px", borderBottom: "1px solid var(--border)" }}>
          <div className="panel-head" style={{ marginBottom: 0 }}>
            <div className="panel-head-left">
              <span className="panel-title">{t("panel.recentSessions")}</span>
              <span className="panel-subtitle">{t("panel.recentSessionsSub")}</span>
            </div>
            <div className="panel-actions">
              <button type="button" className="icon-btn"><SlidersHorizontal size={14} /></button>
              <button type="button" className="icon-btn"><MoreHorizontal size={14} /></button>
            </div>
          </div>
        </div>

        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>{t("table.date")}</th>
                <th>{t("table.title")}</th>
                <th>{t("table.duration")}</th>
                <th>{t("table.avgStress")}</th>
                <th>{t("table.result")}</th>
                <th>{t("table.action")}</th>
              </tr>
            </thead>
            <tbody>
              {isLoading ? (
                <tr><td colSpan={6} style={{ padding: "20px 16px", color: "var(--text-secondary)" }}>{t("table.loading")}</td></tr>
              ) : presentationRecords.length === 0 ? (
                <tr><td colSpan={6} style={{ padding: "20px 16px", color: "var(--text-secondary)" }}>{t("table.noSessions")}</td></tr>
              ) : (
                presentationRecords.map((record) => (
                  <tr key={record.id}>
                    <td className="td-mono">{record.date}</td>
                    <td style={{ fontWeight: 500 }}>{record.title}</td>
                    <td className="td-mono">{record.duration}</td>
                    <td className="td-mono" style={{ color: "var(--accent)" }}>{record.stressAverage}</td>
                    <td>
                      <span className={`severity-pill ${severityClass(record.result)}`}>
                        {record.result}
                      </span>
                    </td>
                    <td>
                      <button type="button" className="table-action" onClick={() => setSelectedRecord(record)}>
                        <CalendarDays size={12} style={{ display: "inline", marginRight: 4 }} />
                        {t("table.review")}
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </main>
    </>
  );
}
