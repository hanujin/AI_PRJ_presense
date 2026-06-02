"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Activity, BrainCircuit, CalendarDays, FolderClock, MoreHorizontal, Play, SlidersHorizontal, Sparkles, TrendingDown, TrendingUp } from "lucide-react";
import { useSupabaseAuth } from "@/components/supabase-provider";
import { loadDashboardData, type DiagnosisRecord, type PresentationRecord } from "@/lib/supabase/data";

const SCENE_COLORS: Record<string, number> = {
  camera: 78,
  audience: 62,
  interview: 55,
  slides: 45,
};

function StressTrend({ records }: { records: PresentationRecord[] }) {
  if (records.length === 0) {
    return (
      <div className="state-empty">No sessions yet — finish a practice run to see your trend.</div>
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
        <span>Oldest</span>
        <span>Latest</span>
      </div>
    </div>
  );
}

export function DashboardOverview() {
  const { hasEnv, user } = useSupabaseAuth();
  const [presentationRecords, setPresentationRecords] = useState<PresentationRecord[]>([]);
  const [diagnosisHistory, setDiagnosisHistory] = useState<DiagnosisRecord[]>([]);
  const [isLoading, setIsLoading] = useState(hasEnv);
  const [loadError, setLoadError] = useState("");

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
  const latestDiagnosis = diagnosisHistory[0]?.detail ?? "No sessions yet";
  const latestFocus = diagnosisHistory.find((d) => d.label.toLowerCase().includes("focus"))?.value ?? "—";

  const sceneTotals = Object.entries(SCENE_COLORS);
  const maxScene = Math.max(...sceneTotals.map(([, v]) => v));

  function severityClass(result: string) {
    const r = result.toLowerCase();
    if (r.includes("high") || r.includes("noticeable")) return "sev-high";
    if (r.includes("moderate") || r.includes("mid")) return "sev-medium";
    return "sev-low";
  }

  return (
    <main className="page-content">
      <div className="page-top">
        <div>
          <h1 className="page-title">System Overview</h1>
          <p className="page-subtitle">Real-time intelligence from your presentation practice.</p>
        </div>
        <div className="page-actions">
          <Link href="/practice" className="btn btn-primary">
            <Play size={14} />
            Start Practice
          </Link>
        </div>
      </div>

      {/* KPI Row */}
      <div className="kpi-row">
        <div className="kpi-card">
          <div className="kpi-top">
            <div className="kpi-icon"><FolderClock size={16} /></div>
            <span className={`kpi-badge neutral`}>
              {hasEnv ? "Synced" : "Local"}
            </span>
          </div>
          <div className="kpi-value">{totalSessions}</div>
          <div className="kpi-label">Total Sessions</div>
          <div className="kpi-sub">{hasEnv ? "Saved to your account" : "Add Supabase to sync"}</div>
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
          <div className="kpi-label">Latest Avg Stress</div>
          <div className="kpi-sub">Score out of 1.00</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-top">
            <div className="kpi-icon" style={{ background: "rgba(245,158,11,0.1)", color: "var(--accent-warn)" }}>
              <Activity size={16} />
            </div>
            <span className="kpi-badge neutral">Latest</span>
          </div>
          <div className="kpi-value" style={{ fontSize: 18, marginBottom: 6, lineHeight: 1.3 }}>{latestFocus}</div>
          <div className="kpi-label">Next Focus Area</div>
          <div className="kpi-sub">From latest diagnosis</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-top">
            <div className="kpi-icon" style={{ background: "rgba(139,92,246,0.1)", color: "#7c3aed" }}>
              <Sparkles size={16} />
            </div>
          </div>
          <div className="kpi-value" style={{ fontSize: 18, marginBottom: 6, lineHeight: 1.3, overflow: "hidden", display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical" as const }}>{latestDiagnosis.slice(0, 40)}{latestDiagnosis.length > 40 ? "…" : ""}</div>
          <div className="kpi-label">Latest Diagnosis</div>
          <div className="kpi-sub">AI coaching summary</div>
        </div>
      </div>

      {/* Main grid: trend chart + scene breakdown */}
      <div className="dash-grid">
        <div className="card">
          <div className="panel-head">
            <div className="panel-head-left">
              <span className="panel-title">Stress Trend Over Time</span>
              <span className="panel-subtitle">Average stress score per session</span>
            </div>
            <div className="chart-legend">
              <div className="legend-item">
                <div className="legend-dot" style={{ background: "var(--accent)" }} />
                Stress score
              </div>
            </div>
          </div>

          {isLoading ? (
            <p className="state-loading">Loading sessions...</p>
          ) : loadError ? (
            <p className="error-msg">{loadError}</p>
          ) : (
            <StressTrend records={presentationRecords} />
          )}
        </div>

        <div className="card">
          <div className="panel-head">
            <div className="panel-head-left">
              <span className="panel-title">Sessions by Scene</span>
              <span className="panel-subtitle">Practice mode breakdown</span>
            </div>
          </div>

          {sceneTotals.map(([scene, value]) => (
            <div key={scene} className="asset-row">
              <div className="asset-info">
                <div className="asset-name" style={{ textTransform: "capitalize" }}>{scene}</div>
                <div className="asset-bar-track">
                  <div className="asset-bar-fill" style={{ width: `${Math.round((value / maxScene) * 100)}%` }} />
                </div>
              </div>
              <div className="asset-value">{value} min</div>
            </div>
          ))}
        </div>
      </div>

      {/* Recent sessions table */}
      <div className="card" style={{ padding: 0 }}>
        <div style={{ padding: "18px 20px 14px", borderBottom: "1px solid var(--border)" }}>
          <div className="panel-head" style={{ marginBottom: 0 }}>
            <div className="panel-head-left">
              <span className="panel-title">Recent Sessions</span>
              <span className="panel-subtitle">Session history &amp; results</span>
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
                <th>Date</th>
                <th>Title</th>
                <th>Duration</th>
                <th>Avg Stress</th>
                <th>Result</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {isLoading ? (
                <tr><td colSpan={6} style={{ padding: "20px 16px", color: "var(--text-secondary)" }}>Loading...</td></tr>
              ) : presentationRecords.length === 0 ? (
                <tr><td colSpan={6} style={{ padding: "20px 16px", color: "var(--text-secondary)" }}>No sessions yet. Finish a practice run to see it here.</td></tr>
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
                      <button type="button" className="table-action">
                        <CalendarDays size={12} style={{ display: "inline", marginRight: 4 }} />
                        Review
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
  );
}
