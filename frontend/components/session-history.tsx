"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import { BrainCircuit, CalendarDays, Film, FolderClock, Gauge, LoaderCircle, Play, Sparkles, Trash2, X } from "lucide-react";
import { useLang } from "@/lib/i18n";
import { StressChart } from "@/components/stress-chart";
import { deletePracticeSession, loadSessionHistory, type SessionHistoryRecord } from "@/lib/supabase/data";

function severity(score: number) { return score >= 0.7 ? "sev-high" : score >= 0.5 ? "sev-medium" : "sev-low"; }

function localizeRecordText(value: string, t: (key: string, fallback?: string) => string) {
  const fixed: Record<string, string> = {
    "Mostly stable delivery": "diag.stableDelivery",
    "Practice slower transitions between your main explanation blocks and leave a short pause after each key term.": "diag.nextAction",
  };
  if (fixed[value]) return t(fixed[value]);
  const phaseKey = (phase: string) => `diag.phase.${phase.toLowerCase()}`;
  const pressure = value.match(/^Noticeable (Open|Build|Core|Close)-section pressure$/i);
  if (pressure) return t("diag.noticeablePressure").replace("{phase}", t(phaseKey(pressure[1])));
  const peak = value.match(/^Peak pressure reached (\d+(?:\.\d+)?)% around the (Open|Build|Core|Close) phase\. Deck: (.+)\.$/);
  if (peak) return t("diag.peakPressure").replace("{value}", peak[1]).replace("{phase}", t(phaseKey(peak[2]))).replace("{deck}", peak[3] === "No deck used" ? t("diag.noDeck") : peak[3]);
  return value;
}

function Timeline({ values, onSelect }: { values: number[]; onSelect?: (index: number) => void }) {
  if (!values.length) return <span className="history-no-timeline">No timeline saved</span>;
  return <div className="history-timeline" aria-label="Stress timeline">
    {values.map((value, index) => onSelect ? <button key={index} type="button" className={value >= 70 ? "high" : value >= 50 ? "mid" : "low"} style={{ flex: Math.max(value, 20) }} title={`Stress ${value}%`} onClick={() => onSelect(index)} /> : <span key={index} className={value >= 70 ? "high" : value >= 50 ? "mid" : "low"} style={{ flex: Math.max(value, 20) }} title={`Stress ${value}%`} />)}
  </div>;
}

function ReviewModal({ record, ko, t, onClose }: { record: SessionHistoryRecord; ko: boolean; t: (key: string, fallback?: string) => string; onClose: () => void }) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const peak = record.timeline.length ? Math.max(...record.timeline) : 0;
  const [playbackError, setPlaybackError] = useState(false);
  const phases = ko ? ["도입", "전개", "핵심", "마무리"] : ["Open", "Build", "Core", "Close"];
  const averages = phases.map((_, p) => {
    const values = record.timeline.slice(Math.floor(p * record.timeline.length / 4), Math.floor((p + 1) * record.timeline.length / 4));
    return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
  });
  const peakBin = record.timeline.indexOf(peak);
  const strongest = record.timeline.length ? averages.indexOf(Math.max(...averages.map((v) => v ?? -1))) : -1;
  function seekToBin(bin: number) {
    const video = videoRef.current;
    if (!video || !Number.isFinite(video.duration)) return;
    video.currentTime = ((bin + 0.5) / Math.max(record.timeline.length, 1)) * video.duration;
    void video.play().catch(() => undefined);
  }
  return <div className="modal-backdrop" onClick={onClose} role="presentation"><section className="card history-review-modal" role="dialog" aria-modal="true" aria-labelledby="history-review-title" onClick={(event) => event.stopPropagation()}>
    <button type="button" className="modal-close" onClick={onClose} aria-label="Close session review"><X size={18} /></button>
    <div className="diagnosis-modal-eyebrow">{ko ? "세션 리뷰" : "Session review"}</div>
    <h2 id="history-review-title" className="diagnosis-modal-title">{record.title}</h2>
    <div className="history-review-meta"><span><CalendarDays size={13} /> {record.date} · {record.duration}</span><span className={`severity-pill ${severity(Number(record.stressAverage))}`}>{ko ? "평균 스트레스" : "Average stress"} {record.stressAverage}</span></div>
    <div className="history-review-video">
      {record.videoUrl ? <><video ref={videoRef} controls playsInline preload="metadata" src={record.videoUrl} onError={() => setPlaybackError(true)} />{playbackError && <p role="alert" className="error-msg">{ko ? "영상 재생에 실패했습니다. 페이지를 새로고침해 재시도해 주세요." : "Playback failed. Refresh the page to retry."}</p>}</> : <div className="history-video-empty"><Film size={24} /><strong>{record.videoPath ? (ko ? "영상 불러오기 실패" : "Could not load video") : (ko ? "저장된 영상이 없습니다" : "No saved video")}</strong><span>{record.videoPath ? (ko ? "영상 경로는 저장되어 있지만 접근할 수 없습니다. 저장소 권한을 확인해 주세요." : "The video path exists but cannot be accessed. Check storage permissions.") : (ko ? "이 기록에는 영상 경로가 없습니다. 당시 녹화 또는 업로드가 완료되지 않았을 수 있습니다." : "This record has no video path. Recording or upload may not have completed.")}</span>{record.videoError && <small>{record.videoError}</small>}</div>}
    </div>
    <div className="history-review-section"><div className="history-review-section-head"><div><strong>{ko ? "스트레스 진행바" : "Stress timeline"}</strong><span>{record.videoUrl ? (ko ? "구간을 누르면 영상의 해당 시점으로 이동합니다" : "Select a segment to jump to that moment in the video") : (ko ? "영상 없이도 스트레스 구간을 확인할 수 있습니다" : "Stress moments are available without video")}</span></div><strong className="history-peak">{peak ? `${peak}% ${ko ? "최고" : "peak"}` : "–"}</strong></div><Timeline values={record.timeline} onSelect={record.videoUrl ? seekToBin : undefined} /><div className="video-timeline-legend"><span><i className="low" />{ko ? "좋음" : "Good"}</span><span><i className="mid" />{ko ? "보완 필요" : "Needs attention"}</span><span><i className="high" />{ko ? "스트레스 높음" : "High stress"}</span></div></div>
    <div className="history-review-coaching"><div className="coach-card coach-alert"><h3>{ko ? "세션 결과" : "Session result"}</h3><p>{localizeRecordText(record.result, t)}</p></div><div className="coach-card coach-data"><h3>{ko ? "AI 진단" : "AI diagnosis"}</h3><p>{localizeRecordText(record.diagnosis, t)}</p></div><div className="coach-card coach-ok"><h3>{ko ? "다음 연습 포인트" : "Next practice focus"}</h3><p>{localizeRecordText(record.nextAction, t)}</p></div></div>
    <section className="history-review-section" style={{ marginTop: 16 }}>
      <h3>{ko ? "구간별 스트레스 분석" : "Stress by phase"}</h3>
      <p>{ko ? "전체 연습 시간을 네 구간으로 나눈 요약입니다. 그래프 범위는 0~100%입니다." : "The session is divided into four equal phases. Chart range: 0–100%."}</p>
      <StressChart series={record.timeline} />
      <div className="report-grid" style={{ marginTop: 16 }}>
        <div className="report-card"><div className="rc-label">{ko ? "전체 평균" : "Average"}</div><div className="rc-value">{Math.round(Number(record.stressAverage) * 100)}%</div></div>
        <div className="report-card"><div className="rc-label">{ko ? "최고 스트레스" : "Peak stress"}</div><div className="rc-value">{record.timeline.length ? `${peak}%` : "—"}</div><div className="rc-desc">{peakBin >= 0 ? `${phases[Math.min(3, Math.floor(peakBin * 4 / record.timeline.length))]} · ${peakBin + 1}/${record.timeline.length}` : "—"}</div></div>
        <div className="report-card"><div className="rc-label">{ko ? "평균 스트레스가 가장 높은 구간" : "Highest average phase"}</div><div className="rc-value">{strongest >= 0 ? phases[strongest] : "—"}</div></div>
        <div className="report-card"><div className="rc-label">{ko ? "연습 환경 / 시간" : "Scene / duration"}</div><div className="rc-value" style={{ fontSize: 16 }}>{record.sceneLabel} · {record.duration}</div></div>
      </div>
      <div className="history-review-coaching">{phases.map((phase, i) => <div key={phase} className="coach-card coach-data"><h3>{phase}</h3><p>{averages[i] === null ? (ko ? "저장된 데이터 없음" : "No saved data") : `${ko ? "구간 평균 스트레스" : "Average stress"}: ${averages[i]!.toFixed(1)}%`}</p></div>)}</div>
    </section>
  </section></div>;
}

export function SessionHistory() {
  const { lang, t } = useLang(); const ko = lang === "ko";
  const [records, setRecords] = useState<SessionHistoryRecord[]>([]); const [loading, setLoading] = useState(true); const [error, setError] = useState(""); const [selectedId, setSelectedId] = useState("");
  useEffect(() => { void loadSessionHistory().then(setRecords).catch((reason) => setError(reason instanceof Error ? reason.message : "Failed to load history.")).finally(() => setLoading(false)); }, []);
  const selected = records.find((record) => record.id === selectedId);
  const average = useMemo(() => records.length ? records.reduce((sum, record) => sum + Number(record.stressAverage), 0) / records.length : 0, [records]);
  const videoCount = records.filter((record) => record.videoUrl).length;
  const latestPeak = records[0]?.timeline.length ? Math.max(...records[0].timeline) : 0;
  async function handleDelete(record: SessionHistoryRecord) {
    const message = ko
      ? `“${record.title}” 기록을 삭제할까요? 저장된 영상도 함께 삭제되며 되돌릴 수 없습니다.`
      : `Delete “${record.title}”? Its saved video will also be deleted and this cannot be undone.`;
    if (!window.confirm(message)) return;
    try {
      await deletePracticeSession(record.id, record.videoPath);
      setRecords((items) => items.filter((item) => item.id !== record.id));
      if (selectedId === record.id) setSelectedId("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Failed to delete session.");
    }
  }
  return <main className="page-content">
    <div className="page-top"><div><h1 className="page-title">{ko ? "기록" : "History"}</h1><p className="page-subtitle">{ko ? "저장한 세션 결과와 AI 진단을 한곳에서 확인하세요." : "Review your saved session results and AI diagnoses in one place."}</p></div><div className="page-actions"><Link href="/practice" className="btn btn-primary"><Play size={14} />{ko ? "새 연습 시작" : "Start practice"}</Link></div></div>
    <div className="kpi-row"><div className="kpi-card"><div className="kpi-top"><div className="kpi-icon"><FolderClock size={16} /></div></div><div className="kpi-value">{records.length}</div><div className="kpi-label">{ko ? "저장된 세션" : "Saved sessions"}</div><div className="kpi-sub">{ko ? "기록에서 다시 볼 수 있습니다" : "Available in your history"}</div></div><div className="kpi-card"><div className="kpi-top"><div className="kpi-icon" style={{ background: "var(--accent-ok-light)", color: "var(--accent-ok)" }}><Gauge size={16} /></div></div><div className="kpi-value">{average.toFixed(2)}</div><div className="kpi-label">{ko ? "평균 스트레스" : "Average stress"}</div><div className="kpi-sub">{ko ? "전체 기록 기준" : "Across all saved sessions"}</div></div><div className="kpi-card"><div className="kpi-top"><div className="kpi-icon" style={{ background: "rgba(245,158,11,0.1)", color: "var(--accent-warn)" }}><Film size={16} /></div></div><div className="kpi-value">{videoCount}</div><div className="kpi-label">{ko ? "저장된 영상" : "Saved videos"}</div><div className="kpi-sub">{ko ? "선택한 경우에만 보관" : "Kept only when selected"}</div></div><div className="kpi-card"><div className="kpi-top"><div className="kpi-icon" style={{ background: "rgba(139,92,246,0.1)", color: "#7c3aed" }}><Sparkles size={16} /></div></div><div className="kpi-value">{latestPeak ? `${latestPeak}%` : "–"}</div><div className="kpi-label">{ko ? "최근 최고 스트레스" : "Latest peak stress"}</div><div className="kpi-sub">{ko ? "최근 세션 기준" : "For the latest session"}</div></div></div>
    {loading ? <p className="state-loading"><LoaderCircle size={16} /> {ko ? "기록을 불러오는 중..." : "Loading history..."}</p> : null}{error ? <p className="error-msg">{error}</p> : null}{!loading && !error && records.length === 0 ? <div className="state-empty history-platform-empty"><BrainCircuit size={24} /><strong>{ko ? "아직 저장된 세션이 없습니다" : "No saved sessions yet"}</strong><span>{ko ? "연습 종료 후 저장하면 이곳에 표시됩니다." : "Save a session after practice and it will appear here."}</span></div> : null}
    {records.length > 0 ? <div className="card history-table-card"><div className="history-panel-header"><div className="panel-head-left"><span className="panel-title">{ko ? "세션 기록" : "Session records"}</span><span className="panel-subtitle">{ko ? "저장된 분석 결과" : "Saved analysis results"}</span></div><CalendarDays size={16} color="var(--text-secondary)" /></div><div className="table-wrap"><table className="data-table"><thead><tr><th>{ko ? "날짜" : "Date"}</th><th>{ko ? "세션" : "Session"}</th><th>{ko ? "시간" : "Duration"}</th><th>{ko ? "평균 스트레스" : "Avg. stress"}</th><th>{ko ? "영상" : "Video"}</th><th>{ko ? "보기" : "Review"}</th></tr></thead><tbody>{records.map((record) => <tr key={record.id}><td className="td-mono">{record.date}</td><td style={{ fontWeight: 500 }}>{record.title}<small className="history-scene">{record.sceneLabel}</small></td><td className="td-mono">{record.duration}</td><td><span className={`severity-pill ${severity(Number(record.stressAverage))}`}>{record.stressAverage}</span></td><td>{record.videoUrl ? <Film size={15} color="var(--accent)" /> : <span className="history-muted">–</span>}</td><td><div className="history-table-actions"><button type="button" className="table-action" onClick={() => setSelectedId(record.id)}>{ko ? "상세 보기" : "View details"}</button><button type="button" className="history-delete" onClick={() => void handleDelete(record)} aria-label={ko ? "기록 삭제" : "Delete record"}><Trash2 size={14} /></button></div></td></tr>)}</tbody></table></div></div> : null}
    {selected ? <ReviewModal record={selected} ko={ko} t={t} onClose={() => setSelectedId("")} /> : null}
  </main>;
}
