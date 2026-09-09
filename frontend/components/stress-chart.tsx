"use client";
import { useId } from "react";
import { useLang } from "@/lib/i18n";

export function StressChart({ series }: { series: number[] }) {
  const { lang } = useLang();
  const titleId = useId();
  const phases = lang === "ko" ? ["도입", "전개", "핵심", "마무리"] : ["Open", "Build", "Core", "Close"];
  if (!series.length) return <p>{lang === "ko" ? "저장된 구간 데이터가 없습니다." : "No segment data was saved."}</p>;
  const left = 44, top = 20, width = 584, height = 180;
  const points = series.map((value, index) => ({
    x: left + ((index + 0.5) / series.length) * width,
    y: top + height * (1 - Math.min(100, Math.max(0, Number.isFinite(value) ? value : 0)) / 100),
    value,
  }));
  return <figure className="stress-line-chart">
    <svg viewBox="0 0 650 242" role="img" aria-labelledby={titleId}>
      <title id={titleId}>{lang === "ko" ? "구간별 스트레스 추이, 0~100%" : "Stress over the session, 0–100%"}</title>
      {[0, 25, 50, 75, 100].map((value) => {
        const y = top + height * (1 - value / 100);
        return <g key={value}>
          <line className="stress-line-grid" x1={left} x2={left + width} y1={y} y2={y} />
          <text className="stress-line-label" x={left - 10} y={y + 4} textAnchor="end">{value}</text>
        </g>;
      })}
      {phases.map((phase, index) => <g key={phase}>
        {index > 0 && <line className="stress-line-divider" x1={left + width * index / 4} x2={left + width * index / 4} y1={top} y2={top + height} />}
        <text className="stress-line-label" x={left + width * (index + 0.5) / 4} y={226} textAnchor="middle">{phase}</text>
      </g>)}
      <polyline className="stress-line-path" points={points.map(({ x, y }) => `${x},${y}`).join(" ")} />
      {points.map(({ x, y, value }, index) => <circle key={index} className="stress-line-point" cx={x} cy={y} r="4">
        <title>{`${phases[Math.min(3, Math.floor(index * 4 / series.length))]} · ${index + 1}/${series.length}: ${value}%`}</title>
      </circle>)}
    </svg>
    <figcaption>{lang === "ko" ? "가로축: 연습 진행 구간 · 세로축: 스트레스 지수 (%)" : "Horizontal: session progress · Vertical: stress score (%)"}</figcaption>
  </figure>;
}
