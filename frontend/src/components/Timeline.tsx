import { useMemo, useState } from "react";
import type { RegimeHistory } from "../types";
import { REGIME_COLORS, REGIME_ORDER } from "../api";

const RANGES = [12, 36, 60, 120];

export default function Timeline({
  data,
  months,
  onRange,
}: {
  data: RegimeHistory;
  months: number;
  onRange: (m: number) => void;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const points = data.regimes;

  const segments = useMemo(() => {
    // Collapse consecutive same-regime months into bands.
    const segs: { regime: string; start: number; len: number }[] = [];
    points.forEach((pt, i) => {
      const last = segs[segs.length - 1];
      if (last && last.regime === pt.regime) last.len += 1;
      else segs.push({ regime: pt.regime, start: i, len: 1 });
    });
    return segs;
  }, [points]);

  const n = points.length || 1;
  const hovered = hover != null ? points[hover] : null;

  return (
    <section className="panel">
      <div className="panel__head">
        <span className="eyebrow">Regime timeline</span>
        <div className="seg-ctrl">
          {RANGES.map((m) => (
            <button
              key={m}
              className={`seg-btn ${m === months ? "is-active" : ""}`}
              onClick={() => onRange(m)}
            >
              {m}m
            </button>
          ))}
        </div>
      </div>

      <div className="timeline">
        <svg
          viewBox={`0 0 ${n} 10`}
          preserveAspectRatio="none"
          className="timeline__svg"
        >
          {segments.map((s, i) => (
            <rect
              key={i}
              x={s.start}
              y={0}
              width={s.len}
              height={10}
              fill={REGIME_COLORS[s.regime]}
              opacity={0.9}
            />
          ))}
          {points.map((_, i) => (
            <rect
              key={`h-${i}`}
              x={i}
              y={0}
              width={1}
              height={10}
              fill="transparent"
              onMouseEnter={() => setHover(i)}
              onMouseLeave={() => setHover(null)}
            />
          ))}
        </svg>
      </div>

      <div className="timeline__foot">
        <span className="mono ink-3">{points[0]?.date ?? "—"}</span>
        <span className="timeline__cursor mono">
          {hovered ? `${hovered.date} · ${hovered.regime}` : `${n} months`}
        </span>
        <span className="mono ink-3">{points[n - 1]?.date ?? "—"}</span>
      </div>

      <div className="legend">
        {REGIME_ORDER.map((r) => (
          <span className="legend__item" key={r}>
            <span
              className="legend__swatch"
              style={{ background: REGIME_COLORS[r] }}
            />
            {r}
          </span>
        ))}
      </div>
    </section>
  );
}
