import type { CurrentRegime } from "../types";
import { REGIME_COLORS, REGIME_ORDER, pct } from "../api";

function RegimeGlyph({ regime, size = 40 }: { regime: string; size?: number }) {
  const color = REGIME_COLORS[regime] ?? "#98a2ad";
  // Distinct monoline mark per regime: a phase indicator around a ring.
  const angle: Record<string, number> = {
    Expansion: -45,
    Recovery: -135,
    Slowdown: 45,
    Recession: 135,
  };
  const a = ((angle[regime] ?? 0) * Math.PI) / 180;
  const cx = 24 + 15 * Math.cos(a);
  const cy = 24 + 15 * Math.sin(a);
  return (
    <svg width={size} height={size} viewBox="0 0 48 48" aria-hidden="true">
      <circle
        cx="24"
        cy="24"
        r="15"
        fill="none"
        stroke="var(--line-strong)"
        strokeWidth="1.5"
      />
      <circle cx={cx} cy={cy} r="4.5" fill={color} />
      <circle cx="24" cy="24" r="2" fill="var(--ink-3)" />
    </svg>
  );
}

export default function RegimeNow({ data }: { data: CurrentRegime }) {
  const color = REGIME_COLORS[data.current_regime] ?? "#98a2ad";
  const probs = REGIME_ORDER.map((r) => ({
    regime: r,
    p: data.regime_probabilities[r] ?? 0,
  }));

  return (
    <section className="panel panel--regime">
      <div className="panel__head">
        <span className="eyebrow">Current market regime</span>
        <span className="tag">HMM &middot; 4-state</span>
      </div>

      <div className="regime-hero">
        <RegimeGlyph regime={data.current_regime} size={56} />
        <div className="regime-hero__body">
          <div className="regime-name" style={{ color }}>
            {data.current_regime}
          </div>
          <div className="regime-sub">
            <span className="mono">{pct(data.confidence, 1)}</span> confidence
            <span className="dot-sep">&middot;</span>
            expected persistence{" "}
            <span className="mono">
              {data.expected_duration_months.toFixed(0)} mo
            </span>
          </div>
        </div>
      </div>

      <div className="prob-list">
        <div className="prob-list__label">Posterior probabilities</div>
        {probs.map(({ regime, p }) => (
          <div className="prob-row" key={regime}>
            <span className="prob-row__name">{regime}</span>
            <div className="prob-track">
              <div
                className="prob-fill"
                style={{
                  width: `${Math.max(p * 100, 0.4)}%`,
                  background: REGIME_COLORS[regime],
                }}
              />
            </div>
            <span className="prob-row__val mono">{pct(p, 1)}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
