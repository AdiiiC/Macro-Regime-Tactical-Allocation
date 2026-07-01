import type { ComparisonResult, RegimeName } from "../types";
import { REGIME_COLORS, pct } from "../api";

const ROW_KEYS: { key: "hmm" | "kmeans" | "lstm"; label: string }[] = [
  { key: "hmm", label: "HMM" },
  { key: "kmeans", label: "KMeans" },
  { key: "lstm", label: "LSTM" },
];

export default function ModelComparison({ data }: { data: ComparisonResult }) {
  const n = data.timeline.length;
  const available = ROW_KEYS.filter((r) =>
    data.timeline.some((t) => t[r.key] != null)
  );

  return (
    <section className="panel">
      <div className="panel__head">
        <span className="eyebrow">Model cross-check</span>
        <span className="tag">robustness</span>
      </div>

      <div className="mc-list">
        {data.models.map((mdl) => {
          const unavailable = mdl.current_regime == null;
          return (
            <div className={`mc-item ${unavailable ? "is-off" : ""}`} key={mdl.name}>
              <div className="mc-item__top">
                <span className="mc-item__name">{mdl.name}</span>
                {mdl.current_regime ? (
                  <span
                    className="mc-chip"
                    style={{
                      color: REGIME_COLORS[mdl.current_regime],
                      borderColor: REGIME_COLORS[mdl.current_regime],
                    }}
                  >
                    {mdl.current_regime}
                  </span>
                ) : (
                  <span className="mc-chip mc-chip--off">n/a</span>
                )}
              </div>

              {mdl.agreement_with_hmm != null && (
                <div className="mc-agree">
                  <div className="mc-agree__track">
                    <div
                      className="mc-agree__fill"
                      style={{ width: `${mdl.agreement_with_hmm * 100}%` }}
                    />
                  </div>
                  <span className="mc-agree__val mono">
                    {pct(mdl.agreement_with_hmm, 0)}
                  </span>
                </div>
              )}

              <div className="mc-item__meta">
                <span className="ink-3">
                  {mdl.quality != null
                    ? `${mdl.quality_label}: ${mdl.quality.toFixed(2)}`
                    : mdl.type === "primary"
                    ? "primary signal"
                    : mdl.quality_label}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      <div className="mc-timeline">
        <span className="mc-timeline__label">Regime label agreement · last {n} months</span>
        {available.map((r) => (
          <div className="mc-strip" key={r.key}>
            <span className="mc-strip__name mono">{r.label}</span>
            <div className="mc-strip__cells">
              {data.timeline.map((t, i) => {
                const reg = t[r.key] as RegimeName | null | undefined;
                return (
                  <span
                    key={i}
                    className="mc-cell"
                    title={reg ? `${t.date}: ${reg}` : t.date}
                    style={{ background: reg ? REGIME_COLORS[reg] : "var(--inset)" }}
                  />
                );
              })}
            </div>
          </div>
        ))}
      </div>

      <p className="panel__note">
        Agreement = share of months each alternative model assigns the same regime label as the
        primary HMM. High agreement across independent methods raises confidence in the current
        call.
      </p>
    </section>
  );
}
