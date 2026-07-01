import type { DriversResult } from "../types";
import { REGIME_COLORS, num, signedNum } from "../api";

// Turn a raw z-scored feature name into a readable label.
function featureLabel(feat: string): string {
  const map: Record<string, string> = {
    YoY: "y/y",
    Mom3: "3m mom.",
    Chg3: "3m chg.",
    Level: "level",
  };
  const parts = feat.split("_");
  const suffix = parts[parts.length - 1];
  const tail = map[suffix];
  const base = (tail ? parts.slice(0, -1) : parts).join(" ");
  return tail ? `${base} · ${tail}` : base;
}

export default function Drivers({ data }: { data: DriversResult }) {
  const maxAbs = Math.max(...data.drivers.map((d) => Math.abs(d.z_score)), 1);
  const color = REGIME_COLORS[data.regime];

  return (
    <section className="panel">
      <div className="panel__head">
        <span className="eyebrow">Regime drivers</span>
        <span className="tag">as of {data.as_of}</span>
      </div>

      <p className="drv-lede">
        Indicators most stretched from their historical norm — the macro backdrop the model
        reads as{" "}
        <span style={{ color }} className="drv-regime">
          {data.regime}
        </span>
        . Values are standard deviations (z-scores).
      </p>

      <div className="drv-list">
        {data.drivers.map((d) => {
          const wpct = (Math.abs(d.z_score) / maxAbs) * 50; // half-width
          const positive = d.z_score >= 0;
          return (
            <div className="drv-row" key={d.feature}>
              <span className="drv-row__name">{featureLabel(d.feature)}</span>
              <div className="drv-track">
                <span className="drv-track__axis" />
                <span
                  className={`drv-track__fill ${positive ? "pos" : "neg"}`}
                  style={{
                    left: positive ? "50%" : `${50 - wpct}%`,
                    width: `${wpct}%`,
                  }}
                />
              </div>
              <span className={`drv-row__z mono ${positive ? "pos" : "neg"}`}>
                {signedNum(d.z_score, 2)}σ
              </span>
              <span className="drv-row__avg mono ink-3" title="Average for this regime">
                {num(d.regime_avg, 2)}
              </span>
            </div>
          );
        })}
      </div>

      <div className="drv-key">
        <span className="drv-key__item">
          <span className="drv-key__dot pos" /> above norm
        </span>
        <span className="drv-key__item">
          <span className="drv-key__dot neg" /> below norm
        </span>
        <span className="drv-key__item ink-3">right column = regime average</span>
      </div>
    </section>
  );
}
