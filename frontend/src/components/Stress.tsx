import { useState } from "react";
import type { StressResult } from "../types";
import { ASSET_LABELS, ASSET_ORDER, signedPct } from "../api";

export default function Stress({ data }: { data: StressResult }) {
  const entries = Object.entries(data.scenarios);
  const [open, setOpen] = useState<string | null>(entries[0]?.[0] ?? null);
  const maxAbs = Math.max(
    ...entries.map(([, s]) => Math.abs(s.portfolio_impact)),
    0.05
  );

  return (
    <section className="panel">
      <div className="panel__head">
        <span className="eyebrow">Stress scenarios</span>
        <span className="tag">portfolio impact</span>
      </div>

      <div className="stress">
        {entries.map(([name, s]) => {
          const w = (Math.abs(s.portfolio_impact) / maxAbs) * 100;
          const pos = s.portfolio_impact >= 0;
          const isOpen = open === name;
          return (
            <div className="stress-item" key={name}>
              <button
                className="stress-row"
                onClick={() => setOpen(isOpen ? null : name)}
              >
                <span className="stress-row__name">
                  <span className={`caret ${isOpen ? "is-open" : ""}`}>&rsaquo;</span>
                  {name}
                </span>
                <div className="stress-row__track">
                  <div
                    className={`stress-row__fill ${pos ? "pos" : "neg"}`}
                    style={{ width: `${w}%` }}
                  />
                </div>
                <span className={`stress-row__val mono ${pos ? "pos" : "neg"}`}>
                  {signedPct(s.portfolio_impact)}
                </span>
              </button>

              {isOpen && (
                <div className="shocks">
                  {ASSET_ORDER.map((a) => {
                    const v = s.shocks[a] ?? 0;
                    return (
                      <div className="shock" key={a}>
                        <span className="shock__a">{ASSET_LABELS[a]}</span>
                        <span
                          className={`shock__v mono ${
                            v > 0 ? "pos" : v < 0 ? "neg" : "flat"
                          }`}
                        >
                          {signedPct(v)}
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
