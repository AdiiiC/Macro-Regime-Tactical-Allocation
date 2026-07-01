import { useEffect, useState } from "react";
import type { VarResult } from "../types";
import { api, pct, signedPct } from "../api";

const HORIZONS = [3, 6, 12, 24];

export default function Risk({ initial }: { initial: VarResult }) {
  const [horizon, setHorizon] = useState(initial.horizon_months);
  const [data, setData] = useState<VarResult>(initial);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let active = true;
    if (horizon === initial.horizon_months && data === initial) return;
    setLoading(true);
    api
      .var(horizon)
      .then((d) => active && setData(d))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [horizon]);

  const metrics = [
    { k: "VaR 95%", v: data.var_95, hint: "1-in-20 loss floor" },
    { k: "CVaR 95%", v: data.cvar_95, hint: "avg loss beyond VaR95" },
    { k: "VaR 99%", v: data.var_99, hint: "1-in-100 loss floor" },
    { k: "CVaR 99%", v: data.cvar_99, hint: "avg loss beyond VaR99" },
  ];

  return (
    <section className="panel">
      <div className="panel__head">
        <span className="eyebrow">Risk &middot; Monte Carlo</span>
        <div className="seg-ctrl">
          {HORIZONS.map((h) => (
            <button
              key={h}
              className={`seg-btn ${h === horizon ? "is-active" : ""}`}
              onClick={() => setHorizon(h)}
            >
              {h}m
            </button>
          ))}
        </div>
      </div>

      <div className={`riskgrid ${loading ? "is-loading" : ""}`}>
        {metrics.map((m) => (
          <div className="riskcell" key={m.k}>
            <span className="riskcell__k">{m.k}</span>
            <span className="riskcell__v mono neg">{signedPct(m.v)}</span>
            <span className="riskcell__hint">{m.hint}</span>
          </div>
        ))}
      </div>

      <div className="riskfoot">
        <div className="riskfoot__item">
          <span className="riskfoot__k">Expected return</span>
          <span className="mono pos">{signedPct(data.expected_return)}</span>
        </div>
        <div className="riskfoot__item">
          <span className="riskfoot__k">P(loss)</span>
          <span className="mono">{pct(data.probability_of_loss)}</span>
        </div>
        <div className="riskfoot__item">
          <span className="riskfoot__k">Horizon</span>
          <span className="mono">{data.horizon_months} mo</span>
        </div>
      </div>
    </section>
  );
}
