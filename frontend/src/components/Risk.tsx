import { useEffect, useState } from "react";
import type { MarketKey, VarResult } from "../types";
import { api, pct, signedPct } from "../api";

const HORIZONS = [3, 6, 12, 24];

export default function Risk({
  initial,
  market,
}: {
  initial: VarResult;
  market: MarketKey;
}) {
  const [horizon, setHorizon] = useState(initial.horizon_months);
  const [data, setData] = useState<VarResult>(initial);
  const [loading, setLoading] = useState(false);

  // Re-sync when the market switches (new `initial`).
  useEffect(() => {
    setHorizon(initial.horizon_months);
    setData(initial);
  }, [market, initial]);

  useEffect(() => {
    let active = true;
    if (horizon === initial.horizon_months && data === initial) return;
    setLoading(true);
    api
      .var(market, horizon)
      .then((d) => active && setData(d))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [horizon]);

  const metrics = [
    {
      k: "VaR 95%",
      v: data.var_95,
      hint: "1-in-20 loss floor",
      tip: "Value at Risk (95%): the loss the portfolio is not expected to exceed 95% of the time over the horizon.",
    },
    {
      k: "CVaR 95%",
      v: data.cvar_95,
      hint: "avg loss beyond VaR95",
      tip: "Conditional VaR (95%): the average loss in the worst 5% of simulated outcomes.",
    },
    {
      k: "VaR 99%",
      v: data.var_99,
      hint: "1-in-100 loss floor",
      tip: "Value at Risk (99%): the loss not expected to be exceeded 99% of the time.",
    },
    {
      k: "CVaR 99%",
      v: data.cvar_99,
      hint: "avg loss beyond VaR99",
      tip: "Conditional VaR (99%): the average loss in the worst 1% of simulated outcomes.",
    },
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
            <span className="riskcell__k has-tip" data-tip={m.tip}>
              {m.k}
            </span>
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
