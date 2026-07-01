import { useEffect, useState } from "react";
import type { Allocation as AllocationT, MarketKey, RegimeName } from "../types";
import {
  assetLabel,
  orderAssets,
  REGIME_COLORS,
  REGIME_ORDER,
  api,
  pct,
  signedPct,
} from "../api";

export default function Allocation({
  live,
  market,
  initialSelected,
  onSelect,
}: {
  live: AllocationT;
  market: MarketKey;
  initialSelected?: "live" | RegimeName;
  onSelect?: (sel: "live" | RegimeName) => void;
}) {
  const [selected, setSelected] = useState<"live" | RegimeName>(
    initialSelected ?? "live"
  );
  const [data, setData] = useState<AllocationT>(live);
  const [loading, setLoading] = useState(false);

  const choose = (sel: "live" | RegimeName) => {
    setSelected(sel);
    onSelect?.(sel);
  };

  // Reset to live whenever the market (and therefore `live`) changes.
  useEffect(() => {
    setSelected("live");
    setData(live);
  }, [market, live]);

  useEffect(() => {
    let active = true;
    if (selected === "live") {
      setData(live);
      return;
    }
    setLoading(true);
    api
      .allocationFor(market, selected)
      .then((d) => active && setData(d))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [selected, live, market]);

  const assets = orderAssets(
    Array.from(
      new Set([
        ...Object.keys(data.target_weights),
        ...Object.keys(data.benchmark_weights),
      ])
    )
  );
  const maxW = Math.max(
    ...assets.map((a) => data.target_weights[a] ?? 0),
    0.35
  );

  return (
    <section className="panel">
      <div className="panel__head">
        <span className="eyebrow">Tactical allocation</span>
        <div className="seg-ctrl">
          <button
            className={`seg-btn ${selected === "live" ? "is-active" : ""}`}
            onClick={() => choose("live")}
          >
            Live
          </button>
          {REGIME_ORDER.map((r) => (
            <button
              key={r}
              className={`seg-btn ${selected === r ? "is-active" : ""}`}
              onClick={() => choose(r as RegimeName)}
              style={
                selected === r
                  ? { color: REGIME_COLORS[r], borderColor: REGIME_COLORS[r] }
                  : undefined
              }
            >
              {r.slice(0, 3)}
            </button>
          ))}
        </div>
      </div>

      <div className={`alloc ${loading ? "is-loading" : ""}`}>
        <div className="alloc__head">
          <span className="alloc__regime" style={{ color: REGIME_COLORS[data.regime] }}>
            {data.regime}
          </span>
          <span className="alloc__cols">
            <span>Target</span>
            <span>vs 60/40</span>
          </span>
        </div>

        {assets.map((asset) => {
          const t = data.target_weights[asset] ?? 0;
          const b = data.benchmark_weights[asset] ?? 0;
          const delta = t - b;
          return (
            <div className="alloc-row" key={asset}>
              <span className="alloc-row__name">{assetLabel(asset)}</span>
              <div className="alloc-row__bar">
                <div
                  className="alloc-row__fill"
                  style={{ width: `${(t / maxW) * 100}%` }}
                />
                <span
                  className="alloc-row__bench"
                  style={{ left: `${(b / maxW) * 100}%` }}
                  title={`Benchmark ${pct(b)}`}
                />
              </div>
              <span className="alloc-row__w mono">{pct(t)}</span>
              <span
                className={`alloc-row__delta mono ${
                  delta > 0.0005 ? "pos" : delta < -0.0005 ? "neg" : "flat"
                }`}
              >
                {Math.abs(delta) < 0.0005 ? "—" : signedPct(delta)}
              </span>
            </div>
          );
        })}
      </div>

      <p className="rationale">{data.rationale}</p>

      <div className="tilts">
        <div className="tilts__col">
          <span className="tilts__label pos">Overweight</span>
          <div className="chips">
            {data.overweight.map((x) => (
              <span className="chip chip--pos" key={x}>
                {assetLabel(x)}
              </span>
            ))}
          </div>
        </div>
        <div className="tilts__col">
          <span className="tilts__label neg">Underweight</span>
          <div className="chips">
            {data.underweight.map((x) => (
              <span className="chip chip--neg" key={x}>
                {assetLabel(x)}
              </span>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
