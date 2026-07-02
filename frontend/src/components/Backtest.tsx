import { useEffect, useMemo, useState } from "react";
import type { BacktestResult, BenchmarkVariant, MarketKey } from "../types";
import { api, REGIME_COLORS, num, pct, signedPct } from "../api";

type ChartMode = "equity" | "drawdown" | "rolling";

const BENCH_LABEL: Record<BenchmarkVariant, string> = {
  sixty_forty: "60 / 40",
  equal_weight: "Equal wt.",
  risk_parity: "Risk parity",
  kelly: "Kelly lev.",
};

const W = 720;
const H = 210;
const PAD = { l: 40, r: 10, t: 12, b: 20 };

function scaleY(v: number, min: number, max: number) {
  const span = max - min || 1;
  return PAD.t + (1 - (v - min) / span) * (H - PAD.t - PAD.b);
}
function scaleX(i: number, n: number) {
  return PAD.l + (n <= 1 ? 0 : (i / (n - 1)) * (W - PAD.l - PAD.r));
}
function path(vals: (number | null)[], min: number, max: number) {
  let d = "";
  let started = false;
  vals.forEach((v, i) => {
    if (v === null || Number.isNaN(v)) {
      started = false;
      return;
    }
    const x = scaleX(i, vals.length);
    const y = scaleY(v, min, max);
    d += `${started ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)} `;
    started = true;
  });
  return d.trim();
}

export default function Backtest({
  data: initial,
  market,
}: {
  data: BacktestResult;
  market: MarketKey;
}) {
  const [mode, setMode] = useState<ChartMode>("equity");
  const [data, setData] = useState<BacktestResult>(initial);
  const [benchmark, setBenchmark] = useState<BenchmarkVariant>(initial.benchmark);
  const [loading, setLoading] = useState(false);

  // Reset when the parent hands us a fresh backtest (market switch / refresh).
  useEffect(() => {
    setData(initial);
    setBenchmark(initial.benchmark);
  }, [initial]);

  function chooseBenchmark(v: BenchmarkVariant) {
    if (v === benchmark) return;
    setBenchmark(v);
    setLoading(true);
    api
      .backtest(market, v)
      .then(setData)
      .catch(() => void 0)
      .finally(() => setLoading(false));
  }

  const m = data.metrics;

  const chart = useMemo(() => {
    if (mode === "equity") {
      const s = data.equity_curve.map((p) => p.strategy);
      const b = data.equity_curve.map((p) => p.benchmark);
      const min = Math.min(...s, ...b);
      const max = Math.max(...s, ...b);
      return { s, b, min, max, dates: data.equity_curve.map((p) => p.date) };
    }
    if (mode === "drawdown") {
      const s = data.drawdown.map((p) => p.strategy * 100);
      const b = data.drawdown.map((p) => p.benchmark * 100);
      const min = Math.min(...s, ...b, 0);
      return { s, b, min, max: 0, dates: data.drawdown.map((p) => p.date) };
    }
    const s = data.rolling_sharpe.map((p) => p.strategy);
    const b = data.rolling_sharpe.map((p) => p.benchmark);
    const nums = [...s, ...b].filter((v): v is number => v !== null && !Number.isNaN(v));
    return {
      s,
      b,
      min: Math.min(...nums, 0),
      max: Math.max(...nums, 1),
      dates: data.rolling_sharpe.map((p) => p.date),
    };
  }, [mode, data]);

  const yTicks = useMemo(() => {
    const { min, max } = chart;
    return [min, min + (max - min) / 2, max];
  }, [chart]);

  const fmtY = (v: number) =>
    mode === "drawdown" ? `${v.toFixed(0)}%` : mode === "rolling" ? v.toFixed(1) : v.toFixed(0);

  const first = chart.dates[0]?.slice(0, 7) ?? "";
  const last = chart.dates[chart.dates.length - 1]?.slice(0, 7) ?? "";

  const rows: { k: string; s: string; b: string; win: boolean; tip: string }[] = [
    {
      k: "Ann. return",
      s: signedPct(m.annual_return_strategy),
      b: signedPct(m.annual_return_benchmark),
      win: m.annual_return_strategy >= m.annual_return_benchmark,
      tip: "Compound annual growth rate over the full backtest.",
    },
    {
      k: "Volatility",
      s: pct(m.annual_vol_strategy),
      b: pct(m.annual_vol_benchmark),
      win: m.annual_vol_strategy <= m.annual_vol_benchmark,
      tip: "Annualized standard deviation of monthly returns.",
    },
    {
      k: "Sharpe",
      s: num(m.sharpe_strategy),
      b: num(m.sharpe_benchmark),
      win: m.sharpe_strategy >= m.sharpe_benchmark,
      tip: "Annualized return per unit of total volatility (rf = 0).",
    },
    {
      k: "Max drawdown",
      s: pct(m.max_drawdown_strategy),
      b: pct(m.max_drawdown_benchmark),
      win: m.max_drawdown_strategy >= m.max_drawdown_benchmark,
      tip: "Largest peak-to-trough decline in portfolio value.",
    },
  ];

  const extras: { k: string; v: string; tip: string }[] = [
    { k: "Sortino", v: num(m.sortino_strategy), tip: "Return per unit of downside volatility." },
    { k: "Calmar", v: num(m.calmar_strategy), tip: "Annual return divided by max drawdown." },
    { k: "Info ratio", v: num(m.information_ratio), tip: "Active return per unit of tracking error vs benchmark." },
    { k: "Win rate", v: pct(m.win_rate, 0), tip: "Share of months the strategy beat the benchmark." },
  ];

  const attrMax = Math.max(...data.regime_attribution.map((a) => Math.abs(a.contribution)), 0.05);

  return (
    <section className="panel">
      <div className="panel__head">
        <span className="eyebrow">Strategy backtest</span>
        <span className="tag">
          {data.start.slice(0, 7)} – {data.end.slice(0, 7)}
        </span>
      </div>

      <div className="bt-bench">
        <span className="bt-bench__label ink-3">Benchmark</span>
        <div className="seg-ctrl">
          {(
            ["sixty_forty", "equal_weight", "risk_parity", "kelly"] as BenchmarkVariant[]
          ).map((v) => (
            <button
              key={v}
              className={`seg-btn ${benchmark === v ? "is-active" : ""}`}
              onClick={() => chooseBenchmark(v)}
              disabled={loading}
            >
              {BENCH_LABEL[v]}
            </button>
          ))}
        </div>
        {benchmark === "kelly" && data.benchmark_leverage != null && (
          <span className="bt-bench__note ink-3">
            {data.benchmark_leverage.toFixed(2)}× half-Kelly, borrow @ 4%
          </span>
        )}
      </div>

      <div className="bt-metrics">
        <div className="bt-table">
          <div className="bt-table__head">
            <span />
            <span>Strategy</span>
            <span>{BENCH_LABEL[benchmark]}</span>
          </div>
          {rows.map((r) => (
            <div className="bt-table__row" key={r.k}>
              <span className="bt-table__k has-tip" data-tip={r.tip}>
                {r.k}
              </span>
              <span className={`mono ${r.win ? "pos" : "ink"}`}>{r.s}</span>
              <span className="mono ink-3">{r.b}</span>
            </div>
          ))}
        </div>
        <div className="bt-extras">
          {extras.map((e) => (
            <div className="bt-extra" key={e.k}>
              <span className="bt-extra__k has-tip" data-tip={e.tip}>
                {e.k}
              </span>
              <span className="bt-extra__v mono">{e.v}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="bt-chart__head">
        <div className="seg-ctrl">
          {(["equity", "drawdown", "rolling"] as ChartMode[]).map((mo) => (
            <button
              key={mo}
              className={`seg-btn ${mode === mo ? "is-active" : ""}`}
              onClick={() => setMode(mo)}
            >
              {mo === "equity" ? "Equity" : mo === "drawdown" ? "Drawdown" : "Roll. Sharpe"}
            </button>
          ))}
        </div>
        <div className="bt-legend">
          <span className="bt-legend__item">
            <span className="bt-swatch" style={{ background: "var(--gold)" }} /> Strategy
          </span>
          <span className="bt-legend__item">
            <span className="bt-swatch" style={{ background: "var(--ink-2)" }} /> {BENCH_LABEL[benchmark]}
          </span>
        </div>
      </div>

      <svg className="bt-svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img">
        {yTicks.map((t, i) => {
          const y = scaleY(t, chart.min, chart.max);
          return (
            <g key={i}>
              <line x1={PAD.l} y1={y} x2={W - PAD.r} y2={y} stroke="var(--line)" strokeWidth={1} />
              <text x={PAD.l - 6} y={y + 3} textAnchor="end" className="bt-axis">
                {fmtY(t)}
              </text>
            </g>
          );
        })}
        {mode === "drawdown" && (
          <path
            d={`${path(chart.s, chart.min, chart.max)} L${scaleX(chart.s.length - 1, chart.s.length)},${scaleY(0, chart.min, chart.max)} L${scaleX(0, chart.s.length)},${scaleY(0, chart.min, chart.max)} Z`}
            fill="var(--gold-wash)"
            stroke="none"
          />
        )}
        <path d={path(chart.b, chart.min, chart.max)} fill="none" stroke="var(--ink-2)" strokeWidth={1.4} />
        <path d={path(chart.s, chart.min, chart.max)} fill="none" stroke="var(--gold)" strokeWidth={1.8} />
      </svg>
      <div className="bt-xaxis mono">
        <span>{first}</span>
        <span>{last}</span>
      </div>

      <div className="bt-attr">
        <span className="bt-attr__label">Cumulative return by regime</span>
        {data.regime_attribution.map((a) => (
          <div className="bt-attr__row" key={a.regime}>
            <span className="bt-attr__name" style={{ color: REGIME_COLORS[a.regime] }}>
              {a.regime}
            </span>
            <div className="bt-attr__track">
              <div
                className="bt-attr__fill"
                style={{
                  width: `${(Math.abs(a.contribution) / attrMax) * 100}%`,
                  background: REGIME_COLORS[a.regime],
                  opacity: a.contribution >= 0 ? 0.85 : 0.4,
                }}
              />
            </div>
            <span className="bt-attr__months mono ink-3">{a.months}m</span>
            <span className={`bt-attr__val mono ${a.contribution >= 0 ? "pos" : "neg"}`}>
              {signedPct(a.contribution)}
            </span>
          </div>
        ))}
      </div>

      <p className="panel__note">
        Event-driven monthly walk-forward, {data.currency}. Regime signals drive tactical
        weights vs the {BENCH_LABEL[benchmark]} benchmark, net of 10 bps transaction costs.
      </p>
    </section>
  );
}
