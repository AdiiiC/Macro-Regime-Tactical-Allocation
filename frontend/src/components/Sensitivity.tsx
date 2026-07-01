import { useMemo } from "react";
import type { SensitivityResult } from "../types";
import { num, signedPct } from "../api";

// Map a Sharpe value to a gold-tinted heat intensity relative to the grid range.
function heat(v: number, lo: number, hi: number): number {
  const span = hi - lo || 1;
  return Math.max(0, Math.min(1, (v - lo) / span));
}

export default function Sensitivity({ data }: { data: SensitivityResult }) {
  const { cells, cost_grid, cadence_grid } = data;

  const lookup = useMemo(() => {
    const map = new Map<string, (typeof cells)[number]>();
    for (const c of cells) map.set(`${c.rebalance_months}:${c.cost_bps}`, c);
    return map;
  }, [cells]);

  const sharpes = cells.map((c) => c.sharpe);
  const lo = Math.min(...sharpes);
  const hi = Math.max(...sharpes);

  return (
    <section className="panel">
      <div className="panel__head">
        <span className="eyebrow">Robustness sweep</span>
        <span className="tag">net Sharpe</span>
      </div>

      <p className="drv-lede">
        Does the edge survive friction? Each cell is the strategy&rsquo;s net Sharpe under a
        transaction-cost assumption (columns) and rebalancing cadence (rows). Stable colour
        across the grid signals a robust, non-overfit strategy.
      </p>

      <div className="sw-grid" role="table">
        <div className="sw-row sw-row--head" role="row">
          <span className="sw-corner" role="columnheader">
            months ⟍ bps
          </span>
          {cost_grid.map((bps) => (
            <span key={bps} className="sw-col-head mono" role="columnheader">
              {bps}
            </span>
          ))}
        </div>
        {cadence_grid.map((n) => (
          <div className="sw-row" role="row" key={n}>
            <span className="sw-row-head mono" role="rowheader">
              {n}m
            </span>
            {cost_grid.map((bps) => {
              const cell = lookup.get(`${n}:${bps}`);
              const s = cell?.sharpe ?? 0;
              const t = heat(s, lo, hi);
              return (
                <span
                  key={bps}
                  className="sw-cell mono has-tip"
                  data-tip={
                    cell
                      ? `Sharpe ${num(s)} · return ${signedPct(cell.annual_return)} · maxDD ${signedPct(cell.max_drawdown)}`
                      : "n/a"
                  }
                  style={{ background: `rgba(194, 161, 90, ${0.12 + t * 0.6})` }}
                >
                  {num(s, 2)}
                </span>
              );
            })}
          </div>
        ))}
      </div>

      <div className="drv-key">
        <span className="drv-key__item ink-3">rows = rebalance cadence</span>
        <span className="drv-key__item ink-3">columns = cost per unit turnover</span>
      </div>
    </section>
  );
}
