import type { TransitionMatrix } from "../types";
import { REGIME_COLORS, REGIME_ORDER } from "../api";

const SHORT: Record<string, string> = {
  Expansion: "EXP",
  Recovery: "REC",
  Slowdown: "SLW",
  Recession: "RCN",
};

export default function Transition({ data }: { data: TransitionMatrix }) {
  // pandas to_dict() => matrix[to][from]. Rows = from, cols = to.
  const order = REGIME_ORDER.filter((r) => data.regimes.includes(r));
  const val = (from: string, to: string) => data.matrix[to]?.[from] ?? 0;

  return (
    <section className="panel">
      <div className="panel__head">
        <span className="eyebrow">Transition matrix</span>
        <span className="tag">monthly P(next)</span>
      </div>

      <div className="matrix">
        <div className="matrix__corner" />
        {order.map((to) => (
          <div className="matrix__colhead" key={to}>
            <span
              className="matrix__dot"
              style={{ background: REGIME_COLORS[to] }}
            />
            {SHORT[to]}
          </div>
        ))}

        {order.map((from) => (
          <FragmentRow key={from} from={from} order={order} val={val} />
        ))}
      </div>

      <p className="panel__note">
        Rows are the current state, columns the next-month state. Diagonal
        dominance signals regime persistence.
      </p>
    </section>
  );
}

function FragmentRow({
  from,
  order,
  val,
}: {
  from: string;
  order: string[];
  val: (from: string, to: string) => number;
}) {
  return (
    <>
      <div className="matrix__rowhead">
        <span
          className="matrix__dot"
          style={{ background: REGIME_COLORS[from] }}
        />
        {SHORT[from]}
      </div>
      {order.map((to) => {
        const v = val(from, to);
        const strong = v > 0.5;
        return (
          <div
            className="matrix__cell"
            key={to}
            style={{
              background: `rgba(194,161,90,${Math.min(v, 1) * 0.55})`,
            }}
          >
            <span
              className="mono"
              style={{ color: strong ? "var(--bg)" : "var(--ink-2)" }}
            >
              {v < 0.005 ? "·" : (v * 100).toFixed(0)}
            </span>
          </div>
        );
      })}
    </>
  );
}
