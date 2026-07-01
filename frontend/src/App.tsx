import { useEffect, useState } from "react";
import { api } from "./api";
import type {
  Allocation as AllocationT,
  CurrentRegime,
  Health,
  RegimeHistory,
  StressResult,
  TransitionMatrix,
  VarResult,
} from "./types";
import RegimeNow from "./components/RegimeNow";
import Timeline from "./components/Timeline";
import Transition from "./components/Transition";
import Allocation from "./components/Allocation";
import Risk from "./components/Risk";
import Stress from "./components/Stress";

interface Bundle {
  health: Health;
  regime: CurrentRegime;
  history: RegimeHistory;
  transition: TransitionMatrix;
  allocation: AllocationT;
  var: VarResult;
  stress: StressResult;
}

function BrandMark() {
  return (
    <svg width="30" height="30" viewBox="0 0 32 32" aria-hidden="true">
      <rect x="1" y="1" width="30" height="30" rx="3" fill="none" stroke="var(--line-strong)" />
      <path d="M5 21 L12 12 L18 17 L27 7" fill="none" stroke="var(--gold)" strokeWidth="2" strokeLinecap="square" strokeLinejoin="miter" />
      <circle cx="27" cy="7" r="2.4" fill="var(--gold)" />
      <line x1="5" y1="26" x2="27" y2="26" stroke="var(--line)" strokeWidth="1" />
    </svg>
  );
}

export default function App() {
  const [data, setData] = useState<Bundle | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [months, setMonths] = useState(60);

  useEffect(() => {
    let active = true;
    Promise.all([
      api.health(),
      api.currentRegime(),
      api.history(months),
      api.transition(),
      api.allocation(),
      api.var(12),
      api.stress(),
    ])
      .then(([health, regime, history, transition, allocation, v, stress]) => {
        if (active)
          setData({ health, regime, history, transition, allocation, var: v, stress });
      })
      .catch((e) => active && setError(String(e.message ?? e)));
    return () => {
      active = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!data) return;
    let active = true;
    api.history(months).then((h) => active && setData((d) => (d ? { ...d, history: h } : d)));
    return () => {
      active = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [months]);

  if (error) {
    return (
      <div className="boot">
        <BrandMark />
        <h1>Desk offline</h1>
        <p className="mono">{error}</p>
        <p className="boot__hint">
          Start the API: <code>uvicorn api.main:app --port 8001</code>
        </p>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="boot">
        <BrandMark />
        <p className="mono ink-3">Loading regime desk…</p>
      </div>
    );
  }

  const updated = new Date(data.health.last_updated);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <BrandMark />
          <div className="brand__text">
            <span className="brand__name">MERIDIAN</span>
            <span className="brand__tag">Macro Regime &amp; Allocation Desk</span>
          </div>
        </div>
        <div className="status">
          <span className={`status__dot ${data.health.model_loaded ? "ok" : "bad"}`} />
          <span className="mono">
            {data.health.model_loaded ? "MODEL LIVE" : "DEGRADED"}
          </span>
          <span className="status__sep">/</span>
          <span className="mono ink-3">
            {updated.toLocaleDateString(undefined, {
              year: "numeric",
              month: "short",
              day: "2-digit",
            })}
          </span>
        </div>
      </header>

      <main className="grid">
        <div className="col col--main">
          <RegimeNow data={data.regime} />
          <Timeline data={data.history} months={months} onRange={setMonths} />
          <Allocation live={data.allocation} />
        </div>
        <div className="col col--side">
          <Transition data={data.transition} />
          <Risk initial={data.var} />
          <Stress data={data.stress} />
        </div>
      </main>

      <footer className="foot">
        <span className="mono ink-3">
          Hidden Markov regime model &middot; tactical overlay vs 60/40 &middot; Monte Carlo VaR
        </span>
        <span className="mono ink-3">MERIDIAN DESK</span>
      </footer>
    </div>
  );
}
