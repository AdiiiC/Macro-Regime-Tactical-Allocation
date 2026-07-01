import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import type {
  Allocation as AllocationT,
  BacktestResult,
  ComparisonResult,
  CurrentRegime,
  DriversResult,
  Health,
  MarketInfo,
  MarketKey,
  RegimeHistory,
  RegimeName,
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
import Backtest from "./components/Backtest";
import Drivers from "./components/Drivers";
import ModelComparison from "./components/ModelComparison";

interface Core {
  health: Health;
  regime: CurrentRegime;
  history: RegimeHistory;
  transition: TransitionMatrix;
  allocation: AllocationT;
}

const THEME_KEY = "meridian-theme";

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

function PanelSkeleton({ title, rows = 4 }: { title: string; rows?: number }) {
  return (
    <section className="panel">
      <div className="panel__head">
        <span className="eyebrow">{title}</span>
        <span className="tag">loading</span>
      </div>
      <div className="skel">
        {Array.from({ length: rows }).map((_, i) => (
          <div className="skel__row" key={i} style={{ width: `${90 - i * 8}%` }} />
        ))}
      </div>
    </section>
  );
}

function PanelError({ title, message }: { title: string; message: string }) {
  return (
    <section className="panel panel--err">
      <div className="panel__head">
        <span className="eyebrow">{title}</span>
        <span className="tag">error</span>
      </div>
      <p className="panel__note">{message}</p>
    </section>
  );
}

// Neutral "this view isn't available for this market" placeholder.
function PanelNote({ title, message }: { title: string; message: string }) {
  return (
    <section className="panel">
      <div className="panel__head">
        <span className="eyebrow">{title}</span>
        <span className="tag">n/a</span>
      </div>
      <p className="panel__note">{message}</p>
    </section>
  );
}

// ── URL helpers (deep-linking market + explored regime) ─────────────────
function readParams(): { market: MarketKey; regime?: RegimeName } {
  const p = new URLSearchParams(window.location.search);
  const market = (p.get("market") === "india" ? "india" : "us") as MarketKey;
  const regime = p.get("regime") as RegimeName | null;
  const valid: RegimeName[] = ["Expansion", "Slowdown", "Recession", "Recovery"];
  return { market, regime: regime && valid.includes(regime) ? regime : undefined };
}
function writeParams(market: MarketKey, regime?: "live" | RegimeName) {
  const p = new URLSearchParams(window.location.search);
  p.set("market", market);
  if (regime && regime !== "live") p.set("regime", regime);
  else p.delete("regime");
  window.history.replaceState(null, "", `?${p.toString()}`);
}

export default function App() {
  const initial = readParams();
  const [market, setMarket] = useState<MarketKey>(initial.market);
  const [markets, setMarkets] = useState<MarketInfo[]>([]);
  const [theme, setTheme] = useState<"dark" | "light">(
    () => (localStorage.getItem(THEME_KEY) as "dark" | "light") ?? "dark"
  );
  const [refreshing, setRefreshing] = useState(false);

  const [core, setCore] = useState<Core | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [months, setMonths] = useState(60);

  const [backtest, setBacktest] = useState<BacktestResult | null>(null);
  const [btErr, setBtErr] = useState<string | null>(null);
  const [drivers, setDrivers] = useState<DriversResult | null>(null);
  const [drvErr, setDrvErr] = useState<string | null>(null);
  const [comparison, setComparison] = useState<ComparisonResult | null>(null);
  const [cmpErr, setCmpErr] = useState<string | null>(null);
  const [risk, setRisk] = useState<VarResult | null>(null);
  const [riskErr, setRiskErr] = useState<string | null>(null);
  const [stress, setStress] = useState<StressResult | null>(null);
  const [stressErr, setStressErr] = useState<string | null>(null);

  // Theme
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  // Markets list (once)
  useEffect(() => {
    api.markets().then((r) => setMarkets(r.markets)).catch(() => void 0);
  }, []);

  const loadMarket = useCallback(
    (m: MarketKey) => {
      setCore(null);
      setError(null);
      setBacktest(null);
      setBtErr(null);
      setDrivers(null);
      setDrvErr(null);
      setComparison(null);
      setCmpErr(null);
      setRisk(null);
      setRiskErr(null);
      setStress(null);
      setStressErr(null);

      let active = true;
      Promise.all([
        api.health(),
        api.currentRegime(m),
        api.history(m, months),
        api.transition(m),
        api.allocation(m),
      ])
        .then(([health, regime, history, transition, allocation]) => {
          if (active)
            setCore({ health, regime, history, transition, allocation });
        })
        .catch((e) => active && setError(String(e.message ?? e)));

      // Lazy analytical panels (independent, non-blocking).
      api.backtest(m).then(setBacktest).catch((e) => setBtErr(String(e.message ?? e)));
      api.drivers(m).then(setDrivers).catch((e) => setDrvErr(String(e.message ?? e)));
      api.comparison(m).then(setComparison).catch((e) => setCmpErr(String(e.message ?? e)));
      api.var(m, 12).then(setRisk).catch((e) => setRiskErr(String(e.message ?? e)));
      api.stress(m).then(setStress).catch((e) => setStressErr(String(e.message ?? e)));

      return () => {
        active = false;
      };
    },
    // months intentionally excluded — history updates separately below
    // eslint-disable-next-line react-hooks/exhaustive-deps
    []
  );

  // Load on market change
  useEffect(() => {
    writeParams(market, initial.regime);
    const cleanup = loadMarket(market);
    return cleanup;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [market]);

  // History range changes
  useEffect(() => {
    if (!core) return;
    let active = true;
    api.history(market, months).then((h) => active && setCore((d) => (d ? { ...d, history: h } : d)));
    return () => {
      active = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [months]);

  const onRefresh = async () => {
    setRefreshing(true);
    try {
      await api.refresh(market);
      loadMarket(market);
    } finally {
      setRefreshing(false);
    }
  };

  const changeMarket = (m: MarketKey) => {
    if (m === market) return;
    setMarket(m);
    writeParams(m);
  };

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

  if (!core) {
    return (
      <div className="boot">
        <BrandMark />
        <p className="mono ink-3">Loading regime desk…</p>
      </div>
    );
  }

  const updated = new Date(core.health.last_updated);

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

        <div className="controls">
          <div className="seg-ctrl">
            {(markets.length
              ? markets
              : [
                  { key: "us", label: "United States" },
                  { key: "india", label: "India" },
                ]
            ).map((mk) => (
              <button
                key={mk.key}
                className={`seg-btn ${market === mk.key ? "is-active" : ""}`}
                onClick={() => changeMarket(mk.key as MarketKey)}
              >
                {mk.key.toUpperCase()}
              </button>
            ))}
          </div>

          <button
            className="icon-btn"
            title="Toggle theme"
            onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
          >
            {theme === "dark" ? "☾" : "☀"}
          </button>

          <button
            className="icon-btn"
            title="Refresh models from latest data"
            onClick={onRefresh}
            disabled={refreshing}
          >
            <span className={refreshing ? "spin" : ""}>⟳</span>
          </button>

          <a className="ghost-btn" href={api.reportUrl(market)} target="_blank" rel="noreferrer">
            Memo ↓
          </a>

          <div className="status">
            <span className={`status__dot ${core.health.model_loaded ? "ok" : "bad"}`} />
            <span className="mono ink-3">
              {updated.toLocaleDateString(undefined, {
                year: "numeric",
                month: "short",
                day: "2-digit",
              })}
            </span>
          </div>
        </div>
      </header>

      <main className="grid">
        <div className="col col--main">
          <RegimeNow data={core.regime} />
          {drivers ? (
            <Drivers data={drivers} />
          ) : drvErr ? (
            <PanelError title="Regime drivers" message={drvErr} />
          ) : (
            <PanelSkeleton title="Regime drivers" rows={6} />
          )}
          <Timeline data={core.history} months={months} onRange={setMonths} />
          <Allocation
            live={core.allocation}
            market={market}
            initialSelected={initial.regime}
            onSelect={(sel) => writeParams(market, sel)}
          />
          {backtest ? (
            <Backtest data={backtest} />
          ) : btErr ? (
            <PanelNote title="Strategy backtest" message={btErr} />
          ) : (
            <PanelSkeleton title="Strategy backtest" rows={5} />
          )}
        </div>

        <div className="col col--side">
          <Transition data={core.transition} />
          {comparison ? (
            <ModelComparison data={comparison} />
          ) : cmpErr ? (
            <PanelError title="Model cross-check" message={cmpErr} />
          ) : (
            <PanelSkeleton title="Model cross-check" rows={5} />
          )}
          {risk ? (
            <Risk initial={risk} market={market} />
          ) : riskErr ? (
            <PanelNote title="Risk · Monte Carlo" message={riskErr} />
          ) : (
            <PanelSkeleton title="Risk · Monte Carlo" rows={4} />
          )}
          {stress ? (
            <Stress data={stress} />
          ) : stressErr ? (
            <PanelNote title="Stress scenarios" message={stressErr} />
          ) : (
            <PanelSkeleton title="Stress scenarios" rows={4} />
          )}
        </div>
      </main>

      <footer className="foot">
        <span className="mono ink-3">
          Hidden Markov regime model &middot; tactical overlay vs benchmark &middot; Monte Carlo VaR
        </span>
        <span className="mono ink-3">MERIDIAN DESK</span>
      </footer>
    </div>
  );
}
