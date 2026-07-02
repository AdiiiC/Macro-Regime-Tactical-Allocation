import type {
  Allocation,
  BacktestResult,
  BenchmarkVariant,
  ComparisonResult,
  CurrentRegime,
  DataStatus,
  DriversResult,
  Health,
  MarketKey,
  MarketsResponse,
  RegimeHistory,
  RegimeLog,
  SensitivityResult,
  StressResult,
  TransitionMatrix,
  VarResult,
} from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "/api";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { Accept: "application/json" },
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

const q = (market: MarketKey, extra = "") =>
  `?market=${market}${extra ? `&${extra}` : ""}`;

export const api = {
  markets: () => get<MarketsResponse>("/markets"),
  health: () => get<Health>("/health"),
  currentRegime: (m: MarketKey) => get<CurrentRegime>(`/regime/current${q(m)}`),
  history: (m: MarketKey, months: number) =>
    get<RegimeHistory>(`/regime/history${q(m, `months=${months}`)}`),
  regimeLog: (m: MarketKey, limit = 120) =>
    get<RegimeLog>(`/regime/log${q(m, `limit=${limit}`)}`),
  transition: (m: MarketKey) =>
    get<TransitionMatrix>(`/regime/transition-matrix${q(m)}`),
  drivers: (m: MarketKey, top = 8) =>
    get<DriversResult>(`/regime/drivers${q(m, `top=${top}`)}`),
  allocation: (m: MarketKey) => get<Allocation>(`/allocation/current${q(m)}`),
  allocationFor: (m: MarketKey, regime: string) =>
    get<Allocation>(`/allocation/regime/${regime}${q(m)}`),
  backtest: (m: MarketKey, benchmark: BenchmarkVariant = "sixty_forty") =>
    get<BacktestResult>(`/backtest${q(m, `benchmark=${benchmark}`)}`),
  sensitivity: (m: MarketKey) =>
    get<SensitivityResult>(`/backtest/sensitivity${q(m)}`),
  comparison: (m: MarketKey) => get<ComparisonResult>(`/model/comparison${q(m)}`),
  var: (m: MarketKey, horizon: number, simulations = 10000) =>
    get<VarResult>(`/risk/var${q(m, `horizon=${horizon}&simulations=${simulations}`)}`),
  stress: (m: MarketKey) => get<StressResult>(`/risk/stress-scenarios${q(m)}`),
  dataStatus: () => get<DataStatus>("/data/status"),
  refresh: (m: MarketKey, live = false) =>
    fetch(`${BASE}/model/refresh${q(m, live ? "live=true" : "")}`, {
      method: "POST",
    }).then((r) => r.json()),
  reportUrl: (m: MarketKey) => `${BASE}/report/pdf${q(m)}`,
};

export const REGIME_COLORS: Record<string, string> = {
  Expansion: "#4bab73",
  Slowdown: "#cf9b45",
  Recession: "#cf5b4e",
  Recovery: "#4f8fc7",
};

export const ASSET_LABELS: Record<string, string> = {
  // US
  US_Equity: "US Equity",
  Intl_Equity: "Intl Equity",
  EM_Equity: "EM Equity",
  US_Bonds: "US Bonds",
  TIPS: "TIPS",
  Gold: "Gold",
  Commodities: "Commodities",
  Real_Estate: "Real Estate",
  Cash: "Cash",
  // India
  Nifty_50: "Nifty 50",
  Bank_Nifty: "Bank Nifty",
  Nifty_Midcap: "Nifty Midcap",
  Gold_INR: "Gold (INR)",
  G_Sec_Long: "G-Sec (Long)",
  Liquid_Fund: "Liquid Fund",
  Nifty_IT: "Nifty IT",
};

export const assetLabel = (key: string) => ASSET_LABELS[key] ?? key.replace(/_/g, " ");

// Default US asset ordering; India derives order from the data itself.
export const ASSET_ORDER = [
  "US_Equity",
  "Intl_Equity",
  "EM_Equity",
  "US_Bonds",
  "TIPS",
  "Gold",
  "Commodities",
  "Real_Estate",
  "Cash",
];

// Return a stable asset ordering for whichever market's data we were handed.
export const orderAssets = (keys: string[]): string[] => {
  const known = ASSET_ORDER.filter((a) => keys.includes(a));
  const rest = keys.filter((k) => !ASSET_ORDER.includes(k));
  return [...known, ...rest];
};

export const REGIME_ORDER = ["Expansion", "Recovery", "Slowdown", "Recession"];

export const pct = (v: number, digits = 1) => `${(v * 100).toFixed(digits)}%`;
export const signedPct = (v: number, digits = 1) =>
  `${v >= 0 ? "+" : ""}${(v * 100).toFixed(digits)}%`;
export const num = (v: number, digits = 2) => v.toFixed(digits);
export const signedNum = (v: number, digits = 2) =>
  `${v >= 0 ? "+" : ""}${v.toFixed(digits)}`;
