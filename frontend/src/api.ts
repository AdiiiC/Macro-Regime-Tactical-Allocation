import type {
  Allocation,
  CurrentRegime,
  Health,
  RegimeHistory,
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
    throw new Error(`${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => get<Health>("/health"),
  currentRegime: () => get<CurrentRegime>("/regime/current"),
  history: (months: number) =>
    get<RegimeHistory>(`/regime/history?months=${months}`),
  transition: () => get<TransitionMatrix>("/regime/transition-matrix"),
  allocation: () => get<Allocation>("/allocation/current"),
  allocationFor: (regime: string) =>
    get<Allocation>(`/allocation/regime/${regime}`),
  var: (horizon: number, simulations = 10000) =>
    get<VarResult>(`/risk/var?horizon=${horizon}&simulations=${simulations}`),
  stress: () => get<StressResult>("/risk/stress-scenarios"),
};

export const REGIME_COLORS: Record<string, string> = {
  Expansion: "#4bab73",
  Slowdown: "#cf9b45",
  Recession: "#cf5b4e",
  Recovery: "#4f8fc7",
};

export const ASSET_LABELS: Record<string, string> = {
  US_Equity: "US Equity",
  Intl_Equity: "Intl Equity",
  EM_Equity: "EM Equity",
  US_Bonds: "US Bonds",
  TIPS: "TIPS",
  Gold: "Gold",
  Commodities: "Commodities",
  Real_Estate: "Real Estate",
  Cash: "Cash",
};

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

export const REGIME_ORDER = ["Expansion", "Recovery", "Slowdown", "Recession"];

export const pct = (v: number, digits = 1) => `${(v * 100).toFixed(digits)}%`;
export const signedPct = (v: number, digits = 1) =>
  `${v >= 0 ? "+" : ""}${(v * 100).toFixed(digits)}%`;
