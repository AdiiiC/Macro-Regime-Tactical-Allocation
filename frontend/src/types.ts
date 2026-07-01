export type RegimeName = "Expansion" | "Slowdown" | "Recession" | "Recovery";

export type MarketKey = "us" | "india";

export interface MarketInfo {
  key: MarketKey;
  label: string;
  currency: string;
  loaded: boolean;
  has_returns: boolean;
  assets: string[];
}

export interface MarketsResponse {
  markets: MarketInfo[];
}

export interface Health {
  status: string;
  model_loaded: boolean;
  data_loaded: boolean;
  last_updated: string;
  markets: string[];
}

export interface CurrentRegime {
  current_regime: RegimeName;
  confidence: number;
  regime_probabilities: Record<string, number>;
  expected_duration_months: number;
  timestamp: string;
}

export interface RegimeHistoryPoint {
  date: string;
  regime: RegimeName;
}

export interface RegimeHistory {
  regimes: RegimeHistoryPoint[];
  count: number;
}

export interface TransitionMatrix {
  matrix: Record<string, Record<string, number>>;
  regimes: string[];
}

export interface Allocation {
  regime: RegimeName;
  confidence: number;
  target_weights: Record<string, number>;
  benchmark_weights: Record<string, number>;
  rationale: string;
  overweight: string[];
  underweight: string[];
}

export interface VarResult {
  var_95: number;
  cvar_95: number;
  var_99: number;
  cvar_99: number;
  probability_of_loss: number;
  expected_return: number;
  horizon_months: number;
}

export interface StressScenario {
  portfolio_impact: number;
  shocks: Record<string, number>;
}

export interface StressResult {
  regime: RegimeName;
  scenarios: Record<string, StressScenario>;
}

export interface BacktestMetrics {
  annual_return_strategy: number;
  annual_return_benchmark: number;
  annual_vol_strategy: number;
  annual_vol_benchmark: number;
  sharpe_strategy: number;
  sharpe_benchmark: number;
  sortino_strategy: number;
  calmar_strategy: number;
  max_drawdown_strategy: number;
  max_drawdown_benchmark: number;
  information_ratio: number;
  tracking_error: number;
  win_rate: number;
  total_return_strategy: number;
  total_return_benchmark: number;
}

export interface EquityPoint {
  date: string;
  strategy: number;
  benchmark: number;
}

export interface RollingSharpePoint {
  date: string;
  strategy: number | null;
  benchmark: number | null;
}

export interface RegimeAttribution {
  regime: RegimeName;
  months: number;
  contribution: number;
  avg_monthly_return: number;
}

export interface BacktestResult {
  market: string;
  currency: string;
  start: string;
  end: string;
  metrics: BacktestMetrics;
  equity_curve: EquityPoint[];
  drawdown: EquityPoint[];
  rolling_sharpe: RollingSharpePoint[];
  regime_attribution: RegimeAttribution[];
}

export interface RegimeDriver {
  feature: string;
  z_score: number;
  direction: "elevated" | "depressed";
  regime_avg: number;
}

export interface DriversResult {
  market: string;
  regime: RegimeName;
  as_of: string;
  drivers: RegimeDriver[];
}

export interface ModelSummary {
  name: string;
  type: string;
  current_regime: RegimeName | null;
  agreement_with_hmm: number | null;
  quality: number | null;
  quality_label: string;
  note: string;
}

export interface ComparisonTimelinePoint {
  date: string;
  hmm?: RegimeName | null;
  kmeans?: RegimeName | null;
  lstm?: RegimeName | null;
}

export interface ComparisonResult {
  market: string;
  models: ModelSummary[];
  timeline: ComparisonTimelinePoint[];
}
