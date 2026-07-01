export type RegimeName = "Expansion" | "Slowdown" | "Recession" | "Recovery";

export interface Health {
  status: string;
  model_loaded: boolean;
  data_loaded: boolean;
  last_updated: string;
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
