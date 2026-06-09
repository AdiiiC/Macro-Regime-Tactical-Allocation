"""
Tactical Asset Allocation Engine.
Maps detected macro regimes to portfolio allocations with risk constraints.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Optional, Tuple

import sys
sys.path.insert(0, "..")
from config.settings import REGIME_ALLOCATIONS, BENCHMARK_ALLOCATION, ASSET_TICKERS


class TacticalAllocator:
    """
    Maps macro regimes to asset allocations.
    Supports both rule-based and mean-variance optimized allocations.
    """

    def __init__(
        self,
        regime_allocations: Dict = REGIME_ALLOCATIONS,
        benchmark: Dict = BENCHMARK_ALLOCATION,
        max_deviation: float = 0.15,
        risk_aversion: float = 2.5,
    ):
        self.regime_allocations = regime_allocations
        self.benchmark = benchmark
        self.max_deviation = max_deviation
        self.risk_aversion = risk_aversion
        self.assets = list(ASSET_TICKERS.keys())

    def get_target_allocation(
        self, regime: str, confidence: float = 1.0
    ) -> pd.Series:
        """
        Get target allocation for a given regime.

        Args:
            regime: Detected macro regime name
            confidence: Model confidence (0-1). Lower confidence
                       blends toward benchmark.
        """
        if regime not in self.regime_allocations:
            # Default to benchmark if unknown regime
            return pd.Series(self.benchmark, name="weight")

        tactical = self.regime_allocations[regime]

        # Blend with benchmark based on confidence
        blended = {}
        for asset in self.assets:
            tac_w = tactical.get(asset, 0.0)
            bench_w = self.benchmark.get(asset, 0.0)
            blended[asset] = confidence * tac_w + (1 - confidence) * bench_w

        weights = pd.Series(blended, name="weight")
        # Ensure weights sum to 1
        weights = weights / weights.sum()
        return weights

    def optimize_allocation(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        regime: str,
        confidence: float = 1.0,
    ) -> pd.Series:
        """
        Mean-variance optimization with regime-based constraints.

        Args:
            expected_returns: Expected annual returns per asset
            cov_matrix: Covariance matrix of asset returns
            regime: Current detected regime
            confidence: Model confidence for constraint tightness
        """
        n_assets = len(self.assets)
        available_assets = [a for a in self.assets if a in expected_returns.index]

        mu = expected_returns[available_assets].values
        sigma = cov_matrix.loc[available_assets, available_assets].values

        # Target allocation as center of constraint
        target = self.get_target_allocation(regime, confidence)
        target_weights = target[available_assets].values

        # Objective: maximize utility = E[r] - (λ/2) * variance + penalty for deviation
        def objective(w):
            port_return = mu @ w
            port_var = w @ sigma @ w
            deviation_penalty = 10 * np.sum((w - target_weights) ** 2)
            return -(port_return - (self.risk_aversion / 2) * port_var - deviation_penalty)

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # fully invested
        ]

        # Bounds: no short selling, max position
        bounds = [(0.0, 0.50) for _ in range(n_assets)]

        # Initial guess: target allocation
        x0 = target_weights

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10},
        )

        if result.success:
            optimal_weights = pd.Series(
                result.x, index=available_assets, name="weight"
            )
        else:
            # Fallback to rule-based
            optimal_weights = pd.Series(
                target_weights, index=available_assets, name="weight"
            )

        return optimal_weights

    def compute_rebalance_trades(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        portfolio_value: float,
        min_trade_pct: float = 0.01,
    ) -> pd.DataFrame:
        """
        Compute trades needed to rebalance from current to target.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            portfolio_value: Total portfolio value
            min_trade_pct: Minimum trade size as % of portfolio
        """
        trades = pd.DataFrame(index=self.assets)
        trades["current_weight"] = current_weights.reindex(self.assets, fill_value=0)
        trades["target_weight"] = target_weights.reindex(self.assets, fill_value=0)
        trades["delta_weight"] = trades["target_weight"] - trades["current_weight"]
        trades["trade_value"] = trades["delta_weight"] * portfolio_value
        trades["action"] = trades["delta_weight"].apply(
            lambda x: "BUY" if x > min_trade_pct else ("SELL" if x < -min_trade_pct else "HOLD")
        )

        # Filter out negligible trades
        trades = trades[trades["action"] != "HOLD"]
        return trades

    def get_regime_tilt_explanation(self, regime: str) -> Dict:
        """Return human-readable explanation of allocation rationale."""
        explanations = {
            "Expansion": {
                "rationale": "Strong growth with moderate inflation favors risk assets.",
                "overweight": ["Equities", "Commodities", "Real Estate"],
                "underweight": ["Bonds", "Cash", "Gold"],
                "key_risks": ["Overheating", "Policy tightening", "Valuation stretch"],
            },
            "Slowdown": {
                "rationale": "Decelerating growth warrants defensive positioning.",
                "overweight": ["Bonds", "TIPS", "Gold"],
                "underweight": ["EM Equity", "Commodities"],
                "key_risks": ["Recession transition", "Credit deterioration"],
            },
            "Recession": {
                "rationale": "Capital preservation paramount. Maximum defensiveness.",
                "overweight": ["Cash", "Treasuries", "Gold"],
                "underweight": ["Equities", "Credit", "Commodities"],
                "key_risks": ["Deflation", "Systemic risk", "Liquidity trap"],
            },
            "Recovery": {
                "rationale": "Early cycle recovery favors cyclicals and risk-on.",
                "overweight": ["Equities", "EM", "Commodities", "Real Estate"],
                "underweight": ["Cash", "Long-duration bonds"],
                "key_risks": ["False recovery", "Policy misstep"],
            },
        }
        return explanations.get(regime, {"rationale": "Unknown regime", "overweight": [], "underweight": [], "key_risks": []})
