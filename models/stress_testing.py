"""
Monte Carlo Stress Testing Engine.
Simulates forward paths under each regime and computes VaR/CVaR.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy import stats


class MonteCarloStressTest:
    """
    Monte Carlo simulation for portfolio stress testing.
    Generates forward scenarios conditioned on current regime.
    """

    def __init__(
        self,
        n_simulations: int = 10_000,
        horizon_months: int = 12,
        confidence_levels: list = None,
        seed: int = 42,
    ):
        self.n_simulations = n_simulations
        self.horizon_months = horizon_months
        self.confidence_levels = confidence_levels or [0.95, 0.99]
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def simulate_regime_paths(
        self,
        current_regime: str,
        transition_matrix: pd.DataFrame,
        regime_return_params: Dict[str, Dict[str, Tuple[float, float]]],
    ) -> Dict[str, np.ndarray]:
        """
        Simulate regime-conditional forward paths.

        Args:
            current_regime: Starting regime
            transition_matrix: Regime transition probability matrix
            regime_return_params: {regime: {asset: (mean, std)}}

        Returns:
            Dict with simulated paths per asset
        """
        regimes = list(transition_matrix.index)
        current_idx = regimes.index(current_regime)
        trans_mat = transition_matrix.values

        assets = list(list(regime_return_params.values())[0].keys())
        n_assets = len(assets)

        # Storage: (n_simulations, horizon, n_assets)
        simulated_returns = np.zeros(
            (self.n_simulations, self.horizon_months, n_assets)
        )
        regime_paths = np.zeros(
            (self.n_simulations, self.horizon_months), dtype=int
        )

        for sim in range(self.n_simulations):
            regime_idx = current_idx
            for t in range(self.horizon_months):
                # Transition to next regime
                regime_idx = self.rng.choice(
                    len(regimes), p=trans_mat[regime_idx]
                )
                regime_paths[sim, t] = regime_idx
                regime_name = regimes[regime_idx]

                # Generate returns for this regime
                for a, asset in enumerate(assets):
                    mu, sigma = regime_return_params[regime_name][asset]
                    simulated_returns[sim, t, a] = self.rng.normal(mu, sigma)

        return {
            "returns": simulated_returns,
            "regime_paths": regime_paths,
            "assets": assets,
            "regimes": regimes,
        }

    def compute_portfolio_var(
        self,
        simulated_returns: np.ndarray,
        weights: pd.Series,
        assets: list,
    ) -> Dict:
        """
        Compute VaR and CVaR from simulated paths.

        Args:
            simulated_returns: (n_sims, horizon, n_assets) array
            weights: Portfolio weights
            assets: Asset names

        Returns:
            Dict with VaR, CVaR, and distribution stats
        """
        # Align weights with assets
        w = np.array([weights.get(a, 0.0) for a in assets])

        # Compute portfolio returns per simulation
        # Cumulative return over horizon
        cumulative_returns = np.zeros(self.n_simulations)

        for sim in range(self.n_simulations):
            portfolio_monthly = simulated_returns[sim] @ w
            cumulative_returns[sim] = np.prod(1 + portfolio_monthly) - 1

        results = {
            "mean_return": np.mean(cumulative_returns),
            "median_return": np.median(cumulative_returns),
            "std_return": np.std(cumulative_returns),
            "skewness": stats.skew(cumulative_returns),
            "kurtosis": stats.kurtosis(cumulative_returns),
            "best_case": np.percentile(cumulative_returns, 99),
            "worst_case": np.percentile(cumulative_returns, 1),
            "prob_negative": np.mean(cumulative_returns < 0),
            "distribution": cumulative_returns,
        }

        for cl in self.confidence_levels:
            var = np.percentile(cumulative_returns, (1 - cl) * 100)
            # CVaR = expected loss given we exceed VaR
            cvar = cumulative_returns[cumulative_returns <= var].mean()
            results[f"VaR_{cl:.0%}"] = var
            results[f"CVaR_{cl:.0%}"] = cvar

        return results

    def compute_asset_contribution_to_var(
        self,
        simulated_returns: np.ndarray,
        weights: pd.Series,
        assets: list,
        confidence: float = 0.95,
    ) -> pd.Series:
        """Compute marginal VaR contribution per asset."""
        w = np.array([weights.get(a, 0.0) for a in assets])
        base_var = self.compute_portfolio_var(simulated_returns, weights, assets)
        base_var_value = base_var[f"VaR_{confidence:.0%}"]

        contributions = {}
        delta = 0.01  # 1% perturbation

        for i, asset in enumerate(assets):
            if w[i] < 0.001:
                contributions[asset] = 0.0
                continue

            # Perturb weight up
            w_up = w.copy()
            w_up[i] += delta
            w_up = w_up / w_up.sum()
            weights_up = pd.Series(w_up, index=assets)
            var_up = self.compute_portfolio_var(simulated_returns, weights_up, assets)

            contributions[asset] = (
                var_up[f"VaR_{confidence:.0%}"] - base_var_value
            ) / delta * w[i]

        return pd.Series(contributions, name="VaR_Contribution")

    def run_scenario_analysis(
        self,
        simulated_returns: np.ndarray,
        weights: pd.Series,
        assets: list,
        scenarios: Dict[str, Dict[str, float]],
    ) -> pd.DataFrame:
        """
        Run deterministic stress scenarios.

        Args:
            scenarios: {scenario_name: {asset: shock_return}}
        """
        w = np.array([weights.get(a, 0.0) for a in assets])
        results = []

        for name, shocks in scenarios.items():
            shock_returns = np.array([shocks.get(a, 0.0) for a in assets])
            portfolio_loss = shock_returns @ w
            results.append({
                "Scenario": name,
                "Portfolio Impact": portfolio_loss,
                **{f"{a}_Shock": shocks.get(a, 0.0) for a in assets},
            })

        return pd.DataFrame(results).set_index("Scenario")


# ─── Pre-defined Stress Scenarios ─────────────────────────────────────────────
STRESS_SCENARIOS = {
    "2008 GFC Replay": {
        "US_Equity": -0.38, "Intl_Equity": -0.43, "EM_Equity": -0.53,
        "US_Bonds": 0.05, "TIPS": -0.02, "Gold": 0.05,
        "Commodities": -0.36, "Real_Estate": -0.37, "Cash": 0.02,
    },
    "2020 COVID Crash": {
        "US_Equity": -0.34, "Intl_Equity": -0.33, "EM_Equity": -0.32,
        "US_Bonds": 0.03, "TIPS": 0.01, "Gold": 0.03,
        "Commodities": -0.25, "Real_Estate": -0.27, "Cash": 0.01,
    },
    "Stagflation Shock": {
        "US_Equity": -0.20, "Intl_Equity": -0.25, "EM_Equity": -0.30,
        "US_Bonds": -0.12, "TIPS": 0.05, "Gold": 0.15,
        "Commodities": 0.20, "Real_Estate": -0.15, "Cash": 0.03,
    },
    "Rate Shock (+300bps)": {
        "US_Equity": -0.15, "Intl_Equity": -0.18, "EM_Equity": -0.25,
        "US_Bonds": -0.15, "TIPS": -0.08, "Gold": -0.05,
        "Commodities": -0.05, "Real_Estate": -0.20, "Cash": 0.04,
    },
    "Dollar Crisis": {
        "US_Equity": -0.10, "Intl_Equity": 0.10, "EM_Equity": 0.08,
        "US_Bonds": -0.08, "TIPS": 0.02, "Gold": 0.25,
        "Commodities": 0.15, "Real_Estate": -0.05, "Cash": 0.01,
    },
    "Tech Bubble Burst": {
        "US_Equity": -0.30, "Intl_Equity": -0.20, "EM_Equity": -0.15,
        "US_Bonds": 0.08, "TIPS": 0.04, "Gold": 0.10,
        "Commodities": -0.05, "Real_Estate": -0.10, "Cash": 0.02,
    },
}
