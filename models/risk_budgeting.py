"""
Regime-Conditional Risk Budgeting (Risk Parity within each regime).
Equalizes risk contribution per asset rather than dollar weight.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Optional


class RiskBudgetAllocator:
    """
    Risk parity allocator that equalizes marginal risk contribution
    within each regime context.
    """

    def __init__(self, risk_budgets: Optional[Dict[str, float]] = None):
        """
        Args:
            risk_budgets: Target risk budget per asset (must sum to 1).
                         If None, equal risk budget is used.
        """
        self.risk_budgets = risk_budgets

    def compute_risk_parity_weights(
        self,
        cov_matrix: pd.DataFrame,
        risk_budgets: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        """
        Compute risk parity weights given a covariance matrix.

        The optimization minimizes the difference between actual risk
        contributions and target risk budgets.
        """
        assets = cov_matrix.columns.tolist()
        n = len(assets)
        sigma = cov_matrix.values

        # Default: equal risk budget
        if risk_budgets is None:
            budgets = np.ones(n) / n
        else:
            budgets = np.array([risk_budgets.get(a, 1.0 / n) for a in assets])
            budgets = budgets / budgets.sum()

        def objective(w):
            """Minimize sum of squared differences between risk contributions and targets."""
            port_vol = np.sqrt(w @ sigma @ w)
            if port_vol < 1e-10:
                return 1e10

            # Marginal risk contribution
            mrc = sigma @ w / port_vol
            # Risk contribution
            rc = w * mrc
            # Percentage risk contribution
            prc = rc / port_vol

            # Objective: minimize deviation from target budgets
            return np.sum((prc - budgets) ** 2)

        # Constraints: fully invested, long only
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.01, 0.50) for _ in range(n)]  # min 1%, max 50%

        # Initial guess: inverse volatility
        inv_vol = 1.0 / np.sqrt(np.diag(sigma))
        x0 = inv_vol / inv_vol.sum()

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        if result.success:
            weights = pd.Series(result.x, index=assets, name="weight")
        else:
            # Fallback: inverse volatility
            weights = pd.Series(inv_vol / inv_vol.sum(), index=assets, name="weight")

        return weights

    def compute_regime_risk_parity(
        self,
        regime_cov_matrices: Dict[str, pd.DataFrame],
        regime_risk_budgets: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, pd.Series]:
        """
        Compute risk parity allocations for each regime.

        Args:
            regime_cov_matrices: Covariance matrix per regime
            regime_risk_budgets: Optional regime-specific risk budgets
        """
        allocations = {}
        for regime, cov in regime_cov_matrices.items():
            budgets = None
            if regime_risk_budgets and regime in regime_risk_budgets:
                budgets = regime_risk_budgets[regime]
            allocations[regime] = self.compute_risk_parity_weights(cov, budgets)
        return allocations

    @staticmethod
    def compute_risk_contributions(
        weights: pd.Series, cov_matrix: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Decompose portfolio risk into per-asset contributions.

        Returns DataFrame with marginal risk, risk contribution, and percentage.
        """
        w = weights.values
        sigma = cov_matrix.values
        assets = weights.index.tolist()

        port_vol = np.sqrt(w @ sigma @ w)
        mrc = sigma @ w / port_vol
        rc = w * mrc
        prc = rc / port_vol

        return pd.DataFrame(
            {
                "Weight": w,
                "Marginal_Risk_Contribution": mrc,
                "Risk_Contribution": rc,
                "Pct_Risk_Contribution": prc,
            },
            index=assets,
        )

    @staticmethod
    def estimate_regime_covariance(
        returns: pd.DataFrame, regimes: pd.Series
    ) -> Dict[str, pd.DataFrame]:
        """
        Estimate covariance matrix for each regime from historical data.
        Uses shrinkage estimator (Ledoit-Wolf) for stability.
        """
        from sklearn.covariance import LedoitWolf

        aligned = pd.concat([returns, regimes.rename("Regime")], axis=1, join="inner")
        regime_covs = {}

        for regime in aligned["Regime"].unique():
            regime_returns = aligned[aligned["Regime"] == regime].drop(columns=["Regime"])
            if len(regime_returns) > 10:
                lw = LedoitWolf()
                lw.fit(regime_returns.values)
                regime_covs[regime] = pd.DataFrame(
                    lw.covariance_,
                    index=regime_returns.columns,
                    columns=regime_returns.columns,
                )
            else:
                # Not enough data — use sample covariance
                regime_covs[regime] = regime_returns.cov()

        return regime_covs
