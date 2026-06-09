"""
Leverage & Position Sizing Module.
Implements Kelly Criterion and dynamic leverage based on regime confidence.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from scipy.optimize import minimize_scalar


class KellyCriterion:
    """
    Kelly Criterion for optimal position sizing.
    Adjusts leverage based on edge and variance.
    """

    def __init__(
        self,
        max_leverage: float = 2.0,
        min_leverage: float = 0.0,
        kelly_fraction: float = 0.5,  # Half-Kelly for conservatism
    ):
        """
        Args:
            max_leverage: Maximum allowed leverage
            min_leverage: Minimum (can be 0 for no shorting, or negative for short)
            kelly_fraction: Fraction of full Kelly to use (0.5 = half-Kelly)
        """
        self.max_leverage = max_leverage
        self.min_leverage = min_leverage
        self.kelly_fraction = kelly_fraction

    def compute_kelly_leverage(
        self,
        expected_return: float,
        volatility: float,
        risk_free_rate: float = 0.04,
    ) -> float:
        """
        Compute optimal Kelly leverage for a single strategy.

        Kelly fraction = (μ - rf) / σ²

        Args:
            expected_return: Expected annual return of strategy
            volatility: Annual volatility
            risk_free_rate: Annual risk-free rate
        """
        if volatility <= 0:
            return 1.0

        excess_return = expected_return - risk_free_rate
        full_kelly = excess_return / (volatility ** 2)

        # Apply fraction
        position = full_kelly * self.kelly_fraction

        # Clamp to bounds
        position = np.clip(position, self.min_leverage, self.max_leverage)

        return position

    def compute_regime_leverage(
        self,
        regime: str,
        regime_stats: Dict[str, Dict[str, float]],
        confidence: float = 1.0,
    ) -> float:
        """
        Compute leverage based on regime-conditional return/risk.

        Args:
            regime: Current regime
            regime_stats: {regime: {mean_return, volatility}}
            confidence: Model confidence (scales toward 1x at low confidence)
        """
        if regime not in regime_stats:
            return 1.0

        stats = regime_stats[regime]
        full_leverage = self.compute_kelly_leverage(
            stats["mean_return"], stats["volatility"]
        )

        # Blend toward 1x leverage at low confidence
        leverage = confidence * full_leverage + (1 - confidence) * 1.0

        return np.clip(leverage, self.min_leverage, self.max_leverage)

    def compute_multi_asset_kelly(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float = 0.04,
    ) -> pd.Series:
        """
        Multi-asset Kelly: optimal weights for multiple assets.

        w* = (1/γ) * Σ⁻¹ * (μ - rf)

        Where γ is the risk aversion (inverse of Kelly fraction).
        """
        assets = expected_returns.index.tolist()
        mu = expected_returns.values - risk_free_rate
        sigma_inv = np.linalg.inv(cov_matrix.loc[assets, assets].values)

        # Full Kelly weights
        gamma = 1.0 / self.kelly_fraction
        weights = (1.0 / gamma) * sigma_inv @ mu

        # Clamp individual positions
        weights = np.clip(weights, self.min_leverage / len(assets), self.max_leverage / len(assets))

        # If total leverage exceeds max, scale down
        total_leverage = np.sum(np.abs(weights))
        if total_leverage > self.max_leverage:
            weights = weights * (self.max_leverage / total_leverage)

        return pd.Series(weights, index=assets, name="kelly_weight")


class DynamicLeverageManager:
    """
    Manages leverage dynamically based on:
    - Regime confidence
    - Volatility regime
    - Drawdown control
    """

    def __init__(
        self,
        base_leverage: float = 1.0,
        max_leverage: float = 2.0,
        drawdown_threshold: float = -0.10,
        vol_target: float = 0.10,  # 10% target vol
    ):
        self.base_leverage = base_leverage
        self.max_leverage = max_leverage
        self.drawdown_threshold = drawdown_threshold
        self.vol_target = vol_target

    def compute_vol_target_leverage(
        self, realized_vol: float
    ) -> float:
        """
        Volatility targeting: scale leverage to achieve constant vol.

        leverage = target_vol / realized_vol
        """
        if realized_vol <= 0:
            return self.base_leverage

        leverage = self.vol_target / realized_vol
        return np.clip(leverage, 0.1, self.max_leverage)

    def compute_drawdown_adjusted_leverage(
        self,
        current_drawdown: float,
        base_leverage: float,
    ) -> float:
        """
        Reduce leverage as drawdown deepens (CPPI-like).

        If drawdown exceeds threshold, linearly reduce to 0.
        """
        if current_drawdown >= 0:
            return base_leverage

        if current_drawdown <= self.drawdown_threshold:
            # Fully de-risked
            return 0.0

        # Linear interpolation
        ratio = current_drawdown / self.drawdown_threshold
        return base_leverage * (1 - ratio)

    def get_dynamic_leverage(
        self,
        regime: str,
        confidence: float,
        realized_vol: float,
        current_drawdown: float,
        regime_stats: Dict[str, Dict[str, float]],
    ) -> Dict:
        """
        Combine all leverage signals into final position size.

        Returns:
            Dict with component leverages and final combined leverage
        """
        kelly = KellyCriterion(
            max_leverage=self.max_leverage,
            kelly_fraction=0.5,
        )

        # Kelly leverage
        kelly_lev = kelly.compute_regime_leverage(regime, regime_stats, confidence)

        # Vol-target leverage
        vol_lev = self.compute_vol_target_leverage(realized_vol)

        # Drawdown-adjusted
        dd_lev = self.compute_drawdown_adjusted_leverage(
            current_drawdown, self.base_leverage
        )

        # Final: minimum of all signals (most conservative wins)
        final_leverage = min(kelly_lev, vol_lev, dd_lev, self.max_leverage)
        final_leverage = max(final_leverage, 0.0)

        return {
            "kelly_leverage": kelly_lev,
            "vol_target_leverage": vol_lev,
            "drawdown_leverage": dd_lev,
            "final_leverage": final_leverage,
            "regime": regime,
            "confidence": confidence,
            "realized_vol": realized_vol,
            "current_drawdown": current_drawdown,
        }
