"""
Sector Rotation Layer.
Within equities, rotate across S&P 500 sectors based on regime.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Optional


# S&P 500 Sector ETFs
SECTOR_ETFS = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Consumer_Discretionary": "XLY",
    "Consumer_Staples": "XLP",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Materials": "XLB",
    "Real_Estate": "XLRE",
    "Communication": "XLC",
}

# Regime-based sector tilts (overweight/underweight)
SECTOR_REGIME_TILTS = {
    "Expansion": {
        "Technology": 0.18,
        "Consumer_Discretionary": 0.15,
        "Industrials": 0.14,
        "Financials": 0.13,
        "Communication": 0.10,
        "Healthcare": 0.08,
        "Energy": 0.08,
        "Materials": 0.06,
        "Consumer_Staples": 0.04,
        "Real_Estate": 0.02,
        "Utilities": 0.02,
    },
    "Slowdown": {
        "Healthcare": 0.18,
        "Consumer_Staples": 0.16,
        "Utilities": 0.14,
        "Technology": 0.12,
        "Communication": 0.10,
        "Financials": 0.08,
        "Industrials": 0.07,
        "Real_Estate": 0.05,
        "Consumer_Discretionary": 0.04,
        "Materials": 0.03,
        "Energy": 0.03,
    },
    "Recession": {
        "Consumer_Staples": 0.20,
        "Healthcare": 0.20,
        "Utilities": 0.18,
        "Communication": 0.10,
        "Technology": 0.10,
        "Real_Estate": 0.05,
        "Financials": 0.05,
        "Industrials": 0.04,
        "Materials": 0.03,
        "Consumer_Discretionary": 0.03,
        "Energy": 0.02,
    },
    "Recovery": {
        "Financials": 0.16,
        "Consumer_Discretionary": 0.15,
        "Industrials": 0.14,
        "Technology": 0.13,
        "Materials": 0.10,
        "Energy": 0.10,
        "Communication": 0.08,
        "Real_Estate": 0.06,
        "Healthcare": 0.04,
        "Consumer_Staples": 0.02,
        "Utilities": 0.02,
    },
}


class SectorRotator:
    """
    Implements sector rotation within the equity allocation.
    Adjusts sector weights based on detected macro regime.
    """

    def __init__(
        self,
        sector_tilts: Dict = SECTOR_REGIME_TILTS,
        sector_etfs: Dict = SECTOR_ETFS,
    ):
        self.sector_tilts = sector_tilts
        self.sector_etfs = sector_etfs
        self.sector_returns: Optional[pd.DataFrame] = None

    def get_sector_allocation(
        self, regime: str, equity_weight: float, confidence: float = 1.0
    ) -> pd.Series:
        """
        Get sector-level allocation for given regime.

        Args:
            regime: Current macro regime
            equity_weight: Total equity portfolio weight (e.g., 0.35)
            confidence: Blend toward equal-weight at low confidence
        """
        if regime not in self.sector_tilts:
            # Equal weight fallback
            n_sectors = len(self.sector_etfs)
            equal = {s: equity_weight / n_sectors for s in self.sector_etfs}
            return pd.Series(equal, name="sector_weight")

        tactical = self.sector_tilts[regime]

        # Equal weight baseline
        n_sectors = len(tactical)
        equal_weight = 1.0 / n_sectors

        # Blend based on confidence
        blended = {}
        for sector, tilt in tactical.items():
            blended[sector] = confidence * tilt + (1 - confidence) * equal_weight

        # Normalize and scale by equity weight
        weights = pd.Series(blended)
        weights = weights / weights.sum() * equity_weight
        weights.name = "sector_weight"
        return weights

    def fetch_sector_data(self, start: str = "2005-01-01") -> pd.DataFrame:
        """Fetch historical sector ETF returns."""
        tickers = list(self.sector_etfs.values())

        data = yf.download(tickers, start=start, auto_adjust=True, progress=False)

        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"]
        else:
            prices = data

        # Rename to sector names
        reverse_map = {v: k for k, v in self.sector_etfs.items()}
        prices = prices.rename(columns=reverse_map)

        self.sector_returns = prices.resample("ME").last().pct_change().dropna()
        return self.sector_returns

    def compute_sector_momentum(self, lookback: int = 6) -> pd.Series:
        """
        Compute sector momentum scores (6-month trailing return).
        Used as secondary signal for sector selection.
        """
        if self.sector_returns is None:
            raise ValueError("Fetch sector data first.")

        # Trailing N-month return
        cumulative = (1 + self.sector_returns.tail(lookback)).prod() - 1
        return cumulative.sort_values(ascending=False)

    def get_regime_sector_performance(
        self, regimes: pd.Series
    ) -> pd.DataFrame:
        """Compute average sector performance per regime."""
        if self.sector_returns is None:
            raise ValueError("Fetch sector data first.")

        aligned = pd.concat(
            [self.sector_returns, regimes.rename("Regime")], axis=1, join="inner"
        )

        # Annualized return per sector per regime
        result = aligned.groupby("Regime").mean() * 12
        return result.drop(columns=["Regime"], errors="ignore")

    def backtest_sector_rotation(
        self,
        regimes: pd.Series,
        benchmark: str = "equal_weight",
    ) -> Dict:
        """
        Backtest sector rotation strategy vs. benchmark.

        Args:
            regimes: Regime labels over time
            benchmark: 'equal_weight' or 'cap_weight'
        """
        if self.sector_returns is None:
            raise ValueError("Fetch sector data first.")

        common = self.sector_returns.index.intersection(regimes.index)
        returns = self.sector_returns.loc[common]
        regime_series = regimes.loc[common]

        sectors = returns.columns.tolist()
        n_sectors = len(sectors)

        # Strategy returns
        strat_returns = []
        bench_returns = []

        for date, regime in regime_series.items():
            if date not in returns.index:
                continue

            period_ret = returns.loc[date]

            # Tactical weights
            if regime in self.sector_tilts:
                weights = pd.Series(self.sector_tilts[regime])
                weights = weights.reindex(sectors, fill_value=0)
                weights = weights / weights.sum()
            else:
                weights = pd.Series(1.0 / n_sectors, index=sectors)

            strat_ret = (weights * period_ret).sum()
            bench_ret = period_ret.mean()  # equal weight

            strat_returns.append(strat_ret)
            bench_returns.append(bench_ret)

        strat_cum = (1 + pd.Series(strat_returns, index=common)).cumprod()
        bench_cum = (1 + pd.Series(bench_returns, index=common)).cumprod()

        excess_return = (strat_cum.iloc[-1] / bench_cum.iloc[-1] - 1) * 100

        return {
            "strategy_cumulative": strat_cum,
            "benchmark_cumulative": bench_cum,
            "excess_return_pct": excess_return,
            "strategy_annual_return": pd.Series(strat_returns).mean() * 12,
            "benchmark_annual_return": pd.Series(bench_returns).mean() * 12,
        }
