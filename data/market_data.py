"""
Market data pipeline: fetches asset class returns using yfinance.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Optional

import sys
sys.path.insert(0, "..")
from config.settings import ASSET_TICKERS, BACKTEST_START


class MarketDataPipeline:
    """Fetches and processes asset class price/return data."""

    def __init__(self, tickers: Dict[str, str] = ASSET_TICKERS):
        self.tickers = tickers
        self.prices: Optional[pd.DataFrame] = None
        self.returns: Optional[pd.DataFrame] = None

    def fetch_prices(
        self, start: str = BACKTEST_START, end: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch adjusted close prices for all asset classes."""
        ticker_list = list(self.tickers.values())

        data = yf.download(
            ticker_list, start=start, end=end, auto_adjust=True, progress=False
        )

        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"]
        else:
            prices = data

        # Rename columns from tickers to asset names
        reverse_map = {v: k for k, v in self.tickers.items()}
        prices = prices.rename(columns=reverse_map)

        self.prices = prices.ffill().dropna()
        return self.prices

    def compute_returns(self, frequency: str = "M") -> pd.DataFrame:
        """
        Compute periodic returns.

        Args:
            frequency: 'D' for daily, 'M' for monthly
        """
        if self.prices is None:
            self.fetch_prices()

        if frequency == "M":
            monthly_prices = self.prices.resample("ME").last()
            self.returns = monthly_prices.pct_change().dropna()
        else:
            self.returns = self.prices.pct_change().dropna()

        return self.returns

    def get_correlation_matrix(self) -> pd.DataFrame:
        """Compute rolling correlation matrix of asset returns."""
        if self.returns is None:
            self.compute_returns()
        return self.returns.corr()

    def get_regime_conditional_returns(
        self, regimes: pd.Series
    ) -> pd.DataFrame:
        """
        Compute mean annualized returns per asset per regime.

        Args:
            regimes: Series with regime labels indexed by date
        """
        if self.returns is None:
            self.compute_returns()

        # Align dates
        aligned = pd.concat(
            [self.returns, regimes.rename("Regime")], axis=1, join="inner"
        )

        stats = aligned.groupby("Regime").agg(["mean", "std"])

        # Annualize (monthly -> annual)
        result = pd.DataFrame()
        for asset in self.tickers.keys():
            if asset in aligned.columns:
                result.loc[asset, "columns"] = None  # placeholder
                for regime in aligned["Regime"].unique():
                    mask = aligned["Regime"] == regime
                    regime_rets = aligned.loc[mask, asset]
                    result.loc[asset, f"{regime}_AnnReturn"] = regime_rets.mean() * 12
                    result.loc[asset, f"{regime}_AnnVol"] = regime_rets.std() * np.sqrt(12)
                    result.loc[asset, f"{regime}_Sharpe"] = (
                        (regime_rets.mean() * 12) / (regime_rets.std() * np.sqrt(12))
                        if regime_rets.std() > 0
                        else 0
                    )

        if "columns" in result.columns:
            result = result.drop(columns=["columns"])

        return result
