"""
India market data pipeline: fetches Nifty, Sensex, Gold, G-Secs via Yahoo Finance.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import BACKTEST_START
from config.india_settings import INDIA_ASSET_TICKERS


class IndiaMarketDataPipeline:
    """Fetches and processes Indian asset class price/return data."""

    def __init__(self, tickers: Dict[str, str] = INDIA_ASSET_TICKERS):
        self.tickers = tickers
        self.prices: Optional[pd.DataFrame] = None
        self.returns: Optional[pd.DataFrame] = None

    def fetch_prices(
        self, start: str = BACKTEST_START, end: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch adjusted close prices for all Indian asset classes."""
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

        # Some India tickers may have gaps — forward fill
        self.prices = prices.ffill().dropna(how="all")
        return self.prices

    def compute_returns(self, frequency: str = "M") -> pd.DataFrame:
        """Compute periodic returns."""
        if self.prices is None:
            self.fetch_prices()

        if frequency == "M":
            monthly_prices = self.prices.resample("ME").last()
            self.returns = monthly_prices.pct_change().dropna()
        else:
            self.returns = self.prices.pct_change().dropna()

        return self.returns

    def get_correlation_matrix(self) -> pd.DataFrame:
        """Compute correlation matrix of Indian asset returns."""
        if self.returns is None:
            self.compute_returns()
        return self.returns.corr()

    def get_regime_conditional_returns(
        self, regimes: pd.Series
    ) -> pd.DataFrame:
        """Compute mean annualized returns per asset per regime."""
        if self.returns is None:
            self.compute_returns()

        common_idx = self.returns.index.intersection(regimes.index)
        if len(common_idx) == 0:
            return pd.DataFrame()

        aligned_returns = self.returns.loc[common_idx]
        aligned_regimes = regimes.loc[common_idx]

        regime_returns = {}
        for regime in aligned_regimes.unique():
            mask = aligned_regimes == regime
            regime_returns[regime] = aligned_returns[mask].mean() * 12  # annualize

        return pd.DataFrame(regime_returns)
