"""
India macro data pipeline: fetches economic indicators from FRED + World Bank,
computes transformations (YoY changes, z-scores, momentum).
Supplements US pipeline for India-specific regime detection.
"""

import pandas as pd
import numpy as np
from fredapi import Fred
from datetime import datetime
from typing import Optional, Dict
import requests

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import FRED_API_KEY, BACKTEST_START
from config.india_settings import INDIA_FRED_INDICATORS, INDIA_WORLDBANK_INDICATORS


class IndiaDataPipeline:
    """Fetches and transforms Indian macroeconomic data from multiple sources."""

    def __init__(self, api_key: str = FRED_API_KEY):
        self.fred = Fred(api_key=api_key)
        self.raw_data: Optional[pd.DataFrame] = None
        self.transformed_data: Optional[pd.DataFrame] = None

    def fetch_fred_indicators(
        self, start: str = BACKTEST_START, end: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch India-specific indicators from FRED."""
        if end is None:
            end = datetime.today().strftime("%Y-%m-%d")

        series_dict = {}
        failed = []

        for name, series_id in INDIA_FRED_INDICATORS.items():
            try:
                data = self.fred.get_series(
                    series_id, observation_start=start, observation_end=end
                )
                series_dict[name] = data
            except Exception as e:
                failed.append((name, series_id, str(e)))

        if failed:
            print(f"⚠️  Failed to fetch {len(failed)} India FRED indicators:")
            for name, sid, err in failed:
                print(f"   - {name} ({sid}): {err}")

        df = pd.DataFrame(series_dict)
        df = df.resample("ME").last().ffill()
        return df

    def fetch_worldbank_indicators(
        self, start_year: int = 2000, end_year: Optional[int] = None
    ) -> pd.DataFrame:
        """Fetch supplementary data from World Bank API for India."""
        if end_year is None:
            end_year = datetime.today().year

        series_dict = {}

        for name, indicator_id in INDIA_WORLDBANK_INDICATORS.items():
            try:
                url = (
                    f"https://api.worldbank.org/v2/country/IND/indicator/{indicator_id}"
                    f"?date={start_year}:{end_year}&format=json&per_page=100"
                )
                resp = requests.get(url, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    if len(data) > 1 and data[1]:
                        records = {
                            pd.Timestamp(f"{item['date']}-12-31"): item["value"]
                            for item in data[1]
                            if item["value"] is not None
                        }
                        series_dict[name] = pd.Series(records)
            except Exception as e:
                print(f"   ⚠️  World Bank {name}: {e}")

        if series_dict:
            df = pd.DataFrame(series_dict).sort_index()
            # Resample annual to monthly (forward fill)
            df = df.resample("ME").ffill()
            return df
        return pd.DataFrame()

    def fetch_all_indicators(
        self, start: str = BACKTEST_START, end: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch all India indicators from all sources and merge."""
        # FRED indicators (monthly)
        fred_df = self.fetch_fred_indicators(start, end)

        # World Bank indicators (annual, resampled to monthly)
        try:
            start_year = int(start[:4])
            wb_df = self.fetch_worldbank_indicators(start_year=start_year)
        except Exception:
            wb_df = pd.DataFrame()

        # Merge on date index
        if not wb_df.empty:
            self.raw_data = fred_df.join(wb_df, how="outer").ffill()
        else:
            self.raw_data = fred_df

        self.raw_data = self.raw_data.ffill()
        return self.raw_data

    def compute_transformations(self) -> pd.DataFrame:
        """
        Transform raw indicators into model features:
        - YoY percent change (for levels/indices)
        - 3-month momentum
        - Z-score normalization (rolling 48-month window for India - shorter history)
        """
        if self.raw_data is None:
            raise ValueError("Must fetch data first. Call fetch_all_indicators().")

        df = self.raw_data.copy()
        features = pd.DataFrame(index=df.index)

        # Level indicators that need YoY transformation
        level_indicators = [
            "India_CPI", "India_WPI", "India_Industrial_Production",
            "India_M2", "India_GDP_Growth",
        ]

        for col in df.columns:
            if col in level_indicators:
                features[f"{col}_YoY"] = df[col].pct_change(12) * 100
                features[f"{col}_Mom3"] = features[f"{col}_YoY"].diff(3)
            else:
                # Rate/spread/index — use level and 3-month change
                features[f"{col}_Level"] = df[col]
                features[f"{col}_Chg3"] = df[col].diff(3)

        # Z-score normalization (rolling 48-month window — shorter for India data)
        rolling_mean = features.rolling(window=48, min_periods=18).mean()
        rolling_std = features.rolling(window=48, min_periods=18).std()
        z_scores = (features - rolling_mean) / rolling_std

        z_scores = z_scores.replace([np.inf, -np.inf], np.nan)
        self.transformed_data = z_scores.dropna(how="all")
        return self.transformed_data

    def get_model_ready_data(self) -> pd.DataFrame:
        """Return cleaned feature matrix ready for HMM."""
        if self.transformed_data is None:
            self.compute_transformations()

        df = self.transformed_data.copy()
        df = df.ffill().bfill()
        df = df.dropna()
        return df

    def get_leading_indicators_dashboard(self) -> pd.DataFrame:
        """Return latest values of key India leading indicators."""
        if self.raw_data is None:
            raise ValueError("Must fetch data first.")

        leading = [
            "India_Repo_Rate", "India_CPI", "India_Industrial_Production",
            "India_USD_INR", "India_M2", "India_Unemployment", "India_WPI",
        ]

        available = [c for c in leading if c in self.raw_data.columns]
        return self.raw_data[available].tail(12)


def load_cached_india_data(filepath: str) -> pd.DataFrame:
    """Load previously cached India macro data."""
    return pd.read_parquet(filepath)


def save_cached_india_data(df: pd.DataFrame, filepath: str) -> None:
    """Save India macro data to parquet."""
    df.to_parquet(filepath)
