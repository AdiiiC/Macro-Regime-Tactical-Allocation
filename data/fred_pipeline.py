"""
Macro data pipeline: fetches economic indicators from FRED,
computes transformations (YoY changes, z-scores, momentum).
"""

import pandas as pd
import numpy as np
from fredapi import Fred
from datetime import datetime
from typing import Optional

import sys
sys.path.insert(0, "..")
from config.settings import FRED_API_KEY, MACRO_INDICATORS, BACKTEST_START


class MacroDataPipeline:
    """Fetches and transforms macroeconomic data from FRED."""

    def __init__(self, api_key: str = FRED_API_KEY):
        self.fred = Fred(api_key=api_key)
        self.raw_data: Optional[pd.DataFrame] = None
        self.transformed_data: Optional[pd.DataFrame] = None

    def fetch_all_indicators(
        self, start: str = BACKTEST_START, end: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch all configured macro indicators from FRED."""
        if end is None:
            end = datetime.today().strftime("%Y-%m-%d")

        series_dict = {}
        failed = []

        for name, series_id in MACRO_INDICATORS.items():
            try:
                data = self.fred.get_series(
                    series_id, observation_start=start, observation_end=end
                )
                series_dict[name] = data
            except Exception as e:
                failed.append((name, series_id, str(e)))

        if failed:
            print(f"⚠️  Failed to fetch {len(failed)} indicators:")
            for name, sid, err in failed:
                print(f"   - {name} ({sid}): {err}")

        self.raw_data = pd.DataFrame(series_dict)
        # Resample to monthly frequency, forward fill
        self.raw_data = self.raw_data.resample("ME").last().ffill()
        return self.raw_data

    def compute_transformations(self) -> pd.DataFrame:
        """
        Transform raw indicators into model features:
        - YoY percent change (for levels like GDP, CPI)
        - 3-month momentum
        - Z-score normalization (rolling 60-month window)
        """
        if self.raw_data is None:
            raise ValueError("Must fetch data first. Call fetch_all_indicators().")

        df = self.raw_data.copy()
        features = pd.DataFrame(index=df.index)

        # Levels that need YoY transformation
        level_indicators = [
            "GDP", "Industrial_Production", "Retail_Sales", "Nonfarm_Payrolls",
            "CPI", "Core_CPI", "PPI", "M2", "Housing_Starts", "Case_Shiller",
        ]

        for col in df.columns:
            if col in level_indicators:
                # YoY percent change
                features[f"{col}_YoY"] = df[col].pct_change(12) * 100
                # 3-month momentum of YoY
                features[f"{col}_Mom3"] = features[f"{col}_YoY"].diff(3)
            else:
                # Already a rate/spread — use level and change
                features[f"{col}_Level"] = df[col]
                features[f"{col}_Chg3"] = df[col].diff(3)

        # Z-score normalization (rolling 60-month window)
        rolling_mean = features.rolling(window=60, min_periods=24).mean()
        rolling_std = features.rolling(window=60, min_periods=24).std()
        z_scores = (features - rolling_mean) / rolling_std

        # Replace inf with NaN
        z_scores = z_scores.replace([np.inf, -np.inf], np.nan)

        self.transformed_data = z_scores.dropna(how="all")
        return self.transformed_data

    def get_model_ready_data(self) -> pd.DataFrame:
        """Return cleaned, imputed feature matrix ready for HMM."""
        if self.transformed_data is None:
            self.compute_transformations()

        df = self.transformed_data.copy()
        # Forward fill then backward fill remaining NaNs
        df = df.ffill().bfill()
        # Drop any remaining rows with NaN
        df = df.dropna()
        return df

    def get_leading_indicators_dashboard(self) -> pd.DataFrame:
        """Return latest values of key leading indicators for dashboard."""
        if self.raw_data is None:
            raise ValueError("Must fetch data first.")

        leading = [
            "Yield_Spread_10Y2Y", "BAA_Spread", "VIX", "Financial_Stress",
            "Initial_Claims", "PMI_Manufacturing", "Housing_Starts",
            "Breakeven_5Y", "M2",
        ]

        available = [c for c in leading if c in self.raw_data.columns]
        latest = self.raw_data[available].tail(12)
        return latest


def load_cached_data(filepath: str) -> pd.DataFrame:
    """Load previously cached macro data from parquet."""
    return pd.read_parquet(filepath)


def save_cached_data(df: pd.DataFrame, filepath: str) -> None:
    """Save macro data to parquet for offline use."""
    df.to_parquet(filepath)
