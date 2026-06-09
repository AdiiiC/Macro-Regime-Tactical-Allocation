"""
Generate synthetic cached data for demo mode (no API key required).
Run this script once to create sample data for the dashboard.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)


def generate_synthetic_macro_features():
    """Generate realistic synthetic macro features with regime structure."""
    n_months = 240  # 20 years
    dates = pd.date_range("2005-01-01", periods=n_months, freq="ME")

    # Define regime sequence (realistic cycle)
    regimes = []
    regime_order = ["Expansion", "Slowdown", "Recession", "Recovery"]
    durations = [24, 12, 10, 14]  # Average months per regime

    idx = 0
    while len(regimes) < n_months:
        r = regime_order[idx % 4]
        d = durations[idx % 4] + np.random.randint(-3, 4)
        d = max(6, d)
        regimes.extend([r] * d)
        idx += 1

    regimes = regimes[:n_months]

    # Generate features based on regime
    regime_means = {
        "Expansion": np.array([1.5, 0.8, -0.5, -0.8, 1.2, -0.3, 0.6, -0.4, 1.0, -0.5]),
        "Slowdown": np.array([-0.3, -0.2, 0.8, 0.5, -0.5, 0.6, -0.3, 0.4, -0.2, 0.3]),
        "Recession": np.array([-1.8, -1.2, 1.8, 1.5, -1.5, 1.5, -1.0, 1.2, -1.5, 1.0]),
        "Recovery": np.array([0.8, 0.5, -0.3, -0.5, 0.8, -0.2, 0.4, -0.3, 0.6, -0.2]),
    }

    columns = [
        "GDP_YoY", "Industrial_Production_YoY", "VIX_Level", "BAA_Spread_Level",
        "Nonfarm_Payrolls_YoY", "Financial_Stress_Level", "CPI_YoY",
        "Yield_Spread_10Y2Y_Level", "Housing_Starts_YoY", "Initial_Claims_Chg3",
    ]

    data = np.zeros((n_months, 10))
    for i, regime in enumerate(regimes):
        mean = regime_means[regime]
        # Add autocorrelation for realism
        if i > 0:
            data[i] = 0.7 * data[i - 1] + 0.3 * mean + np.random.randn(10) * 0.4
        else:
            data[i] = mean + np.random.randn(10) * 0.4

    features = pd.DataFrame(data, index=dates, columns=columns)
    return features, regimes


def generate_synthetic_market_returns(regimes):
    """Generate synthetic asset returns conditioned on regimes."""
    n_months = len(regimes)
    dates = pd.date_range("2005-01-01", periods=n_months, freq="ME")

    # Expected monthly returns by regime
    regime_returns = {
        "Expansion": {
            "US_Equity": 0.012, "Intl_Equity": 0.010, "EM_Equity": 0.013,
            "US_Bonds": 0.002, "TIPS": 0.003, "Gold": 0.002,
            "Commodities": 0.008, "Real_Estate": 0.010, "Cash": 0.003,
        },
        "Slowdown": {
            "US_Equity": 0.003, "Intl_Equity": 0.001, "EM_Equity": -0.002,
            "US_Bonds": 0.004, "TIPS": 0.004, "Gold": 0.006,
            "Commodities": -0.002, "Real_Estate": 0.001, "Cash": 0.003,
        },
        "Recession": {
            "US_Equity": -0.025, "Intl_Equity": -0.030, "EM_Equity": -0.035,
            "US_Bonds": 0.006, "TIPS": 0.003, "Gold": 0.010,
            "Commodities": -0.020, "Real_Estate": -0.025, "Cash": 0.002,
        },
        "Recovery": {
            "US_Equity": 0.015, "Intl_Equity": 0.012, "EM_Equity": 0.018,
            "US_Bonds": 0.003, "TIPS": 0.004, "Gold": 0.004,
            "Commodities": 0.012, "Real_Estate": 0.014, "Cash": 0.002,
        },
    }

    # Volatilities by asset
    vols = {
        "US_Equity": 0.045, "Intl_Equity": 0.055, "EM_Equity": 0.065,
        "US_Bonds": 0.015, "TIPS": 0.018, "Gold": 0.040,
        "Commodities": 0.055, "Real_Estate": 0.050, "Cash": 0.001,
    }

    assets = list(vols.keys())
    returns_data = {}

    for asset in assets:
        rets = []
        for regime in regimes:
            mean = regime_returns[regime][asset]
            vol = vols[asset]
            # Increase vol in recession
            if regime == "Recession":
                vol *= 1.5
            rets.append(np.random.normal(mean, vol))
        returns_data[asset] = rets

    returns = pd.DataFrame(returns_data, index=dates)
    return returns


if __name__ == "__main__":
    print("Generating synthetic data for demo mode...")

    features, regimes = generate_synthetic_macro_features()
    returns = generate_synthetic_market_returns(regimes)

    # Save to cache directory
    cache_dir = os.path.join(os.path.dirname(__file__), "data", "cache")
    os.makedirs(cache_dir, exist_ok=True)

    features.to_parquet(os.path.join(cache_dir, "macro_features.parquet"))
    returns.to_parquet(os.path.join(cache_dir, "market_returns.parquet"))

    print(f"✅ Saved {len(features)} months of macro features")
    print(f"✅ Saved {len(returns)} months of market returns")
    print(f"📁 Cache directory: {cache_dir}")
    print(f"\nRegime distribution:")
    regime_series = pd.Series(regimes)
    print(regime_series.value_counts())
    print(f"\nRun the dashboard: streamlit run dashboard/app.py")
