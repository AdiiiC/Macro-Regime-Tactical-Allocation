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


def generate_india_macro_features():
    """Generate realistic synthetic India macro features with regime structure."""
    n_months = 240  # 20 years
    dates = pd.date_range("2005-01-01", periods=n_months, freq="ME")

    # India regime sequence (slightly different cycle timing)
    regimes = []
    regime_order = ["Expansion", "Slowdown", "Recession", "Recovery"]
    durations = [28, 10, 8, 14]  # India: longer expansions, shorter recessions

    idx = 0
    while len(regimes) < n_months:
        r = regime_order[idx % 4]
        d = durations[idx % 4] + np.random.randint(-3, 5)
        d = max(5, d)
        regimes.extend([r] * d)
        idx += 1

    regimes = regimes[:n_months]

    # India-specific regime characteristics
    regime_means = {
        "Expansion": np.array([1.2, 0.9, -0.4, -0.3, 0.5, 0.8, -0.6, 1.0]),
        "Slowdown": np.array([-0.2, -0.3, 0.6, 0.8, -0.2, -0.3, 0.5, -0.3]),
        "Recession": np.array([-1.5, -1.0, 1.5, 1.2, -0.8, -1.2, 1.0, -1.0]),
        "Recovery": np.array([0.6, 0.4, -0.2, -0.4, 0.3, 0.5, -0.3, 0.5]),
    }

    columns = [
        "India_CPI_YoY", "India_Industrial_Production_YoY",
        "India_Repo_Rate_Level", "India_USD_INR_Level",
        "India_M2_YoY", "India_GDP_Growth_Level",
        "India_Unemployment_Level", "India_WPI_YoY",
    ]

    data = np.zeros((n_months, 8))
    for i, regime in enumerate(regimes):
        mean = regime_means[regime]
        if i > 0:
            data[i] = 0.7 * data[i - 1] + 0.3 * mean + np.random.randn(8) * 0.35
        else:
            data[i] = mean + np.random.randn(8) * 0.35

    features = pd.DataFrame(data, index=dates, columns=columns)
    return features, regimes


def generate_india_market_returns(regimes):
    """Generate synthetic Indian asset returns conditioned on regimes."""
    n_months = len(regimes)
    dates = pd.date_range("2005-01-01", periods=n_months, freq="ME")

    regime_returns = {
        "Expansion": {
            "Nifty_50": 0.015, "Bank_Nifty": 0.018, "Nifty_Midcap": 0.020,
            "Gold_INR": 0.003, "G_Sec_Long": 0.004, "Liquid_Fund": 0.005,
            "Nifty_IT": 0.012,
        },
        "Slowdown": {
            "Nifty_50": 0.002, "Bank_Nifty": -0.005, "Nifty_Midcap": -0.003,
            "Gold_INR": 0.008, "G_Sec_Long": 0.005, "Liquid_Fund": 0.004,
            "Nifty_IT": 0.004,
        },
        "Recession": {
            "Nifty_50": -0.030, "Bank_Nifty": -0.040, "Nifty_Midcap": -0.045,
            "Gold_INR": 0.012, "G_Sec_Long": 0.006, "Liquid_Fund": 0.003,
            "Nifty_IT": -0.015,
        },
        "Recovery": {
            "Nifty_50": 0.018, "Bank_Nifty": 0.022, "Nifty_Midcap": 0.025,
            "Gold_INR": 0.005, "G_Sec_Long": 0.004, "Liquid_Fund": 0.004,
            "Nifty_IT": 0.015,
        },
    }

    vols = {
        "Nifty_50": 0.055, "Bank_Nifty": 0.070, "Nifty_Midcap": 0.075,
        "Gold_INR": 0.035, "G_Sec_Long": 0.012, "Liquid_Fund": 0.002,
        "Nifty_IT": 0.060,
    }

    assets = list(vols.keys())
    returns_data = {}

    for asset in assets:
        rets = []
        for regime in regimes:
            mean = regime_returns[regime][asset]
            vol = vols[asset]
            if regime == "Recession":
                vol *= 1.6
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

    print(f"✅ Saved {len(features)} months of US macro features")
    print(f"✅ Saved {len(returns)} months of US market returns")

    # ─── India Demo Data ───────────────────────────────────────────────────
    india_features, india_regimes = generate_india_macro_features()
    india_returns = generate_india_market_returns(india_regimes)

    india_features.to_parquet(os.path.join(cache_dir, "india_macro_features.parquet"))
    india_returns.to_parquet(os.path.join(cache_dir, "india_market_returns.parquet"))

    print(f"✅ Saved {len(india_features)} months of India macro features")
    print(f"✅ Saved {len(india_returns)} months of India market returns")

    print(f"\n📁 Cache directory: {cache_dir}")
    print(f"\nUS Regime distribution:")
    regime_series = pd.Series(regimes)
    print(regime_series.value_counts())
    print(f"\nIndia Regime distribution:")
    india_regime_series = pd.Series(india_regimes)
    print(india_regime_series.value_counts())
    print(f"\nRun the dashboard: streamlit run dashboard/app.py")
