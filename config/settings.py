"""
Configuration settings for Macro Regime Detection & Tactical Allocation.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# FRED API key — get free at https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY = os.getenv("FRED_API_KEY", "YOUR_FRED_API_KEY")

# ─── Macro Indicators (FRED Series IDs) ───────────────────────────────────────
MACRO_INDICATORS = {
    # Growth
    "GDP": "GDP",
    "Industrial_Production": "INDPRO",
    "Retail_Sales": "RSAFS",
    "Nonfarm_Payrolls": "PAYEMS",
    "PMI_Manufacturing": "MANEMP",
    # Inflation
    "CPI": "CPIAUCSL",
    "Core_CPI": "CPILFESL",
    "PPI": "PPIACO",
    "Breakeven_5Y": "T5YIE",
    "Breakeven_10Y": "T10YIE",
    # Rates & Yield Curve
    "Fed_Funds_Rate": "FEDFUNDS",
    "Treasury_2Y": "DGS2",
    "Treasury_10Y": "DGS10",
    "Treasury_30Y": "DGS30",
    "Yield_Spread_10Y2Y": "T10Y2Y",
    # Credit & Stress
    "BAA_Spread": "BAA10Y",
    "High_Yield_Spread": "BAMLH0A0HYM2",
    "VIX": "VIXCLS",
    "Financial_Stress": "STLFSI2",
    # Money Supply & Liquidity
    "M2": "M2SL",
    "Excess_Reserves": "EXCSRESNS",
    # Labor
    "Unemployment_Rate": "UNRATE",
    "Initial_Claims": "ICSA",
    # Housing
    "Housing_Starts": "HOUST",
    "Case_Shiller": "CSUSHPINSA",
}

# ─── Regime Labels ─────────────────────────────────────────────────────────────
REGIME_NAMES = {
    0: "Expansion",
    1: "Slowdown",
    2: "Recession",
    3: "Recovery",
}

REGIME_COLORS = {
    "Expansion": "#2ecc71",
    "Slowdown": "#f39c12",
    "Recession": "#e74c3c",
    "Recovery": "#3498db",
}

# ─── Asset Classes for Tactical Allocation ─────────────────────────────────────
ASSET_TICKERS = {
    "US_Equity": "SPY",
    "Intl_Equity": "EFA",
    "EM_Equity": "EEM",
    "US_Bonds": "AGG",
    "TIPS": "TIP",
    "Gold": "GLD",
    "Commodities": "DBC",
    "Real_Estate": "VNQ",
    "Cash": "BIL",
}

# ─── Regime-Based Target Allocations ──────────────────────────────────────────
REGIME_ALLOCATIONS = {
    "Expansion": {
        "US_Equity": 0.35,
        "Intl_Equity": 0.15,
        "EM_Equity": 0.10,
        "US_Bonds": 0.10,
        "TIPS": 0.05,
        "Gold": 0.05,
        "Commodities": 0.10,
        "Real_Estate": 0.10,
        "Cash": 0.00,
    },
    "Slowdown": {
        "US_Equity": 0.20,
        "Intl_Equity": 0.10,
        "EM_Equity": 0.05,
        "US_Bonds": 0.25,
        "TIPS": 0.10,
        "Gold": 0.10,
        "Commodities": 0.05,
        "Real_Estate": 0.05,
        "Cash": 0.10,
    },
    "Recession": {
        "US_Equity": 0.05,
        "Intl_Equity": 0.05,
        "EM_Equity": 0.00,
        "US_Bonds": 0.30,
        "TIPS": 0.10,
        "Gold": 0.20,
        "Commodities": 0.00,
        "Real_Estate": 0.00,
        "Cash": 0.30,
    },
    "Recovery": {
        "US_Equity": 0.30,
        "Intl_Equity": 0.15,
        "EM_Equity": 0.10,
        "US_Bonds": 0.15,
        "TIPS": 0.05,
        "Gold": 0.05,
        "Commodities": 0.10,
        "Real_Estate": 0.10,
        "Cash": 0.00,
    },
}

# ─── Benchmark (60/40) ─────────────────────────────────────────────────────────
BENCHMARK_ALLOCATION = {
    "US_Equity": 0.60,
    "Intl_Equity": 0.00,
    "EM_Equity": 0.00,
    "US_Bonds": 0.40,
    "TIPS": 0.00,
    "Gold": 0.00,
    "Commodities": 0.00,
    "Real_Estate": 0.00,
    "Cash": 0.00,
}

# ─── HMM Settings ──────────────────────────────────────────────────────────────
HMM_N_REGIMES = 4
HMM_COVARIANCE_TYPE = "full"
HMM_N_ITER = 200
HMM_RANDOM_STATE = 42

# ─── Backtest Settings ─────────────────────────────────────────────────────────
BACKTEST_START = "2005-01-01"
BACKTEST_END = "2026-01-01"
REBALANCE_FREQUENCY = "M"  # Monthly
TRANSACTION_COST_BPS = 10  # 10bps per trade
INITIAL_CAPITAL = 1_000_000
