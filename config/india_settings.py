"""
India-specific configuration for Macro Regime Detection.
Multi-source indicators: FRED, World Bank, Yahoo Finance.
"""

# ─── India Macro Indicators (FRED Series IDs) ─────────────────────────────────
# These are India-specific series available on FRED
INDIA_FRED_INDICATORS = {
    # Growth & Production
    "India_GDP_Growth": "NGDPRPCHA163NRUG",  # Real GDP growth (annual %)
    "India_Industrial_Production": "INDPROINDMISMEI",  # IIP index
    "India_Manufacturing_PMI": "BSCICP03INM665S",  # Manufacturing Confidence
    # Inflation
    "India_CPI": "INDCPIALLMINMEI",  # CPI All Items
    "India_WPI": "PIEAMP02INM659N",  # Producer/Wholesale Prices
    # Interest Rates & Monetary
    "India_Repo_Rate": "INTDSRINM193N",  # Discount/Policy Rate
    "India_Short_Rate": "IRSTCI01INM156N",  # Short-term Interest Rate
    "India_M2": "MYAGM2INM189N",  # Broad Money (M2)
    # Labor
    "India_Unemployment": "LRUNTTTTINM156S",  # Unemployment Rate
    # Trade & Currency
    "India_USD_INR": "DEXINUS",  # USD/INR Exchange Rate
    "India_Current_Account": "BPBLTT01INQ637S",  # Current Account Balance
    # Financial
    "India_FDI": "ROWFDIQ027S",  # FDI flows proxy
}

# ─── World Bank Indicators (for supplementary data) ───────────────────────────
INDIA_WORLDBANK_INDICATORS = {
    "GDP_Growth_Annual": "NY.GDP.MKTP.KD.ZG",
    "Inflation_GDP_Deflator": "NY.GDP.DEFL.KD.ZG",
    "Trade_Percent_GDP": "NE.TRD.GNFS.ZS",
    "FDI_Net_Inflows": "BX.KLT.DINV.WD.GD.ZS",
    "Gross_Savings": "NY.GNS.ICTR.ZS",
}

# ─── Yahoo Finance India Market Tickers ────────────────────────────────────────
INDIA_MARKET_TICKERS = {
    "Nifty_50": "^NSEI",
    "Sensex": "^BSESN",
    "Bank_Nifty": "^NSEBANK",
    "Nifty_Midcap": "NIFTYMIDCAP150.NS",
    "Gold_INR": "GOLDBEES.NS",
    "India_10Y_Bond": "IN10Y.NS",  # fallback: use FRED
    "USD_INR": "INR=X",
    "Nifty_IT": "^CNXIT",
}

# ─── India Asset Classes for Tactical Allocation ──────────────────────────────
INDIA_ASSET_TICKERS = {
    "Nifty_50": "^NSEI",
    "Bank_Nifty": "^NSEBANK",
    "Nifty_Midcap": "NIFTYMIDCAP150.NS",
    "Gold_INR": "GOLDBEES.NS",
    "G_Sec_Long": "0GILTS.NS",  # Long-term gilt fund proxy
    "Liquid_Fund": "LIQUIDBEES.NS",
    "Nifty_IT": "^CNXIT",
}

# ─── India Regime Allocations ─────────────────────────────────────────────────
INDIA_REGIME_ALLOCATIONS = {
    "Expansion": {
        "Nifty_50": 0.30,
        "Bank_Nifty": 0.15,
        "Nifty_Midcap": 0.20,
        "Gold_INR": 0.05,
        "G_Sec_Long": 0.10,
        "Liquid_Fund": 0.05,
        "Nifty_IT": 0.15,
    },
    "Slowdown": {
        "Nifty_50": 0.20,
        "Bank_Nifty": 0.05,
        "Nifty_Midcap": 0.05,
        "Gold_INR": 0.20,
        "G_Sec_Long": 0.25,
        "Liquid_Fund": 0.15,
        "Nifty_IT": 0.10,
    },
    "Recession": {
        "Nifty_50": 0.10,
        "Bank_Nifty": 0.00,
        "Nifty_Midcap": 0.00,
        "Gold_INR": 0.30,
        "G_Sec_Long": 0.20,
        "Liquid_Fund": 0.35,
        "Nifty_IT": 0.05,
    },
    "Recovery": {
        "Nifty_50": 0.25,
        "Bank_Nifty": 0.20,
        "Nifty_Midcap": 0.15,
        "Gold_INR": 0.10,
        "G_Sec_Long": 0.15,
        "Liquid_Fund": 0.05,
        "Nifty_IT": 0.10,
    },
}

INDIA_BENCHMARK_ALLOCATION = {
    "Nifty_50": 0.40,
    "Bank_Nifty": 0.10,
    "Nifty_Midcap": 0.10,
    "Gold_INR": 0.10,
    "G_Sec_Long": 0.20,
    "Liquid_Fund": 0.05,
    "Nifty_IT": 0.05,
}

# ─── Ticker Tape Labels (India) ───────────────────────────────────────────────
INDIA_TICKER_LABELS = {
    "India_CPI_YoY": "India CPI YoY",
    "India_CPI_Mom3": "CPI Momentum",
    "India_Industrial_Production_YoY": "IIP YoY",
    "India_Industrial_Production_Mom3": "IIP Momentum",
    "India_Repo_Rate_Level": "Repo Rate",
    "India_Repo_Rate_Chg3": "Repo Rate Δ3m",
    "India_M2_YoY": "M2 Growth",
    "India_Unemployment_Level": "Unemployment",
    "India_USD_INR_Level": "USD/INR",
    "India_USD_INR_Chg3": "USD/INR Δ3m",
    "India_WPI_YoY": "WPI YoY",
    "India_GDP_Growth_Level": "GDP Growth",
    "India_Short_Rate_Level": "Short Rate",
    "India_Short_Rate_Chg3": "Short Rate Δ3m",
}

# ─── Regime Tilt Explanations (India-specific) ────────────────────────────────
INDIA_REGIME_EXPLANATIONS = {
    "Expansion": {
        "rationale": "Strong IIP growth, rising corporate earnings, RBI neutral-to-accommodative. "
                     "Favor cyclicals (banking, midcap) and equities broadly.",
        "overweight": ["Bank Nifty", "Midcap", "Nifty IT (global demand)"],
        "underweight": ["Gold", "Liquid Funds"],
        "key_risks": ["RBI surprise hike", "INR depreciation", "Global risk-off"],
    },
    "Slowdown": {
        "rationale": "Declining IIP, rising inflation (WPI/CPI), potential RBI tightening. "
                     "Rotate to defensives and duration.",
        "overweight": ["G-Secs (rate cut bets)", "Gold (hedge)", "Liquid Funds"],
        "underweight": ["Midcap (high beta)", "Bank Nifty (NPA risk)"],
        "key_risks": ["Stagflation", "Fiscal slippage", "Oil price shock"],
    },
    "Recession": {
        "rationale": "GDP contraction, high NPAs, capital flight, INR under pressure. "
                     "Maximum defensiveness — gold, liquid, minimal equity.",
        "overweight": ["Gold (safe haven)", "Liquid Funds", "G-Secs"],
        "underweight": ["All equities", "Especially Bank Nifty and Midcap"],
        "key_risks": ["Sovereign downgrade", "Banking crisis", "Prolonged slowdown"],
    },
    "Recovery": {
        "rationale": "RBI rate cuts flowing through, credit growth resuming, FII inflows returning. "
                     "Early cyclical positioning in banks and rate-sensitive sectors.",
        "overweight": ["Bank Nifty (credit cycle)", "Nifty 50", "Midcap (recovery beta)"],
        "underweight": ["Liquid Funds (opportunity cost)", "Excessive gold"],
        "key_risks": ["False recovery", "Global contagion", "Policy reversal"],
    },
}
