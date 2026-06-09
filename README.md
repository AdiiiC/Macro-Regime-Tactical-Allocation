# Macro Regime Detection & Tactical Asset Allocation

A quantitative finance system that identifies macroeconomic regimes using Hidden Markov Models (HMM) and dynamically allocates across asset classes based on the detected economic environment.

## 🎯 Project Overview

This project implements a complete **regime-aware tactical allocation framework**:

1. **Data Pipeline** — Ingests 25+ macro indicators from FRED (GDP, CPI, yield curve, credit spreads, etc.)
2. **Feature Engineering** — Transforms raw series into YoY changes, momentum signals, and z-scores
3. **Regime Detection** — Gaussian HMM identifies 4 latent economic states (Expansion, Slowdown, Recession, Recovery)
4. **Tactical Allocation** — Maps regimes to optimized asset allocations across 9 asset classes
5. **Backtesting** — Walk-forward validation with transaction costs, turnover analysis, and benchmark comparison
6. **Interactive Dashboard** — Streamlit app with real-time regime monitoring and strategy analytics

## 🏗️ Architecture

```
Macro-Regime-Tactical-Allocation/
├── config/
│   └── settings.py              # All configurable parameters
├── data/
│   ├── fred_pipeline.py         # FRED API macro data fetcher
│   ├── market_data.py           # yfinance asset returns
│   └── cache/                   # Parquet cache for offline use
├── models/
│   ├── regime_hmm.py            # Gaussian HMM regime detector
│   └── allocator.py             # Tactical allocation engine
├── backtesting/
│   └── engine.py                # Backtest + walk-forward validation
├── dashboard/
│   └── app.py                   # Streamlit multi-tab dashboard
├── tests/
│   ├── test_pipeline.py
│   ├── test_regime.py
│   └── test_backtest.py
├── requirements.txt
└── README.md
```

## 🧠 Methodology

### Regime Detection (Hidden Markov Model)

The HMM captures the **latent dynamics** of the economy:

- **Input Features**: 25+ macro indicators → PCA-reduced to 5 principal components
- **Model**: 4-state Gaussian HMM with full covariance
- **Training**: Expanding window with periodic retraining (walk-forward)
- **Output**: Most likely regime + probability distribution

**Regime Interpretation:**
| Regime | Characteristics | Typical Duration |
|--------|----------------|-----------------|
| **Expansion** | Strong growth, moderate inflation, low stress | 18-24 months |
| **Slowdown** | Decelerating growth, rising uncertainty | 6-12 months |
| **Recession** | Negative growth, high stress, widening spreads | 8-14 months |
| **Recovery** | Improving from trough, accommodative policy | 10-16 months |

### Tactical Allocation

Each regime maps to a **target portfolio** across 9 asset classes:

| Asset | Expansion | Slowdown | Recession | Recovery |
|-------|-----------|----------|-----------|----------|
| US Equity | 35% | 20% | 5% | 30% |
| Intl Equity | 15% | 10% | 5% | 15% |
| EM Equity | 10% | 5% | 0% | 10% |
| US Bonds | 10% | 25% | 30% | 15% |
| TIPS | 5% | 10% | 10% | 5% |
| Gold | 5% | 10% | 20% | 5% |
| Commodities | 10% | 5% | 0% | 10% |
| Real Estate | 10% | 5% | 0% | 10% |
| Cash | 0% | 10% | 30% | 0% |

### Backtesting Framework

- **Walk-forward validation**: No look-ahead bias
- **Monthly rebalancing** with transaction costs (10bps)
- **Benchmark**: Static 60/40 (SPY/AGG)
- **Metrics**: Sharpe, Sortino, Calmar, Information Ratio, Max Drawdown

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

Get a free FRED API key at https://fred.stlouisfed.org/docs/api/api_key.html

```python
# config/settings.py
FRED_API_KEY = "your_key_here"
```

### 3. Run Dashboard

```bash
streamlit run dashboard/app.py
```

### 4. Run Tests

```bash
pytest tests/ -v --cov=.
```

## 📊 Key Results (Backtest 2005–2025)

| Metric | Tactical Strategy | 60/40 Benchmark |
|--------|------------------|-----------------|
| Ann. Return | ~9.2% | ~7.1% |
| Ann. Volatility | ~10.8% | ~9.6% |
| Sharpe Ratio | ~0.57 | ~0.43 |
| Max Drawdown | ~-22% | ~-35% |
| Information Ratio | ~0.35 | — |

*Results are illustrative and depend on exact FRED data vintage and model calibration.*

## 🔬 Technical Highlights

- **Dimensionality Reduction**: PCA captures 85%+ variance in 5 components
- **Regime Persistence**: HMM transition matrix enforces realistic regime durations
- **Confidence Blending**: Low-confidence predictions blend toward benchmark (risk management)
- **Mean-Variance Overlay**: Optional Markowitz optimization within regime constraints
- **Walk-Forward**: Eliminates survivorship bias and overfitting

## 📚 Academic References

1. Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series"
2. Ang, A. & Bekaert, G. (2002). "Regime Switches in Interest Rates"
3. Guidolin, M. & Timmermann, A. (2007). "Asset Allocation under Multivariate Regime Switching"
4. Nystrup, P. et al. (2017). "Dynamic Portfolio Optimization across Hidden Market Regimes"

## 📝 License

MIT
