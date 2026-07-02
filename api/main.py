"""
FastAPI REST API for Macro Regime Detection.
Exposes regime signals, tactical allocation, backtests, explainability,
model comparison and investment memos for US and India markets.
"""

import os
import sys
import math
import tempfile
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    REGIME_ALLOCATIONS,
    REGIME_COLORS,
    ASSET_TICKERS,
    BENCHMARK_ALLOCATION,
)
from config.india_settings import (
    INDIA_REGIME_ALLOCATIONS,
    INDIA_BENCHMARK_ALLOCATION,
    INDIA_ASSET_TICKERS,
    INDIA_REGIME_EXPLANATIONS,
    INDIA_STRESS_SCENARIOS,
)
from models.regime_hmm import RegimeDetector
from models.allocator import TacticalAllocator
from models.stress_testing import MonteCarloStressTest, STRESS_SCENARIOS
from backtesting.engine import BacktestEngine

# ─── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Macro Regime Tactical Allocation API",
    description=(
        "Real-time macroeconomic regime detection, tactical asset allocation, "
        "walk-forward backtesting, regime explainability and model comparison "
        "for US and India markets via Hidden Markov Models."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Tightened CORS: explicit dev origins + any localhost port. Override in
# production with the ALLOWED_ORIGINS env var (comma-separated).
_env_origins = os.getenv("ALLOWED_ORIGINS")
if _env_origins:
    _origins = [o.strip() for o in _env_origins.split(",") if o.strip()]
else:
    _origins = [
        "http://localhost:5173",
        "http://localhost:5180",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5180",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Response Models ───────────────────────────────────────────────────────────

class RegimeResponse(BaseModel):
    current_regime: str
    confidence: float
    regime_probabilities: Dict[str, float]
    expected_duration_months: float
    timestamp: str


class AllocationResponse(BaseModel):
    regime: str
    confidence: float
    target_weights: Dict[str, float]
    benchmark_weights: Dict[str, float]
    rationale: str
    overweight: List[str]
    underweight: List[str]


class RiskMetrics(BaseModel):
    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float
    probability_of_loss: float
    expected_return: float
    horizon_months: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    data_loaded: bool
    last_updated: str
    markets: List[str]


# ─── Market Bundles ────────────────────────────────────────────────────────────

VALID_REGIMES = ["Expansion", "Slowdown", "Recession", "Recovery"]

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cache")


def get_data_status(key: str) -> dict:
    """Per-market data-currency metadata written by ``data.refresh`` (empty
    dict if the cache has never been refreshed via the live pipeline)."""
    try:
        from data.refresh import load_meta

        return load_meta().get(key, {})
    except Exception:  # noqa: BLE001
        return {}

MARKET_CONFIG = {
    "us": {
        "label": "United States",
        "macro_file": "macro_features.parquet",
        "returns_file": "market_returns.parquet",
        "regime_allocations": REGIME_ALLOCATIONS,
        "benchmark": BENCHMARK_ALLOCATION,
        "asset_tickers": ASSET_TICKERS,
        "explanations": None,  # use allocator.get_regime_tilt_explanation
        "stress_scenarios": STRESS_SCENARIOS,
        "currency": "USD",
    },
    "india": {
        "label": "India",
        "macro_file": "india_macro_features.parquet",
        "returns_file": "india_market_returns.parquet",
        "regime_allocations": INDIA_REGIME_ALLOCATIONS,
        "benchmark": INDIA_BENCHMARK_ALLOCATION,
        "asset_tickers": INDIA_ASSET_TICKERS,
        "explanations": INDIA_REGIME_EXPLANATIONS,
        "stress_scenarios": INDIA_STRESS_SCENARIOS,
        "currency": "INR",
    },
}


class MarketBundle:
    """Fitted models + data + config for a single market."""

    def __init__(self, key: str, cfg: dict):
        self.key = key
        self.label = cfg["label"]
        self.currency = cfg["currency"]
        self.regime_allocations = cfg["regime_allocations"]
        self.benchmark = cfg["benchmark"]
        self.asset_tickers = cfg["asset_tickers"]
        self.explanations = cfg["explanations"]
        self.stress_scenarios = cfg.get("stress_scenarios", STRESS_SCENARIOS)

        macro_path = os.path.join(CACHE_DIR, cfg["macro_file"])
        returns_path = os.path.join(CACHE_DIR, cfg["returns_file"])
        if not (os.path.exists(macro_path) and os.path.exists(returns_path)):
            raise FileNotFoundError(f"Cached data missing for market '{key}'")

        self.macro_features = pd.read_parquet(macro_path)
        self.market_returns = pd.read_parquet(returns_path)

        # Asset return data may be unavailable (e.g. never fetched offline);
        # regime/allocation views still work, but risk/backtest do not.
        self.has_returns = (
            self.market_returns is not None
            and len(self.market_returns) > 0
            and self.market_returns.index.notna().all()
        )

        # Data-currency: prefer the refresh metadata, fall back to the actual
        # last observation in the loaded frames.
        meta = get_data_status(key)
        self.data_status = {
            "as_of": meta.get("refreshed_at"),
            "macro_through": meta.get("macro_through")
            or (str(self.macro_features.index.max())[:10] if len(self.macro_features) else None),
            "returns_through": meta.get("returns_through")
            or (str(self.market_returns.index.max())[:10] if self.has_returns else None),
            "prices_through": meta.get("prices_through"),
        }

        self.detector = RegimeDetector(n_regimes=4, n_components_pca=5)
        self.detector.fit(self.macro_features)
        self.allocator = TacticalAllocator(
            regime_allocations=self.regime_allocations,
            benchmark=self.benchmark,
            assets=list(self.asset_tickers.keys()),
        )
        self.last_updated = datetime.now().isoformat()

        # Lazily populated caches
        self._backtest = None
        self._comparison = None

    # -- convenience --------------------------------------------------------
    def current(self):
        regimes = self.detector.predict(self.macro_features)
        proba = self.detector.predict_proba(self.macro_features)
        current = regimes.iloc[-1]
        confidence = float(proba.iloc[-1].max())
        return current, confidence, regimes, proba

    def explain(self, regime: str) -> dict:
        if self.explanations is not None:
            return self.explanations[regime]
        return self.allocator.get_regime_tilt_explanation(regime)


class ModelState:
    """Registry of loaded markets (loaded lazily on first use)."""

    def __init__(self):
        self.markets: Dict[str, MarketBundle] = {}
        self.load_errors: Dict[str, str] = {}

    def load(self, key: str) -> Optional[MarketBundle]:
        if key in self.markets:
            return self.markets[key]
        if key not in MARKET_CONFIG:
            return None
        try:
            bundle = MarketBundle(key, MARKET_CONFIG[key])
            self.markets[key] = bundle
            self.load_errors.pop(key, None)
            return bundle
        except Exception as exc:  # noqa: BLE001
            self.load_errors[key] = str(exc)
            print(f"Warning: could not load market '{key}': {exc}")
            return None

    def refresh(self, key: str) -> Optional[MarketBundle]:
        self.markets.pop(key, None)
        return self.load(key)


state = ModelState()


def get_bundle(market: str) -> MarketBundle:
    market = (market or "us").lower()
    if market not in MARKET_CONFIG:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown market '{market}'. Valid: {list(MARKET_CONFIG)}",
        )
    bundle = state.load(market)
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail=f"Market '{market}' unavailable: "
            f"{state.load_errors.get(market, 'not loaded')}",
        )
    return bundle


def require_returns(bundle: MarketBundle) -> None:
    if not bundle.has_returns:
        raise HTTPException(
            status_code=501,
            detail=(
                f"Asset return data is not available for {bundle.label}. "
                "Regime, allocation, drivers and model views are available; "
                "risk and backtest analytics require market price history."
            ),
        )


def _finite(v) -> Optional[float]:
    """None for non-finite floats so the response stays JSON-compliant."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _finite0(v) -> float:
    f = _finite(v)
    return f if f is not None else 0.0


@app.on_event("startup")
async def startup_event():
    # Eagerly load US so the default experience is instant; India loads on demand.
    state.load("us")


# ─── Backtest helpers ──────────────────────────────────────────────────────────

def _run_backtest(bundle: MarketBundle):
    """Run (and cache) the walk-forward backtest for a market."""
    if bundle._backtest is not None:
        return bundle._backtest

    regimes = bundle.detector.predict(bundle.macro_features)
    regime_allocs = {
        name: pd.Series(weights) for name, weights in bundle.regime_allocations.items()
    }
    engine = BacktestEngine()
    result = engine.run(
        asset_returns=bundle.market_returns,
        regime_signals=regimes,
        regime_allocations=regime_allocs,
        benchmark_weights=bundle.benchmark,
    )
    bundle._backtest = result
    return result


BENCHMARK_VARIANTS = ("sixty_forty", "equal_weight", "risk_parity", "kelly")

# Annual risk-free rate used for Kelly leverage and the borrowing leg of the
# levered benchmark. Kept conservative and explicit rather than pulled live so
# the benchmark construction is reproducible.
KELLY_RF = 0.04
KELLY_MAX_LEVERAGE = 1.5


def _benchmark_weights(bundle: MarketBundle, variant: str) -> dict:
    """Resolve a benchmark construction to a weight dict over the market assets."""
    assets = list(bundle.market_returns.columns)

    if variant == "equal_weight":
        w = 1.0 / len(assets)
        return {a: w for a in assets}

    if variant == "risk_parity":
        from models.risk_budgeting import RiskBudgetAllocator

        cov = bundle.market_returns.cov() * 12.0  # annualized covariance
        weights = RiskBudgetAllocator().compute_risk_parity_weights(cov)
        return {k: float(v) for k, v in weights.to_dict().items()}

    # Default: the market's static benchmark (60/40 style).
    return dict(bundle.benchmark)


def _run_backtest_custom(
    bundle: MarketBundle,
    benchmark_weights: dict,
    rebalance_every_n: int = 1,
):
    """Fresh (uncached) backtest with explicit benchmark and rebalance cadence."""
    regimes = bundle.detector.predict(bundle.macro_features)
    regime_allocs = {
        name: pd.Series(weights) for name, weights in bundle.regime_allocations.items()
    }
    engine = BacktestEngine()
    return engine.run(
        asset_returns=bundle.market_returns,
        regime_signals=regimes,
        regime_allocations=regime_allocs,
        benchmark_weights=benchmark_weights,
        rebalance_every_n=rebalance_every_n,
    )


def _kelly_levered_benchmark(bundle: MarketBundle, pv: pd.Series):
    """
    Build an honest Kelly-levered benchmark equity curve aligned to ``pv``.

    The BacktestEngine renormalizes any benchmark weight vector to sum to 1, so
    leverage cannot be expressed through the weight path. Instead we:

      1. Compute long-only multi-asset (half-)Kelly proportions as the base
         portfolio and take its realized monthly return stream.
      2. Size a single capped half-Kelly leverage ``L`` from that base
         portfolio's annualized mean/vol.
      3. Lever the stream with an explicit borrowing cost at the risk-free rate:
         ``r_lev = rf_m + L * (r_base - rf_m)`` — when ``L > 1`` the shortfall
         ``(1 - L)`` is borrowed at ``rf``.

    Returns ``(benchmark_value, leverage)`` where ``benchmark_value`` is indexed
    like ``pv`` (first point = 1.0 seed, scale is irrelevant since the endpoint
    normalizes to 100 and metrics use pct-change).
    """
    from models.leverage import KellyCriterion

    rets = bundle.market_returns
    kelly = KellyCriterion(
        max_leverage=KELLY_MAX_LEVERAGE, min_leverage=0.0, kelly_fraction=0.5
    )

    mu = rets.mean() * 12.0
    cov = rets.cov() * 12.0
    raw = kelly.compute_multi_asset_kelly(mu, cov, risk_free_rate=KELLY_RF)

    # Long-only proportions for the base portfolio (drop negative Kelly tilts).
    w = raw.clip(lower=0.0)
    w = (w / w.sum()) if w.sum() > 0 else pd.Series(
        1.0 / len(rets.columns), index=rets.columns
    )

    base_ret = (rets * w).sum(axis=1)
    ann_mu = float(base_ret.mean() * 12.0)
    ann_vol = float(base_ret.std() * math.sqrt(12.0))
    lev = float(np.clip(
        kelly.compute_kelly_leverage(ann_mu, ann_vol, risk_free_rate=KELLY_RF),
        0.1, KELLY_MAX_LEVERAGE,
    ))

    rf_m = KELLY_RF / 12.0
    lev_ret = rf_m + lev * (base_ret - rf_m)

    bv = pd.Series(index=pv.index, dtype=float)
    bv.iloc[0] = 1.0
    growth = (1.0 + lev_ret.reindex(pv.index[1:]).fillna(0.0)).cumprod()
    bv.iloc[1:] = growth.values
    return bv, lev


def _annualized_metrics(pv: pd.Series, bv: pd.Series) -> dict:
    sr = pv.pct_change().dropna()
    br = bv.pct_change().dropna()
    idx = sr.index.intersection(br.index)
    sr, br = sr.loc[idx], br.loc[idx]

    def ann_return(series, values):
        if len(values) == 0:
            return 0.0
        total = series.iloc[-1] / series.iloc[0]
        return float(total ** (12.0 / len(values)) - 1)

    def ann_vol(values):
        return float(values.std() * math.sqrt(12)) if len(values) > 1 else 0.0

    def max_dd(series):
        return float((series / series.cummax() - 1).min())

    def sortino(values, ar):
        downside = values[values < 0]
        dd = downside.std() * math.sqrt(12) if len(downside) > 1 else 0.0
        return float(ar / dd) if dd > 0 else 0.0

    ar_s, ar_b = ann_return(pv, sr), ann_return(bv, br)
    vol_s, vol_b = ann_vol(sr), ann_vol(br)
    mdd_s, mdd_b = max_dd(pv), max_dd(bv)
    excess = (sr - br).dropna()
    te = float(excess.std() * math.sqrt(12)) if len(excess) > 1 else 0.0
    info = float(excess.mean() * 12 / te) if te > 0 else 0.0

    return {
        "annual_return_strategy": _finite0(ar_s),
        "annual_return_benchmark": _finite0(ar_b),
        "annual_vol_strategy": _finite0(vol_s),
        "annual_vol_benchmark": _finite0(vol_b),
        "sharpe_strategy": _finite0(ar_s / vol_s) if vol_s > 0 else 0.0,
        "sharpe_benchmark": _finite0(ar_b / vol_b) if vol_b > 0 else 0.0,
        "sortino_strategy": _finite0(sortino(sr, ar_s)),
        "calmar_strategy": _finite0(ar_s / abs(mdd_s)) if mdd_s < 0 else 0.0,
        "max_drawdown_strategy": _finite0(mdd_s),
        "max_drawdown_benchmark": _finite0(mdd_b),
        "information_ratio": _finite0(info),
        "tracking_error": _finite0(te),
        "win_rate": _finite0((sr > br).mean()),
        "total_return_strategy": _finite0(pv.iloc[-1] / pv.iloc[0] - 1),
        "total_return_benchmark": _finite0(bv.iloc[-1] / bv.iloc[0] - 1),
    }


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=bool(state.markets),
        data_loaded=bool(state.markets),
        last_updated=next(
            (b.last_updated for b in state.markets.values()), "never"
        ),
        markets=list(state.markets.keys()),
    )


@app.get("/markets")
async def list_markets():
    """List available markets, load state and whether asset returns exist."""
    out = []
    for key, cfg in MARKET_CONFIG.items():
        bundle = state.load(key)
        out.append({
            "key": key,
            "label": cfg["label"],
            "currency": cfg["currency"],
            "loaded": key in state.markets,
            "has_returns": bool(bundle.has_returns) if bundle else False,
            "data_through": (bundle.data_status.get("prices_through")
                             or bundle.data_status.get("returns_through")
                             or bundle.data_status.get("macro_through")) if bundle else None,
            "assets": list(cfg["asset_tickers"].keys()),
        })
    return {"markets": out}


@app.get("/data/status")
async def data_status():
    """Freshness of the underlying data per market (from the live refresh)."""
    out = {}
    for key in MARKET_CONFIG:
        bundle = state.load(key)
        out[key] = bundle.data_status if bundle else get_data_status(key)
    return {"as_of": datetime.now().isoformat(), "markets": out}


@app.get("/regime/current", response_model=RegimeResponse)
async def get_current_regime(market: str = Query(default="us")):
    bundle = get_bundle(market)
    current, confidence, _, proba = bundle.current()
    durations = bundle.detector.get_expected_duration()
    return RegimeResponse(
        current_regime=current,
        confidence=confidence,
        regime_probabilities=proba.iloc[-1].to_dict(),
        expected_duration_months=float(durations.get(current, 0)),
        timestamp=datetime.now().isoformat(),
    )


@app.get("/regime/history")
async def get_regime_history(
    market: str = Query(default="us"),
    months: int = Query(default=60, ge=1, le=240),
):
    bundle = get_bundle(market)
    regimes = bundle.detector.predict(bundle.macro_features)
    history = regimes.tail(months)
    return {
        "regimes": [
            {"date": str(date.date()), "regime": regime}
            for date, regime in history.items()
        ],
        "count": len(history),
    }


@app.get("/regime/transition-matrix")
async def get_transition_matrix(market: str = Query(default="us")):
    bundle = get_bundle(market)
    trans = bundle.detector.get_transition_matrix()
    return {"matrix": trans.to_dict(), "regimes": trans.index.tolist()}


@app.get("/regime/drivers")
async def get_regime_drivers(
    market: str = Query(default="us"),
    top: int = Query(default=8, ge=3, le=20),
):
    """
    Explainability: which macro features are driving the current regime.

    Macro features are z-scored, so the latest row values are the number of
    standard deviations each indicator sits from its rolling norm. We rank by
    absolute deviation and compare against the current regime's historical mean.
    """
    bundle = get_bundle(market)
    mf = bundle.macro_features
    regimes = bundle.detector.predict(mf)
    current = regimes.iloc[-1]

    latest = mf.iloc[-1]
    regime_mask = (regimes == current).reindex(mf.index).fillna(False)
    regime_means = mf[regime_mask.values].mean()

    drivers = []
    for feat in mf.columns:
        z = float(latest[feat]) if pd.notna(latest[feat]) else 0.0
        drivers.append({
            "feature": feat,
            "z_score": z,
            "direction": "elevated" if z >= 0 else "depressed",
            "regime_avg": float(regime_means[feat]) if pd.notna(regime_means[feat]) else 0.0,
        })

    drivers.sort(key=lambda d: abs(d["z_score"]), reverse=True)
    return {
        "market": bundle.key,
        "regime": current,
        "as_of": str(mf.index[-1].date()),
        "drivers": drivers[:top],
    }


@app.get("/allocation/current", response_model=AllocationResponse)
async def get_current_allocation(market: str = Query(default="us")):
    bundle = get_bundle(market)
    current, confidence, _, _ = bundle.current()
    weights = bundle.allocator.get_target_allocation(current, confidence)
    explanation = bundle.explain(current)
    return AllocationResponse(
        regime=current,
        confidence=_finite0(confidence),
        target_weights={k: _finite0(v) for k, v in weights.to_dict().items()},
        benchmark_weights={k: _finite0(v) for k, v in bundle.benchmark.items()},
        rationale=explanation["rationale"],
        overweight=explanation["overweight"],
        underweight=explanation["underweight"],
    )


@app.get("/allocation/regime/{regime_name}", response_model=AllocationResponse)
async def get_regime_allocation(regime_name: str, market: str = Query(default="us")):
    bundle = get_bundle(market)
    if regime_name not in VALID_REGIMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid regime. Must be one of: {VALID_REGIMES}",
        )
    weights = bundle.allocator.get_target_allocation(regime_name, confidence=1.0)
    explanation = bundle.explain(regime_name)
    return AllocationResponse(
        regime=regime_name,
        confidence=1.0,
        target_weights={k: _finite0(v) for k, v in weights.to_dict().items()},
        benchmark_weights={k: _finite0(v) for k, v in bundle.benchmark.items()},
        rationale=explanation["rationale"],
        overweight=explanation["overweight"],
        underweight=explanation["underweight"],
    )


@app.get("/backtest")
async def get_backtest(
    market: str = Query(default="us"),
    benchmark: str = Query(default="sixty_forty"),
):
    """
    Walk-forward backtest of the regime strategy vs the market benchmark.
    Returns an equity curve, underwater drawdown, rolling Sharpe, per-regime
    P&L attribution and numeric performance metrics.

    `benchmark` selects the comparison construction: sixty_forty (static),
    equal_weight, risk_parity (equal risk contribution), or kelly (a capped
    half-Kelly-levered tangency portfolio with an explicit borrowing cost).
    """
    bundle = get_bundle(market)
    require_returns(bundle)

    if benchmark not in BENCHMARK_VARIANTS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid benchmark. Must be one of: {list(BENCHMARK_VARIANTS)}",
        )

    benchmark_leverage = None
    if benchmark == "sixty_forty":
        result = _run_backtest(bundle)
        pv = result.portfolio_value
        bv = result.benchmark_value.reindex(pv.index).ffill()
    elif benchmark == "kelly":
        result = _run_backtest(bundle)
        pv = result.portfolio_value
        bv, benchmark_leverage = _kelly_levered_benchmark(bundle, pv)
    else:
        result = _run_backtest_custom(bundle, _benchmark_weights(bundle, benchmark))
        pv = result.portfolio_value
        bv = result.benchmark_value.reindex(pv.index).ffill()

    # Normalize both curves to 100 at inception for chart friendliness.
    pv0, bv0 = pv.iloc[0], bv.iloc[0]
    strat_norm = pv / pv0 * 100
    bench_norm = bv / bv0 * 100

    equity_curve = [
        {"date": str(d.date()), "strategy": float(s), "benchmark": float(b)}
        for d, s, b in zip(pv.index, strat_norm.values, bench_norm.values)
    ]

    strat_dd = (pv / pv.cummax() - 1)
    bench_dd = (bv / bv.cummax() - 1)
    drawdown = [
        {"date": str(d.date()), "strategy": float(s), "benchmark": float(b)}
        for d, s, b in zip(pv.index, strat_dd.values, bench_dd.values)
    ]

    # Rolling 12-month annualized Sharpe.
    sr = pv.pct_change().dropna()
    br = bv.pct_change().dropna()
    window = 12
    rs_s = (sr.rolling(window).mean() / sr.rolling(window).std()) * math.sqrt(12)
    rs_b = (br.rolling(window).mean() / br.rolling(window).std()) * math.sqrt(12)
    rolling_sharpe = [
        {"date": str(d.date()),
         "strategy": _finite(s),
         "benchmark": _finite(b)}
        for d, s, b in zip(rs_s.index, rs_s.values, rs_b.reindex(rs_s.index).values)
    ]

    # Per-regime P&L attribution (sum of monthly strategy returns by regime).
    reg = result.regime_history.reindex(sr.index).ffill()
    attribution = []
    for regime in VALID_REGIMES:
        mask = (reg == regime)
        months = int(mask.sum())
        contribution = float(sr[mask.values].sum()) if months else 0.0
        avg_month = float(sr[mask.values].mean()) if months else 0.0
        attribution.append({
            "regime": regime,
            "months": months,
            "contribution": contribution,
            "avg_monthly_return": avg_month,
        })

    return {
        "market": bundle.key,
        "currency": bundle.currency,
        "benchmark": benchmark,
        "benchmark_leverage": benchmark_leverage,
        "start": str(pv.index[0].date()),
        "end": str(pv.index[-1].date()),
        "metrics": _annualized_metrics(pv, bv),
        "equity_curve": equity_curve,
        "drawdown": drawdown,
        "rolling_sharpe": rolling_sharpe,
        "regime_attribution": attribution,
    }


@app.get("/backtest/sensitivity")
async def get_backtest_sensitivity(
    market: str = Query(default="us"),
):
    """
    Robustness sweep: how the strategy's net Sharpe and annualized return hold
    up across transaction-cost assumptions and rebalancing cadence. A strategy
    whose edge survives higher costs and slower rebalancing is more credible.
    """
    bundle = get_bundle(market)
    require_returns(bundle)

    cost_grid = [0, 5, 10, 25, 50]          # basis points per unit turnover
    cadence_grid = [1, 3, 6, 12]            # rebalance every N months
    benchmark_weights = dict(bundle.benchmark)

    cells = []
    for n in cadence_grid:
        for bps in cost_grid:
            engine = BacktestEngine(transaction_cost_bps=bps)
            regimes = bundle.detector.predict(bundle.macro_features)
            regime_allocs = {
                name: pd.Series(w) for name, w in bundle.regime_allocations.items()
            }
            res = engine.run(
                asset_returns=bundle.market_returns,
                regime_signals=regimes,
                regime_allocations=regime_allocs,
                benchmark_weights=benchmark_weights,
                rebalance_every_n=n,
            )
            m = _annualized_metrics(res.portfolio_value, res.benchmark_value)
            cells.append({
                "cost_bps": bps,
                "rebalance_months": n,
                "sharpe": _finite0(m["sharpe_strategy"]),
                "annual_return": _finite0(m["annual_return_strategy"]),
                "max_drawdown": _finite0(m["max_drawdown_strategy"]),
            })

    return {
        "market": bundle.key,
        "cost_grid": cost_grid,
        "cadence_grid": cadence_grid,
        "cells": cells,
    }


@app.get("/model/comparison")
async def get_model_comparison(market: str = Query(default="us")):
    """
    Cross-check the HMM regime labels against unsupervised (KMeans) and
    supervised (LSTM) alternatives to gauge robustness of the current call.
    """
    bundle = get_bundle(market)
    if bundle._comparison is not None:
        return bundle._comparison

    mf = bundle.macro_features
    hmm_labels = bundle.detector.predict(mf)
    models = [{
        "name": "Gaussian HMM",
        "type": "primary",
        "current_regime": hmm_labels.iloc[-1],
        "agreement_with_hmm": 1.0,
        "quality": None,
        "quality_label": "log-likelihood fit",
        "note": "Primary model used for signals.",
    }]
    timeline_frames = {"hmm": hmm_labels}

    # --- KMeans (fast, unsupervised) ---
    try:
        from models.alternative_models import KMeansRegimeDetector
        km = KMeansRegimeDetector(n_regimes=4).fit(mf)
        km_labels = km.predict(mf).reindex(hmm_labels.index)
        agree = float((hmm_labels.values == km_labels.values).mean())
        sil = float(km.get_silhouette_score(mf))
        models.append({
            "name": "KMeans Clustering",
            "type": "unsupervised",
            "current_regime": km_labels.iloc[-1],
            "agreement_with_hmm": agree,
            "quality": sil,
            "quality_label": "silhouette score",
            "note": "Independent unsupervised clustering of macro features.",
        })
        timeline_frames["kmeans"] = km_labels
    except Exception as exc:  # noqa: BLE001
        models.append({
            "name": "KMeans Clustering", "type": "unsupervised",
            "current_regime": None, "agreement_with_hmm": None,
            "quality": None, "quality_label": "silhouette score",
            "note": f"Unavailable: {exc}",
        })

    # --- LSTM (supervised distillation of HMM labels; heavier) ---
    try:
        from models.alternative_models import LSTMRegimeDetector
        lstm = LSTMRegimeDetector(epochs=40)
        lstm.fit(mf, hmm_labels)
        lstm_labels = lstm.predict(mf).reindex(hmm_labels.index)
        agree = float((hmm_labels.values == lstm_labels.values).mean())
        models.append({
            "name": "LSTM (distilled)",
            "type": "supervised",
            "current_regime": lstm_labels.iloc[-1],
            "agreement_with_hmm": agree,
            "quality": None,
            "quality_label": "sequence model",
            "note": "Trained to reproduce HMM labels from 12-month sequences.",
        })
        timeline_frames["lstm"] = lstm_labels
    except Exception as exc:  # noqa: BLE001
        models.append({
            "name": "LSTM (distilled)", "type": "supervised",
            "current_regime": None, "agreement_with_hmm": None,
            "quality": None, "quality_label": "sequence model",
            "note": f"Unavailable: {exc}",
        })

    # Aligned timeline for the last 36 months.
    tail = hmm_labels.tail(36).index
    timeline = []
    for d in tail:
        row = {"date": str(d.date())}
        for name, series in timeline_frames.items():
            val = series.get(d)
            row[name] = None if val is None or (isinstance(val, float) and pd.isna(val)) else val
        timeline.append(row)

    result = {"market": bundle.key, "models": models, "timeline": timeline}
    bundle._comparison = result
    return result


@app.get("/risk/var", response_model=RiskMetrics)
async def get_risk_metrics(
    market: str = Query(default="us"),
    horizon: int = Query(default=12, ge=1, le=36),
    simulations: int = Query(default=10000, ge=1000, le=50000),
):
    """Get VaR/CVaR risk metrics via Monte Carlo simulation."""
    bundle = get_bundle(market)
    require_returns(bundle)
    current, confidence, regimes, _ = bundle.current()
    common_idx = bundle.market_returns.index.intersection(regimes.index)
    aligned = pd.concat(
        [bundle.market_returns.loc[common_idx],
         regimes.loc[common_idx].rename("Regime")],
        axis=1,
    )

    regime_params = {}
    for regime in regimes.unique():
        regime_data = aligned[aligned["Regime"] == regime].drop(columns=["Regime"])
        regime_params[regime] = {
            asset: (regime_data[asset].mean(), regime_data[asset].std())
            for asset in regime_data.columns
        }

    mc = MonteCarloStressTest(n_simulations=simulations, horizon_months=horizon)
    trans_matrix = bundle.detector.get_transition_matrix()
    sim = mc.simulate_regime_paths(current, trans_matrix, regime_params)

    weights = bundle.allocator.get_target_allocation(current, confidence)
    var_results = mc.compute_portfolio_var(sim["returns"], weights, sim["assets"])

    return RiskMetrics(
        var_95=var_results["VaR_95%"],
        cvar_95=var_results["CVaR_95%"],
        var_99=var_results["VaR_99%"],
        cvar_99=var_results["CVaR_99%"],
        probability_of_loss=var_results["prob_negative"],
        expected_return=var_results["mean_return"],
        horizon_months=horizon,
    )


@app.get("/risk/stress-scenarios")
async def get_stress_scenarios(market: str = Query(default="us")):
    bundle = get_bundle(market)
    require_returns(bundle)
    current, confidence, _, _ = bundle.current()
    weights = bundle.allocator.get_target_allocation(current, confidence)

    results = {}
    for name, shocks in bundle.stress_scenarios.items():
        impact = sum(weights.get(a, 0) * shocks.get(a, 0) for a in weights.index)
        results[name] = {"portfolio_impact": impact, "shocks": shocks}

    return {"regime": current, "scenarios": results}


@app.get("/report/pdf")
async def get_pdf_report(market: str = Query(default="us")):
    """Generate a downloadable investment memo (PDF) for the current regime."""
    from reports.pdf_generator import PDFReportGenerator

    bundle = get_bundle(market)
    require_returns(bundle)
    current, confidence, regimes, _ = bundle.current()
    weights = bundle.allocator.get_target_allocation(current, confidence)

    result = _run_backtest(bundle)
    metrics_fmt = result.compute_metrics()

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()
    try:
        path = PDFReportGenerator().generate_report(
            current_regime=current,
            regime_confidence=confidence,
            allocation=weights,
            backtest_metrics=metrics_fmt,
            regime_history=regimes,
            var_results=None,
            leading_indicators=None,
            output_path=tmp.name,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Report generation failed: {exc}")

    stamp = datetime.now().strftime("%Y%m%d")
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=f"macro_regime_memo_{bundle.key}_{stamp}.pdf",
    )


@app.post("/model/refresh")
async def refresh_model(
    market: str = Query(default="us"),
    live: bool = Query(
        default=False,
        description="Pull fresh FRED/market data up to yesterday before reloading.",
    ),
):
    """Reload a market's models. With ``live=true`` the underlying caches are
    first refreshed from FRED + Yahoo Finance (up to the previous day)."""
    market = (market or "us").lower()
    if market not in MARKET_CONFIG:
        raise HTTPException(status_code=404, detail=f"Unknown market '{market}'")

    refresh_report = None
    if live:
        try:
            from data.refresh import refresh_all

            refresh_report = refresh_all([market]).get("markets", {}).get(market)
            if refresh_report and refresh_report.get("status") == "error":
                raise HTTPException(
                    status_code=502,
                    detail=f"Live data refresh failed: {refresh_report.get('error')}",
                )
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"Live refresh error: {exc}")

    bundle = state.refresh(market)
    if bundle is None:
        raise HTTPException(
            status_code=500,
            detail=state.load_errors.get(market, "refresh failed"),
        )
    return {
        "status": "refreshed",
        "market": market,
        "live": live,
        "timestamp": bundle.last_updated,
        "data_status": bundle.data_status,
        "refresh_report": refresh_report,
    }


# ─── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
