"""
FastAPI REST API for Macro Regime Detection.
Exposes regime signals and allocation recommendations via HTTP.
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import REGIME_ALLOCATIONS, REGIME_COLORS, ASSET_TICKERS
from data.fred_pipeline import load_cached_data
from models.regime_hmm import RegimeDetector
from models.allocator import TacticalAllocator
from models.stress_testing import MonteCarloStressTest, STRESS_SCENARIOS

# ─── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Macro Regime Tactical Allocation API",
    description=(
        "Real-time macroeconomic regime detection and tactical asset allocation "
        "signals via Hidden Markov Models."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


class BacktestMetrics(BaseModel):
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    sortino_ratio: float
    information_ratio: float


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


# ─── Model Loading ─────────────────────────────────────────────────────────────

class ModelState:
    """Global model state."""
    detector: Optional[RegimeDetector] = None
    allocator: Optional[TacticalAllocator] = None
    macro_features: Optional[pd.DataFrame] = None
    market_returns: Optional[pd.DataFrame] = None
    last_updated: Optional[str] = None


state = ModelState()


def load_models():
    """Load models and data on startup."""
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "data", "cache")

    try:
        macro_path = os.path.join(cache_dir, "macro_features.parquet")
        market_path = os.path.join(cache_dir, "market_returns.parquet")

        if os.path.exists(macro_path):
            state.macro_features = pd.read_parquet(macro_path)
            state.market_returns = pd.read_parquet(market_path)

            state.detector = RegimeDetector(n_regimes=4, n_components_pca=5)
            state.detector.fit(state.macro_features)

            state.allocator = TacticalAllocator()
            state.last_updated = datetime.now().isoformat()
    except Exception as e:
        print(f"Warning: Could not load models: {e}")


@app.on_event("startup")
async def startup_event():
    load_models()


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=state.detector is not None,
        data_loaded=state.macro_features is not None,
        last_updated=state.last_updated or "never",
    )


@app.get("/regime/current", response_model=RegimeResponse)
async def get_current_regime():
    """Get the current detected macro regime."""
    if state.detector is None or state.macro_features is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    regimes = state.detector.predict(state.macro_features)
    proba = state.detector.predict_proba(state.macro_features)
    durations = state.detector.get_expected_duration()

    current = regimes.iloc[-1]
    current_proba = proba.iloc[-1].to_dict()
    confidence = max(current_proba.values())

    return RegimeResponse(
        current_regime=current,
        confidence=confidence,
        regime_probabilities=current_proba,
        expected_duration_months=durations.get(current, 0),
        timestamp=datetime.now().isoformat(),
    )


@app.get("/regime/history")
async def get_regime_history(
    months: int = Query(default=60, ge=1, le=240, description="Number of months")
):
    """Get historical regime classifications."""
    if state.detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    regimes = state.detector.predict(state.macro_features)
    history = regimes.tail(months)

    return {
        "regimes": [
            {"date": str(date.date()), "regime": regime}
            for date, regime in history.items()
        ],
        "count": len(history),
    }


@app.get("/regime/transition-matrix")
async def get_transition_matrix():
    """Get regime transition probability matrix."""
    if state.detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    trans = state.detector.get_transition_matrix()
    return {
        "matrix": trans.to_dict(),
        "regimes": trans.index.tolist(),
    }


@app.get("/allocation/current", response_model=AllocationResponse)
async def get_current_allocation():
    """Get current tactical allocation recommendation."""
    if state.detector is None or state.allocator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    regimes = state.detector.predict(state.macro_features)
    proba = state.detector.predict_proba(state.macro_features)

    current = regimes.iloc[-1]
    confidence = proba.iloc[-1].max()

    weights = state.allocator.get_target_allocation(current, confidence)
    explanation = state.allocator.get_regime_tilt_explanation(current)

    from config.settings import BENCHMARK_ALLOCATION

    return AllocationResponse(
        regime=current,
        confidence=confidence,
        target_weights=weights.to_dict(),
        benchmark_weights=BENCHMARK_ALLOCATION,
        rationale=explanation["rationale"],
        overweight=explanation["overweight"],
        underweight=explanation["underweight"],
    )


@app.get("/allocation/regime/{regime_name}", response_model=AllocationResponse)
async def get_regime_allocation(regime_name: str):
    """Get allocation for a specific regime."""
    if state.allocator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    valid_regimes = ["Expansion", "Slowdown", "Recession", "Recovery"]
    if regime_name not in valid_regimes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid regime. Must be one of: {valid_regimes}"
        )

    weights = state.allocator.get_target_allocation(regime_name, confidence=1.0)
    explanation = state.allocator.get_regime_tilt_explanation(regime_name)

    from config.settings import BENCHMARK_ALLOCATION

    return AllocationResponse(
        regime=regime_name,
        confidence=1.0,
        target_weights=weights.to_dict(),
        benchmark_weights=BENCHMARK_ALLOCATION,
        rationale=explanation["rationale"],
        overweight=explanation["overweight"],
        underweight=explanation["underweight"],
    )


@app.get("/risk/var", response_model=RiskMetrics)
async def get_risk_metrics(
    horizon: int = Query(default=12, ge=1, le=36, description="Horizon in months"),
    simulations: int = Query(default=10000, ge=1000, le=50000),
):
    """Get VaR/CVaR risk metrics via Monte Carlo simulation."""
    if state.detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    regimes = state.detector.predict(state.macro_features)
    proba = state.detector.predict_proba(state.macro_features)
    current = regimes.iloc[-1]
    confidence = proba.iloc[-1].max()

    # Estimate regime parameters
    common_idx = state.market_returns.index.intersection(regimes.index)
    aligned = pd.concat(
        [state.market_returns.loc[common_idx], regimes.loc[common_idx].rename("Regime")],
        axis=1,
    )

    regime_params = {}
    for regime in regimes.unique():
        regime_data = aligned[aligned["Regime"] == regime].drop(columns=["Regime"])
        params = {}
        for asset in regime_data.columns:
            params[asset] = (regime_data[asset].mean(), regime_data[asset].std())
        regime_params[regime] = params

    # Run Monte Carlo
    mc = MonteCarloStressTest(n_simulations=simulations, horizon_months=horizon)
    trans_matrix = state.detector.get_transition_matrix()
    sim = mc.simulate_regime_paths(current, trans_matrix, regime_params)

    weights = state.allocator.get_target_allocation(current, confidence)
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
async def get_stress_scenarios():
    """Run deterministic stress scenarios on current portfolio."""
    if state.detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    regimes = state.detector.predict(state.macro_features)
    proba = state.detector.predict_proba(state.macro_features)
    current = regimes.iloc[-1]
    confidence = proba.iloc[-1].max()

    weights = state.allocator.get_target_allocation(current, confidence)

    results = {}
    for name, shocks in STRESS_SCENARIOS.items():
        assets = list(weights.index)
        impact = sum(weights.get(a, 0) * shocks.get(a, 0) for a in assets)
        results[name] = {"portfolio_impact": impact, "shocks": shocks}

    return {"regime": current, "scenarios": results}


@app.post("/model/refresh")
async def refresh_model():
    """Reload model with latest data."""
    try:
        load_models()
        return {"status": "refreshed", "timestamp": state.last_updated}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
