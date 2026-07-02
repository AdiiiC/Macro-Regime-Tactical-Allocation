"""
Contract tests for the FastAPI surface (api/main.py).

These exercise the public HTTP contract the frontend depends on: response
status codes and the presence of the fields each panel reads. The app fits
the US model on startup, so the module-scoped client is shared across tests.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert isinstance(body["markets"], list)
    assert "us" in body["markets"]


def test_markets_lists_us_and_india(client):
    r = client.get("/markets")
    assert r.status_code == 200
    keys = {m["key"] for m in r.json()["markets"]}
    assert {"us", "india"} <= keys
    us = next(m for m in r.json()["markets"] if m["key"] == "us")
    assert us["has_returns"] is True


def test_current_regime_schema(client):
    r = client.get("/regime/current?market=us")
    assert r.status_code == 200
    body = r.json()
    for field in ("current_regime", "confidence", "regime_probabilities"):
        assert field in body
    assert 0.0 <= body["confidence"] <= 1.0


def test_allocation_weights_sum_to_one(client):
    r = client.get("/allocation/current?market=us")
    assert r.status_code == 200
    body = r.json()
    total = sum(body["target_weights"].values())
    assert total == pytest.approx(1.0, abs=1e-6)


def test_backtest_us_has_curve_and_metrics(client):
    r = client.get("/backtest?market=us")
    assert r.status_code == 200
    body = r.json()
    assert body["equity_curve"], "equity curve should be non-empty"
    assert "sharpe_strategy" in body["metrics"]
    assert "regime_attribution" in body


def test_backtest_benchmark_variants(client):
    """Each benchmark variant returns 200 and echoes its own name."""
    for variant in ("sixty_forty", "equal_weight", "risk_parity"):
        r = client.get(f"/backtest?market=us&benchmark={variant}")
        assert r.status_code == 200, variant
        assert r.json()["benchmark"] == variant


def test_backtest_unknown_benchmark_is_400(client):
    r = client.get("/backtest?market=us&benchmark=moon")
    assert r.status_code == 400


def test_sensitivity_sweep_grid(client):
    r = client.get("/backtest/sensitivity?market=us")
    assert r.status_code == 200
    body = r.json()
    assert body["cost_grid"] and body["cadence_grid"]
    assert len(body["cells"]) == len(body["cost_grid"]) * len(body["cadence_grid"])
    cell = body["cells"][0]
    assert {"cost_bps", "rebalance_months", "sharpe"} <= set(cell.keys())


def test_sensitivity_india_now_live(client):
    """India now carries real market returns, so the sweep computes."""
    r = client.get("/backtest/sensitivity?market=india")
    assert r.status_code == 200
    assert r.json()["cells"]


def test_drivers_schema(client):
    r = client.get("/regime/drivers?market=us&top=5")
    assert r.status_code == 200
    body = r.json()
    assert len(body["drivers"]) <= 5
    assert {"feature", "z_score"} <= set(body["drivers"][0].keys())


def test_model_comparison_has_primary(client):
    r = client.get("/model/comparison?market=us")
    assert r.status_code == 200
    names = {m["name"] for m in r.json()["models"]}
    assert any("HMM" in n for n in names)


def test_india_regime_available(client):
    """India has macro features, so regime detection works."""
    r = client.get("/regime/current?market=india")
    assert r.status_code == 200
    assert r.json()["current_regime"] in {
        "Expansion",
        "Slowdown",
        "Recession",
        "Recovery",
    }


def test_india_risk_now_live(client):
    """India carries live market-return history: risk analytics compute."""
    r = client.get("/risk/var?market=india&horizon=12")
    assert r.status_code == 200
    assert "var_95" in r.json()


def test_india_backtest_now_live(client):
    r = client.get("/backtest?market=india")
    assert r.status_code == 200
    body = r.json()
    assert body["equity_curve"]
    assert body["currency"] == "INR"


def test_india_stress_uses_india_scenarios(client):
    """India stress uses India-calibrated scenarios (non-zero, INR assets)."""
    r = client.get("/risk/stress-scenarios?market=india")
    assert r.status_code == 200
    scenarios = r.json()["scenarios"]
    assert "Rupee Crisis (FII Outflows)" in scenarios
    # At least one scenario has a materially non-zero portfolio impact.
    impacts = [abs(s["portfolio_impact"]) for s in scenarios.values()]
    assert max(impacts) > 0.005


def test_data_status_reports_currency(client):
    r = client.get("/data/status")
    assert r.status_code == 200
    body = r.json()
    assert "us" in body["markets"] and "india" in body["markets"]
    # Every loaded market should report some 'through' date.
    for entry in body["markets"].values():
        assert any(
            entry.get(k)
            for k in ("prices_through", "returns_through", "macro_through")
        )


def test_unknown_market_is_404(client):
    r = client.get("/regime/current?market=mars")
    assert r.status_code == 404
