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


def test_india_risk_degrades_gracefully(client):
    """India lacks market-return history: risk analytics return 501, not 500."""
    r = client.get("/risk/var?market=india&horizon=12")
    assert r.status_code == 501


def test_india_backtest_degrades_gracefully(client):
    r = client.get("/backtest?market=india")
    assert r.status_code == 501


def test_unknown_market_is_404(client):
    r = client.get("/regime/current?market=mars")
    assert r.status_code == 404
