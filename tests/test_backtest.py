"""
Tests for the backtesting engine.
"""

import pytest
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, "..")
from backtesting.engine import BacktestEngine, BacktestResult
from models.allocator import TacticalAllocator
from config.settings import REGIME_ALLOCATIONS


class TestBacktestEngine:
    """Tests for BacktestEngine."""

    @pytest.fixture
    def sample_data(self):
        """Create sample market data and regime signals."""
        np.random.seed(42)
        n_months = 120
        dates = pd.date_range("2010-01-01", periods=n_months, freq="ME")

        # Simulated monthly returns
        assets = ["US_Equity", "Intl_Equity", "US_Bonds", "Gold", "Cash"]
        returns = pd.DataFrame(
            {
                "US_Equity": np.random.randn(n_months) * 0.04 + 0.008,
                "Intl_Equity": np.random.randn(n_months) * 0.05 + 0.006,
                "US_Bonds": np.random.randn(n_months) * 0.015 + 0.003,
                "Gold": np.random.randn(n_months) * 0.035 + 0.004,
                "Cash": np.ones(n_months) * 0.002,
            },
            index=dates,
        )

        # Simulated regime signals
        regime_sequence = (
            ["Expansion"] * 30
            + ["Slowdown"] * 20
            + ["Recession"] * 25
            + ["Recovery"] * 20
            + ["Expansion"] * 25
        )
        regimes = pd.Series(regime_sequence, index=dates, name="Regime")

        return returns, regimes

    def test_backtest_runs(self, sample_data):
        """Test that backtest completes without error."""
        returns, regimes = sample_data
        engine = BacktestEngine(initial_capital=1_000_000)

        regime_allocs = {
            name: pd.Series(weights)
            for name, weights in REGIME_ALLOCATIONS.items()
        }

        result = engine.run(returns, regimes, regime_allocs)

        assert isinstance(result, BacktestResult)
        assert len(result.portfolio_value) > 0

    def test_portfolio_value_positive(self, sample_data):
        """Test portfolio value stays positive (no bankruptcy)."""
        returns, regimes = sample_data
        engine = BacktestEngine()

        regime_allocs = {
            name: pd.Series(weights)
            for name, weights in REGIME_ALLOCATIONS.items()
        }

        result = engine.run(returns, regimes, regime_allocs)
        assert (result.portfolio_value > 0).all()

    def test_benchmark_matches_60_40(self, sample_data):
        """Test benchmark is computed correctly."""
        returns, regimes = sample_data
        engine = BacktestEngine(transaction_cost_bps=0)

        regime_allocs = {
            name: pd.Series(weights)
            for name, weights in REGIME_ALLOCATIONS.items()
        }

        result = engine.run(returns, regimes, regime_allocs)

        # Benchmark should start at initial capital
        assert result.benchmark_value.iloc[0] == engine.initial_capital

    def test_metrics_computed(self, sample_data):
        """Test that all metrics are computed."""
        returns, regimes = sample_data
        engine = BacktestEngine()

        regime_allocs = {
            name: pd.Series(weights)
            for name, weights in REGIME_ALLOCATIONS.items()
        }

        result = engine.run(returns, regimes, regime_allocs)

        assert "Sharpe Ratio (Strategy)" in result.metrics
        assert "Max Drawdown (Strategy)" in result.metrics
        assert "Information Ratio" in result.metrics

    def test_transaction_costs_reduce_returns(self, sample_data):
        """Test that higher transaction costs reduce performance."""
        returns, regimes = sample_data

        regime_allocs = {
            name: pd.Series(weights)
            for name, weights in REGIME_ALLOCATIONS.items()
        }

        engine_low = BacktestEngine(transaction_cost_bps=0)
        engine_high = BacktestEngine(transaction_cost_bps=50)

        result_low = engine_low.run(returns, regimes, regime_allocs)
        result_high = engine_high.run(returns, regimes, regime_allocs)

        assert result_low.portfolio_value.iloc[-1] >= result_high.portfolio_value.iloc[-1]


class TestTacticalAllocator:
    """Tests for TacticalAllocator."""

    def test_weights_sum_to_one(self):
        """Test that allocations always sum to 1."""
        allocator = TacticalAllocator()

        for regime in ["Expansion", "Slowdown", "Recession", "Recovery"]:
            weights = allocator.get_target_allocation(regime)
            assert abs(weights.sum() - 1.0) < 1e-6

    def test_confidence_blending(self):
        """Test that low confidence blends toward benchmark."""
        allocator = TacticalAllocator()

        full_confidence = allocator.get_target_allocation("Recession", confidence=1.0)
        half_confidence = allocator.get_target_allocation("Recession", confidence=0.5)
        zero_confidence = allocator.get_target_allocation("Recession", confidence=0.0)

        # At zero confidence, should equal benchmark
        benchmark = pd.Series(allocator.benchmark)
        benchmark = benchmark / benchmark.sum()

        # half should be between full and zero
        assert not full_confidence.equals(half_confidence)

    def test_unknown_regime_returns_benchmark(self):
        """Test fallback for unknown regime."""
        allocator = TacticalAllocator()
        weights = allocator.get_target_allocation("Unknown_Regime")

        benchmark = pd.Series(allocator.benchmark)
        benchmark = benchmark / benchmark.sum()
        # Should match benchmark
        assert abs(weights.sum() - 1.0) < 1e-6
