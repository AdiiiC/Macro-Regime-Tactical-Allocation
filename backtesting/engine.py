"""
Backtesting engine for tactical allocation strategy.
Simulates historical performance with realistic constraints.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

import sys
sys.path.insert(0, "..")
from config.settings import (
    TRANSACTION_COST_BPS,
    INITIAL_CAPITAL,
    BENCHMARK_ALLOCATION,
    REBALANCE_FREQUENCY,
)


@dataclass
class BacktestResult:
    """Container for backtest results and analytics."""

    portfolio_value: pd.Series
    benchmark_value: pd.Series
    weights_history: pd.DataFrame
    regime_history: pd.Series
    trades_history: pd.DataFrame
    metrics: Dict = field(default_factory=dict)

    def compute_metrics(self) -> Dict:
        """Compute comprehensive performance metrics."""
        strat_rets = self.portfolio_value.pct_change().dropna()
        bench_rets = self.benchmark_value.pct_change().dropna()

        # Annualized return
        n_years = len(strat_rets) / 12
        total_return_strat = (self.portfolio_value.iloc[-1] / self.portfolio_value.iloc[0]) - 1
        total_return_bench = (self.benchmark_value.iloc[-1] / self.benchmark_value.iloc[0]) - 1

        ann_return_strat = (1 + total_return_strat) ** (1 / n_years) - 1
        ann_return_bench = (1 + total_return_bench) ** (1 / n_years) - 1

        # Volatility
        ann_vol_strat = strat_rets.std() * np.sqrt(12)
        ann_vol_bench = bench_rets.std() * np.sqrt(12)

        # Sharpe (assuming 3% risk-free)
        rf = 0.03
        sharpe_strat = (ann_return_strat - rf) / ann_vol_strat if ann_vol_strat > 0 else 0
        sharpe_bench = (ann_return_bench - rf) / ann_vol_bench if ann_vol_bench > 0 else 0

        # Max Drawdown
        max_dd_strat = self._max_drawdown(self.portfolio_value)
        max_dd_bench = self._max_drawdown(self.benchmark_value)

        # Sortino Ratio
        downside_strat = strat_rets[strat_rets < 0].std() * np.sqrt(12)
        sortino_strat = (ann_return_strat - rf) / downside_strat if downside_strat > 0 else 0

        # Calmar Ratio
        calmar_strat = ann_return_strat / abs(max_dd_strat) if max_dd_strat != 0 else 0

        # Information Ratio
        active_returns = strat_rets - bench_rets
        tracking_error = active_returns.std() * np.sqrt(12)
        info_ratio = (ann_return_strat - ann_return_bench) / tracking_error if tracking_error > 0 else 0

        # Win Rate
        win_rate = (active_returns > 0).sum() / len(active_returns)

        # Turnover
        if not self.trades_history.empty:
            avg_turnover = self.trades_history.groupby(level=0)["abs_delta"].sum().mean()
        else:
            avg_turnover = 0

        self.metrics = {
            "Total Return (Strategy)": f"{total_return_strat:.2%}",
            "Total Return (Benchmark)": f"{total_return_bench:.2%}",
            "Ann. Return (Strategy)": f"{ann_return_strat:.2%}",
            "Ann. Return (Benchmark)": f"{ann_return_bench:.2%}",
            "Ann. Volatility (Strategy)": f"{ann_vol_strat:.2%}",
            "Ann. Volatility (Benchmark)": f"{ann_vol_bench:.2%}",
            "Sharpe Ratio (Strategy)": f"{sharpe_strat:.3f}",
            "Sharpe Ratio (Benchmark)": f"{sharpe_bench:.3f}",
            "Sortino Ratio": f"{sortino_strat:.3f}",
            "Calmar Ratio": f"{calmar_strat:.3f}",
            "Information Ratio": f"{info_ratio:.3f}",
            "Max Drawdown (Strategy)": f"{max_dd_strat:.2%}",
            "Max Drawdown (Benchmark)": f"{max_dd_bench:.2%}",
            "Win Rate (vs Benchmark)": f"{win_rate:.2%}",
            "Tracking Error": f"{tracking_error:.2%}",
            "Avg Monthly Turnover": f"{avg_turnover:.2%}",
        }
        return self.metrics

    @staticmethod
    def _max_drawdown(values: pd.Series) -> float:
        """Compute maximum drawdown."""
        peak = values.expanding().max()
        drawdown = (values - peak) / peak
        return drawdown.min()


class BacktestEngine:
    """
    Event-driven backtesting engine.
    Simulates monthly rebalancing based on regime signals.
    """

    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        transaction_cost_bps: int = TRANSACTION_COST_BPS,
        rebalance_freq: str = REBALANCE_FREQUENCY,
    ):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost_bps / 10_000
        self.rebalance_freq = rebalance_freq

    def run(
        self,
        asset_returns: pd.DataFrame,
        regime_signals: pd.Series,
        regime_allocations: Dict[str, pd.Series],
        benchmark_weights: Optional[Dict] = None,
    ) -> BacktestResult:
        """
        Run full backtest simulation.

        Args:
            asset_returns: Monthly returns for each asset class
            regime_signals: Regime label for each month
            regime_allocations: Target weights per regime
            benchmark_weights: Static benchmark allocation
        """
        if benchmark_weights is None:
            benchmark_weights = BENCHMARK_ALLOCATION

        # Align dates
        common_dates = asset_returns.index.intersection(regime_signals.index)
        asset_returns = asset_returns.loc[common_dates]
        regime_signals = regime_signals.loc[common_dates]

        assets = asset_returns.columns.tolist()
        n_periods = len(common_dates)

        # Initialize tracking
        portfolio_values = [self.initial_capital]
        benchmark_values = [self.initial_capital]
        weights_history = []
        trades_records = []

        current_weights = pd.Series(0.0, index=assets)
        bench_weights = pd.Series(benchmark_weights).reindex(assets, fill_value=0.0)
        bench_weights = bench_weights / bench_weights.sum()

        current_bench_weights = bench_weights.copy()

        for i in range(n_periods):
            date = common_dates[i]
            regime = regime_signals.iloc[i]
            period_returns = asset_returns.iloc[i]

            # ─── Strategy Portfolio ────────────────────────────────────
            # Get target allocation for current regime
            if regime in regime_allocations:
                target = regime_allocations[regime]
                if isinstance(target, dict):
                    target = pd.Series(target)
                target = target.reindex(assets, fill_value=0.0)
                target = target / target.sum()
            else:
                target = current_weights  # hold if unknown

            # Compute turnover and transaction costs
            turnover = (target - current_weights).abs().sum()
            tc = turnover * self.transaction_cost

            # Record trades
            deltas = target - current_weights
            for asset in assets:
                if abs(deltas[asset]) > 0.005:
                    trades_records.append({
                        "date": date,
                        "asset": asset,
                        "delta": deltas[asset],
                        "abs_delta": abs(deltas[asset]),
                        "regime": regime,
                    })

            # Update weights to target (rebalance)
            current_weights = target.copy()

            # Apply returns
            port_return = (current_weights * period_returns).sum() - tc
            new_value = portfolio_values[-1] * (1 + port_return)
            portfolio_values.append(new_value)

            # Drift weights post-return
            drifted = current_weights * (1 + period_returns)
            current_weights = drifted / drifted.sum()

            weights_history.append(current_weights.copy())

            # ─── Benchmark Portfolio ──────────────────────────────────
            bench_return = (bench_weights * period_returns).sum()
            bench_value = benchmark_values[-1] * (1 + bench_return)
            benchmark_values.append(bench_value)

        # Build result
        dates_with_start = [common_dates[0] - pd.DateOffset(months=1)] + list(common_dates)
        portfolio_series = pd.Series(portfolio_values, index=dates_with_start, name="Strategy")
        benchmark_series = pd.Series(benchmark_values, index=dates_with_start, name="Benchmark_60_40")

        weights_df = pd.DataFrame(weights_history, index=common_dates, columns=assets)

        trades_df = pd.DataFrame(trades_records)
        if not trades_df.empty:
            trades_df = trades_df.set_index(["date", "asset"])

        result = BacktestResult(
            portfolio_value=portfolio_series,
            benchmark_value=benchmark_series,
            weights_history=weights_df,
            regime_history=regime_signals,
            trades_history=trades_df,
        )
        result.compute_metrics()
        return result


class WalkForwardBacktest:
    """
    Walk-forward validation: retrain HMM on expanding window,
    predict out-of-sample regime, then allocate.
    """

    def __init__(
        self,
        min_train_months: int = 60,
        retrain_frequency: int = 12,  # retrain every 12 months
    ):
        self.min_train_months = min_train_months
        self.retrain_frequency = retrain_frequency

    def run(
        self,
        macro_features: pd.DataFrame,
        asset_returns: pd.DataFrame,
        regime_detector_class,
        allocator,
    ) -> BacktestResult:
        """
        Walk-forward backtest with periodic model retraining.

        Args:
            macro_features: Full history of transformed macro features
            asset_returns: Full history of asset returns
            regime_detector_class: Class to instantiate for regime detection
            allocator: TacticalAllocator instance
        """
        common_dates = macro_features.index.intersection(asset_returns.index)
        macro_features = macro_features.loc[common_dates]
        asset_returns = asset_returns.loc[common_dates]

        n_periods = len(common_dates)
        regime_predictions = []

        detector = None
        last_train_idx = 0

        for i in range(self.min_train_months, n_periods):
            # Retrain if needed
            if detector is None or (i - last_train_idx) >= self.retrain_frequency:
                train_data = macro_features.iloc[: i]
                detector = regime_detector_class()
                detector.fit(train_data)
                last_train_idx = i

            # Predict regime for current period (using only data up to now)
            current_features = macro_features.iloc[i: i + 1]
            regime = detector.predict(current_features).iloc[0]
            regime_predictions.append((common_dates[i], regime))

        # Build regime series
        regime_series = pd.Series(
            dict(regime_predictions), name="Regime"
        )

        # Get allocations per regime
        regime_allocs = {}
        for regime_name in regime_series.unique():
            weights = allocator.get_target_allocation(regime_name)
            regime_allocs[regime_name] = weights

        # Run backtest on out-of-sample period
        oos_returns = asset_returns.loc[regime_series.index]

        engine = BacktestEngine()
        return engine.run(oos_returns, regime_series, regime_allocs)
