"""
Dashboard Pages - Performance Tear Sheet
Quantitative strategy factsheet similar to hedge fund reporting.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def render_tear_sheet(result):
    """Render a professional tear sheet for the backtest result."""
    st.header("📋 Performance Tear Sheet")

    if result is None:
        st.warning("Run backtest first.")
        return

    strat_rets = result.portfolio_value.pct_change().dropna()
    bench_rets = result.benchmark_value.pct_change().dropna()

    # ─── Key Metrics Row ──────────────────────────────────────────────
    st.subheader("Key Performance Indicators")
    cols = st.columns(6)

    metrics_display = [
        ("Ann. Return", result.metrics.get("Ann. Return (Strategy)", "—")),
        ("Sharpe", result.metrics.get("Sharpe Ratio (Strategy)", "—")),
        ("Sortino", result.metrics.get("Sortino Ratio", "—")),
        ("Max DD", result.metrics.get("Max Drawdown (Strategy)", "—")),
        ("Calmar", result.metrics.get("Calmar Ratio", "—")),
        ("Info Ratio", result.metrics.get("Information Ratio", "—")),
    ]
    for col, (label, value) in zip(cols, metrics_display):
        col.metric(label, value)

    st.markdown("---")

    # ─── Cumulative Returns ───────────────────────────────────────────
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            "Cumulative Returns", "Rolling Sharpe (12m)",
            "Underwater (Drawdown)", "Monthly Returns Heatmap",
            "Return Distribution", "Rolling Volatility (12m)",
            "Active Returns vs Benchmark", "Regime Allocation Over Time",
        ),
        row_heights=[0.3, 0.2, 0.25, 0.25],
        vertical_spacing=0.08,
    )

    # 1. Cumulative returns
    strat_cum = (1 + strat_rets).cumprod()
    bench_cum = (1 + bench_rets).cumprod()

    fig.add_trace(
        go.Scatter(x=strat_cum.index, y=strat_cum.values,
                   name="Strategy", line=dict(color="#2ecc71", width=2)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=bench_cum.index, y=bench_cum.values,
                   name="Benchmark", line=dict(color="#3498db", width=2, dash="dash")),
        row=1, col=1,
    )

    # 2. Rolling Sharpe
    rolling_sharpe = (
        strat_rets.rolling(12).mean() / strat_rets.rolling(12).std() * np.sqrt(12)
    )
    fig.add_trace(
        go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values,
                   name="Rolling Sharpe", line=dict(color="#9b59b6")),
        row=1, col=2,
    )
    fig.add_hline(y=0, row=1, col=2, line_dash="dash", line_color="grey")

    # 3. Drawdown
    peak = result.portfolio_value.expanding().max()
    dd = (result.portfolio_value - peak) / peak
    fig.add_trace(
        go.Scatter(x=dd.index, y=dd.values, fill="tozeroy",
                   name="Drawdown", fillcolor="rgba(231,76,60,0.3)",
                   line=dict(color="#e74c3c")),
        row=2, col=1,
    )

    # 4. Monthly returns as bar
    monthly_rets = strat_rets.copy()
    colors_bar = ["#2ecc71" if r >= 0 else "#e74c3c" for r in monthly_rets]
    fig.add_trace(
        go.Bar(x=monthly_rets.index, y=monthly_rets.values,
               name="Monthly Return", marker_color=colors_bar),
        row=2, col=2,
    )

    # 5. Return distribution
    fig.add_trace(
        go.Histogram(x=strat_rets.values, nbinsx=50, name="Strategy",
                     marker_color="#2ecc71", opacity=0.7),
        row=3, col=1,
    )
    fig.add_trace(
        go.Histogram(x=bench_rets.values, nbinsx=50, name="Benchmark",
                     marker_color="#3498db", opacity=0.5),
        row=3, col=1,
    )

    # 6. Rolling volatility
    rolling_vol = strat_rets.rolling(12).std() * np.sqrt(12)
    bench_vol = bench_rets.rolling(12).std() * np.sqrt(12)
    fig.add_trace(
        go.Scatter(x=rolling_vol.index, y=rolling_vol.values,
                   name="Strategy Vol", line=dict(color="#2ecc71")),
        row=3, col=2,
    )
    fig.add_trace(
        go.Scatter(x=bench_vol.index, y=bench_vol.values,
                   name="Benchmark Vol", line=dict(color="#3498db", dash="dash")),
        row=3, col=2,
    )

    # 7. Active returns
    active = strat_rets - bench_rets
    active_colors = ["#2ecc71" if r >= 0 else "#e74c3c" for r in active]
    fig.add_trace(
        go.Bar(x=active.index, y=active.values,
               name="Active Return", marker_color=active_colors),
        row=4, col=1,
    )

    # 8. Weight evolution (if available)
    if result.weights_history is not None and not result.weights_history.empty:
        for col_name in result.weights_history.columns[:5]:  # top 5 assets
            fig.add_trace(
                go.Scatter(
                    x=result.weights_history.index,
                    y=result.weights_history[col_name],
                    name=col_name,
                    stackgroup="one",
                ),
                row=4, col=2,
            )

    fig.update_layout(height=1400, showlegend=True, title_text="Strategy Tear Sheet")
    st.plotly_chart(fig, use_container_width=True)

    # ─── Monthly Returns Table ────────────────────────────────────────
    st.subheader("Monthly Returns Table")
    monthly_df = strat_rets.to_frame("return")
    monthly_df["year"] = monthly_df.index.year
    monthly_df["month"] = monthly_df.index.month

    pivot = monthly_df.pivot_table(values="return", index="year", columns="month")
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot["Annual"] = (1 + strat_rets.groupby(strat_rets.index.year).apply(
        lambda x: (1 + x).prod() - 1
    )).values - 1 if len(pivot) > 0 else 0

    st.dataframe(
        pivot.style.format("{:.2%}").background_gradient(
            cmap="RdYlGn", vmin=-0.05, vmax=0.05
        ),
        use_container_width=True,
    )

    # ─── Risk Decomposition ───────────────────────────────────────────
    st.subheader("Drawdown Periods")
    dd_series = dd.copy()
    # Find top 5 drawdowns
    drawdowns = []
    in_dd = False
    start = None
    for i in range(len(dd_series)):
        if dd_series.iloc[i] < -0.01 and not in_dd:
            in_dd = True
            start = dd_series.index[i]
        elif dd_series.iloc[i] >= -0.001 and in_dd:
            in_dd = False
            end = dd_series.index[i]
            min_dd = dd_series.loc[start:end].min()
            drawdowns.append({"Start": start, "End": end, "Max Drawdown": min_dd,
                            "Duration (months)": (end - start).days // 30})

    if drawdowns:
        dd_df = pd.DataFrame(drawdowns).sort_values("Max Drawdown").head(5)
        dd_df["Max Drawdown"] = dd_df["Max Drawdown"].apply(lambda x: f"{x:.2%}")
        st.dataframe(dd_df, use_container_width=True, hide_index=True)
