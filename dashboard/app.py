"""
Streamlit Dashboard — Macro Regime Detection & Tactical Allocation
Multi-page app with regime visualization, allocation, and backtest results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    REGIME_COLORS,
    REGIME_ALLOCATIONS,
    ASSET_TICKERS,
    BENCHMARK_ALLOCATION,
    HMM_N_REGIMES,
)
from config.india_settings import (
    INDIA_REGIME_ALLOCATIONS,
    INDIA_BENCHMARK_ALLOCATION,
    INDIA_TICKER_LABELS,
    INDIA_REGIME_EXPLANATIONS,
)
from data.fred_pipeline import MacroDataPipeline, load_cached_data, save_cached_data
from data.market_data import MarketDataPipeline
from data.india_pipeline import IndiaDataPipeline, load_cached_india_data, save_cached_india_data
from data.india_market import IndiaMarketDataPipeline
from models.regime_hmm import RegimeDetector
from models.allocator import TacticalAllocator
from backtesting.engine import BacktestEngine, BacktestResult
from dashboard.pages.stress_testing import render_stress_testing_page
from dashboard.pages.tear_sheet import render_tear_sheet
from dashboard.pages.model_selection import render_model_selection_page
from dashboard.components.ui_components import (
    inject_custom_css,
    render_header,
    render_ticker_tape,
    render_kpi_card,
    render_kpi_row,
    render_regime_badge,
    render_confidence_gauge,
    render_sidebar_freshness,
    render_section_header,
    styled_dataframe,
)
from reports.pdf_generator import PDFReportGenerator

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Macro Regime Tactical Allocation",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Inject Custom Theme ───────────────────────────────────────────────────────
inject_custom_css()
render_header()

st.markdown("")  # spacer after header

# ─── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/48/graph-report.png", width=40)
st.sidebar.markdown("## ⚙️ Configuration")

country = st.sidebar.radio(
    "🌍 Market",
    ["🇺🇸 United States", "🇮🇳 India"],
    index=0,
)

data_source = st.sidebar.radio(
    "Data Source",
    ["Live (FRED API)", "Cached (Demo)"],
    index=1,
)

n_regimes = st.sidebar.slider("Number of Regimes", 2, 6, HMM_N_REGIMES)
pca_components = st.sidebar.slider("PCA Components", 3, 10, 5)
risk_aversion = st.sidebar.slider("Risk Aversion (λ)", 0.5, 5.0, 2.5, 0.5)

st.sidebar.markdown("---")
st.sidebar.markdown("## 📅 Backtest Period")
start_year = st.sidebar.slider("Start Year", 2005, 2024, 2005)
end_year = st.sidebar.slider("End Year", 2010, 2026, 2026)

st.sidebar.markdown("---")
render_sidebar_freshness(
    last_updated=datetime.now().strftime("%b %d, %Y %H:%M"),
    model_status="Active",
)

st.sidebar.markdown("---")
st.sidebar.header("📄 Export")
export_pdf = st.sidebar.button("📥 Generate PDF Report")


# ─── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data(source: str, start: str, end: str, market: str = "US"):
    """Load macro and market data for selected country."""
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "data", "cache")
    os.makedirs(cache_dir, exist_ok=True)

    if market == "India":
        macro_cache = os.path.join(cache_dir, "india_macro_features.parquet")
        market_cache = os.path.join(cache_dir, "india_market_returns.parquet")

        if source == "Cached (Demo)" and os.path.exists(macro_cache):
            macro_features = pd.read_parquet(macro_cache)
            market_returns = pd.read_parquet(market_cache)
        else:
            pipeline = IndiaDataPipeline()
            pipeline.fetch_all_indicators(start=start, end=end)
            macro_features = pipeline.get_model_ready_data()
            save_cached_india_data(macro_features, macro_cache)

            market = IndiaMarketDataPipeline()
            market_returns = market.compute_returns(frequency="M")
            save_cached_india_data(market_returns, market_cache)
    else:
        macro_cache = os.path.join(cache_dir, "macro_features.parquet")
        market_cache = os.path.join(cache_dir, "market_returns.parquet")

        if source == "Cached (Demo)" and os.path.exists(macro_cache):
            macro_features = pd.read_parquet(macro_cache)
            market_returns = pd.read_parquet(market_cache)
        else:
            pipeline = MacroDataPipeline()
            pipeline.fetch_all_indicators(start=start, end=end)
            macro_features = pipeline.get_model_ready_data()
            save_cached_data(macro_features, macro_cache)

            mkt = MarketDataPipeline()
            market_returns = mkt.compute_returns(frequency="M")
            save_cached_data(market_returns, market_cache)

    return macro_features, market_returns


# ─── Main Logic ────────────────────────────────────────────────────────────────
is_india = country == "🇮🇳 India"
active_regime_allocs = INDIA_REGIME_ALLOCATIONS if is_india else REGIME_ALLOCATIONS
active_benchmark = INDIA_BENCHMARK_ALLOCATION if is_india else BENCHMARK_ALLOCATION
active_ticker_labels = INDIA_TICKER_LABELS if is_india else None

try:
    macro_features, market_returns = load_data(
        data_source, f"{start_year}-01-01", f"{end_year}-12-31",
        market="India" if is_india else "US",
    )

    # Fit regime model
    detector = RegimeDetector(n_regimes=n_regimes, n_components_pca=pca_components)
    detector.fit(macro_features)
    regimes = detector.predict(macro_features)
    regime_proba = detector.predict_proba(macro_features)

    # ═══════════════════════════════════════════════════════════════════════════
    # TICKER TAPE (above tabs)
    # ═══════════════════════════════════════════════════════════════════════════
    # Current regime
    current_regime = regimes.iloc[-1]
    current_proba = regime_proba.iloc[-1]
    confidence = current_proba.max()
    duration = detector.get_expected_duration()

    # ─── Sidebar: Confidence Gauge ─────────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown("## 🎯 Current Signal")
        render_confidence_gauge(confidence)
        st.markdown(
            f"<p style='text-align:center; color:#6b7d93; font-size:0.8rem;'>"
            f"Regime: <b style='color:#c9a227'>{current_regime}</b></p>",
            unsafe_allow_html=True,
        )

    # Build ticker tape from latest macro features
    US_TICKER_LABELS = {
        "GDP_YoY": "GDP YoY",
        "GDP_Mom3": "GDP Momentum",
        "Industrial_Production_YoY": "Ind. Prod. YoY",
        "Industrial_Production_Mom3": "Ind. Prod. Mom",
        "Retail_Sales_YoY": "Retail Sales YoY",
        "Retail_Sales_Mom3": "Retail Sales Mom",
        "Nonfarm_Payrolls_YoY": "Payrolls YoY",
        "Nonfarm_Payrolls_Mom3": "Payrolls Mom",
        "CPI_YoY": "CPI YoY",
        "CPI_Mom3": "CPI Momentum",
        "Core_CPI_YoY": "Core CPI YoY",
        "Core_CPI_Mom3": "Core CPI Mom",
        "PPI_YoY": "PPI YoY",
        "M2_YoY": "M2 Growth",
        "Housing_Starts_YoY": "Housing YoY",
        "Fed_Funds_Rate_Level": "Fed Funds",
        "Fed_Funds_Rate_Chg3": "Fed Funds Δ3m",
        "Treasury_10Y_Level": "10Y Yield",
        "Treasury_10Y_Chg3": "10Y Yield Δ3m",
        "Treasury_2Y_Level": "2Y Yield",
        "Yield_Spread_10Y2Y_Level": "Yield Curve",
        "Yield_Spread_10Y2Y_Chg3": "Yield Curve Δ",
        "BAA_Spread_Level": "Credit Spread",
        "High_Yield_Spread_Level": "HY Spread",
        "VIX_Level": "VIX",
        "VIX_Chg3": "VIX Δ3m",
        "Unemployment_Rate_Level": "Unemp. Rate",
        "Financial_Stress_Level": "Fin. Stress",
    }
    label_map = INDIA_TICKER_LABELS if is_india else US_TICKER_LABELS
    ticker_data = {}
    if macro_features is not None and len(macro_features) > 1:
        latest = macro_features.iloc[-1]
        prev = macro_features.iloc[-2]
        for col in list(macro_features.columns)[:8]:
            label = label_map.get(col, col.replace("_", " "))
            ticker_data[label] = {
                "value": latest[col],
                "change": latest[col] - prev[col],
                "unit": "",
            }

    if ticker_data:
        render_ticker_tape(ticker_data)

    # ─── Tab Layout ────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🔍 Regime Detection",
        "📈 Allocation",
        "🧪 Backtest",
        "📋 Tear Sheet",
        "🎲 Stress Testing",
        "🔬 Model Selection",
        "📊 Model Diagnostics",
        "📋 Leading Indicators",
    ])

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1: REGIME DETECTION
    # ═══════════════════════════════════════════════════════════════════════════
    with tab1:
        render_section_header("Macro Regime Detection", "🔍")

        # Regime badge
        render_regime_badge(current_regime, confidence)
        st.markdown("")

        # KPI Cards Row
        regime_sparkline = [
            1 if r == "Expansion" else (0.7 if r == "Recovery" else (0.3 if r == "Slowdown" else 0.1))
            for r in regimes.tail(24).values
        ]
        confidence_history = regime_proba.tail(24).max(axis=1).tolist()

        cards = [
            render_kpi_card(
                "Current Regime", current_regime,
                regime_class=current_regime,
                sparkline_data=regime_sparkline,
            ),
            render_kpi_card(
                "Expected Duration",
                f"{duration.get(current_regime, 0):.0f} mo",
                delta="persistent" if duration.get(current_regime, 0) > 12 else "transitional",
                delta_positive=duration.get(current_regime, 0) > 12,
            ),
            render_kpi_card(
                "Model Confidence",
                f"{confidence:.0%}",
                delta=f"{(confidence - regime_proba.iloc[-2].max()):.1%} vs last",
                delta_positive=confidence >= regime_proba.iloc[-2].max(),
                sparkline_data=confidence_history,
            ),
            render_kpi_card(
                "Regimes Detected",
                str(regimes.nunique()),
                delta=f"{len(regimes)} months analyzed",
                delta_positive=True,
            ),
        ]
        render_kpi_row(cards)

        st.markdown("---")

        # ─── Animated Regime Timeline ─────────────────────────────────────────
        # Build animated scatter showing regime evolution
        regime_numeric = regimes.map({
            "Expansion": 4, "Recovery": 3, "Slowdown": 2, "Recession": 1
        }).fillna(0)

        fig_timeline = go.Figure()

        # Colored bars for each regime
        for regime_name, color in REGIME_COLORS.items():
            mask = regimes == regime_name
            if mask.any():
                fig_timeline.add_trace(go.Scatter(
                    x=regimes[mask].index,
                    y=regime_numeric[mask],
                    mode="markers",
                    marker=dict(
                        color=color, size=10, symbol="square",
                        line=dict(width=0),
                    ),
                    name=regime_name,
                    hovertemplate="%{x|%Y-%m}<br>Regime: " + regime_name + "<extra></extra>",
                ))

        fig_timeline.update_layout(
            title=dict(text="<b>Economic Regime Timeline</b>", font=dict(size=16)),
            xaxis_title="",
            yaxis=dict(
                tickvals=[1, 2, 3, 4],
                ticktext=["Recession", "Slowdown", "Recovery", "Expansion"],
                gridcolor="rgba(42,58,78,0.5)",
            ),
            height=280,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8e8e8"),
            xaxis=dict(gridcolor="rgba(42,58,78,0.3)"),
            # Add range slider for animation-like interaction
            xaxis_rangeslider_visible=True,
            xaxis_rangeslider_thickness=0.05,
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

        # ─── Regime Probability Stacked Area ──────────────────────────────────
        fig_proba = go.Figure()
        for col in regime_proba.columns:
            fig_proba.add_trace(
                go.Scatter(
                    x=regime_proba.index,
                    y=regime_proba[col],
                    name=col,
                    stackgroup="one",
                    fillcolor=REGIME_COLORS.get(col, "#999999"),
                    line=dict(width=0.5, color=REGIME_COLORS.get(col, "#999999")),
                    hovertemplate="%{x|%Y-%m}<br>" + col + ": %{y:.1%}<extra></extra>",
                )
            )
        fig_proba.update_layout(
            title=dict(text="<b>Regime Probabilities Over Time</b>", font=dict(size=16)),
            xaxis_title="",
            yaxis_title="Probability",
            height=380,
            yaxis=dict(range=[0, 1], tickformat=".0%", gridcolor="rgba(42,58,78,0.5)"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8e8e8"),
            xaxis=dict(gridcolor="rgba(42,58,78,0.3)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_proba, use_container_width=True)

        # Transition matrix
        render_section_header("Transition Dynamics", "🔄")
        trans_matrix = detector.get_transition_matrix()
        fig_trans = px.imshow(
            trans_matrix,
            text_auto=".2f",
            color_continuous_scale=[[0, "#0d1b2a"], [0.5, "#1a3a5c"], [1, "#c9a227"]],
            title="<b>Monthly Transition Probabilities</b>",
        )
        fig_trans.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8e8e8"),
            height=350,
        )
        st.plotly_chart(fig_trans, use_container_width=True)

        # Stationary distribution
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Stationary Distribution")
            stat_dist = detector.get_stationary_distribution()
            fig_stat = px.pie(
                values=stat_dist.values,
                names=stat_dist.index,
                color=stat_dist.index,
                color_discrete_map=REGIME_COLORS,
            )
            st.plotly_chart(fig_stat, use_container_width=True)

        with col2:
            st.subheader("Expected Duration per Regime")
            st.dataframe(
                duration.to_frame("Months").style.format("{:.1f}"),
                use_container_width=True,
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 2: ALLOCATION
    # ═══════════════════════════════════════════════════════════════════════════
    with tab2:
        render_section_header("Tactical Asset Allocation", "📈")

        allocator = TacticalAllocator(risk_aversion=risk_aversion, regime_allocations=active_regime_allocs)

        # ─── What-If Panel ────────────────────────────────────────────────────
        with st.expander("⚡ What-If Scenario — Override Regime Manually", expanded=False):
            whatif_col1, whatif_col2 = st.columns([1, 2])
            with whatif_col1:
                override_regime = st.selectbox(
                    "Simulate regime:",
                    ["(Use detected)", "Expansion", "Slowdown", "Recession", "Recovery"],
                )
                override_confidence = st.slider("Override confidence:", 0.0, 1.0, confidence)

            active_regime = current_regime if override_regime == "(Use detected)" else override_regime
            active_confidence = override_confidence if override_regime != "(Use detected)" else confidence

            with whatif_col2:
                whatif_alloc = allocator.get_target_allocation(active_regime, active_confidence)
                fig_whatif = px.bar(
                    x=whatif_alloc.index, y=whatif_alloc.values,
                    color=whatif_alloc.values,
                    color_continuous_scale=[[0, "#1a3a5c"], [1, "#c9a227"]],
                    title=f"<b>Allocation if {active_regime} @ {active_confidence:.0%} confidence</b>",
                )
                fig_whatif.update_layout(
                    yaxis_tickformat=".0%", yaxis_title="Weight",
                    xaxis_title="", showlegend=False,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e8e8e8"), height=280,
                )
                st.plotly_chart(fig_whatif, use_container_width=True)

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"Current Allocation ({current_regime})")
            current_alloc = allocator.get_target_allocation(
                current_regime, confidence=confidence
            )
            fig_alloc = px.pie(
                values=current_alloc.values,
                names=current_alloc.index,
                title=f"Target Weights — {current_regime} Regime",
                color_discrete_sequence=px.colors.qualitative.Set3,
                hole=0.4,
            )
            fig_alloc.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e8e8e8"),
            )
            st.plotly_chart(fig_alloc, use_container_width=True)

        with col2:
            bench_label = "Benchmark" if is_india else "Benchmark (60/40)"
            st.subheader(bench_label)
            bench = pd.Series(active_benchmark)
            bench = bench[bench > 0]
            fig_bench = px.pie(
                values=bench.values,
                names=bench.index,
                title=f"Benchmark — {bench_label}",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                hole=0.4,
            )
            fig_bench.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e8e8e8"),
            )
            st.plotly_chart(fig_bench, use_container_width=True)

        # Regime allocation comparison
        st.subheader("Allocation by Regime")
        alloc_df = pd.DataFrame(active_regime_allocs).T
        fig_alloc_compare = px.bar(
            alloc_df,
            barmode="stack",
            title="Target Allocations Across Regimes",
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig_alloc_compare.update_layout(
            xaxis_title="Regime",
            yaxis_title="Weight",
            height=500,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8e8e8"),
        )
        st.plotly_chart(fig_alloc_compare, use_container_width=True)

        # Rationale
        st.subheader("Allocation Rationale")
        explanation = allocator.get_regime_tilt_explanation(current_regime)
        st.info(f"**{current_regime}:** {explanation['rationale']}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Overweight:**")
            for item in explanation["overweight"]:
                st.markdown(f"- ✅ {item}")
        with col2:
            st.markdown("**Underweight:**")
            for item in explanation["underweight"]:
                st.markdown(f"- ⬇️ {item}")
        with col3:
            st.markdown("**Key Risks:**")
            for item in explanation["key_risks"]:
                st.markdown(f"- ⚠️ {item}")

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 3: BACKTEST
    # ═══════════════════════════════════════════════════════════════════════════
    with tab3:
        render_section_header("Strategy Backtest Results", "📊")

        # Run backtest
        engine = BacktestEngine()

        # Prepare regime allocations as dict of Series
        regime_alloc_series = {
            name: pd.Series(weights)
            for name, weights in active_regime_allocs.items()
        }

        # Align data — normalize to month-end for robust matching
        market_monthly = market_returns.copy()
        market_monthly.index = market_monthly.index.to_period("M").to_timestamp("M")
        regimes_monthly = regimes.copy()
        regimes_monthly.index = regimes_monthly.index.to_period("M").to_timestamp("M")
        common_idx = market_monthly.index.intersection(regimes_monthly.index)
        if len(common_idx) > 12:
            result = engine.run(
                asset_returns=market_monthly.loc[common_idx],
                regime_signals=regimes_monthly.loc[common_idx],
                regime_allocations=regime_alloc_series,
            )

            # Performance chart
            fig_perf = go.Figure()
            fig_perf.add_trace(
                go.Scatter(
                    x=result.portfolio_value.index,
                    y=result.portfolio_value.values,
                    name="Tactical Strategy",
                    line=dict(color="#2ecc71", width=2),
                )
            )
            fig_perf.add_trace(
                go.Scatter(
                    x=result.benchmark_value.index,
                    y=result.benchmark_value.values,
                    name="Benchmark (60/40)",
                    line=dict(color="#3498db", width=2, dash="dash"),
                )
            )

            # Shade regimes
            for regime_name, color in REGIME_COLORS.items():
                mask = result.regime_history == regime_name
                if mask.any():
                    for date in result.regime_history[mask].index:
                        fig_perf.add_vrect(
                            x0=date - pd.DateOffset(days=15),
                            x1=date + pd.DateOffset(days=15),
                            fillcolor=color,
                            opacity=0.1,
                            layer="below",
                            line_width=0,
                        )

            fig_perf.update_layout(
                title="Cumulative Performance: Tactical vs. 60/40 Benchmark",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                height=500,
                yaxis_tickprefix="$",
                yaxis_tickformat=",.0f",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e8e8e8"),
            )
            st.plotly_chart(fig_perf, use_container_width=True)

            # Metrics table
            st.subheader("Performance Metrics")
            metrics_df = pd.DataFrame(
                list(result.metrics.items()), columns=["Metric", "Value"]
            )
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(
                    metrics_df.iloc[: len(metrics_df) // 2],
                    use_container_width=True,
                    hide_index=True,
                )
            with col2:
                st.dataframe(
                    metrics_df.iloc[len(metrics_df) // 2:],
                    use_container_width=True,
                    hide_index=True,
                )

            # Drawdown chart
            st.subheader("Drawdown Analysis")
            peak = result.portfolio_value.expanding().max()
            drawdown = (result.portfolio_value - peak) / peak

            peak_b = result.benchmark_value.expanding().max()
            drawdown_b = (result.benchmark_value - peak_b) / peak_b

            fig_dd = go.Figure()
            fig_dd.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    name="Strategy",
                    fill="tozeroy",
                    fillcolor="rgba(46,204,113,0.3)",
                    line=dict(color="#2ecc71"),
                )
            )
            fig_dd.add_trace(
                go.Scatter(
                    x=drawdown_b.index,
                    y=drawdown_b.values,
                    name="Benchmark",
                    fill="tozeroy",
                    fillcolor="rgba(52,152,219,0.2)",
                    line=dict(color="#3498db", dash="dash"),
                )
            )
            fig_dd.update_layout(
                title="Underwater Equity Curve",
                yaxis_title="Drawdown",
                yaxis_tickformat=".0%",
                height=350,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e8e8e8"),
            )
            st.plotly_chart(fig_dd, use_container_width=True)

            # Weight evolution
            st.subheader("Portfolio Weight Evolution")
            fig_weights = px.area(
                result.weights_history,
                title="Asset Allocation Over Time",
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig_weights.update_layout(
                yaxis_title="Weight",
                yaxis=dict(range=[0, 1]),
                height=400,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e8e8e8"),
            )
            st.plotly_chart(fig_weights, use_container_width=True)

        else:
            st.warning("Insufficient overlapping data for backtest. Adjust date range.")

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 4: TEAR SHEET
    # ═══════════════════════════════════════════════════════════════════════════
    with tab4:
        try:
            if result is not None:
                render_tear_sheet(result)
            else:
                st.info("Run backtest in the Backtest tab first to see tear sheet.")
        except NameError:
            st.info("Run backtest in the Backtest tab first to see tear sheet.")

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 5: STRESS TESTING
    # ═══════════════════════════════════════════════════════════════════════════
    with tab5:
        try:
            allocator = TacticalAllocator(risk_aversion=risk_aversion, regime_allocations=active_regime_allocs)
            render_stress_testing_page(
                regimes, regime_proba, allocator, current_regime, market_returns, detector
            )
        except Exception as e:
            st.error(f"Stress testing error: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 6: MODEL SELECTION
    # ═══════════════════════════════════════════════════════════════════════════
    with tab6:
        try:
            render_model_selection_page(macro_features, market_returns, n_regimes, pca_components)
        except Exception as e:
            st.error(f"Model selection error: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 7: MODEL DIAGNOSTICS
    # ═══════════════════════════════════════════════════════════════════════════
    with tab7:
        render_section_header("Model Diagnostics", "🔬")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("PCA Explained Variance")
            pca_var = detector.pca.explained_variance_ratio_
            fig_pca = px.bar(
                x=[f"PC{i+1}" for i in range(len(pca_var))],
                y=pca_var,
                title="PCA Component Variance Explained",
                labels={"x": "Component", "y": "Variance Ratio"},
            )
            fig_pca.add_trace(
                go.Scatter(
                    x=[f"PC{i+1}" for i in range(len(pca_var))],
                    y=np.cumsum(pca_var),
                    name="Cumulative",
                    yaxis="y2",
                )
            )
            fig_pca.update_layout(
                yaxis2=dict(
                    title="Cumulative",
                    overlaying="y",
                    side="right",
                    range=[0, 1],
                ),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e8e8e8"),
            )
            st.plotly_chart(fig_pca, use_container_width=True)

        with col2:
            st.subheader("Regime Statistics")
            regime_counts = regimes.value_counts()
            fig_counts = px.bar(
                x=regime_counts.index,
                y=regime_counts.values,
                color=regime_counts.index,
                color_discrete_map=REGIME_COLORS,
                title="Months in Each Regime",
                labels={"x": "Regime", "y": "Count (months)"},
            )
            st.plotly_chart(fig_counts, use_container_width=True)

        # Model convergence info
        st.subheader("HMM Fit Information")
        st.json({
            "Converged": detector.hmm.monitor_.converged,
            "Iterations": detector.hmm.monitor_.iter,
            "N Regimes": n_regimes,
            "N PCA Components": pca_components,
            "Total Variance Explained": f"{sum(pca_var):.2%}",
        })

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 8: LEADING INDICATORS
    # ═══════════════════════════════════════════════════════════════════════════
    with tab8:
        render_section_header("Leading Indicator Dashboard", "📡")
        st.markdown(
            "Key economic indicators used for regime detection. "
            "Highlighted values indicate stress/opportunity."
        )

        # Display latest macro data
        if hasattr(detector, 'scaler') and macro_features is not None:
            latest = macro_features.tail(1).T
            latest.columns = ["Latest Z-Score"]
            latest["Signal"] = latest["Latest Z-Score"].apply(
                lambda x: "🟢 Normal" if abs(x) < 1
                else ("🟡 Elevated" if abs(x) < 2 else "🔴 Extreme")
            )
            latest = latest.sort_values("Latest Z-Score", ascending=False)
            styled_dataframe(latest, height=600)

    # ═══════════════════════════════════════════════════════════════════════════
    # PDF EXPORT (Sidebar Button)
    # ═══════════════════════════════════════════════════════════════════════════
    if export_pdf:
        with st.spinner("Generating PDF report..."):
            allocator_pdf = TacticalAllocator(risk_aversion=risk_aversion, regime_allocations=active_regime_allocs)
            confidence_pdf = regime_proba.iloc[-1].max()
            alloc_pdf = allocator_pdf.get_target_allocation(current_regime, confidence_pdf)

            # Get backtest metrics if available
            try:
                bt_metrics = result.metrics if 'result' in dir() else {}
            except NameError:
                bt_metrics = {}

            pdf_gen = PDFReportGenerator()
            output_path = os.path.join(
                os.path.dirname(__file__), "..", "reports", "investment_memo.pdf"
            )
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            pdf_gen.generate_report(
                current_regime=current_regime,
                regime_confidence=confidence_pdf,
                allocation=alloc_pdf,
                backtest_metrics=bt_metrics,
                regime_history=regimes,
                output_path=output_path,
            )

            with open(output_path, "rb") as f:
                st.sidebar.download_button(
                    "⬇️ Download PDF",
                    data=f.read(),
                    file_name="macro_regime_investment_memo.pdf",
                    mime="application/pdf",
                )
            st.sidebar.success("✅ Report generated!")

except FileNotFoundError:
    st.error(
        "⚠️ No cached data found. Please set your FRED API key in "
        "`config/settings.py` and select 'Live (FRED API)' as data source."
    )
    st.info(
        "To get started:\n"
        "1. Get a free API key at https://fred.stlouisfed.org/docs/api/api_key.html\n"
        "2. Set `FRED_API_KEY` in `config/settings.py`\n"
        "3. Select 'Live (FRED API)' in the sidebar"
    )
except Exception as e:
    st.error(f"Error: {e}")
    st.exception(e)
