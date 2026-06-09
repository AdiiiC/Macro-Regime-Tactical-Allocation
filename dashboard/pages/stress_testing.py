"""
Dashboard Pages - Stress Testing & Monte Carlo
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.stress_testing import MonteCarloStressTest, STRESS_SCENARIOS
from config.settings import REGIME_COLORS


def render_stress_testing_page(
    regimes, regime_proba, allocator, current_regime, market_returns, detector
):
    """Render the stress testing dashboard tab."""
    st.header("🎲 Monte Carlo Stress Testing")

    col1, col2, col3 = st.columns(3)
    with col1:
        n_sims = st.number_input("Simulations", 1000, 50000, 10000, step=1000)
    with col2:
        horizon = st.slider("Horizon (months)", 3, 36, 12)
    with col3:
        conf_level = st.selectbox("Confidence Level", [0.95, 0.99], index=0)

    # Run Monte Carlo
    mc = MonteCarloStressTest(n_simulations=n_sims, horizon_months=horizon)

    # Estimate regime-conditional return parameters
    common_idx = market_returns.index.intersection(regimes.index)
    aligned = pd.concat(
        [market_returns.loc[common_idx], regimes.loc[common_idx].rename("Regime")],
        axis=1,
    )

    regime_params = {}
    for regime in regimes.unique():
        regime_data = aligned[aligned["Regime"] == regime].drop(columns=["Regime"])
        params = {}
        for asset in regime_data.columns:
            params[asset] = (regime_data[asset].mean(), regime_data[asset].std())
        regime_params[regime] = params

    # Get transition matrix
    trans_matrix = detector.get_transition_matrix()

    # Simulate
    sim_results = mc.simulate_regime_paths(current_regime, trans_matrix, regime_params)

    # Get current allocation
    confidence = regime_proba.iloc[-1].max()
    current_alloc = allocator.get_target_allocation(current_regime, confidence)

    # Compute VaR
    var_results = mc.compute_portfolio_var(
        sim_results["returns"], current_alloc, sim_results["assets"]
    )

    # ─── Display Results ──────────────────────────────────────────────
    st.subheader("Portfolio Risk Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("VaR (95%)", f"{var_results['VaR_95%']:.2%}")
    with col2:
        st.metric("CVaR (95%)", f"{var_results['CVaR_95%']:.2%}")
    with col3:
        st.metric("Prob of Loss", f"{var_results['prob_negative']:.1%}")
    with col4:
        st.metric("Expected Return", f"{var_results['mean_return']:.2%}")

    # Distribution plot
    st.subheader(f"Return Distribution ({horizon}-Month Horizon)")
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=var_results["distribution"],
        nbinsx=100,
        name="Simulated Returns",
        marker_color="#3498db",
        opacity=0.7,
    ))
    # VaR lines
    fig_dist.add_vline(x=var_results["VaR_95%"], line_dash="dash",
                       line_color="orange", annotation_text="VaR 95%")
    fig_dist.add_vline(x=var_results["VaR_99%"], line_dash="dash",
                       line_color="red", annotation_text="VaR 99%")
    fig_dist.add_vline(x=0, line_color="black", line_width=1)
    fig_dist.update_layout(
        xaxis_title="Portfolio Return",
        yaxis_title="Frequency",
        xaxis_tickformat=".0%",
        height=400,
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # ─── Deterministic Stress Scenarios ────────────────────────────────
    st.subheader("Deterministic Stress Scenarios")

    scenario_results = mc.run_scenario_analysis(
        sim_results["returns"], current_alloc, sim_results["assets"], STRESS_SCENARIOS
    )

    fig_scenarios = px.bar(
        x=scenario_results.index,
        y=scenario_results["Portfolio Impact"],
        color=scenario_results["Portfolio Impact"].apply(
            lambda x: "Loss" if x < 0 else "Gain"
        ),
        color_discrete_map={"Loss": "#e74c3c", "Gain": "#2ecc71"},
        title="Portfolio Impact Under Stress Scenarios",
    )
    fig_scenarios.update_layout(
        yaxis_tickformat=".0%",
        yaxis_title="Portfolio Impact",
        xaxis_title="Scenario",
        showlegend=False,
        height=400,
    )
    st.plotly_chart(fig_scenarios, use_container_width=True)

    # Details table
    st.dataframe(
        scenario_results["Portfolio Impact"].to_frame().style.format("{:.2%}").background_gradient(
            cmap="RdYlGn", vmin=-0.4, vmax=0.1
        ),
        use_container_width=True,
    )

    return var_results
