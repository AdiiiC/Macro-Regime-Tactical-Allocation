"""
Dashboard Pages - Model Comparison & Selection
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_selection import ModelSelector
from models.alternative_models import KMeansRegimeDetector
from config.settings import REGIME_COLORS


def render_model_selection_page(macro_features, market_returns, n_regimes, pca_components):
    """Render model selection and comparison page."""
    st.header("🔬 Model Selection & Comparison")

    tab_select, tab_compare, tab_cv = st.tabs([
        "BIC/AIC Selection", "Model Comparison", "Cross-Validation"
    ])

    # ─── BIC/AIC Selection ────────────────────────────────────────────
    with tab_select:
        st.subheader("Optimal Number of Regimes")
        st.markdown(
            "Using Bayesian Information Criterion (BIC) and Akaike Information "
            "Criterion (AIC) to select the optimal model complexity."
        )

        if st.button("Run Model Selection", key="run_selection"):
            with st.spinner("Fitting models for 2-7 regimes..."):
                selector = ModelSelector(regime_range=(2, 7), n_fits=3)
                results = selector.select_optimal_regimes(macro_features, n_pca=pca_components)

            st.success(
                f"Optimal regimes: **{results['optimal_n_regimes_bic']}** (BIC), "
                f"**{results['optimal_n_regimes_aic']}** (AIC)"
            )

            # Plot BIC/AIC
            all_results = results["all_results"]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=all_results["n_regimes"], y=all_results["BIC"],
                name="BIC", mode="lines+markers",
                line=dict(color="#e74c3c", width=2),
            ))
            fig.add_trace(go.Scatter(
                x=all_results["n_regimes"], y=all_results["AIC"],
                name="AIC", mode="lines+markers",
                line=dict(color="#3498db", width=2),
                yaxis="y2",
            ))
            fig.update_layout(
                title="Information Criteria vs. Number of Regimes",
                xaxis_title="Number of Regimes",
                yaxis_title="BIC",
                yaxis2=dict(title="AIC", overlaying="y", side="right"),
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Persistence plot
            fig_persist = px.bar(
                all_results, x="n_regimes", y="avg_persistence",
                title="Average Regime Persistence (Self-Transition Probability)",
                labels={"avg_persistence": "Avg P(stay)", "n_regimes": "N Regimes"},
            )
            st.plotly_chart(fig_persist, use_container_width=True)

            # Results table
            st.dataframe(
                all_results.style.format({
                    "log_likelihood": "{:.2f}",
                    "BIC": "{:.0f}",
                    "AIC": "{:.0f}",
                    "avg_persistence": "{:.3f}",
                }),
                use_container_width=True,
            )

    # ─── Model Comparison ─────────────────────────────────────────────
    with tab_compare:
        st.subheader("HMM vs. K-Means Baseline")
        st.markdown(
            "Comparing the Gaussian HMM against a simpler K-Means clustering baseline. "
            "The HMM captures temporal dynamics; K-Means does not."
        )

        if st.button("Run Comparison", key="run_compare"):
            with st.spinner("Fitting comparison models..."):
                # K-Means
                kmeans = KMeansRegimeDetector(n_regimes=n_regimes)
                kmeans.fit(macro_features)
                kmeans_regimes = kmeans.predict(macro_features)

                # HMM (already fitted externally, but let's refit for comparison)
                from models.regime_hmm import RegimeDetector
                hmm = RegimeDetector(n_regimes=n_regimes, n_components_pca=pca_components)
                hmm.fit(macro_features)
                hmm_regimes = hmm.predict(macro_features)

            # Visual comparison
            fig_comp = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                     subplot_titles=("HMM Regimes", "K-Means Regimes"))

            for regime, color in REGIME_COLORS.items():
                hmm_mask = hmm_regimes == regime
                if hmm_mask.any():
                    fig_comp.add_trace(
                        go.Scatter(
                            x=hmm_regimes[hmm_mask].index,
                            y=[1] * hmm_mask.sum(),
                            mode="markers",
                            marker=dict(color=color, size=8),
                            name=f"HMM-{regime}",
                        ),
                        row=1, col=1,
                    )
                km_mask = kmeans_regimes == regime
                if km_mask.any():
                    fig_comp.add_trace(
                        go.Scatter(
                            x=kmeans_regimes[km_mask].index,
                            y=[1] * km_mask.sum(),
                            mode="markers",
                            marker=dict(color=color, size=8),
                            name=f"KM-{regime}",
                            showlegend=False,
                        ),
                        row=2, col=1,
                    )

            fig_comp.update_layout(height=400)
            st.plotly_chart(fig_comp, use_container_width=True)

            # Agreement rate
            common = hmm_regimes.index.intersection(kmeans_regimes.index)
            agreement = (hmm_regimes.loc[common] == kmeans_regimes.loc[common]).mean()
            st.metric("Regime Agreement Rate", f"{agreement:.1%}")

            # Silhouette score for K-Means
            sil_score = kmeans.get_silhouette_score(macro_features)
            st.metric("K-Means Silhouette Score", f"{sil_score:.3f}")

    # ─── Cross-Validation ─────────────────────────────────────────────
    with tab_cv:
        st.subheader("Time-Series Cross-Validation")
        st.markdown(
            "Walk-forward cross-validation to assess regime stability "
            "and out-of-sample generalization."
        )

        n_splits = st.slider("Number of CV Folds", 3, 10, 5)

        if st.button("Run Cross-Validation", key="run_cv"):
            with st.spinner("Running walk-forward CV..."):
                selector = ModelSelector()
                cv_results = selector.cross_validate_stability(
                    macro_features, n_regimes=n_regimes,
                    n_pca=pca_components, n_splits=n_splits
                )

            st.metric("Mean Test Log-Likelihood", f"{cv_results['mean_test_ll']:.3f}")
            st.metric("LL Ratio (test/train)", f"{cv_results['mean_ll_ratio']:.3f}")
            st.metric("All Folds Converged", str(cv_results['all_folds_converged']))

            # Fold details
            st.dataframe(cv_results["fold_details"], use_container_width=True)


# Need this import for subplot
from plotly.subplots import make_subplots
