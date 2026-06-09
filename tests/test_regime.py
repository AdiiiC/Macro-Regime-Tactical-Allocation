"""
Tests for the HMM regime detection model.
"""

import pytest
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, "..")
from models.regime_hmm import RegimeDetector


class TestRegimeDetector:
    """Tests for RegimeDetector."""

    @pytest.fixture
    def synthetic_features(self):
        """Create synthetic macro features with clear regime structure."""
        np.random.seed(42)
        n_months = 200
        dates = pd.date_range("2005-01-01", periods=n_months, freq="ME")

        # Simulate 4 distinct regimes
        regime_sequence = (
            [0] * 50 + [1] * 30 + [2] * 40 + [3] * 30 + [0] * 50
        )[:n_months]

        # Generate features with regime-dependent means
        means = {
            0: [2.0, -0.5, 1.0, 0.5, -1.0],   # Expansion
            1: [-0.5, 0.5, -0.5, 1.0, 0.5],     # Slowdown
            2: [-2.0, 2.0, -1.5, 2.0, 1.5],     # Recession
            3: [1.0, -1.0, 0.5, -0.5, -0.5],    # Recovery
        }

        data = np.zeros((n_months, 5))
        for i, regime in enumerate(regime_sequence):
            data[i] = np.array(means[regime]) + np.random.randn(5) * 0.3

        columns = [
            "GDP_YoY", "VIX_Level", "CPI_YoY",
            "BAA_Spread_Level", "Yield_Spread_10Y2Y_Level",
        ]
        return pd.DataFrame(data, index=dates, columns=columns)

    def test_fit_predict(self, synthetic_features):
        """Test basic fit/predict workflow."""
        detector = RegimeDetector(n_regimes=4, n_components_pca=3)
        detector.fit(synthetic_features)

        regimes = detector.predict(synthetic_features)

        assert isinstance(regimes, pd.Series)
        assert len(regimes) == len(synthetic_features)
        assert regimes.nunique() <= 4

    def test_predict_proba_sums_to_one(self, synthetic_features):
        """Test regime probabilities sum to 1."""
        detector = RegimeDetector(n_regimes=4, n_components_pca=3)
        detector.fit(synthetic_features)

        proba = detector.predict_proba(synthetic_features)

        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_transition_matrix_rows_sum_to_one(self, synthetic_features):
        """Test transition matrix is valid (rows sum to 1)."""
        detector = RegimeDetector(n_regimes=4, n_components_pca=3)
        detector.fit(synthetic_features)

        trans = detector.get_transition_matrix()

        row_sums = trans.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_stationary_distribution_sums_to_one(self, synthetic_features):
        """Test stationary distribution is valid probability."""
        detector = RegimeDetector(n_regimes=4, n_components_pca=3)
        detector.fit(synthetic_features)

        stat = detector.get_stationary_distribution()

        assert abs(stat.sum() - 1.0) < 1e-6
        assert (stat >= 0).all()

    def test_expected_duration_positive(self, synthetic_features):
        """Test expected durations are positive."""
        detector = RegimeDetector(n_regimes=4, n_components_pca=3)
        detector.fit(synthetic_features)

        durations = detector.get_expected_duration()
        assert (durations > 0).all()

    def test_unfitted_model_raises(self):
        """Test that predict raises on unfitted model."""
        detector = RegimeDetector()
        dummy = pd.DataFrame(np.random.randn(10, 5))

        with pytest.raises(ValueError, match="not fitted"):
            detector.predict(dummy)

    def test_different_n_regimes(self, synthetic_features):
        """Test model works with different regime counts."""
        for n in [2, 3, 4, 5]:
            detector = RegimeDetector(n_regimes=n, n_components_pca=3)
            detector.fit(synthetic_features)
            regimes = detector.predict(synthetic_features)
            assert regimes.nunique() <= n
