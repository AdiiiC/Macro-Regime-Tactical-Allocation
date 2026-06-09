"""
Tests for the macro data pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, "..")
from data.fred_pipeline import MacroDataPipeline


class TestMacroDataPipeline:
    """Tests for MacroDataPipeline."""

    def setup_method(self):
        self.pipeline = MacroDataPipeline(api_key="TEST_KEY")

    def test_initialization(self):
        assert self.pipeline.raw_data is None
        assert self.pipeline.transformed_data is None

    @patch("data.fred_pipeline.Fred")
    def test_fetch_creates_dataframe(self, mock_fred_class):
        """Test that fetch returns a DataFrame with proper structure."""
        # Mock FRED responses
        mock_fred = MagicMock()
        dates = pd.date_range("2020-01-01", periods=24, freq="ME")
        mock_fred.get_series.return_value = pd.Series(
            np.random.randn(24), index=dates
        )
        self.pipeline.fred = mock_fred

        result = self.pipeline.fetch_all_indicators(
            start="2020-01-01", end="2021-12-31"
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_compute_transformations_requires_data(self):
        """Test that transformation raises if no data loaded."""
        with pytest.raises(ValueError, match="Must fetch data first"):
            self.pipeline.compute_transformations()

    def test_transformations_output_shape(self):
        """Test transformation produces expected feature types."""
        # Mock raw data
        dates = pd.date_range("2015-01-01", periods=120, freq="ME")
        self.pipeline.raw_data = pd.DataFrame(
            {
                "GDP": np.cumsum(np.random.randn(120)) + 100,
                "CPI": np.cumsum(np.random.randn(120)) + 250,
                "VIX": np.abs(np.random.randn(120)) * 15 + 15,
                "Yield_Spread_10Y2Y": np.random.randn(120) * 0.5 + 1.5,
            },
            index=dates,
        )

        result = self.pipeline.compute_transformations()

        assert isinstance(result, pd.DataFrame)
        # Should have YoY columns for level indicators
        yoy_cols = [c for c in result.columns if "YoY" in c]
        assert len(yoy_cols) > 0

    def test_get_model_ready_data_no_nans(self):
        """Test model-ready data has no NaN values."""
        dates = pd.date_range("2010-01-01", periods=180, freq="ME")
        self.pipeline.raw_data = pd.DataFrame(
            {
                "GDP": np.cumsum(np.random.randn(180)) + 100,
                "CPI": np.cumsum(np.random.randn(180)) + 250,
                "VIX": np.abs(np.random.randn(180)) * 15 + 15,
                "Treasury_10Y": np.random.randn(180) * 0.5 + 3.0,
            },
            index=dates,
        )

        result = self.pipeline.get_model_ready_data()
        assert not result.isna().any().any()
