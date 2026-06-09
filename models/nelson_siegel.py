"""
Nelson-Siegel Yield Curve Decomposition.
Extracts Level, Slope, and Curvature factors from Treasury yields.
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from typing import Tuple, Optional


class NelsonSiegelModel:
    """
    Nelson-Siegel yield curve model.

    y(τ) = β₀ + β₁ * [(1-e^(-τ/λ)) / (τ/λ)]
              + β₂ * [(1-e^(-τ/λ)) / (τ/λ) - e^(-τ/λ)]

    Where:
        β₀ = Level (long-term rate)
        β₁ = Slope (short vs long spread)
        β₂ = Curvature (belly of the curve)
        λ  = Decay parameter
    """

    # Standard Treasury maturities (in years)
    MATURITIES = {
        "DGS1MO": 1 / 12,
        "DGS3MO": 3 / 12,
        "DGS6MO": 6 / 12,
        "DGS1": 1.0,
        "DGS2": 2.0,
        "DGS3": 3.0,
        "DGS5": 5.0,
        "DGS7": 7.0,
        "DGS10": 10.0,
        "DGS20": 20.0,
        "DGS30": 30.0,
    }

    def __init__(self, lambda_init: float = 1.5):
        self.lambda_param = lambda_init
        self.factors_history: Optional[pd.DataFrame] = None

    @staticmethod
    def _ns_factor_loadings(tau: np.ndarray, lam: float) -> np.ndarray:
        """Compute Nelson-Siegel factor loadings for given maturities."""
        x = tau / lam
        # Avoid division by zero
        x = np.maximum(x, 1e-10)

        f1 = np.ones_like(tau)  # Level
        f2 = (1 - np.exp(-x)) / x  # Slope
        f3 = f2 - np.exp(-x)  # Curvature

        return np.column_stack([f1, f2, f3])

    def fit_single_curve(
        self, yields: np.ndarray, maturities: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """
        Fit Nelson-Siegel to a single yield curve observation.

        Returns:
            (beta0, beta1, beta2, lambda)
        """
        # Remove NaN
        mask = ~np.isnan(yields)
        y = yields[mask]
        tau = maturities[mask]

        if len(y) < 4:
            return np.nan, np.nan, np.nan, np.nan

        def residuals(params):
            beta0, beta1, beta2, lam = params
            lam = max(lam, 0.1)
            loadings = self._ns_factor_loadings(tau, lam)
            betas = np.array([beta0, beta1, beta2])
            fitted = loadings @ betas
            return fitted - y

        # Initial guess
        x0 = [y[-1], y[0] - y[-1], 0.0, self.lambda_param]

        result = least_squares(
            residuals, x0,
            bounds=([-10, -20, -20, 0.1], [20, 20, 20, 10]),
            method="trf",
        )

        if result.success:
            return tuple(result.x)
        else:
            return np.nan, np.nan, np.nan, np.nan

    def decompose_yield_curve_history(
        self, yield_data: pd.DataFrame, maturity_map: Optional[dict] = None
    ) -> pd.DataFrame:
        """
        Decompose a time series of yield curves into Level, Slope, Curvature.

        Args:
            yield_data: DataFrame with columns as FRED series IDs for different maturities
            maturity_map: Mapping from column name to maturity in years

        Returns:
            DataFrame with columns [Level, Slope, Curvature, Lambda]
        """
        if maturity_map is None:
            maturity_map = self.MATURITIES

        # Filter columns that exist in data
        available = [col for col in yield_data.columns if col in maturity_map]
        if not available:
            # Try matching without prefix
            available = yield_data.columns.tolist()
            maturities_arr = np.linspace(0.25, 30, len(available))
        else:
            maturities_arr = np.array([maturity_map[col] for col in available])

        factors = []
        for date, row in yield_data[available].iterrows():
            yields = row.values.astype(float)
            beta0, beta1, beta2, lam = self.fit_single_curve(yields, maturities_arr)
            factors.append({
                "Date": date,
                "Level": beta0,
                "Slope": beta1,
                "Curvature": beta2,
                "Lambda": lam,
            })

        self.factors_history = pd.DataFrame(factors).set_index("Date")
        return self.factors_history

    def get_features_for_hmm(self) -> pd.DataFrame:
        """
        Return yield curve features formatted for HMM input.
        Includes level, slope, curvature, and their changes.
        """
        if self.factors_history is None:
            raise ValueError("Must run decompose_yield_curve_history first.")

        df = self.factors_history[["Level", "Slope", "Curvature"]].copy()

        # Add changes
        df["Level_Chg1"] = df["Level"].diff(1)
        df["Level_Chg3"] = df["Level"].diff(3)
        df["Slope_Chg1"] = df["Slope"].diff(1)
        df["Slope_Chg3"] = df["Slope"].diff(3)
        df["Curvature_Chg1"] = df["Curvature"].diff(1)

        # Inversion signal (slope < 0 indicates inversion)
        df["Curve_Inverted"] = (df["Slope"] < 0).astype(float)

        return df.dropna()

    def fitted_curve(
        self, beta0: float, beta1: float, beta2: float, lam: float,
        maturities: Optional[np.ndarray] = None
    ) -> pd.Series:
        """Generate fitted yield curve for given parameters."""
        if maturities is None:
            maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])

        loadings = self._ns_factor_loadings(maturities, lam)
        betas = np.array([beta0, beta1, beta2])
        fitted = loadings @ betas

        return pd.Series(fitted, index=maturities, name="Yield")


def generate_synthetic_yield_curves(n_months: int = 240) -> pd.DataFrame:
    """Generate synthetic yield curve data for demo."""
    np.random.seed(42)
    dates = pd.date_range("2005-01-01", periods=n_months, freq="ME")

    # Simulate factors with mean reversion
    level = np.zeros(n_months)
    slope = np.zeros(n_months)
    curvature = np.zeros(n_months)

    level[0], slope[0], curvature[0] = 4.0, -1.5, 0.5

    for t in range(1, n_months):
        level[t] = level[t-1] + 0.1 * (3.5 - level[t-1]) + np.random.randn() * 0.15
        slope[t] = slope[t-1] + 0.08 * (-1.0 - slope[t-1]) + np.random.randn() * 0.2
        curvature[t] = curvature[t-1] + 0.12 * (0.5 - curvature[t-1]) + np.random.randn() * 0.15

    # Generate yields from factors
    maturities = np.array([1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])
    columns = ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30"]

    yields_data = np.zeros((n_months, len(maturities)))
    for t in range(n_months):
        lam = 1.5
        loadings = NelsonSiegelModel._ns_factor_loadings(maturities, lam)
        betas = np.array([level[t], slope[t], curvature[t]])
        yields_data[t] = loadings @ betas + np.random.randn(len(maturities)) * 0.05

    return pd.DataFrame(yields_data, index=dates, columns=columns)
