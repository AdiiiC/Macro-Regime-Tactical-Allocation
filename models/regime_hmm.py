"""
Hidden Markov Model for macroeconomic regime detection.
Identifies latent economic states from transformed macro indicators.
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Dict

import sys
sys.path.insert(0, "..")
from config.settings import (
    HMM_N_REGIMES,
    HMM_COVARIANCE_TYPE,
    HMM_N_ITER,
    HMM_RANDOM_STATE,
    REGIME_NAMES,
)


class RegimeDetector:
    """
    Gaussian HMM-based macro regime detector.

    Workflow:
        1. Reduce dimensionality of macro features via PCA
        2. Fit Gaussian HMM to capture latent regime dynamics
        3. Decode most likely regime sequence (Viterbi)
        4. Label regimes based on economic interpretation
    """

    def __init__(
        self,
        n_regimes: int = HMM_N_REGIMES,
        n_components_pca: int = 5,
        covariance_type: str = HMM_COVARIANCE_TYPE,
        n_iter: int = HMM_N_ITER,
        random_state: int = HMM_RANDOM_STATE,
    ):
        self.n_regimes = n_regimes
        self.n_components_pca = n_components_pca
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components_pca)
        self.hmm: Optional[GaussianHMM] = None

        self._regime_mapping: Optional[Dict[int, str]] = None
        self._fitted = False

    def fit(self, features: pd.DataFrame) -> "RegimeDetector":
        """
        Fit the regime detection model.

        Args:
            features: DataFrame of transformed macro features (from MacroDataPipeline)
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(features.values)

        # PCA dimensionality reduction
        X_pca = self.pca.fit_transform(X_scaled)

        # Fit Gaussian HMM
        self.hmm = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            tol=0.01,
            random_state=self.random_state,
        )
        self.hmm.fit(X_pca)

        # Decode regimes for training data to establish mapping
        raw_regimes = self.hmm.predict(X_pca)
        self._establish_regime_mapping(features, raw_regimes)

        self._fitted = True
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """
        Predict regime for given macro features.

        Returns:
            Series with regime labels indexed by date.
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(features.values)
        X_pca = self.pca.transform(X_scaled)

        raw_regimes = self.hmm.predict(X_pca)
        labeled_regimes = pd.Series(
            [self._regime_mapping.get(r, f"Regime_{r}") for r in raw_regimes],
            index=features.index,
            name="Regime",
        )
        return labeled_regimes

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Get regime probabilities for each time step.

        Returns:
            DataFrame with columns for each regime's probability.
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(features.values)
        X_pca = self.pca.transform(X_scaled)

        proba = self.hmm.predict_proba(X_pca)
        columns = [
            self._regime_mapping.get(i, f"Regime_{i}")
            for i in range(self.n_regimes)
        ]
        return pd.DataFrame(proba, index=features.index, columns=columns)

    def get_transition_matrix(self) -> pd.DataFrame:
        """Return the regime transition probability matrix."""
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        labels = [
            self._regime_mapping.get(i, f"Regime_{i}")
            for i in range(self.n_regimes)
        ]
        return pd.DataFrame(
            self.hmm.transmat_, index=labels, columns=labels
        )

    def get_stationary_distribution(self) -> pd.Series:
        """Compute the stationary distribution of regimes."""
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Stationary distribution from eigenvector of transition matrix
        transmat = self.hmm.transmat_
        eigenvalues, eigenvectors = np.linalg.eig(transmat.T)
        # Find eigenvector corresponding to eigenvalue ≈ 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()

        labels = [
            self._regime_mapping.get(i, f"Regime_{i}")
            for i in range(self.n_regimes)
        ]
        return pd.Series(stationary, index=labels, name="Stationary_Prob")

    def get_expected_duration(self) -> pd.Series:
        """Expected duration (months) in each regime."""
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        durations = {}
        for i in range(self.n_regimes):
            # Expected duration = 1 / (1 - P(stay in same state))
            p_stay = self.hmm.transmat_[i, i]
            expected = 1.0 / (1.0 - p_stay) if p_stay < 1.0 else np.inf
            label = self._regime_mapping.get(i, f"Regime_{i}")
            durations[label] = expected

        return pd.Series(durations, name="Expected_Duration_Months")

    def _establish_regime_mapping(
        self, features: pd.DataFrame, raw_regimes: np.ndarray
    ) -> None:
        """
        Map raw HMM states to economic regime labels based on characteristics.

        Logic:
        - Compute mean of key indicators per regime
        - Expansion = high growth + low stress
        - Recession = low growth + high stress
        - Slowdown = declining growth
        - Recovery = improving from low base
        """
        regime_features = pd.DataFrame(
            self.scaler.transform(features.values),
            index=features.index,
            columns=features.columns,
        )
        regime_features["raw_regime"] = raw_regimes

        # Compute mean feature values per regime
        regime_means = regime_features.groupby("raw_regime").mean()

        # Score each regime on growth and stress dimensions
        growth_cols = [c for c in features.columns if "YoY" in c or "Mom3" in c]
        stress_cols = [c for c in features.columns if "VIX" in c or "Spread" in c or "Stress" in c]

        scores = pd.DataFrame(index=range(self.n_regimes))

        if growth_cols:
            available_growth = [c for c in growth_cols if c in regime_means.columns]
            if available_growth:
                scores["growth"] = regime_means[available_growth].mean(axis=1).values

        if stress_cols:
            available_stress = [c for c in stress_cols if c in regime_means.columns]
            if available_stress:
                scores["stress"] = regime_means[available_stress].mean(axis=1).values

        # Assign labels based on growth/stress ranking
        if "growth" in scores.columns and "stress" in scores.columns:
            scores["composite"] = scores["growth"] - scores["stress"]
            ranked = scores["composite"].sort_values(ascending=False)
            mapping = {}
            regime_labels = ["Expansion", "Recovery", "Slowdown", "Recession"]
            for i, regime_idx in enumerate(ranked.index):
                if i < len(regime_labels):
                    mapping[regime_idx] = regime_labels[i]
                else:
                    mapping[regime_idx] = f"Regime_{regime_idx}"
        else:
            # Fallback: use default naming
            mapping = REGIME_NAMES.copy()

        self._regime_mapping = mapping

    def get_model_diagnostics(self) -> Dict:
        """Return model diagnostics for evaluation."""
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        return {
            "n_regimes": self.n_regimes,
            "log_likelihood": self.hmm.score(
                self.pca.transform(
                    self.scaler.transform(
                        np.zeros((1, self.scaler.n_features_in_))
                    )
                )
            ),
            "n_pca_components": self.n_components_pca,
            "pca_explained_variance": self.pca.explained_variance_ratio_.tolist(),
            "convergence": self.hmm.monitor_.converged,
            "n_iterations": self.hmm.monitor_.iter,
            "aic": -2 * self.hmm.score(
                self.pca.transform(
                    self.scaler.transform(
                        np.zeros((1, self.scaler.n_features_in_))
                    )
                )
            ) + 2 * self._count_parameters(),
        }

    def _count_parameters(self) -> int:
        """Count number of free parameters in the HMM."""
        n = self.n_regimes
        k = self.n_components_pca
        # Transition matrix: n*(n-1)
        # Means: n*k
        # Covariances: depends on type
        if self.covariance_type == "full":
            cov_params = n * k * (k + 1) // 2
        elif self.covariance_type == "diag":
            cov_params = n * k
        else:
            cov_params = n * k
        return n * (n - 1) + n * k + cov_params
