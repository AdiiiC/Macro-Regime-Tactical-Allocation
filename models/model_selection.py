"""
Model Selection: Automatically select optimal number of regimes using BIC/AIC.
Also provides cross-validation for regime stability.
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, List, Tuple, Optional
import warnings


class ModelSelector:
    """
    Selects optimal HMM configuration using information criteria
    and cross-validation.
    """

    def __init__(
        self,
        regime_range: Tuple[int, int] = (2, 7),
        pca_range: Tuple[int, int] = (3, 8),
        n_fits: int = 5,
        random_state: int = 42,
    ):
        """
        Args:
            regime_range: (min, max) number of regimes to test
            pca_range: (min, max) number of PCA components to test
            n_fits: Number of random restarts per configuration
            random_state: Base random seed
        """
        self.regime_range = regime_range
        self.pca_range = pca_range
        self.n_fits = n_fits
        self.random_state = random_state
        self.results: Optional[pd.DataFrame] = None

    def select_optimal_regimes(
        self, features: pd.DataFrame, n_pca: int = 5
    ) -> Dict:
        """
        Find optimal number of regimes using BIC, AIC, and log-likelihood.

        Args:
            features: Transformed macro features
            n_pca: Number of PCA components (fixed)

        Returns:
            Dict with optimal n_regimes and selection criteria
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features.values)

        pca = PCA(n_components=n_pca)
        X_pca = pca.fit_transform(X_scaled)

        results = []
        n_samples = X_pca.shape[0]

        for n_regimes in range(self.regime_range[0], self.regime_range[1] + 1):
            best_ll = -np.inf
            best_model = None

            for restart in range(self.n_fits):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = GaussianHMM(
                            n_components=n_regimes,
                            covariance_type="full",
                            n_iter=200,
                            random_state=self.random_state + restart,
                        )
                        model.fit(X_pca)
                        ll = model.score(X_pca)

                        if ll > best_ll:
                            best_ll = ll
                            best_model = model
                except Exception:
                    continue

            if best_model is None:
                continue

            # Count parameters
            n_params = self._count_params(n_regimes, n_pca, "full")

            # Information criteria
            bic = -2 * best_ll * n_samples + n_params * np.log(n_samples)
            aic = -2 * best_ll * n_samples + 2 * n_params

            # Regime stability (average self-transition probability)
            avg_persistence = np.diag(best_model.transmat_).mean()

            results.append({
                "n_regimes": n_regimes,
                "log_likelihood": best_ll,
                "BIC": bic,
                "AIC": aic,
                "n_parameters": n_params,
                "avg_persistence": avg_persistence,
                "converged": best_model.monitor_.converged,
            })

        self.results = pd.DataFrame(results)

        # Select optimal by BIC (most conservative)
        optimal_bic = self.results.loc[self.results["BIC"].idxmin()]
        optimal_aic = self.results.loc[self.results["AIC"].idxmin()]

        return {
            "optimal_n_regimes_bic": int(optimal_bic["n_regimes"]),
            "optimal_n_regimes_aic": int(optimal_aic["n_regimes"]),
            "bic_values": self.results[["n_regimes", "BIC"]].to_dict("records"),
            "aic_values": self.results[["n_regimes", "AIC"]].to_dict("records"),
            "all_results": self.results,
        }

    def cross_validate_stability(
        self, features: pd.DataFrame, n_regimes: int, n_pca: int = 5, n_splits: int = 5
    ) -> Dict:
        """
        Time-series cross-validation for regime stability.
        Checks if regimes are consistent across different time windows.
        """
        scaler = StandardScaler()
        pca = PCA(n_components=n_pca)

        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(features)):
            train = features.iloc[train_idx]
            test = features.iloc[test_idx]

            X_train_scaled = scaler.fit_transform(train.values)
            X_train_pca = pca.fit_transform(X_train_scaled)

            X_test_scaled = scaler.transform(test.values)
            X_test_pca = pca.transform(X_test_scaled)

            try:
                model = GaussianHMM(
                    n_components=n_regimes,
                    covariance_type="full",
                    n_iter=200,
                    random_state=self.random_state,
                )
                model.fit(X_train_pca)

                train_ll = model.score(X_train_pca)
                test_ll = model.score(X_test_pca)

                train_regimes = model.predict(X_train_pca)
                test_regimes = model.predict(X_test_pca)

                fold_results.append({
                    "fold": fold,
                    "train_ll": train_ll,
                    "test_ll": test_ll,
                    "ll_ratio": test_ll / train_ll if train_ll != 0 else 0,
                    "train_n_regimes_used": len(np.unique(train_regimes)),
                    "test_n_regimes_used": len(np.unique(test_regimes)),
                    "converged": model.monitor_.converged,
                })
            except Exception as e:
                fold_results.append({
                    "fold": fold,
                    "train_ll": np.nan,
                    "test_ll": np.nan,
                    "ll_ratio": np.nan,
                    "train_n_regimes_used": 0,
                    "test_n_regimes_used": 0,
                    "converged": False,
                })

        cv_results = pd.DataFrame(fold_results)

        return {
            "mean_test_ll": cv_results["test_ll"].mean(),
            "std_test_ll": cv_results["test_ll"].std(),
            "mean_ll_ratio": cv_results["ll_ratio"].mean(),
            "all_folds_converged": cv_results["converged"].all(),
            "regime_usage_consistency": cv_results["test_n_regimes_used"].std(),
            "fold_details": cv_results,
        }

    def full_grid_search(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Grid search over both n_regimes and n_pca_components.

        Returns DataFrame with BIC/AIC for all combinations.
        """
        results = []

        for n_pca in range(self.pca_range[0], self.pca_range[1] + 1):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(features.values)

            if n_pca > X_scaled.shape[1]:
                continue

            pca = PCA(n_components=n_pca)
            X_pca = pca.fit_transform(X_scaled)
            n_samples = X_pca.shape[0]

            for n_regimes in range(self.regime_range[0], self.regime_range[1] + 1):
                best_ll = -np.inf

                for restart in range(self.n_fits):
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model = GaussianHMM(
                                n_components=n_regimes,
                                covariance_type="full",
                                n_iter=200,
                                random_state=self.random_state + restart,
                            )
                            model.fit(X_pca)
                            ll = model.score(X_pca)
                            if ll > best_ll:
                                best_ll = ll
                    except Exception:
                        continue

                if best_ll > -np.inf:
                    n_params = self._count_params(n_regimes, n_pca, "full")
                    bic = -2 * best_ll * n_samples + n_params * np.log(n_samples)
                    aic = -2 * best_ll * n_samples + 2 * n_params

                    results.append({
                        "n_regimes": n_regimes,
                        "n_pca": n_pca,
                        "log_likelihood": best_ll,
                        "BIC": bic,
                        "AIC": aic,
                        "n_parameters": n_params,
                    })

        grid_results = pd.DataFrame(results)
        return grid_results

    @staticmethod
    def _count_params(n_regimes: int, n_features: int, cov_type: str) -> int:
        """Count free parameters in Gaussian HMM."""
        # Initial state: n_regimes - 1
        # Transition matrix: n_regimes * (n_regimes - 1)
        # Means: n_regimes * n_features
        # Covariances
        if cov_type == "full":
            cov_params = n_regimes * n_features * (n_features + 1) // 2
        elif cov_type == "diag":
            cov_params = n_regimes * n_features
        else:
            cov_params = n_regimes * n_features

        return (
            (n_regimes - 1)
            + n_regimes * (n_regimes - 1)
            + n_regimes * n_features
            + cov_params
        )
