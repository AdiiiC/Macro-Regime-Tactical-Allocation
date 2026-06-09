"""
Alternative regime detection models for comparison:
1. LSTM-based regime classifier
2. Markov-Switching VAR
3. K-Means clustering baseline
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LSTM Regime Classifier
# ═══════════════════════════════════════════════════════════════════════════════

class LSTMRegimeDetector:
    """
    LSTM-based regime detector.
    Uses sequence of macro features to classify current regime.
    Trained on HMM-generated labels (student-teacher framework).
    """

    def __init__(
        self,
        lookback: int = 12,
        n_regimes: int = 4,
        hidden_size: int = 64,
        n_layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ):
        self.lookback = lookback
        self.n_regimes = n_regimes
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = StandardScaler()
        self._fitted = False

    def _build_model(self, n_features: int):
        """Build LSTM model using PyTorch."""
        import torch
        import torch.nn as nn

        class RegimeLSTM(nn.Module):
            def __init__(self, input_size, hidden_size, n_layers, n_classes, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=n_layers,
                    batch_first=True,
                    dropout=dropout if n_layers > 1 else 0,
                )
                self.attention = nn.Linear(hidden_size, 1)
                self.fc = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size // 2, n_classes),
                )

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                # Attention mechanism
                attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
                context = torch.sum(attn_weights * lstm_out, dim=1)
                return self.fc(context)

        self.model = RegimeLSTM(
            n_features, self.hidden_size, self.n_layers,
            self.n_regimes, self.dropout
        )

    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        """Create lookback sequences for LSTM."""
        sequences = []
        labels = []
        for i in range(self.lookback, len(X)):
            sequences.append(X[i - self.lookback: i])
            labels.append(y[i])
        return np.array(sequences), np.array(labels)

    def fit(self, features: pd.DataFrame, regime_labels: pd.Series) -> "LSTMRegimeDetector":
        """
        Train LSTM on HMM-generated regime labels.

        Args:
            features: Macro features
            regime_labels: Regime labels from HMM (used as training targets)
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        # Encode labels
        unique_labels = sorted(regime_labels.unique())
        label_map = {label: i for i, label in enumerate(unique_labels)}
        self._label_map = label_map
        self._inv_label_map = {v: k for k, v in label_map.items()}

        y_encoded = regime_labels.map(label_map).values

        # Scale features
        X_scaled = self.scaler.fit_transform(features.values)

        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_encoded)

        if len(X_seq) == 0:
            raise ValueError("Not enough data for lookback window.")

        # Build model
        self._build_model(X_seq.shape[2])

        # Training
        X_tensor = torch.FloatTensor(X_seq)
        y_tensor = torch.LongTensor(y_seq)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        self._fitted = True
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predict regimes using trained LSTM."""
        import torch

        if not self._fitted:
            raise ValueError("Model not fitted.")

        X_scaled = self.scaler.transform(features.values)
        X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X_scaled)))

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq)
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1).numpy()

        labels = [self._inv_label_map[p] for p in predictions]

        # Pad beginning with first prediction
        full_labels = [labels[0]] * self.lookback + labels
        return pd.Series(full_labels, index=features.index, name="Regime")

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """Get regime probabilities from LSTM."""
        import torch

        if not self._fitted:
            raise ValueError("Model not fitted.")

        X_scaled = self.scaler.transform(features.values)
        X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X_scaled)))

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq)
            outputs = self.model(X_tensor)
            proba = torch.softmax(outputs, dim=1).numpy()

        # Pad beginning
        padded = np.vstack([proba[0:1]] * self.lookback + [proba])

        columns = [self._inv_label_map[i] for i in range(self.n_regimes)]
        return pd.DataFrame(padded, index=features.index, columns=columns)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Markov-Switching Model (simplified)
# ═══════════════════════════════════════════════════════════════════════════════

class MarkovSwitchingModel:
    """
    Markov-Switching regression model.
    Uses statsmodels MarkovRegression for regime-switching dynamics.
    """

    def __init__(self, n_regimes: int = 4, switching_variance: bool = True):
        self.n_regimes = n_regimes
        self.switching_variance = switching_variance
        self.model = None
        self.results = None
        self._fitted = False

    def fit(self, target_series: pd.Series, exog: Optional[pd.DataFrame] = None) -> "MarkovSwitchingModel":
        """
        Fit Markov-switching model to a target economic series.

        Args:
            target_series: Economic indicator to model (e.g., GDP growth)
            exog: Optional exogenous variables
        """
        import statsmodels.api as sm

        try:
            model = sm.tsa.MarkovRegression(
                target_series.dropna(),
                k_regimes=self.n_regimes,
                trend="c",
                switching_variance=self.switching_variance,
                exog=exog,
            )
            self.results = model.fit(maxiter=500, disp=False)
            self.model = model
            self._fitted = True
        except Exception as e:
            warnings.warn(f"Markov-Switching fit failed: {e}")

        return self

    def predict(self, target_series: pd.Series) -> pd.Series:
        """Get smoothed regime probabilities and classify."""
        if not self._fitted or self.results is None:
            raise ValueError("Model not fitted.")

        smoothed = self.results.smoothed_marginal_probabilities
        regimes = smoothed.values.argmax(axis=1)

        return pd.Series(
            regimes, index=target_series.dropna().index, name="Regime"
        )

    def get_regime_params(self) -> pd.DataFrame:
        """Get estimated parameters per regime."""
        if not self._fitted or self.results is None:
            raise ValueError("Model not fitted.")

        params = {}
        for i in range(self.n_regimes):
            params[f"Regime_{i}"] = {
                "mean": self.results.params.get(f"const[{i}]", np.nan),
                "variance": self.results.params.get(f"sigma2[{i}]", np.nan),
            }
        return pd.DataFrame(params).T


# ═══════════════════════════════════════════════════════════════════════════════
# 3. K-Means Clustering Baseline
# ═══════════════════════════════════════════════════════════════════════════════

class KMeansRegimeDetector:
    """
    Simple K-Means clustering baseline for regime detection.
    No temporal structure — pure feature-space clustering.
    """

    def __init__(self, n_regimes: int = 4, random_state: int = 42):
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.kmeans = None
        self.scaler = StandardScaler()
        self._fitted = False
        self._label_map = None

    def fit(self, features: pd.DataFrame) -> "KMeansRegimeDetector":
        """Fit K-Means clustering."""
        X_scaled = self.scaler.fit_transform(features.values)

        self.kmeans = KMeans(
            n_clusters=self.n_regimes,
            random_state=self.random_state,
            n_init=20,
        )
        self.kmeans.fit(X_scaled)
        self._fitted = True

        # Map clusters to regime names based on centroid characteristics
        self._establish_mapping(features)
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predict regime clusters."""
        if not self._fitted:
            raise ValueError("Model not fitted.")

        X_scaled = self.scaler.transform(features.values)
        clusters = self.kmeans.predict(X_scaled)

        if self._label_map:
            labels = [self._label_map.get(c, f"Cluster_{c}") for c in clusters]
        else:
            labels = [f"Cluster_{c}" for c in clusters]

        return pd.Series(labels, index=features.index, name="Regime")

    def get_silhouette_score(self, features: pd.DataFrame) -> float:
        """Compute silhouette score for cluster quality."""
        X_scaled = self.scaler.transform(features.values)
        labels = self.kmeans.predict(X_scaled)
        return silhouette_score(X_scaled, labels)

    def _establish_mapping(self, features: pd.DataFrame):
        """Map cluster IDs to regime names based on characteristics."""
        X_scaled = self.scaler.transform(features.values)
        clusters = self.kmeans.predict(X_scaled)

        # Score clusters on growth/stress
        growth_cols = [c for c in features.columns if "YoY" in c]
        stress_cols = [c for c in features.columns if "VIX" in c or "Spread" in c]

        cluster_scores = {}
        for c in range(self.n_regimes):
            mask = clusters == c
            if growth_cols:
                growth = features.loc[mask, [col for col in growth_cols if col in features.columns]].mean().mean()
            else:
                growth = 0
            if stress_cols:
                stress = features.loc[mask, [col for col in stress_cols if col in features.columns]].mean().mean()
            else:
                stress = 0
            cluster_scores[c] = growth - stress

        # Rank and assign names
        ranked = sorted(cluster_scores.items(), key=lambda x: x[1], reverse=True)
        names = ["Expansion", "Recovery", "Slowdown", "Recession"]
        self._label_map = {}
        for i, (cluster_id, _) in enumerate(ranked):
            if i < len(names):
                self._label_map[cluster_id] = names[i]


# ═══════════════════════════════════════════════════════════════════════════════
# Model Comparison Framework
# ═══════════════════════════════════════════════════════════════════════════════

class ModelComparison:
    """Compare multiple regime detection models."""

    def __init__(self):
        self.results = {}

    def compare_models(
        self,
        features: pd.DataFrame,
        asset_returns: pd.DataFrame,
        regime_allocations: Dict,
        models: Dict[str, object],
    ) -> pd.DataFrame:
        """
        Compare regime models on out-of-sample portfolio performance.

        Args:
            models: {name: fitted_model} dict with predict() method
        """
        from backtesting.engine import BacktestEngine

        engine = BacktestEngine()
        comparison = []

        for name, model in models.items():
            try:
                regimes = model.predict(features)
                common_idx = asset_returns.index.intersection(regimes.index)

                if len(common_idx) < 24:
                    continue

                result = engine.run(
                    asset_returns.loc[common_idx],
                    regimes.loc[common_idx],
                    regime_allocations,
                )

                comparison.append({
                    "Model": name,
                    **result.metrics,
                })
            except Exception as e:
                comparison.append({
                    "Model": name,
                    "Error": str(e),
                })

        return pd.DataFrame(comparison)
