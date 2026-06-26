# backend/ml/models/tree.py

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


class TreeBaseline:
    """
    Gradient-boosted tree baseline for sequence data.

    Flattens (samples, seq_len, features) to (samples, seq_len*features),
    same approach as LinearBaseline, but captures non-linear interactions.
    No feature scaling required.
    """

    def __init__(self, **kwargs):
        self.model = HistGradientBoostingRegressor(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_flat = X.reshape(X.shape[0], -1)
        self.model.fit(X_flat, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_flat = X.reshape(X.shape[0], -1)
        return self.model.predict(X_flat)
