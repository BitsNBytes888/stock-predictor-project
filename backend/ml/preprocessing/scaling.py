# backend/ml/preprocessing/scaling.py

import numpy as np


class StandardScaler3D:
    """
    Standardizes (samples, seq_len, features) arrays feature-wise.

    Mean/std are computed across the samples and seq_len axes, fit only
    on the data passed to `fit`/`fit_transform` (e.g. a walk-forward
    training window) to avoid leaking test-set statistics.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray) -> "StandardScaler3D":
        flat = X.reshape(-1, X.shape[-1])
        self.mean_ = flat.mean(axis=0)
        self.std_ = flat.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
