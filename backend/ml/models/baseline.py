# backend/ml/models/baseline.py

import numpy as np
from typing import Tuple


class LinearBaseline:
    """
    Simple linear regression baseline for sequence data.
    """

    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the linear model using least squares.

        Parameters
        ----------
        X : np.ndarray
            Shape (num_samples, seq_len, num_features)
        y : np.ndarray
            Shape (num_samples,)
        """

        # Flatten sequences
        X_flat = X.reshape(X.shape[0], -1)

        # Add bias column
        X_aug = np.c_[X_flat, np.ones(X_flat.shape[0])]

        # Closed-form least squares
        params = np.linalg.lstsq(X_aug, y, rcond=None)[0]

        self.weights = params[:-1]
        self.bias = params[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the trained linear model.
        """

        X_flat = X.reshape(X.shape[0], -1)
        return X_flat @ self.weights + self.bias
