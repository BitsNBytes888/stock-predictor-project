# backend/ml/models/arima.py

import warnings

import joblib
import numpy as np
from statsmodels.tsa.arima.model import ARIMA


class ARIMABaseline:
    """
    Univariate ARIMA baseline.

    Ignores X entirely - `y` passed in by `walk_forward_eval` is already
    the per-step target (log_return) series aligned with each sequence
    index, which is exactly the series ARIMA needs to fit and forecast.
    """

    def __init__(self, order: tuple[int, int, int] = (5, 0, 0)):
        self.order = order
        self.fitted = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.fitted = ARIMA(y, order=self.order).fit()

    def predict(self, X: np.ndarray) -> np.ndarray:
        steps = X.shape[0]
        return np.asarray(self.fitted.forecast(steps=steps))

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "ARIMABaseline":
        return joblib.load(path)
