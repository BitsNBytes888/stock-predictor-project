# backend/ml/models/naive.py

import numpy as np


class NaiveBaseline:
    """
    Predicts zero for every sample.

    For a `log_return` target, "predict 0" corresponds to a
    random-walk-with-no-drift forecast (predicted price == last price) -
    the standard sanity floor for return prediction.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0])
