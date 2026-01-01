import numpy as np
from typing import Callable, Dict, Any

from backend.evaluation.metrics import mse, mae, r2_score, directional_accuracy


def walk_forward_eval(
    X: np.ndarray,
    y: np.ndarray,
    model_factory: Callable[[], Any],
    min_train_size: int = 100,
) -> Dict[str, float]:
    """
    Generic walk-forward evaluation.

    Parameters
    ----------
    X : np.ndarray
        Shape (n_samples, seq_len, n_features)
    y : np.ndarray
        Shape (n_samples,)
    model_factory : Callable
        A function that returns a *new, untrained* model instance.
    min_train_size : int
        Minimum samples before starting walk-forward.

    Returns
    -------
    Dict[str, float]
        Evaluation metrics
    """

    preds = []
    truths = []

    for t in range(min_train_size, len(X)):
        # Train set: [0 ... t-1]
        X_train = X[:t]
        y_train = y[:t]

        # Test point: time t
        X_test = X[t:t + 1]
        y_test = y[t]

        # Fresh model every step (CRITICAL)
        model = model_factory()
        model.fit(X_train, y_train)

        pred = model.predict(X_test)[0]

        preds.append(pred)
        truths.append(y_test)

    preds = np.array(preds)
    truths = np.array(truths)

    return {
        "mse": mse(truths, preds),
        "mae": mae(truths, preds),
        "r2": r2_score(truths, preds),
        "directional_accuracy": directional_accuracy(truths, preds)
    }
