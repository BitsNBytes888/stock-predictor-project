import numpy as np
from typing import Callable, Dict, Any

from backend.evaluation.metrics import mse, mae, r2_score, directional_accuracy


def walk_forward_eval(
    X: np.ndarray,
    y: np.ndarray,
    model_factory: Callable[[], Any],
    min_train_size: int = 100,
    retrain_every: int = 1,
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
    retrain_every : int
        Refit a fresh model every this many test steps. Default 1 preserves
        the original per-step refit behavior. Larger values (e.g. 10) make
        expensive models (LSTM, tree ensembles) practical by spreading compute
        over fewer fits while allowing a bigger epoch/iteration budget per fit.

    Returns
    -------
    Dict[str, float]
        Evaluation metrics
    """

    preds = []
    truths = []

    model = None
    steps_since_fit = 0

    for t in range(min_train_size, len(X)):
        if model is None or steps_since_fit >= retrain_every:
            X_train = X[:t]
            y_train = y[:t]
            model = model_factory()
            model.fit(X_train, y_train)
            steps_since_fit = 0

        X_test = X[t:t + 1]
        y_test = y[t]

        pred = model.predict(X_test)[0]

        preds.append(pred)
        truths.append(y_test)
        steps_since_fit += 1

    preds = np.array(preds)
    truths = np.array(truths)

    return {
        "mse": mse(truths, preds),
        "mae": mae(truths, preds),
        "r2": r2_score(truths, preds),
        "directional_accuracy": directional_accuracy(truths, preds)
    }
