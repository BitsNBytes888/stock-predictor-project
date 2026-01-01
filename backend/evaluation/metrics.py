# backend/evaluation/metrics.py

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error (MSE)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean((y_true - y_pred) ** 2)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error (MAE)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of determination (R^2)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - ss_res / ss_tot


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Fraction of predictions with correct direction (sign)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean(np.sign(y_true) == np.sign(y_pred))
