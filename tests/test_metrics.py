import numpy as np
import pytest

from backend.evaluation.metrics import mse, mae, r2_score, directional_accuracy


def test_mse_perfect_prediction():
    y = np.array([1.0, -1.0, 0.5])
    assert mse(y, y) == pytest.approx(0.0)


def test_mse_known_error():
    y_true = np.array([1.0, 0.0])
    y_pred = np.array([0.0, 0.0])
    assert mse(y_true, y_pred) == pytest.approx(0.5)


def test_mae_perfect_prediction():
    y = np.array([1.0, -1.0, 0.5])
    assert mae(y, y) == pytest.approx(0.0)


def test_mae_known_error():
    y_true = np.array([1.0, -1.0, 1.0, -1.0])
    y_pred = np.array([1.0, -1.0, -1.0, -1.0])
    assert mae(y_true, y_pred) == pytest.approx(0.5)


def test_r2_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    assert r2_score(y, y) == pytest.approx(1.0)


def test_r2_mean_prediction_gives_zero():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.full_like(y_true, y_true.mean())
    assert r2_score(y_true, y_pred) == pytest.approx(0.0)


def test_r2_zero_variance_target():
    y_true = np.array([5.0, 5.0, 5.0])
    y_pred = np.array([4.0, 5.0, 6.0])
    assert r2_score(y_true, y_pred) == pytest.approx(0.0)


def test_directional_accuracy_all_correct():
    y_true = np.array([1.0, -1.0, 1.0, -1.0])
    y_pred = np.array([0.5, -0.5, 0.1, -0.1])
    assert directional_accuracy(y_true, y_pred) == pytest.approx(1.0)


def test_directional_accuracy_mixed():
    y_true = np.array([1.0, -1.0, 1.0, -1.0])
    y_pred = np.array([1.0, -1.0, -1.0, -1.0])
    assert directional_accuracy(y_true, y_pred) == pytest.approx(0.75)
