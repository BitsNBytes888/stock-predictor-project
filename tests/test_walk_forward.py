import numpy as np
import pytest

from backend.evaluation.walk_forward import walk_forward_eval
from backend.ml.models.naive import NaiveBaseline


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(11)
    n, seq_len, n_feat = 30, 3, 2
    X = rng.normal(size=(n, seq_len, n_feat)).astype(np.float32)
    y = rng.normal(size=(n,)).astype(np.float32)
    return X, y


def test_returns_expected_keys(synthetic_data):
    X, y = synthetic_data
    result = walk_forward_eval(X, y, NaiveBaseline, min_train_size=10)
    assert set(result.keys()) == {"mse", "mae", "r2", "directional_accuracy"}


def test_retrain_every_1_calls_factory_each_step(synthetic_data):
    X, y = synthetic_data
    call_count = [0]

    def counting_factory():
        call_count[0] += 1
        return NaiveBaseline()

    min_train = 10
    walk_forward_eval(X, y, counting_factory, min_train_size=min_train, retrain_every=1)
    expected_steps = len(X) - min_train
    assert call_count[0] == expected_steps


def test_retrain_every_5_limits_factory_calls(synthetic_data):
    X, y = synthetic_data
    call_count = [0]

    def counting_factory():
        call_count[0] += 1
        return NaiveBaseline()

    walk_forward_eval(X, y, counting_factory, min_train_size=10, retrain_every=5)
    assert call_count[0] < len(X) - 10


def test_naive_mse_equals_mean_squared_y(synthetic_data):
    X, y = synthetic_data
    min_train = 10
    result = walk_forward_eval(X, y, NaiveBaseline, min_train_size=min_train, retrain_every=1)
    test_y = y[min_train:]
    expected_mse = float(np.mean(test_y ** 2))
    assert result["mse"] == pytest.approx(expected_mse, rel=1e-5)
