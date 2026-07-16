import numpy as np
import pytest

from backend.ml.models.naive import NaiveBaseline
from backend.ml.models.baseline import LinearBaseline
from backend.ml.models.arima import ARIMABaseline
from backend.ml.models.tree import TreeBaseline
from backend.ml.models.lstm import LSTMModel


@pytest.fixture
def Xy():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 5, 3)).astype(np.float32)
    y = rng.normal(size=(50,)).astype(np.float32)
    return X, y


# ---------------------------------------------------------------------------
# NaiveBaseline
# ---------------------------------------------------------------------------

def test_naive_predict_shape(Xy):
    X, y = Xy
    m = NaiveBaseline()
    m.fit(X, y)
    preds = m.predict(X)
    assert preds.shape == (50,)


def test_naive_always_predicts_zero(Xy):
    X, y = Xy
    m = NaiveBaseline()
    m.fit(X, y)
    assert np.all(m.predict(X) == 0.0)


def test_naive_save_load_roundtrip(Xy, tmp_path):
    X, y = Xy
    m = NaiveBaseline()
    m.fit(X, y)
    path = str(tmp_path / "naive.pkl")
    m.save(path)
    m2 = NaiveBaseline.load(path)
    np.testing.assert_array_equal(m.predict(X), m2.predict(X))


# ---------------------------------------------------------------------------
# LinearBaseline
# ---------------------------------------------------------------------------

def test_linear_predict_shape(Xy):
    X, y = Xy
    m = LinearBaseline()
    m.fit(X, y)
    assert m.predict(X).shape == (50,)


def test_linear_predict_finite(Xy):
    X, y = Xy
    m = LinearBaseline()
    m.fit(X, y)
    assert np.isfinite(m.predict(X)).all()


def test_linear_save_load_roundtrip(Xy, tmp_path):
    X, y = Xy
    m = LinearBaseline()
    m.fit(X, y)
    path = str(tmp_path / "linear.pkl")
    m.save(path)
    m2 = LinearBaseline.load(path)
    np.testing.assert_array_almost_equal(m.predict(X), m2.predict(X))


# ---------------------------------------------------------------------------
# ARIMABaseline
# ---------------------------------------------------------------------------

def test_arima_predict_shape(Xy):
    X, y = Xy
    m = ARIMABaseline(order=(2, 0, 0))
    m.fit(X, y)
    preds = m.predict(X)
    assert preds.shape == (50,)


def test_arima_predict_finite(Xy):
    X, y = Xy
    m = ARIMABaseline(order=(2, 0, 0))
    m.fit(X, y)
    assert np.isfinite(m.predict(X)).all()


def test_arima_ignores_x_content(Xy):
    X, y = Xy
    m = ARIMABaseline(order=(2, 0, 0))
    m.fit(X, y)
    rng = np.random.default_rng(99)
    X_alt = rng.normal(size=X.shape).astype(np.float32)
    np.testing.assert_array_almost_equal(m.predict(X), m.predict(X_alt))


def test_arima_save_load_roundtrip(Xy, tmp_path):
    X, y = Xy
    m = ARIMABaseline(order=(2, 0, 0))
    m.fit(X, y)
    path = str(tmp_path / "arima.pkl")
    m.save(path)
    m2 = ARIMABaseline.load(path)
    np.testing.assert_array_almost_equal(m.predict(X), m2.predict(X))


# ---------------------------------------------------------------------------
# TreeBaseline
# ---------------------------------------------------------------------------

def test_tree_predict_shape(Xy):
    X, y = Xy
    m = TreeBaseline()
    m.fit(X, y)
    assert m.predict(X).shape == (50,)


def test_tree_predict_finite(Xy):
    X, y = Xy
    m = TreeBaseline()
    m.fit(X, y)
    assert np.isfinite(m.predict(X)).all()


def test_tree_save_load_roundtrip(Xy, tmp_path):
    X, y = Xy
    m = TreeBaseline()
    m.fit(X, y)
    path = str(tmp_path / "tree.pkl")
    m.save(path)
    m2 = TreeBaseline.load(path)
    np.testing.assert_array_almost_equal(m.predict(X), m2.predict(X))


# ---------------------------------------------------------------------------
# LSTMModel  (tiny: hidden_dim=8, epochs=3 — fast but functional)
# ---------------------------------------------------------------------------

def test_lstm_predict_shape(Xy):
    X, y = Xy
    m = LSTMModel(input_dim=3, hidden_dim=8, epochs=3)
    m.fit(X, y)
    assert m.predict(X).shape == (50,)


def test_lstm_predict_finite(Xy):
    X, y = Xy
    m = LSTMModel(input_dim=3, hidden_dim=8, epochs=3)
    m.fit(X, y)
    assert np.isfinite(m.predict(X)).all()


def test_lstm_scaler_set_after_fit(Xy):
    X, y = Xy
    m = LSTMModel(input_dim=3, hidden_dim=8, epochs=3)
    m.fit(X, y)
    assert hasattr(m, "scaler") and m.scaler is not None


def test_lstm_save_load_roundtrip(Xy, tmp_path):
    X, y = Xy
    m = LSTMModel(input_dim=3, hidden_dim=8, epochs=3)
    m.fit(X, y)
    path = str(tmp_path / "lstm.pt")
    m.save(path)
    m2 = LSTMModel.load(path)
    np.testing.assert_array_almost_equal(m.predict(X), m2.predict(X), decimal=5)
