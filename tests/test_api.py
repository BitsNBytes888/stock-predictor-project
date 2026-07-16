import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# /api/tickers
# ---------------------------------------------------------------------------

def test_get_tickers_no_artifacts_dir(monkeypatch, tmp_path):
    monkeypatch.setattr("backend.app.main.ARTIFACTS_DIR", str(tmp_path / "nonexistent"))
    response = client.get("/api/tickers")
    assert response.status_code == 200
    assert response.json() == {"tickers": []}


def test_get_tickers_with_trained_ticker(monkeypatch, tmp_path):
    (tmp_path / "AAPL").mkdir()
    monkeypatch.setattr("backend.app.main.ARTIFACTS_DIR", str(tmp_path))
    response = client.get("/api/tickers")
    assert response.status_code == 200
    assert "AAPL" in response.json()["tickers"]


# ---------------------------------------------------------------------------
# /api/models
# ---------------------------------------------------------------------------

def test_get_models():
    response = client.get("/api/models")
    assert response.status_code == 200
    models = response.json()["models"]
    assert set(models) == {"naive", "linear", "arima", "tree", "lstm"}


# ---------------------------------------------------------------------------
# /api/history/{ticker}
# ---------------------------------------------------------------------------

def test_get_history_valid(monkeypatch, sample_ohlcv):
    monkeypatch.setattr("backend.app.main.fetch_ohlcv_cached", lambda t: sample_ohlcv)
    monkeypatch.setattr("backend.app.main.validate_ohlcv", lambda df: df)
    monkeypatch.setattr("backend.app.main.clean_ohlcv", lambda df: df)
    response = client.get("/api/history/AAPL")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "AAPL"
    assert len(data["history"]) == len(sample_ohlcv)
    first = data["history"][0]
    assert set(first.keys()) == {"date", "open", "high", "low", "close", "volume"}


def test_get_history_invalid_ticker(monkeypatch):
    def raise_value_error(t):
        raise ValueError("no data for ticker")
    monkeypatch.setattr("backend.app.main.fetch_ohlcv_cached", raise_value_error)
    response = client.get("/api/history/FAKE")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# /api/predict/{ticker}
# ---------------------------------------------------------------------------

def test_get_prediction_valid(monkeypatch):
    expected = {
        "ticker": "AAPL",
        "model": "lstm",
        "last_close": 150.0,
        "predicted_log_return": -0.001,
        "predicted_price": 149.85,
    }
    monkeypatch.setattr(
        "backend.app.main.predict_next",
        lambda ticker, model, artifacts_dir: expected,
    )
    response = client.get("/api/predict/AAPL?model=lstm")
    assert response.status_code == 200
    assert response.json()["predicted_price"] == pytest.approx(149.85)
    assert response.json()["ticker"] == "AAPL"


def test_get_prediction_missing_model(monkeypatch):
    def raise_fnf(ticker, model, artifacts_dir):
        raise FileNotFoundError("No saved model")
    monkeypatch.setattr("backend.app.main.predict_next", raise_fnf)
    response = client.get("/api/predict/AAPL?model=lstm")
    assert response.status_code == 404


def test_get_prediction_bad_data(monkeypatch):
    def raise_val(ticker, model, artifacts_dir):
        raise ValueError("Not enough data")
    monkeypatch.setattr("backend.app.main.predict_next", raise_val)
    response = client.get("/api/predict/AAPL?model=lstm")
    assert response.status_code == 422
