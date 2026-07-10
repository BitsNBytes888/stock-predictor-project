import os
import sys

import numpy as np

from backend.ml.data.fetch_yfinance import fetch_ohlcv
from backend.ml.data.validate import validate_ohlcv
from backend.ml.data.clean import clean_ohlcv
from backend.ml.features.basic import engineer_basic_features
from backend.ml.features.technical import add_technical_features
from backend.ml.models.naive import NaiveBaseline
from backend.ml.models.baseline import LinearBaseline
from backend.ml.models.arima import ARIMABaseline
from backend.ml.models.tree import TreeBaseline
from backend.ml.models.lstm import LSTMModel


MODEL_CLASSES = {
    "naive":  NaiveBaseline,
    "linear": LinearBaseline,
    "arima":  ARIMABaseline,
    "tree":   TreeBaseline,
    "lstm":   LSTMModel,
}


def predict_next(
    ticker: str,
    model_type: str,
    seq_len: int = 20,
    artifacts_dir: str = "artifacts",
) -> dict:
    if model_type not in MODEL_CLASSES:
        raise ValueError(f"Unknown model '{model_type}'. Choose from: {list(MODEL_CLASSES)}")

    ext = "pt" if model_type == "lstm" else "pkl"
    path = os.path.join(artifacts_dir, ticker, f"{model_type}.{ext}")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved model at '{path}'. Run train_model first."
        )

    model = MODEL_CLASSES[model_type].load(path)

    df = fetch_ohlcv(ticker)
    df = validate_ohlcv(df)
    df = clean_ohlcv(df)
    feature_df = add_technical_features(engineer_basic_features(df))

    if len(feature_df) < seq_len:
        raise ValueError(
            f"Not enough data to form a sequence: need {seq_len} rows, got {len(feature_df)}"
        )

    last_close = float(feature_df["Close"].iloc[-1])
    X_infer = feature_df.to_numpy()[-seq_len:][np.newaxis, :, :]  # (1, seq_len, n_features)

    pred_log_return = float(model.predict(X_infer)[0])
    pred_price = float(last_close * np.exp(pred_log_return))

    return {
        "ticker": ticker,
        "model": model_type,
        "last_close": last_close,
        "predicted_log_return": pred_log_return,
        "predicted_price": pred_price,
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m backend.ml.inference.predict <TICKER> <model_type>")
        print(f"  model_type: {list(MODEL_CLASSES)}")
        sys.exit(1)
    result = predict_next(ticker=sys.argv[1], model_type=sys.argv[2])
    for k, v in result.items():
        print(f"  {k}: {v}")
