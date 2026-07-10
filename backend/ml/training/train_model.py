import os
import sys

from backend.ml.data.build_dataset import build_dataset
from backend.ml.models.naive import NaiveBaseline
from backend.ml.models.baseline import LinearBaseline
from backend.ml.models.arima import ARIMABaseline
from backend.ml.models.tree import TreeBaseline
from backend.ml.models.lstm import LSTMModel


MODEL_REGISTRY = {
    "naive":  lambda dim: NaiveBaseline(),
    "linear": lambda dim: LinearBaseline(),
    "arima":  lambda dim: ARIMABaseline(order=(5, 0, 0)),
    "tree":   lambda dim: TreeBaseline(),
    "lstm":   lambda dim: LSTMModel(input_dim=dim, hidden_dim=64, epochs=100, lr=1e-3),
}


def train_model(
    ticker: str,
    model_type: str,
    seq_len: int = 20,
    artifacts_dir: str = "artifacts",
) -> str:
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_type}'. Choose from: {list(MODEL_REGISTRY)}")

    print(f"Building dataset for {ticker}...")
    X, y = build_dataset(ticker, seq_len=seq_len)
    print(f"  X={X.shape}, y={y.shape}")

    model = MODEL_REGISTRY[model_type](X.shape[2])
    print(f"Training {model_type}...")
    model.fit(X, y)

    ext = "pt" if model_type == "lstm" else "pkl"
    save_path = os.path.join(artifacts_dir, ticker, f"{model_type}.{ext}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Saved → {save_path}")
    return save_path


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m backend.ml.training.train_model <TICKER> <model_type>")
        print(f"  model_type: {list(MODEL_REGISTRY)}")
        sys.exit(1)
    train_model(ticker=sys.argv[1], model_type=sys.argv[2])
