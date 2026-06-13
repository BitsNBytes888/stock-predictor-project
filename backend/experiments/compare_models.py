from backend.ml.data.build_dataset import build_dataset
from backend.ml.models.naive import NaiveBaseline
from backend.ml.models.baseline import LinearBaseline
from backend.ml.models.arima import ARIMABaseline
from backend.ml.models.lstm import LSTMModel
from backend.evaluation.walk_forward import walk_forward_eval



def main():
    ticker = "AAPL"
    seq_len = 20

    # Build dataset
    X, y = build_dataset(
        ticker=ticker,
        seq_len=seq_len,
    )

    print(f"Dataset shapes: X={X.shape}, y={y.shape}")

    models = {
        "Naive": lambda: NaiveBaseline(),
        "Linear": lambda: LinearBaseline(),
        "ARIMA": lambda: ARIMABaseline(order=(5, 0, 0)),
        "LSTM": lambda: LSTMModel(
            input_dim=X.shape[2],
            hidden_dim=64,
            epochs=10,
            lr=1e-3,
        ),
    }

    for name, model_factory in models.items():
        print(f"\nRunning {name}...")
        metrics = walk_forward_eval(
            X,
            y,
            model_factory=model_factory,
            min_train_size=200,
        )

        print(f"{name} metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
