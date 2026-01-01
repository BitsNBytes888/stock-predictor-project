from backend.ml.data.build_dataset import build_dataset
from backend.ml.models.baseline import LinearBaseline
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

    # ------------------
    # Linear Baseline
    # ------------------
    print("\nRunning Linear Baseline...")
    baseline_metrics = walk_forward_eval(
        X,
        y,
        model_factory=lambda: LinearBaseline(),
        min_train_size=200,
    )

    print("Baseline metrics:")
    for k, v in baseline_metrics.items():
        print(f"{k}: {v:.4f}")

    # ------------------
    # LSTM
    # ------------------
    print("\nRunning LSTM...")
    lstm_metrics = walk_forward_eval(
        X,
        y,
        model_factory=lambda: LSTMModel(
            input_dim=X.shape[2],
            hidden_dim=64,
            epochs=10,
            lr=1e-3,
        ),
        min_train_size=200,
    )

    print("LSTM metrics:")
    for k, v in lstm_metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()