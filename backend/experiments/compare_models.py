from backend.ml.data.build_dataset import build_dataset
from backend.ml.models.naive import NaiveBaseline
from backend.ml.models.baseline import LinearBaseline
from backend.ml.models.arima import ARIMABaseline
from backend.ml.models.tree import TreeBaseline
from backend.ml.models.lstm import LSTMModel
from backend.evaluation.walk_forward import walk_forward_eval



def main():
    ticker = "AAPL"
    seq_len = 20

    # Build dataset (now includes technical indicators — ~19 features)
    X, y = build_dataset(
        ticker=ticker,
        seq_len=seq_len,
    )

    print(f"Dataset shapes: X={X.shape}, y={y.shape}")

    # retrain_every=1: refit at every walk-forward step (cheap models)
    # retrain_every=10: refit every 10 steps (LSTM), allows higher epoch budget
    models = {
        "Naive":  (lambda: NaiveBaseline(),                                              1),
        "Linear": (lambda: LinearBaseline(),                                             1),
        "ARIMA":  (lambda: ARIMABaseline(order=(5, 0, 0)),                              1),
        "Tree":   (lambda: TreeBaseline(),                                               1),
        "LSTM":   (lambda: LSTMModel(input_dim=X.shape[2], hidden_dim=64, epochs=50,
                                     lr=1e-3),                                          10),
    }

    for name, (model_factory, retrain_every) in models.items():
        print(f"\nRunning {name} (retrain_every={retrain_every})...")
        metrics = walk_forward_eval(
            X,
            y,
            model_factory=model_factory,
            min_train_size=200,
            retrain_every=retrain_every,
        )

        print(f"{name} metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
