# backend/evaluation/metrics_test.py

import numpy as np

from backend.evaluation.metrics import (
    mean_squared_error,
    mean_absolute_error,
    directional_accuracy,
)


def run_sanity_checks():
    y_true = np.array([1.0, -1.0, 1.0, -1.0])
    y_pred = np.array([1.0, -1.0, -1.0, -1.0])

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    da = directional_accuracy(y_true, y_pred)

    print("Sanity check results:")
    print(f"MSE: {mse:.4f} (expected 1.0)")
    print(f"MAE: {mae:.4f} (expected 0.5)")
    print(f"Directional Accuracy: {da:.2%} (expected 75%)")


if __name__ == "__main__":
    run_sanity_checks()
