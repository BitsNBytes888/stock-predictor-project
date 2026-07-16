import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv():
    """100-row valid OHLCV DataFrame with DatetimeIndex and no NaNs."""
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    close = 100.0 + rng.normal(0, 1, n).cumsum()
    return pd.DataFrame(
        {
            "Open":   close + rng.normal(0, 0.5, n),
            "High":   close + np.abs(rng.normal(0, 1, n)),
            "Low":    close - np.abs(rng.normal(0, 1, n)),
            "Close":  close,
            "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=dates,
    )


@pytest.fixture
def feature_df(sample_ohlcv):
    from backend.ml.features.basic import engineer_basic_features
    return engineer_basic_features(sample_ohlcv)


@pytest.fixture
def technical_df(feature_df):
    from backend.ml.features.technical import add_technical_features
    return add_technical_features(feature_df)
