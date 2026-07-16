import numpy as np
import pandas as pd
import pytest

from backend.ml.data.sequence import make_sequences


@pytest.fixture
def simple_df():
    """Small deterministic DataFrame for sequence tests."""
    n = 30
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "feat_a": np.arange(n, dtype=float),
            "target": np.arange(n, dtype=float) * 2,
        },
        index=dates,
    )


def test_output_shapes(simple_df):
    seq_len = 5
    X, y = make_sequences(simple_df, target_col="target", seq_len=seq_len)
    n = len(simple_df)
    assert X.shape == (n - seq_len, seq_len, 2)
    assert y.shape == (n - seq_len,)


def test_seq_len_one(simple_df):
    X, y = make_sequences(simple_df, target_col="target", seq_len=1)
    n = len(simple_df)
    assert X.shape == (n - 1, 1, 2)
    assert y.shape == (n - 1,)


def test_target_values_correct(simple_df):
    seq_len = 5
    X, y = make_sequences(simple_df, target_col="target", seq_len=seq_len)
    target_col_idx = list(simple_df.columns).index("target")
    for i in range(len(y)):
        expected = simple_df["target"].iloc[seq_len + i]
        assert y[i] == pytest.approx(expected)
