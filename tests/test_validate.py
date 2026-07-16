import numpy as np
import pandas as pd
import pytest

from backend.ml.data.validate import validate_ohlcv


def test_valid_df_passes(sample_ohlcv):
    result = validate_ohlcv(sample_ohlcv)
    assert isinstance(result.index, pd.DatetimeIndex)
    assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(result) == len(sample_ohlcv)


def test_multiindex_columns_flattened(sample_ohlcv):
    multi = pd.DataFrame(
        sample_ohlcv.values,
        index=sample_ohlcv.index,
        columns=pd.MultiIndex.from_tuples(
            [(c, "AAPL") for c in sample_ohlcv.columns]
        ),
    )
    result = validate_ohlcv(multi)
    assert isinstance(result.columns, pd.Index)
    assert not isinstance(result.columns, pd.MultiIndex)
    assert "Open" in result.columns


def test_missing_required_column_raises(sample_ohlcv):
    df = sample_ohlcv.drop(columns=["Volume"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_ohlcv(df)


def test_non_datetime_index_raises(sample_ohlcv):
    df = sample_ohlcv.reset_index(drop=True)
    with pytest.raises(ValueError, match="DatetimeIndex"):
        validate_ohlcv(df)


def test_duplicate_timestamps_raises(sample_ohlcv):
    df = pd.concat([sample_ohlcv, sample_ohlcv.iloc[[0]]])
    with pytest.raises(ValueError, match="Duplicate timestamps"):
        validate_ohlcv(df)


def test_nan_values_raises(sample_ohlcv):
    df = sample_ohlcv.copy()
    df.iloc[5, 0] = np.nan
    with pytest.raises(ValueError, match="NaN values"):
        validate_ohlcv(df)


def test_min_rows_raises():
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "Open": rng.uniform(90, 110, 10),
            "High": rng.uniform(110, 120, 10),
            "Low":  rng.uniform(80, 90, 10),
            "Close": rng.uniform(90, 110, 10),
            "Volume": rng.integers(1_000_000, 5_000_000, 10).astype(float),
        },
        index=dates,
    )
    with pytest.raises(ValueError, match="Not enough data"):
        validate_ohlcv(df, min_rows=50)
