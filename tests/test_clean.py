import numpy as np
import pytest

from backend.ml.data.clean import clean_ohlcv


def test_no_nans_passes_unchanged(sample_ohlcv):
    result = clean_ohlcv(sample_ohlcv)
    assert result.shape == sample_ohlcv.shape
    assert not result.isnull().any().any()


def test_drop_removes_nan_row(sample_ohlcv):
    df = sample_ohlcv.copy()
    df.iloc[50, 0] = np.nan
    result = clean_ohlcv(df, method="drop")
    assert len(result) == len(sample_ohlcv) - 1
    assert not result.isnull().any().any()


def test_ffill_fills_and_keeps_shape(sample_ohlcv):
    df = sample_ohlcv.copy()
    df.iloc[50, 0] = np.nan
    result = clean_ohlcv(df, method="ffill")
    assert len(result) == len(sample_ohlcv)
    assert not result.isnull().any().any()


def test_nan_fraction_too_high_raises(sample_ohlcv):
    df = sample_ohlcv.copy()
    df.iloc[:20, 0] = np.nan  # 20% NaN in first column
    with pytest.raises(ValueError, match="NaN fraction too high"):
        clean_ohlcv(df, max_nan_frac=0.01)


def test_invalid_method_raises(sample_ohlcv):
    with pytest.raises(ValueError, match="Unknown cleaning method"):
        clean_ohlcv(sample_ohlcv, method="interpolate")
