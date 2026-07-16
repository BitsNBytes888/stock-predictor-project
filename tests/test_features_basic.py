import numpy as np
import pytest

from backend.ml.features.basic import engineer_basic_features


def test_output_has_one_fewer_row(sample_ohlcv):
    result = engineer_basic_features(sample_ohlcv)
    assert len(result) == len(sample_ohlcv) - 1


def test_ohlcv_columns_preserved(sample_ohlcv):
    result = engineer_basic_features(sample_ohlcv)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        assert col in result.columns


def test_log_return_correct(sample_ohlcv):
    result = engineer_basic_features(sample_ohlcv)
    expected = np.log(sample_ohlcv["Close"].iloc[1] / sample_ohlcv["Close"].iloc[0])
    assert result["log_return"].iloc[0] == pytest.approx(expected)


def test_high_low_range_non_negative(sample_ohlcv):
    result = engineer_basic_features(sample_ohlcv)
    assert (result["high_low_range"] >= 0).all()


def test_volume_change_correct(sample_ohlcv):
    result = engineer_basic_features(sample_ohlcv)
    v0 = sample_ohlcv["Volume"].iloc[0]
    v1 = sample_ohlcv["Volume"].iloc[1]
    expected = (v1 - v0) / v0
    assert result["volume_change"].iloc[0] == pytest.approx(expected)
