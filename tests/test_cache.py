from datetime import date
from unittest.mock import patch

from backend.ml.data.cache import fetch_ohlcv_cached


def test_cache_miss_calls_fetch_and_writes_parquet(tmp_path, sample_ohlcv):
    with patch("backend.ml.data.cache.fetch_ohlcv", return_value=sample_ohlcv) as mock_fetch:
        result = fetch_ohlcv_cached("TEST", cache_dir=str(tmp_path))

    mock_fetch.assert_called_once_with("TEST")
    today = date.today().isoformat()
    assert (tmp_path / f"TEST_{today}.parquet").exists()
    assert list(result.columns) == list(sample_ohlcv.columns)


def test_cache_hit_does_not_call_fetch(tmp_path, sample_ohlcv):
    today = date.today().isoformat()
    parquet_path = tmp_path / f"TEST_{today}.parquet"
    sample_ohlcv.to_parquet(parquet_path)

    with patch("backend.ml.data.cache.fetch_ohlcv", return_value=sample_ohlcv) as mock_fetch:
        result = fetch_ohlcv_cached("TEST", cache_dir=str(tmp_path))

    mock_fetch.assert_not_called()
    assert list(result.columns) == list(sample_ohlcv.columns)
    assert len(result) == len(sample_ohlcv)
