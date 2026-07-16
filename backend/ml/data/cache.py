import os
from datetime import date

import pandas as pd

from backend.ml.data.fetch_yfinance import fetch_ohlcv


def fetch_ohlcv_cached(ticker: str, cache_dir: str = "cache") -> pd.DataFrame:
    today = date.today().isoformat()
    path = os.path.join(cache_dir, f"{ticker}_{today}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    df = fetch_ohlcv(ticker)
    os.makedirs(cache_dir, exist_ok=True)
    df.to_parquet(path)
    return df
