# backend/ml/features/technical.py

import numpy as np
import pandas as pd


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical-indicator features to a feature-engineered OHLCV DataFrame.

    Expects the output of engineer_basic_features (has OHLCV columns plus
    log_return etc.). All new features are stationary/relative quantities
    consistent with the existing style. Drops NaN rows from rolling windows
    before returning (the 20-day window is the binding constraint, ~19 rows).
    """
    df = df.copy()
    close = df["Close"]

    # --- Price relative to moving averages (stationary position indicator)
    df["price_to_sma10"] = close / close.rolling(10).mean() - 1
    df["price_to_sma20"] = close / close.rolling(20).mean() - 1

    # --- Rolling volatility of log returns
    df["vol_5"] = df["log_return"].rolling(5).std()
    df["vol_20"] = df["log_return"].rolling(20).std()

    # --- RSI(14): centered at 0 (neutral=50 → 0, overbought → +1, oversold → -1)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - 100 / (1 + rs)
    df["rsi14"] = (rsi - 50.0) / 50.0

    # --- MACD (12/26/9): normalized by Close to be scale-invariant across tickers
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = (ema12 - ema26) / close
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    df["macd"] = macd_line
    df["macd_signal"] = macd_signal

    # --- Bollinger Band width: (upper - lower) / MA20 = 4 * std20 / MA20
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["bb_width"] = 4 * std20 / ma20

    # --- Cyclical day-of-week (5-day trading week)
    dow = df.index.dayofweek.astype(float)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 5)

    df = df.dropna()

    return df
