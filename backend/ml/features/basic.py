# backend/ml/features/basic.py

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def engineer_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create basic stationary features from OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        Clean OHLCV data indexed by datetime.

    Returns
    -------
    pd.DataFrame
        DataFrame with engineered features.
    """

    # Defensive copy
    df = df.copy()

    # --- Log returns (stationary price change)
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # --- Intraday volatility proxy
    df["high_low_range"] = (df["High"] - df["Low"]) / df["Close"]

    # --- Directional move during the day
    df["close_open_return"] = (df["Close"] - df["Open"]) / df["Open"]

    # --- Volume change (relative)
    df["volume_change"] = df["Volume"].pct_change()

    # Drop rows created by shift / pct_change
    df = df.dropna()

    return df
