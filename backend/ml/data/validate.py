import pandas as pd
import numpy as np


REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def validate_ohlcv(df: pd.DataFrame, min_rows: int = 50) -> pd.DataFrame:
    """
    Validate and clean OHLCV stock data.

    Guarantees after return:
    - Flat columns: Open, High, Low, Close, Volume
    - DatetimeIndex, sorted ascending
    - No NaNs or infinite values
    - At least `min_rows` rows

    Raises:
        ValueError: if validation fails
    """

    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty or None.")

    # Defensive copy (never mutate caller data)
    df = df.copy()

    # --------------------------------------------------
    # 1. Flatten MultiIndex columns (yfinance behavior)
    # --------------------------------------------------
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0) #Level 1 has the ticker values. By taking level 0, we are getting only the OHLCV columns.

    # --------------------------------------------------
    # 2. Ensure required OHLCV columns exist
    # --------------------------------------------------
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}") #Make sure all our required columns are there.

    df = df[REQUIRED_COLUMNS] #Update our df to have only those.

    # --------------------------------------------------
    # 3. Validate index (time axis)
    # --------------------------------------------------
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    if df.index.duplicated().any():
        raise ValueError("Duplicate timestamps found in index.")

    # --------------------------------------------------
    # 4. Validate values
    # --------------------------------------------------
    if df.isnull().any().any():
        raise ValueError("NaN values detected in OHLCV data.")

    if np.isinf(df.to_numpy()).any():
        raise ValueError("Infinite values detected in OHLCV data.")

    # --------------------------------------------------
    # 5. Minimum data length
    # --------------------------------------------------
    if len(df) < min_rows:
        raise ValueError(
            f"Not enough data points: {len(df)} < {min_rows}"
        )

    return df
