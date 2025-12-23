import pandas as pd


def clean_ohlcv(
    df: pd.DataFrame,
    max_nan_frac: float = 0.01,
    method: str = "drop",
) -> pd.DataFrame:
    """
    Clean OHLCV data after validation.

    Parameters
    ----------
    df : pd.DataFrame
        Validated OHLCV data.
    max_nan_frac : float
        Maximum fraction of NaNs allowed per column.
    method : str
        How to handle NaNs: "drop" or "ffill".

    Returns
    -------
    pd.DataFrame
        Cleaned OHLCV data.

    Raises
    ------
    ValueError
        If NaN fraction exceeds threshold or method is invalid.
    """

    if method not in {"drop", "ffill"}:
        raise ValueError(f"Unknown cleaning method: {method}")

    df = df.copy()

    # --------------------------------------------------
    # 1. Check NaN severity
    # --------------------------------------------------
    nan_frac = df.isnull().mean() #Is null will return a dataframe with trues and falses. Doing .mean() returns a list, with each index representing the fraction of NaNs per column. 

    if nan_frac.max() > max_nan_frac: #Doing .max() gives us whichever column has the highest fraction of NaNs. 
        raise ValueError(
            f"NaN fraction too high: {nan_frac.max():.2%} "
            f"(max allowed {max_nan_frac:.2%})"
        )

    # --------------------------------------------------
    # 2. Apply cleaning policy
    # --------------------------------------------------
    if method == "drop":
        df = df.dropna()

    elif method == "ffill":
        df = df.ffill().dropna() #ffill is forwardfill. We do dropna for those NaNs which were at the top and therefore could not be forward filled.

    return df
