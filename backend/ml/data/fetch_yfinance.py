import yfinance as yf
import pandas as pd


def fetch_ohlcv(
    ticker: str, 
    start: str | None = None,
    end: str | None = None,
    period: str = "2y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a given ticker using yfinance.

    Args:
        ticker: Stock symbol (e.g., "AAPL")
        start: Optional start date (YYYY-MM-DD)
        end: Optional end date (YYYY-MM-DD)
        period: Lookback period (used if start/end not provided)
        interval: Data interval (e.g., 1d, 1h)

    Returns:
        DataFrame with columns:
        [Open, High, Low, Close, Volume]
        indexed by datetime.
    """

    # Fetch raw data from Yahoo Finance
    df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        period=period if start is None else None,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    # Defensive check: yfinance sometimes returns empty frames
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'")

    # Keep only OHLCV columns (drop Adj Close)
    expected_columns = ["Open", "High", "Low", "Close", "Volume"]
    df = df[expected_columns]

    return df
