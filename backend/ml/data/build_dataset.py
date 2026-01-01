from backend.ml.data.fetch_yfinance import fetch_ohlcv
from backend.ml.data.validate import validate_ohlcv
from backend.ml.data.clean import clean_ohlcv
from backend.ml.features.basic import engineer_basic_features
from backend.ml.data.sequence import make_sequences



def build_dataset(
    ticker: str,
    seq_len: int = 20,
):
    df = fetch_ohlcv(ticker)
    validate_ohlcv(df)
    df = clean_ohlcv(df)

    feature_df = engineer_basic_features(df)

    X, y = make_sequences(
        feature_df,
        target_col="log_return",
        seq_len=seq_len,
    )

    return X, y
