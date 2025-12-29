from backend.ml.data.fetch_yfinance import fetch_ohlcv
from backend.ml.data.validate import validate_ohlcv
from backend.ml.data.clean import clean_ohlcv
from backend.ml.features.basic import engineer_basic_features

df = fetch_ohlcv("AAPL")
validate_ohlcv(df)
clean_df = clean_ohlcv(df)

features = engineer_basic_features(clean_df)
print(features.head())
