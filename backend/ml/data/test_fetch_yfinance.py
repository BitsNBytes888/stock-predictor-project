from backend.ml.data.fetch_yfinance import fetch_ohlcv
from backend.ml.data.sequence import make_sequences

def main():
    df = fetch_ohlcv("AAPL")
    print("Fetched rows:", len(df))
    print(df.head())

    # X, y= make_sequences(df, "log_return", 5)
    # print(X)
    # print(y)


if __name__ == "__main__":
    main()