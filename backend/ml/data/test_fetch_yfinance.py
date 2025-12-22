from fetch_yfinance import fetch_ohlcv

def main():
    df = fetch_ohlcv("AAPL")
    print("Fetched rows:", len(df))
    print(df.head())


if __name__ == "__main__":
    main()