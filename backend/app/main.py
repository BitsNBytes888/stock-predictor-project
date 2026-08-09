import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.ml.inference.predict import predict_next, MODEL_CLASSES
from backend.ml.training.train_model import train_model
from backend.ml.data.cache import fetch_ohlcv_cached
from backend.ml.data.validate import validate_ohlcv
from backend.ml.data.clean import clean_ohlcv

app = FastAPI(title="Stock Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACTS_DIR = "artifacts"


@app.get("/api/artifact/{ticker}/{model}")
def check_artifact(ticker: str, model: str):
    ext = "pt" if model == "lstm" else "pkl"
    path = os.path.join(ARTIFACTS_DIR, ticker.upper(), f"{model}.{ext}")
    return {"exists": os.path.exists(path)}


@app.get("/api/tickers")
def get_tickers():
    if not os.path.isdir(ARTIFACTS_DIR):
        return {"tickers": []}
    tickers = [
        d for d in os.listdir(ARTIFACTS_DIR)
        if os.path.isdir(os.path.join(ARTIFACTS_DIR, d))
    ]
    return {"tickers": sorted(tickers)}


@app.get("/api/models")
def get_models():
    return {"models": list(MODEL_CLASSES)}


@app.get("/api/history/{ticker}")
def get_history(ticker: str):
    try:
        df = fetch_ohlcv_cached(ticker.upper())
        df = validate_ohlcv(df)
        df = clean_ohlcv(df)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    records = [
        {
            "date": str(idx.date()),
            "open": float(row.Open),
            "high": float(row.High),
            "low": float(row.Low),
            "close": float(row.Close),
            "volume": int(row.Volume),
        }
        for idx, row in df.iterrows()
    ]
    return {"ticker": ticker.upper(), "history": records}


@app.get("/api/predict/{ticker}")
def get_prediction(ticker: str, model: str = "lstm"):
    ticker = ticker.upper()
    try:
        result = predict_next(ticker, model, artifacts_dir=ARTIFACTS_DIR)
    except FileNotFoundError:
        try:
            train_model(ticker, model, artifacts_dir=ARTIFACTS_DIR)
        except (ValueError, Exception) as e:
            raise HTTPException(status_code=404, detail=f"Could not train model for {ticker}: {e}")
        try:
            result = predict_next(ticker, model, artifacts_dir=ARTIFACTS_DIR)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return result
