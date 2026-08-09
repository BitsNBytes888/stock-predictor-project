# Stock Predictor — Dev Log

A running journal of every decision made while building this project: what existed, what was broken, what each phase added, and, most importantly, **why**.

---

## Project Goals

Three reasons this project exists:

1. **Personal interest.** I've always wanted to build something end-to-end, from raw data through ML to a working UI. Stock prediction is a domain where the data is freely available, the problem is well-defined, and the results are instantly interpretable.

2. **Resume / internship signal.** A project that shows data engineering, ML modeling, a REST API, and a frontend in a single coherent codebase is more interesting to a hiring team than five disconnected toy scripts.

3. **Learning to work with coding agents.** Claude Code is being used throughout this project. The goal isn't to have an AI write code I don't understand — it's to move faster while staying in the driver's seat: approving plans, reviewing diffs, asking "why", and being able to explain every decision independently.

---

## Tech Stack at a Glance

| Layer | Technology | Why |
|-------|-----------|-----|
| Data fetch | `yfinance` | Free, no API key, good OHLCV coverage, daily data is all we need |
| Data layer | `pandas` + `numpy` | Standard for tabular + array work in Python |
| ML models | `scikit-learn`, `statsmodels`, `PyTorch` | sklearn for tree/linear, statsmodels for ARIMA, PyTorch for LSTM |
| Model persistence | `joblib` (sklearn-style), `torch.save` (LSTM) | Each ecosystem has a preferred serializer |
| Data cache | `parquet` via `pyarrow` | Preserves dtypes and DatetimeIndex; columnar format is efficient for time-series slices |
| API | `FastAPI` + `uvicorn` | Auto-generated OpenAPI docs, Pydantic validation, async-ready, pairs well with React |
| Tests | `pytest` | Industry standard; fixtures, mocking, and parametrize keep tests readable |
| Frontend | React (Vite) + Recharts | Fast dev server; Recharts is React-native (composable JSX, no imperative API), easier to learn than TradingView's canvas-based lightweight-charts |

---

## Codebase Layout

```
stock-predictor-project/
├── backend/
│   ├── app/
│   │   └── main.py                  FastAPI app — 4 endpoints
│   ├── evaluation/
│   │   ├── metrics.py               mse, mae, r2_score, directional_accuracy
│   │   └── walk_forward.py          time-series cross-validation
│   ├── experiments/
│   │   └── compare_models.py        runs all models, prints comparison table
│   └── ml/
│       ├── data/
│       │   ├── fetch_yfinance.py    raw OHLCV download
│       │   ├── validate.py          schema/NaN/index checks
│       │   ├── clean.py             NaN handling (drop or ffill)
│       │   ├── sequence.py          sliding-window sequence builder
│       │   ├── build_dataset.py     orchestrates the full pipeline
│       │   └── cache.py             date-keyed parquet disk cache
│       ├── features/
│       │   ├── basic.py             log_return, high_low_range, close_open_return, volume_change
│       │   └── technical.py         SMA ratios, RSI, MACD, Bollinger, cyclical day-of-week
│       ├── models/
│       │   ├── base.py              StockModel Protocol (interface definition)
│       │   ├── naive.py             predict 0 always (random-walk baseline)
│       │   ├── baseline.py          closed-form linear regression
│       │   ├── arima.py             ARIMA(5,0,0) univariate baseline
│       │   ├── tree.py              HistGradientBoostingRegressor
│       │   └── lstm.py              1-layer LSTM with StandardScaler3D
│       ├── preprocessing/
│       │   └── scaling.py           StandardScaler3D for (samples, seq_len, features) arrays
│       ├── training/
│       │   └── train_model.py       train any model on full history, save to artifacts/
│       └── inference/
│           └── predict.py           load saved model, predict next price
├── tests/
│   ├── conftest.py                  shared fixtures (sample_ohlcv, feature_df, technical_df)
│   ├── test_validate.py
│   ├── test_clean.py
│   ├── test_sequence.py
│   ├── test_cache.py
│   ├── test_features_basic.py
│   ├── test_features_technical.py
│   ├── test_scaling.py
│   ├── test_models.py
│   ├── test_metrics.py
│   ├── test_walk_forward.py
│   └── test_api.py
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/
│       ├── main.jsx                 React entry point
│       ├── App.jsx                  top-level state + layout
│       ├── App.css                  all styles
│       ├── api.js                   fetch wrappers for 5 backend endpoints
│       ├── data/
│       │   └── tickers.json         639 ticker symbols + names for autocomplete
│       └── components/
│           ├── TickerInput.jsx      search input + typeahead dropdown
│           ├── Controls.jsx         TickerInput + model dropdown + submit button
│           ├── PriceChart.jsx       Recharts LineChart (history + predicted dot)
│           └── PredictionCard.jsx   last close / predicted price / % change / log return
├── artifacts/                       trained model files (gitignored)
│   └── AAPL/
│       ├── naive.pkl
│       ├── linear.pkl
│       ├── arima.pkl
│       ├── tree.pkl
│       └── lstm.pt
├── cache/                           parquet cache (gitignored)
├── pytest.ini
└── DEVLOG.md                        ← you are here
```

---

## Data Pipeline

Every prediction — training or inference — runs data through the same pipeline:

```
yfinance (raw)
    → validate_ohlcv      [schema, DatetimeIndex, no NaNs, ≥50 rows]
    → clean_ohlcv         [drop or ffill residual NaNs]
    → engineer_basic_features   [log_return, high_low_range, close_open_return, volume_change]
    → add_technical_features    [SMA ratios, RSI, MACD, Bollinger, day-of-week]
    → make_sequences      [sliding windows: X(n, seq_len, features), y(n,)]
```

The separation between validation and cleaning is intentional: `validate_ohlcv` is a strict gate that rejects bad data, while `clean_ohlcv` handles the small fraction of known-benign gaps (e.g., a single missing close due to a market anomaly).

---

## Initial State (Before Phase 1)

The project already had a working data pipeline and a first attempt at two models. What existed:

- `fetch_yfinance.py`, `validate.py`, `clean.py`, `sequence.py`, `build_dataset.py` — functional
- `features/basic.py` — 4 features: log_return, high_low_range, close_open_return, volume_change
- `models/baseline.py` — `LinearBaseline` (closed-form least squares)
- `models/lstm.py` — `LSTMModel` (1-layer LSTM, PyTorch)
- `evaluation/metrics.py` — mse, mae, r2_score, directional_accuracy
- `evaluation/walk_forward.py` — walk-forward evaluation loop
- `experiments/compare_models.py` — script to compare Linear vs LSTM

**What was broken or missing:**

| Issue | File | Impact |
|-------|------|--------|
| `metrics_test.py` imported `mean_squared_error` / `mean_absolute_error` | `evaluation/metrics_test.py` | Test file could not run at all |
| `build_dataset.py` discarded `validate_ohlcv`'s return value | `ml/data/build_dataset.py` | y was shape `(n,1)` not `(n,)` → all metrics were silently wrong due to numpy broadcasting |
| `lstm.py` had dead `nn.Sequential(self.lstm, self.fc)` line | `ml/models/lstm.py` | Not a runtime crash (fit/predict bypass it), but confusing and wrong |
| `scaling.py` was an empty file | `ml/preprocessing/scaling.py` | LSTM trained on raw, unscaled features (log_return ~0.01, volume_change could be 10x) |
| No Naive baseline | — | No floor to compare against; couldn't tell if models were beating random |
| No ARIMA model | — | Classical time-series model missing from comparison |
| No model persistence | — | Models had to be re-trained from scratch every run; no path to an API |
| `app/main.py`, `ml/inference/predict.py`, `ml/training/train_ltsm.py` | all three | Empty stubs |

The `build_dataset.py` bug deserves attention: `validate_ohlcv` returns a cleaned DataFrame with flat columns, but the original code wrote `validate_ohlcv(df)` without capturing the return value. This left yfinance's MultiIndex columns in place, which made every `y` array 2-dimensional — and then numpy broadcasting silently turned metrics like MSE into nonsense (computing `(n,1) - (n,)` produces an `(n,n)` matrix, not an `(n,)` vector). This is exactly the kind of bug that's hard to spot without a test suite.

---

## Phase 1 — Quick Wins + ARIMA + Scaling

**Goal:** fix the known bugs, make LSTM training scale-aware, add a sanity-floor baseline, and add ARIMA — all wired into `compare_models.py` for a side-by-side comparison.

### What changed

**Bug fixes:**
- `metrics_test.py`: renamed imports to `mse` / `mae` to match the actual function names in `metrics.py`
- `build_dataset.py`: added `df = validate_ohlcv(df)` (the missing assignment) — y is now correctly `(n,)` shaped
- `lstm.py`: removed the dead `nn.Sequential` line; `fit` and `predict` already call `self.lstm` and `self.fc` directly

**New: `StandardScaler3D` (`preprocessing/scaling.py`)**

LSTM needs scaled inputs. sklearn's `StandardScaler` works on 2D arrays `(samples, features)`, but our data is 3D: `(samples, seq_len, features)`. The custom `StandardScaler3D` flattens to `(samples * seq_len, features)`, computes per-feature mean and std, then reshapes back. The key constraint: **fit only on training data** to avoid leaking test-set statistics into the model's normalization. This is wired directly into `LSTMModel.fit()` (fits the scaler) and `LSTMModel.predict()` (applies it), so the caller never has to manage it.

Zero-std guard: if a feature has zero variance (e.g., a constant column), the denominator is set to 1.0 instead of 0 to avoid division by zero.

**New: `NaiveBaseline` (`models/naive.py`)**

Predicts exactly zero for every input. Since the prediction target is `log_return`, "predict 0" means "the price won't change" — the random-walk-with-no-drift forecast. This is the standard sanity floor for return prediction: if your model can't beat "do nothing", it's not adding value.

One structural note: `directional_accuracy` uses `np.sign`, and `np.sign(0) == 0`. Since Naive always predicts 0, it will never have the same sign as any nonzero true return, so its directional accuracy is structurally 0% — not a bug, just a consequence of the metric definition.

**New: `ARIMABaseline` (`models/arima.py`)**

ARIMA is a classical univariate time-series model. Unlike the other models that take a feature matrix `X`, ARIMA only needs the `y` series (the sequence of past `log_return` values) to fit and forecast. The implementation uses `ARIMA(y, order=(5,0,0))` — 5 autoregressive lags, no differencing (since `log_return` is already stationary), no moving-average terms.

The adapter conforms to the same `fit(X, y)` / `predict(X)` interface as every other model — it just ignores `X` in `fit`. This means it drops into `walk_forward_eval` with zero changes to that function.

Order `(5,0,0)` was chosen because `log_return` is already stationary (so `d=0`), and 5 lags is a reasonable default for daily data without needing auto-(p,d,q) selection.

**Updated `compare_models.py`:**

Refactored from two hardcoded blocks to a loop over a dict of `{name: model_factory}` — adding a new model is now a one-liner.

### Key concepts

**Why log returns?**
- `log_return = log(Close_t / Close_{t-1})`. This is stationary (no trend), scale-invariant (works the same for a $5 stock and a $500 stock), and additive over time (multi-day return = sum of daily log returns).
- Raw price levels are non-stationary and can't be compared across tickers without normalization.

**Why walk-forward evaluation?**
- Standard train/test split leaks the future into training. Walk-forward instead mirrors how a real system would work: train on everything up to today, predict tomorrow, then advance by one day and repeat.
- Each step `t`: train on `X[:t]`, `y[:t]` → predict on `X[t]`. No future data is ever seen during training.

**What does negative R² mean?**
- R² = 1 means perfect predictions. R² = 0 means the model does no better than always predicting the mean. R² < 0 means the model is *worse* than predicting the mean.
- For daily `log_return` prediction from OHLCV-derived features, negative R² is the *expected* result, not a bug. Daily stock returns are close to a random walk (this is the weak-form efficient market hypothesis): there's very little systematic signal in yesterday's price history for predicting tomorrow's return. The models aren't broken; the signal just isn't there.

### Results (AAPL, seq_len=20, walk-forward from min_train_size=200)

| Model  | MSE    | MAE    | R²      | Dir. Acc. |
|--------|--------|--------|---------|-----------|
| Naive  | 0.0002 | 0.0105 | -0.006  | 0.000     |
| Linear | 0.0028 | 0.0270 | -11.92  | 0.507     |
| ARIMA  | 0.0002 | 0.0110 | -0.044  | 0.504     |
| LSTM   | 0.0018 | 0.0322 | -7.37   | 0.511     |

Naive has the best MSE and R² — confirming the near-random-walk nature of daily returns. Linear and LSTM are actively worse than "predict 0". This is informative, not embarrassing: it tells us the signal extraction problem, not the infrastructure, is the hard part.

---

## Phase 2 — Feature Expansion + Tree Ensemble + `retrain_every`

**Goal:** add more features, make walk-forward evaluation practical for expensive models, and add a tree ensemble — then re-run the comparison.

### What changed

**New: `features/technical.py` — `add_technical_features(df)`**

10 new features added to the output of `engineer_basic_features`:

| Feature | Formula | Why |
|---------|---------|-----|
| `price_to_sma10` | `Close / SMA10 - 1` | Price position relative to 10-day moving average |
| `price_to_sma20` | `Close / SMA20 - 1` | Price position relative to 20-day moving average |
| `vol_5` | Rolling 5-day std of `log_return` | Short-term volatility regime |
| `vol_20` | Rolling 20-day std of `log_return` | Longer-term volatility baseline |
| `rsi14` | `(RSI - 50) / 50` | Momentum; centered at 0 (overbought=+1, oversold=-1) |
| `macd` | `(EMA12 - EMA26) / Close` | Trend-following signal, normalized by price |
| `macd_signal` | 9-day EMA of `macd` | Smoothed MACD |
| `bb_width` | `4 * std20 / MA20` | Bollinger Band width — volatility expansion indicator |
| `dow_sin` | `sin(2π * weekday / 5)` | Cyclical day-of-week encoding |
| `dow_cos` | `cos(2π * weekday / 5)` | Cyclical day-of-week encoding (paired with sin) |

All features are stationary and relative (no raw price levels). Cyclical encoding deserves explanation: if you encode day-of-week as 0–4, then Monday (0) and Friday (4) look far apart to a model, even though they're adjacent in a trading week. The sin/cos pair maps each weekday to a point on a unit circle so that distance is preserved correctly.

RSI is normalized to `[-1, 1]` because: raw RSI ∈ [0, 100], centered at 50 → subtract 50, divide by 50. This makes it consistent in scale with the other features.

Rolling windows (especially the 20-day ones) introduce NaN rows at the start of the series. These are dropped with `df.dropna()` after all features are computed, same pattern as `engineer_basic_features`.

**Added `retrain_every` to `walk_forward_eval`**

The original walk-forward refitted a brand-new model from scratch at every single step. For `LinearBaseline` (microseconds per fit) this is fine, but for `LSTMModel` at 10 epochs/step over ~280 steps, that's 2,800 epochs of training — most of it redundant, since adding one data point barely changes what a well-initialized model would learn.

The `retrain_every: int = 1` parameter lets you control this: with `retrain_every=10`, the model refits every 10 steps (using all data available at that point), and predictions in between use the most recently fitted model unchanged. This lets you spend more epochs per fit without exploding total compute time.

**New: `TreeBaseline` (`models/tree.py`)**

Wraps `sklearn.ensemble.HistGradientBoostingRegressor`. Like `LinearBaseline`, it flattens the 3D input `(samples, seq_len, features)` to 2D `(samples, seq_len * features)` before training. Tree models don't need feature scaling (they split on thresholds, not distances), and `HistGradientBoostingRegressor` natively handles missing values.

Why trees? They capture non-linear feature interactions that linear models miss, train in seconds, and provide feature importances — useful for validating whether the new technical features are actually doing anything.

---

## Phase 3 — Persistence & Inference

**Goal:** make every model saveable and loadable so training and prediction can be decoupled. This is the minimum prerequisite for an API.

### What changed

**`save` / `load` on all 5 models**

Two serialization strategies:

- **joblib** for `NaiveBaseline`, `LinearBaseline`, `ARIMABaseline`, `TreeBaseline`: `joblib.dump(self, path)` / `joblib.load(path)`. Joblib is sklearn's own serializer; it handles numpy arrays more robustly than standard pickle and is already installed as a sklearn dependency.

- **`torch.save` / `torch.load`** for `LSTMModel`: PyTorch's `nn.Module` objects don't serialize cleanly through joblib (the optimizer and device state cause issues). Instead, a checkpoint dict is saved containing: the config hyperparameters (so the architecture can be reconstructed), the LSTM and FC layer `state_dict`s (weights only), and the fitted `StandardScaler3D` object (which is a plain Python class with numpy arrays, safe to embed in a torch checkpoint).

  Load sequence: reconstruct `LSTMModel(**config)` → `load_state_dict` for both layers → attach scaler.

**New: `training/train_model.py`**

A `MODEL_REGISTRY` maps model name strings to factory functions:

```python
MODEL_REGISTRY = {
    "naive":  lambda dim: NaiveBaseline(),
    "linear": lambda dim: LinearBaseline(),
    "arima":  lambda dim: ARIMABaseline(order=(5, 0, 0)),
    "tree":   lambda dim: TreeBaseline(),
    "lstm":   lambda dim: LSTMModel(input_dim=dim, hidden_dim=64, epochs=100, lr=1e-3),
}
```

`train_model(ticker, model_type)`:
1. Calls `build_dataset(ticker)` — same pipeline used in evaluation
2. Creates and fits the model on the full dataset (no walk-forward — this is a production fit on all available history)
3. Saves to `artifacts/{ticker}/{model_type}.pkl` or `.pt`

Runnable from the command line: `python3.12 -m backend.ml.training.train_model AAPL lstm`

**New: `inference/predict.py`**

`predict_next(ticker, model_type)`:
1. Loads saved model from `artifacts/`
2. Fetches the latest OHLCV data (via cache — see Phase 4)
3. Runs the feature pipeline (same functions as training, in the same order)
4. Takes the last `seq_len` rows as the inference batch: shape `(1, seq_len, n_features)`
5. Gets `pred_log_return` from `model.predict(...)[0]`
6. Converts: `pred_price = last_close * exp(pred_log_return)`

The log-return to price conversion is exact (no approximation): if `log_return = log(P_t / P_{t-1})`, then `P_t = P_{t-1} * exp(log_return)`.

**New: `models/base.py` — `StockModel` Protocol**

A `typing.Protocol` that defines the `fit / predict / save / load` interface. Not enforced at runtime (Python doesn't do that for Protocols), but serves as documentation and enables static type checkers (like mypy or Pylance in VS Code) to catch interface violations at development time.

---

## Phase 4 — FastAPI Backend + Disk Cache

**Goal:** expose the ML pipeline over HTTP so a frontend can call it.

### What changed

**New: `ml/data/cache.py` — `fetch_ohlcv_cached`**

yfinance is an external HTTP call to Yahoo Finance. Calling it on every prediction request would be slow and fragile (Yahoo rate-limits aggressively). The cache is a simple pattern:

- Key: `cache/{TICKER}_{YYYY-MM-DD}.parquet`
- On a cache hit (file exists for today's date): read from disk, no network call
- On a cache miss: call `fetch_ohlcv`, save result to parquet, return the DataFrame

Why parquet? It preserves the pandas `DatetimeIndex` and column dtypes exactly — unlike CSV, which loses timezone information and requires type inference on read.

Why date-keyed? Stock data only changes once per day (after market close). Any call on the same calendar day can safely reuse the previous result.

**`inference/predict.py`**: swapped `fetch_ohlcv` → `fetch_ohlcv_cached`. All inference calls now go through the cache transparently.

**`app/main.py` — FastAPI application**

Four endpoints:

| Endpoint | Method | Returns | Error cases |
|----------|--------|---------|-------------|
| `/api/tickers` | GET | `{"tickers": [...]}` — subdirs of `artifacts/` | Empty list if dir doesn't exist |
| `/api/models` | GET | `{"models": ["naive", "linear", "arima", "tree", "lstm"]}` | — |
| `/api/history/{ticker}` | GET | `{"ticker": "AAPL", "history": [{date, open, high, low, close, volume}, ...]}` | 404 if fetch/validate fails |
| `/api/predict/{ticker}?model=lstm` | GET | `{ticker, model, last_close, predicted_log_return, predicted_price}` | 404 if model file missing, 422 if bad data |
| `/api/artifact/{ticker}/{model}` | GET | `{"exists": true\|false}` — file existence check only | — |

**Why FastAPI?**
- Auto-generated interactive API docs at `/docs` (OpenAPI / Swagger UI)
- Pydantic validation on query parameters and return types
- CORS middleware built-in — needed so the browser frontend (on port 5173) can call the API (on port 8000) without being blocked

**CORS explained:** browsers enforce a "same-origin policy" — a page loaded from `http://localhost:5173` is blocked from making API calls to `http://localhost:8000` unless the server explicitly allows it via `Access-Control-Allow-Origin` headers. FastAPI's `CORSMiddleware` adds those headers. `allow_origins=["*"]` is appropriate for local development; in production you'd restrict this to your frontend's actual domain.

**HTTP status codes:** the API uses standard codes rather than returning errors inside 200 responses:
- `404 Not Found`: model file doesn't exist (train it first), or ticker has no data
- `422 Unprocessable Entity`: the data fetch succeeded but something is wrong with the data (e.g., fewer rows than `seq_len`)

---

## Phase Testing — Pytest Test Suite

**Goal:** retrofit automated tests so regressions are caught without manually running the whole pipeline.

The project had two "test" files before this phase: both were plain scripts that printed output rather than asserting correctness. Neither would catch a regression automatically.

### Structure

```
pytest.ini          ← configures test discovery; adds . to pythonpath
tests/
├── conftest.py     ← shared fixtures: sample_ohlcv, feature_df, technical_df
├── test_validate.py
├── test_clean.py
├── test_sequence.py
├── test_cache.py
├── test_features_basic.py
├── test_features_technical.py
├── test_scaling.py
├── test_models.py
├── test_metrics.py
├── test_walk_forward.py
└── test_api.py
```

**72 tests, 0 real network calls.** Every test that would normally call yfinance uses `unittest.mock.patch` to intercept the call and return a synthetic DataFrame. LSTM tests use `hidden_dim=8, epochs=3` so they complete in under a second. Total suite runtime: ~9 seconds.

### Test coverage

| Module | Test file | What's tested |
|--------|-----------|---------------|
| `validate.py` | `test_validate.py` | Valid df passes; MultiIndex flattened; missing column, bad index, duplicates, NaNs, too-few-rows all raise ValueError |
| `clean.py` | `test_clean.py` | No-op on clean data; drop removes row; ffill fills mid-series NaN; fraction too high raises; invalid method raises |
| `sequence.py` | `test_sequence.py` | Output shapes; seq_len=1 edge case; target value alignment |
| `cache.py` | `test_cache.py` | Cache miss calls fetch and writes parquet; cache hit skips fetch entirely |
| `features/basic.py` | `test_features_basic.py` | Row count; OHLCV preserved; log_return math; high_low_range ≥ 0; volume_change math |
| `features/technical.py` | `test_features_technical.py` | All 10 columns present; no NaNs; RSI/bb_width bounds; dow_sin/cos bounded; output shorter than input |
| `scaling.py` | `test_scaling.py` | mean_/std_ shapes; zero-centering; unit-scaling; zero-std guard; fit_transform = fit then transform; no leakage |
| All 5 models | `test_models.py` | Predict shape; finite output; save/load roundtrip; ARIMA X-independence; Naive always zero; LSTM scaler attached |
| `metrics.py` | `test_metrics.py` | Perfect prediction → 0 error; known error values; R² edge cases (perfect, mean, zero variance) |
| `walk_forward.py` | `test_walk_forward.py` | Return keys; retrain_every=1 calls factory each step; retrain_every=5 limits calls; Naive MSE = E[y²] |
| `app/main.py` | `test_api.py` | All 4 endpoints: success paths, 404/422 error codes, ticker auto-discovery |

**A note on test_api.py:** the API tests patch functions at the `backend.app.main` module level (e.g., `backend.app.main.predict_next`), not at their source module. This is because Python's import system means the API module has already bound the name `predict_next` to a local reference — patching the source would not affect the already-imported reference.

---

## Phase 5 — React Frontend

**Goal:** a real browser UI that talks to the FastAPI backend — price history chart, model selector, and a prediction card showing the next-day price forecast.

### Stack decisions

**Vite** as the dev server and bundler. It starts in milliseconds and does HMR (hot module replacement) so the browser updates instantly as you edit JSX. The alternative (Create React App) is officially deprecated.

**Recharts** for charting. It's React-native: charts are composed from JSX components like `<LineChart>`, `<Line>`, `<XAxis>` — the same model as the rest of the UI, no imperative canvas API to learn. TradingView's `lightweight-charts` is more capable for financial data but requires a separate imperative API and doesn't integrate naturally with React state.

**Native `fetch`** instead of axios. The API calls are simple enough that fetch is clearer; no dependency needed.

**Plain CSS** (one `App.css`) instead of Tailwind or styled-components. Explicit for now; easier to understand and modify as a frontend beginner than utility-class-based systems.

### Component structure

All state lives in `App.jsx` (the "single source of truth" pattern for small apps — no Redux or Zustand needed yet):

```
App.jsx               state owner
├── Controls.jsx      ticker input + model dropdown + button
├── PriceChart.jsx    Recharts LineChart
└── PredictionCard.jsx  last close / predicted price / % change / log return
```

### The predicted point on the chart

The prediction is for "next trading day" — a date that doesn't exist in the history array yet. Rather than inventing a date, the predicted point is appended to the chart data as `{ date: 'Predicted', close: null, predicted: price }` and rendered as a second `<Line>` series with `strokeWidth={0}` (so no line connects it to the history) and a large orange dot (`r: 7`). This makes it visually distinct without cluttering the X-axis with a fabricated future date.

### Two-phase loading

History fetch and prediction are sequential, not parallel (`Promise.all` was the first implementation but replaced):

1. `getHistory(ticker)` → fast (~2s, yfinance + cache) → chart renders
2. `checkArtifact(ticker, model)` → instant → determines which status message to show
3. `getPredict(ticker, model)` → may be slow → prediction card renders when done

This gives the user visual feedback (the chart) while the potentially-slow prediction step runs.

### Log return → price conversion

`predicted_price = last_close * exp(predicted_log_return)`

This is exact, not an approximation. Since `log_return = log(P_t / P_{t-1})`, solving for `P_t` gives `P_t = P_{t-1} * exp(log_return)`. The percentage change shown in `PredictionCard` is `(predicted_price / last_close - 1) * 100`.

---

## Phase 5.1 — Any-Ticker Search + On-Demand Training

**Goal:** remove the restriction that predictions only work for pre-trained tickers. Any valid ticker symbol should work.

### The problem

The original `/api/predict` endpoint raised a `FileNotFoundError` if no artifact existed for the requested ticker+model combination, which the frontend surfaced as an error message. The user had to manually run `train_model` on the command line before the UI would work for any new ticker. This breaks the UX for exploratory use.

### Solution: on-demand training in the API layer

In `backend/app/main.py`, `get_prediction` now catches `FileNotFoundError` from `predict_next` and calls `train_model(ticker, model)` before retrying. First request for a new ticker trains the model; all subsequent requests load the saved artifact and return instantly.

```python
except FileNotFoundError:
    train_model(ticker, model, artifacts_dir=ARTIFACTS_DIR)
    result = predict_next(ticker, model, artifacts_dir=ARTIFACTS_DIR)
```

The training is synchronous — the HTTP request stays open until training completes. For fast models (naive, linear, arima, tree) this is <5 seconds. For LSTM (~30–60s), the user sees the loading state in the UI. FastAPI/uvicorn handles this cleanly since sync endpoint functions run in a thread pool, not blocking the event loop.

**Why on-demand and not pre-training a fixed set of tickers?**
- A fixed list goes stale and is always someone's favourite ticker short
- The cache-on-first-request pattern is standard for expensive computations that can be stored (CDN caching, memoization, etc.)
- The user benefits immediately: first request is slow, all subsequent ones are fast

### Ticker input change

The ticker `<select>` dropdown was replaced with a `<input type="text">` search bar. Symbols auto-uppercase as you type. The button is disabled until something is typed. Enter key submits.

---

## Phase 5.2 — Ticker Autocomplete + Training Status UX

**Goal:** two quality-of-life improvements discovered during first real use.

### 1. Ticker autocomplete

**Problem:** a plain text input gives no guidance. Typos like `MCFST` instead of `MSFT` waste a full round-trip (history fetch, yfinance call) before failing.

**Solution:** a `TickerInput.jsx` component that wraps the input with a typeahead dropdown. A static `tickers.json` (639 entries: full S&P 500 + NASDAQ 100 + major ETFs + popular individual names) is bundled with the frontend. Filtering happens entirely client-side — no extra API call, instant response.

Implementation notes:
- `useEffect` on the input value recomputes filtered suggestions (`symbol.startsWith(value)`, max 8 shown)
- Click-outside detection uses a `mousedown` listener on `document` pointing to a `containerRef` — this is the standard pattern because `blur` fires before `click`, so if you used `onClick` on list items, the dropdown would close before the click registered
- Keyboard nav: `↑`/`↓` move the highlighted index, `Enter` selects the highlighted item or submits if none is highlighted, `Escape` closes
- List items use `onMouseDown` (not `onClick`) for the same reason — prevents the input blur from closing the dropdown before selection fires

**Validation approach:** soft. The suggestions guide the user toward valid symbols, but we don't block submission for unlisted ones — some valid tickers won't be in the bundled list (e.g., recent IPOs). yfinance is the hard validator; an invalid ticker fails at `/api/history` with a clear 404 error message surfaced in the red error box.

### 2. Training status clarity

**Problem:** the blue "Training…" box appeared during *every* prediction request, even when the artifact already existed and inference would complete in <1 second. This was misleading (implied it was always training) and provided no time estimate for the first-time case.

**Solution:** a `GET /api/artifact/{ticker}/{model}` endpoint that returns `{"exists": bool}` (just a file-existence check, no ML work). The frontend calls this after the history loads, then chooses one of two status messages:

- **Artifact exists** → blue box: "Running inference for TSLA…" (disappears in <1s)
- **Artifact missing** → amber/yellow box: "Training lstm model for TSLA for the first time. The trained model is saved — future predictions will be instant. LSTM may take 30–60s."

The two variants are visually distinct (blue vs. amber) so users can instantly tell whether they're waiting for a one-time training cost or a cached inference.

---

## Running the Project

### Prerequisites

Install Python dependencies (from `stock-predictor-project/`):
```bash
pip install -r backend/requirements.txt
```

Install frontend dependencies (from `stock-predictor-project/frontend/`):
```bash
npm install
```

### Running the full stack

Open two terminal windows:

**Terminal 1 — Backend API** (from `stock-predictor-project/`):
```bash
python3.12 -m uvicorn backend.app.main:app --reload --port 8000
```

**Terminal 2 — Frontend dev server** (from `stock-predictor-project/frontend/`):
```bash
npm run dev
```

Then open **http://localhost:5173** in your browser.

- The API interactive docs are at **http://localhost:8000/docs**
- First prediction for any new ticker trains the model on-demand. Fast models (naive, linear, arima, tree) take <5s. LSTM takes 30–60s but the UI shows a status message while it runs.

### Running tests

From `stock-predictor-project/`:
```bash
python3.12 -m pytest tests/ -v
```

72 tests, ~9 seconds, no real network calls.

### Training models manually

From `stock-predictor-project/` (the UI does this automatically on first request, but you can pre-train too):
```bash
python3.12 -m backend.ml.training.train_model AAPL naive
python3.12 -m backend.ml.training.train_model AAPL linear
python3.12 -m backend.ml.training.train_model AAPL arima
python3.12 -m backend.ml.training.train_model AAPL tree
python3.12 -m backend.ml.training.train_model AAPL lstm
```

### Running inference from the command line

```bash
python3.12 -m backend.ml.inference.predict AAPL lstm
```

### Walk-forward model comparison (ML experiments)

```bash
python3.12 -m backend.experiments.compare_models
```

This re-runs walk-forward evaluation across all models for AAPL. It's slow (LSTM trains many times) and is purely for understanding model performance — the API and UI don't use it.

---

**Python version note:** always use `python3.12` explicitly. The default `python` in this environment points to a conda base that lacks the project's dependencies.

---

## What's Next

- **Phase 6 — Advanced Models (stretch):** GRU (near-free — swap `nn.LSTM` for `nn.GRU` in `lstm.py`), and optionally a small Transformer encoder or Temporal Convolutional Network (TCN). Unlikely to outperform simpler models on this problem, but demonstrates architectural range and is worth understanding.

- **Phase 7 — Styling pass:** the current CSS is functional but minimal. A proper styling pass with better typography, color theme, responsive layout, and chart polish is deferred to later.

- **Phase 8 — Alternative Data (optional):** news sentiment scoring via a free API (e.g., Alpha Vantage news endpoint, or a local NLP model) aligned to trading dates. Significant scope increase — deferred until the core product is stable.
