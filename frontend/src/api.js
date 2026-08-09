const BASE = "http://localhost:8000";

export async function getTickers() {
  const r = await fetch(`${BASE}/api/tickers`);
  const data = await r.json();
  return data.tickers;
}

export async function getModels() {
  const r = await fetch(`${BASE}/api/models`);
  const data = await r.json();
  return data.models;
}

export async function getHistory(ticker) {
  const r = await fetch(`${BASE}/api/history/${ticker}`);
  if (!r.ok) {
    const err = await r.json();
    throw new Error(err.detail || "Failed to fetch history");
  }
  const data = await r.json();
  return data.history;
}

export async function checkArtifact(ticker, model) {
  const r = await fetch(`${BASE}/api/artifact/${ticker}/${model}`)
  const data = await r.json()
  return data.exists
}

export async function getPredict(ticker, model) {
  const r = await fetch(`${BASE}/api/predict/${ticker}?model=${model}`);
  if (!r.ok) {
    const err = await r.json();
    throw new Error(err.detail || "Prediction failed");
  }
  return r.json();
}
