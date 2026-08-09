import { useState, useEffect } from 'react'
import { getModels, getHistory, getPredict, checkArtifact } from './api'
import Controls from './components/Controls'
import PriceChart from './components/PriceChart'
import PredictionCard from './components/PredictionCard'

export default function App() {
  const [models, setModels] = useState([])
  const [tickerInput, setTickerInput] = useState('')
  const [lastTicker, setLastTicker] = useState('')
  const [model, setModel] = useState('lstm')
  const [history, setHistory] = useState([])
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [predLoading, setPredLoading] = useState(false)
  const [predStatus, setPredStatus] = useState('idle') // 'idle' | 'predicting' | 'training'
  const [error, setError] = useState(null)

  useEffect(() => {
    getModels()
      .then(m => {
        setModels(m)
        if (m.includes('lstm')) setModel('lstm')
        else if (m.length > 0) setModel(m[0])
      })
      .catch(() =>
        setError('Could not reach the API. Make sure the backend is running on port 8000.')
      )
  }, [])

  async function handleSubmit() {
    const t = tickerInput.trim().toUpperCase()
    if (!t) return

    setLoading(true)
    setPredLoading(false)
    setPredStatus('idle')
    setError(null)
    setPrediction(null)
    setHistory([])
    setLastTicker(t)

    try {
      // Phase 1: fetch history (fast — show chart immediately)
      const hist = await getHistory(t)
      setHistory(hist)
      setLoading(false)

      // Phase 2: check if artifact exists so we can show the right status message
      const artifactExists = await checkArtifact(t, model)
      setPredStatus(artifactExists ? 'predicting' : 'training')
      setPredLoading(true)

      // Phase 3: run prediction (may train on-demand if artifact missing)
      const pred = await getPredict(t, model)
      setPrediction(pred)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
      setPredLoading(false)
      setPredStatus('idle')
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <h1>Stock Predictor</h1>
          <p>Multi-model next-day price prediction using LSTM, ARIMA, gradient-boosted trees, and more</p>
        </div>
      </header>

      <main className="app-main">
        <Controls
          models={models}
          tickerInput={tickerInput}
          model={model}
          onTickerChange={setTickerInput}
          onModelChange={setModel}
          onSubmit={handleSubmit}
          loading={loading || predLoading}
        />

        {error && <div className="error-box">⚠ {error}</div>}

        {history.length > 0 && (
          <div className="results">
            <PriceChart history={history} prediction={prediction} />

            {predLoading && predStatus === 'predicting' && (
              <div className="pred-loading">
                ⏳ Running inference for {lastTicker}…
              </div>
            )}

            {predLoading && predStatus === 'training' && (
              <div className="pred-loading training">
                <strong>🏋 Training {model} model for {lastTicker} for the first time.</strong>
                <span>The trained model is saved — future predictions for this ticker will be instant. LSTM may take 30–60s.</span>
              </div>
            )}

            {prediction && <PredictionCard prediction={prediction} />}
          </div>
        )}

        {!error && history.length === 0 && !loading && !predLoading && (
          <div className="empty-state">
            <p>Search for a ticker symbol and click <strong>Get Prediction</strong>.</p>
          </div>
        )}
      </main>
    </div>
  )
}
