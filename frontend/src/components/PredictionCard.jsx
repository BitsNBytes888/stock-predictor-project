export default function PredictionCard({ prediction }) {
  const { last_close, predicted_price, predicted_log_return, model, ticker } = prediction
  const pctChange = ((predicted_price / last_close) - 1) * 100
  const isUp = pctChange >= 0

  return (
    <div className="prediction-card">
      <h2>Next-Day Prediction — {ticker} ({model})</h2>

      <div className="prediction-rows">
        <div className="prediction-row">
          <span className="label">Last Close</span>
          <span className="value">${last_close.toFixed(2)}</span>
        </div>
        <div className="prediction-row">
          <span className="label">Predicted Price</span>
          <span className="value">${predicted_price.toFixed(2)}</span>
        </div>
        <div className="prediction-row">
          <span className="label">Expected Change</span>
          <span className={`value change ${isUp ? 'up' : 'down'}`}>
            {isUp ? '▲' : '▼'} {Math.abs(pctChange).toFixed(3)}%
          </span>
        </div>
        <div className="prediction-row">
          <span className="label">Log Return</span>
          <span className="value mono">{predicted_log_return.toFixed(6)}</span>
        </div>
      </div>

      <p className="disclaimer">
        For educational purposes only. Not financial advice.
      </p>
    </div>
  )
}
