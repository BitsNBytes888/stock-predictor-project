import TickerInput from './TickerInput'

export default function Controls({
  models,
  tickerInput,
  model,
  onTickerChange,
  onModelChange,
  onSubmit,
  loading,
}) {
  return (
    <div className="controls">
      <div className="control-group">
        <label htmlFor="ticker-input">Ticker Symbol</label>
        <TickerInput
          value={tickerInput}
          onChange={onTickerChange}
          onSubmit={onSubmit}
          disabled={loading}
        />
      </div>

      <div className="control-group">
        <label htmlFor="model-select">Model</label>
        <select
          id="model-select"
          value={model}
          onChange={e => onModelChange(e.target.value)}
          disabled={loading}
        >
          {models.map(m => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
      </div>

      <button
        className="predict-btn"
        onClick={onSubmit}
        disabled={loading || !tickerInput.trim()}
      >
        {loading ? 'Loading…' : 'Get Prediction'}
      </button>
    </div>
  )
}
