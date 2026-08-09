import { useState, useRef, useEffect } from 'react'
import tickerList from '../data/tickers.json'

export default function TickerInput({ value, onChange, onSubmit, disabled }) {
  const [suggestions, setSuggestions] = useState([])
  const [showDropdown, setShowDropdown] = useState(false)
  const [highlightedIdx, setHighlightedIdx] = useState(-1)
  const containerRef = useRef(null)

  useEffect(() => {
    if (!value) {
      setSuggestions([])
      setShowDropdown(false)
      return
    }
    const filtered = tickerList.filter(t => t.symbol.startsWith(value)).slice(0, 8)
    setSuggestions(filtered)
    setShowDropdown(filtered.length > 0)
    setHighlightedIdx(-1)
  }, [value])

  useEffect(() => {
    function handleMouseDown(e) {
      if (containerRef.current && !containerRef.current.contains(e.target)) {
        setShowDropdown(false)
      }
    }
    document.addEventListener('mousedown', handleMouseDown)
    return () => document.removeEventListener('mousedown', handleMouseDown)
  }, [])

  function selectTicker(symbol) {
    onChange(symbol)
    setShowDropdown(false)
    setSuggestions([])
  }

  function handleKeyDown(e) {
    if (!showDropdown) {
      if (e.key === 'Enter' && !disabled) onSubmit()
      return
    }
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setHighlightedIdx(i => Math.min(i + 1, suggestions.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setHighlightedIdx(i => Math.max(i - 1, -1))
    } else if (e.key === 'Enter') {
      if (highlightedIdx >= 0) {
        selectTicker(suggestions[highlightedIdx].symbol)
      } else {
        setShowDropdown(false)
        if (!disabled) onSubmit()
      }
    } else if (e.key === 'Escape') {
      setShowDropdown(false)
    }
  }

  return (
    <div className="ticker-search-wrapper" ref={containerRef}>
      <input
        id="ticker-input"
        type="text"
        className="ticker-input"
        placeholder="e.g. AAPL, TSLA, SPY"
        value={value}
        onChange={e => onChange(e.target.value.toUpperCase())}
        onKeyDown={handleKeyDown}
        onFocus={() => suggestions.length > 0 && setShowDropdown(true)}
        disabled={disabled}
        maxLength={10}
        autoComplete="off"
        spellCheck={false}
      />
      {showDropdown && (
        <ul className="ticker-dropdown" role="listbox">
          {suggestions.map((t, i) => (
            <li
              key={t.symbol}
              role="option"
              aria-selected={i === highlightedIdx}
              className={`ticker-option${i === highlightedIdx ? ' highlighted' : ''}`}
              onMouseDown={() => selectTicker(t.symbol)}
              onMouseEnter={() => setHighlightedIdx(i)}
            >
              <span className="ticker-symbol">{t.symbol}</span>
              <span className="ticker-name">{t.name}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
