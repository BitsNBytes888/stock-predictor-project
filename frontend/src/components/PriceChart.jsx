import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts'

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload || !payload.length) return null
  const entry = payload.find(p => p.value != null)
  if (!entry) return null
  return (
    <div className="chart-tooltip">
      <p className="tooltip-date">{label}</p>
      <p className="tooltip-price">${Number(entry.value).toFixed(2)}</p>
    </div>
  )
}

export default function PriceChart({ history, prediction }) {
  // Build chart data: history closes + predicted point appended
  const chartData = history.map(h => ({ date: h.date, close: h.close }))

  if (prediction) {
    chartData.push({
      date: 'Predicted',
      close: null,
      predicted: prediction.predicted_price,
    })
  }

  const tickInterval = Math.max(1, Math.floor(chartData.length / 8))

  return (
    <div className="chart-container">
      <h2>Closing Price History</h2>
      <ResponsiveContainer width="100%" height={380}>
        <LineChart data={chartData} margin={{ top: 5, right: 30, left: 10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis
            dataKey="date"
            interval={tickInterval}
            tick={{ fontSize: 11, fill: '#6b7280' }}
          />
          <YAxis
            domain={['auto', 'auto']}
            tickFormatter={v => `$${v.toFixed(0)}`}
            tick={{ fontSize: 11, fill: '#6b7280' }}
            width={60}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            formatter={name => name === 'close' ? 'Historical Close' : 'Predicted'}
          />
          <Line
            type="monotone"
            dataKey="close"
            stroke="#3b82f6"
            dot={false}
            strokeWidth={1.5}
            name="close"
            connectNulls={false}
          />
          {prediction && (
            <Line
              type="monotone"
              dataKey="predicted"
              stroke="#f97316"
              dot={{ r: 7, fill: '#f97316', strokeWidth: 2, stroke: '#fff' }}
              activeDot={{ r: 9 }}
              strokeWidth={0}
              name="predicted"
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
