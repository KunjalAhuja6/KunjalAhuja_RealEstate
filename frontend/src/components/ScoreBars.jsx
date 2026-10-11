export default function ScoreBars({ data, maxVal = 100, labelFormatter }) {
  if (!data || Object.keys(data).length === 0) return null;

  return (
    <div className="score-bars">
      {Object.entries(data).map(([key, rawValue]) => {
        const val = Number(rawValue);
        if (isNaN(val)) return null;
        // Normalize if value is on a 0..1 scale
        const pctVal = val <= 1 && maxVal === 100 ? val * 100 : val;
        const widthPct = Math.max(0, Math.min(100, pctVal));
        const cleanLabel = labelFormatter
          ? labelFormatter(key)
          : key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

        return (
          <div key={key} className="score-bar-row">
            <span className="score-bar-label">{cleanLabel}</span>
            <div
              className="score-bar-track"
              role="progressbar"
              aria-valuenow={Math.round(pctVal)}
              aria-valuemin="0"
              aria-valuemax="100"
              aria-label={cleanLabel}
            >
              <div
                className="score-bar-fill"
                style={{ width: `${widthPct}%` }}
              />
            </div>
            <span className="score-bar-value">{Math.round(pctVal)}</span>
          </div>
        );
      })}
    </div>
  );
}
