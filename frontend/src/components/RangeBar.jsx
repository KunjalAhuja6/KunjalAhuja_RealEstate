import { usd } from "../api";

export default function RangeBar({ low, high, fmv, ml, median }) {
  const safeLow = Number(low) || 150000;
  const safeHigh = Number(high) || 250000;
  const safeFmv = Number(fmv) || safeLow;
  const safeMl = ml != null && !isNaN(ml) ? Number(ml) : null;
  const safeMedian = median != null && !isNaN(median) ? Number(median) : null;

  // Compute scale domain with comfortable margin
  const candidates = [safeLow, safeHigh, safeFmv];
  if (safeMl != null) candidates.push(safeMl);
  if (safeMedian != null) candidates.push(safeMedian);

  const minVal = Math.min(...candidates);
  const maxVal = Math.max(...candidates);

  const spanLo = Math.max(10000, minVal * 0.96);
  const spanHi = Math.max(spanLo + 1000, maxVal * 1.04);

  // Return percentage along the scale (0 to 100)
  const getScalePct = (val) => {
    if (spanHi <= spanLo) return 50;
    const ratio = (val - spanLo) / (spanHi - spanLo);
    return Math.max(0, Math.min(100, ratio * 100));
  };

  const pctLow = getScalePct(safeLow);
  const pctHigh = getScalePct(safeHigh);
  const pctFmv = getScalePct(safeFmv);
  const pctMl = safeMl != null ? getScalePct(safeMl) : null;
  const pctMed = safeMedian != null ? getScalePct(safeMedian) : null;

  // Collision detection: if two markers are within 2% of the scale, nudge vertically
  let mlNudgeY = 0;
  let medNudgeY = 0;

  const closeMlFmv = pctMl != null && Math.abs(pctMl - pctFmv) < 2.0;
  const closeMedFmv = pctMed != null && Math.abs(pctMed - pctFmv) < 2.0;
  const closeMlMed = pctMl != null && pctMed != null && Math.abs(pctMl - pctMed) < 2.0;

  if (closeMlMed) {
    // Nudge ML up, Comp median down
    mlNudgeY = -9;
    medNudgeY = 9;
  } else {
    if (closeMlFmv) {
      mlNudgeY = -9;
    }
    if (closeMedFmv) {
      medNudgeY = 9;
    }
  }

  return (
    <div
      className="range-container"
      role="img"
      aria-label={`Valuation range from ${usd(safeLow)} to ${usd(safeHigh)} with fair value estimate of ${usd(safeFmv)}`}
    >
      {/* Track with vertical padding */}
      <div className="range-track-wrapper">
        <div className="range-base-track">
          {/* Filled band from confidence low to high */}
          <div
            className="range-fill-band"
            style={{
              left: `${pctLow.toFixed(2)}%`,
              width: `${Math.max(2, pctHigh - pctLow).toFixed(2)}%`,
            }}
          />

          {/* Fair Market Value Marker (taller, accent color) */}
          <div
            className="marker-fmv-pillar"
            style={{ left: `${pctFmv.toFixed(2)}%` }}
            title={`Fair market value: ${usd(safeFmv)}`}
          />

          {/* ML Estimate Marker (circle, nudged if within 2%) */}
          {pctMl != null && (
            <div
              className="marker-ml-circle"
              style={{
                left: `${pctMl.toFixed(2)}%`,
                transform: `translate(-50%, calc(-50% + ${mlNudgeY}px))`,
              }}
              title={`Price model estimate: ${usd(safeMl)}`}
            />
          )}

          {/* Comp Median Marker (circle, nudged if within 2%) */}
          {pctMed != null && (
            <div
              className="marker-med-circle"
              style={{
                left: `${pctMed.toFixed(2)}%`,
                transform: `translate(-50%, calc(-50% + ${medNudgeY}px))`,
              }}
              title={`Comp sales median: ${usd(safeMedian)}`}
            />
          )}
        </div>
      </div>

      {/* Low and High values directly under left and right ends of the track */}
      <div className="range-bounds-row">
        <span className="bound-val bound-low">{usd(safeLow)}</span>
        <span className="bound-val bound-high">{usd(safeHigh)}</span>
      </div>

      {/* Underneath: Single legend row (flex, wrap, evenly spaced) */}
      <div className="range-legend-row">
        <div className="legend-item">
          <span className="legend-swatch swatch-fmv" aria-hidden="true" />
          <span className="legend-text">
            Fair value <strong>{usd(safeFmv)}</strong>
          </span>
        </div>

        {safeMl != null && (
          <div className="legend-item">
            <span className="legend-swatch swatch-ml" aria-hidden="true" />
            <span className="legend-text">
              ML estimate <strong>{usd(safeMl)}</strong>
            </span>
          </div>
        )}

        {safeMedian != null && (
          <div className="legend-item">
            <span className="legend-swatch swatch-med" aria-hidden="true" />
            <span className="legend-text">
              Comp median <strong>{usd(safeMedian)}</strong>
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
