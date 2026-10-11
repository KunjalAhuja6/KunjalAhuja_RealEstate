import { useState, useEffect } from "react";
import { usd, api, NEIGHBORHOODS } from "../api";
import OutOfDateNotice from "../components/OutOfDateNotice";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";

export default function ComparableSalesTab({
  form,
  query,
  valuationData,
  isOutOfDate,
  onReRun,
  onNavigateTab,
}) {
  const [compsData, setCompsData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Optional in-tab filters
  const [filterNeighborhood, setFilterNeighborhood] = useState(form.neighborhood || "");
  const [filterMinYear, setFilterMinYear] = useState("");
  const [filterMinBeds, setFilterMinBeds] = useState("");
  const [filterMinSqft, setFilterMinSqft] = useState("");
  const [searchQuery, setSearchQuery] = useState(query || "");

  const loadComps = async () => {
    setLoading(true);
    setError("");
    try {
      // Build filter payload
      const filters = {};
      if (filterNeighborhood && filterNeighborhood !== "All") {
        filters.neighborhood = filterNeighborhood;
      }
      if (filterMinYear) {
        filters.min_year_built = Number(filterMinYear);
      }
      if (filterMinBeds) {
        filters.bedrooms = Number(filterMinBeds);
      }
      if (filterMinSqft) {
        filters.min_sqft = Number(filterMinSqft);
      }

      // If user applied custom search or filters, call searchComparables;
      // otherwise use basicValuation to get stats and comps together.
      const hasCustomFilters = Object.keys(filters).length > 0 || (searchQuery && searchQuery !== query);

      if (hasCustomFilters) {
        const qText = searchQuery || `${form.bedrooms || 3} bedroom home in ${form.neighborhood || "Ames"}`;
        const searchRes = await api.searchComparables(qText, 8, filters, form);
        const comps = searchRes.comparables || [];

        // Compute summary metrics on the returned sales
        const prices = comps
          .map((c) => Number(c?.data?.SalePrice))
          .filter((p) => p && !isNaN(p));
        const sqfts = comps
          .map((c) => Number(c?.data?.GrLivArea))
          .filter((s) => s && !isNaN(s));

        const medianPrice = prices.length
          ? [...prices].sort((a, b) => a - b)[Math.floor(prices.length / 2)]
          : 0;
        const lowPrice = prices.length ? Math.min(...prices) : 0;
        const highPrice = prices.length ? Math.max(...prices) : 0;
        const avgPpsf =
          prices.length && sqfts.length
            ? Math.round(
                prices.reduce((sum, p, i) => sum + p / (sqfts[i] || 1), 0) /
                  prices.length
              )
            : 0;

        setCompsData({
          comparables: comps,
          median_price: medianPrice,
          price_range_low: lowPrice,
          price_range_high: highPrice,
          implied_ppsf: avgPpsf,
        });
      } else {
        const basicRes = await api.basicValuation(form, query, 6);
        setCompsData({
          comparables: basicRes.comparables_summary || [],
          median_price: basicRes.median_comparable_price,
          price_range_low: basicRes.valuation_range_low,
          price_range_high: basicRes.valuation_range_high,
          implied_ppsf: basicRes.implied_price_per_sqft,
        });
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // If valuationData is available and user hasn't searched yet, initialize from it
    if (valuationData && !compsData) {
      const bv = valuationData.basic_valuation || {};
      setCompsData({
        comparables: valuationData.comparables_used || [],
        median_price: bv.median_comparable_price || 0,
        price_range_low: bv.valuation_range_low || 0,
        price_range_high: bv.valuation_range_high || 0,
        implied_ppsf: bv.implied_price_per_sqft || 0,
      });
    } else if (!compsData && !loading) {
      loadComps();
    }
  }, [valuationData]);

  const compsList = compsData?.comparables || [];

  return (
    <div className="tab-pane comparable-sales-pane">
      {isOutOfDate && <OutOfDateNotice onReRun={onReRun} isRunning={loading} />}

      <section className="comps-summary-banner">
        <div className="comp-stat-card">
          <span className="stat-label">Comparable sales median</span>
          <span className="stat-value">{usd(compsData?.median_price)}</span>
        </div>
        <div className="comp-stat-card">
          <span className="stat-label">Comp price range</span>
          <span className="stat-value">
            {usd(compsData?.price_range_low)} – {usd(compsData?.price_range_high)}
          </span>
        </div>
        <div className="comp-stat-card">
          <span className="stat-label">Implied price / sq ft</span>
          <span className="stat-value">
            {compsData?.implied_ppsf ? `${usd(compsData.implied_ppsf)}/sq ft` : "–"}
          </span>
        </div>
      </section>

      <section className="comps-filter-toolbar">
        <div className="filter-row">
          <div className="filter-field">
            <label htmlFor="filter-neighborhood">Neighborhood</label>
            <select
              id="filter-neighborhood"
              value={filterNeighborhood}
              onChange={(e) => setFilterNeighborhood(e.target.value)}
            >
              <option value="All">All neighborhoods</option>
              {NEIGHBORHOODS.map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </div>

          <div className="filter-field">
            <label htmlFor="filter-min-year">Min year built</label>
            <input
              id="filter-min-year"
              type="number"
              placeholder="e.g. 1990"
              value={filterMinYear}
              onChange={(e) => setFilterMinYear(e.target.value)}
            />
          </div>

          <div className="filter-field">
            <label htmlFor="filter-min-beds">Min bedrooms</label>
            <input
              id="filter-min-beds"
              type="number"
              min="1"
              max="6"
              placeholder="e.g. 3"
              value={filterMinBeds}
              onChange={(e) => setFilterMinBeds(e.target.value)}
            />
          </div>

          <div className="filter-field">
            <label htmlFor="filter-min-sqft">Min sq ft</label>
            <input
              id="filter-min-sqft"
              type="number"
              placeholder="e.g. 1500"
              value={filterMinSqft}
              onChange={(e) => setFilterMinSqft(e.target.value)}
            />
          </div>

          <div className="filter-button-wrap">
            <button
              type="button"
              className="primary-button btn-small"
              onClick={loadComps}
              disabled={loading}
            >
              {loading ? "Searching…" : "Filter sales"}
            </button>
          </div>
        </div>
      </section>

      {loading && <LoadingState message="Finding closest comparable sales…" />}
      {error && <ErrorState error={error} onRetry={loadComps} />}

      {!loading && !error && compsList.length === 0 && (
        <div className="state-card empty-state">
          <h2>No matching comparable sales</h2>
          <p className="state-description">
            No sales matched the selected filters. Try broadening the neighborhood or year criteria.
          </p>
        </div>
      )}

      {!loading && compsList.length > 0 && (
        <section className="comps-table-section">
          <h3>Closest past sales</h3>
          <div className="table-responsive">
            <table className="data-table">
              <thead>
                <tr>
                  <th scope="col">Property ID</th>
                  <th scope="col">Neighborhood</th>
                  <th scope="col">Living area</th>
                  <th scope="col">Year built</th>
                  <th scope="col">Quality</th>
                  <th scope="col">Sale price</th>
                  <th scope="col">Similarity match</th>
                </tr>
              </thead>
              <tbody>
                {compsList.map((comp, idx) => {
                  const pid = comp.property_id || `COMP-${idx + 1}`;
                  const isSynthetic = String(pid).toUpperCase().startsWith("SYN-");
                  const d = comp.data || comp;
                  const matchRatio = comp.prioritized_score ?? comp.similarity_score ?? 0.85;
                  const matchPct = Math.round(Math.min(1, Math.max(0, matchRatio)) * 100);

                  return (
                    <tr key={pid}>
                      <td>
                        <span className="prop-id-cell">
                          {pid}
                          {isSynthetic && (
                            <span className="tag-synthetic" title="Synthetically augmented reference transaction">
                              Synthetic
                            </span>
                          )}
                        </span>
                      </td>
                      <td>{d.Neighborhood || "–"}</td>
                      <td>
                        {d.GrLivArea != null
                          ? `${Number(d.GrLivArea).toLocaleString()} sq ft`
                          : "–"}
                      </td>
                      <td>{d.YearBuilt || "–"}</td>
                      <td>{d.OverallQual != null ? `${d.OverallQual}/10` : "–"}</td>
                      <td>
                        <strong>{usd(d.SalePrice)}</strong>
                      </td>
                      <td>
                        <div className="match-meter-wrap" title={`${matchPct}% match`}>
                          <div
                            className="match-meter-bar"
                            role="progressbar"
                            aria-valuenow={matchPct}
                            aria-valuemin="0"
                            aria-valuemax="100"
                            aria-label={`Match score ${matchPct}%`}
                          >
                            <div
                              className="match-meter-fill"
                              style={{ width: `${matchPct}%` }}
                            />
                          </div>
                          <span className="match-meter-label">{matchPct}%</span>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <div className="action-buttons-row">
        <button
          type="button"
          className="ghost-button"
          onClick={() => onNavigateTab("ml")}
        >
          Compare with the model's estimate
        </button>
      </div>
    </div>
  );
}
