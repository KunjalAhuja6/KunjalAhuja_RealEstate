import { useState, useEffect } from "react";
import { usd, pct, api } from "../api";
import OutOfDateNotice from "../components/OutOfDateNotice";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";

export default function WhatIfTab({
  form,
  isOutOfDate,
  onReRun,
  onApplyAndRevalue,
}) {
  const [mods, setMods] = useState({
    add_sqft: 300,
    upgrade_quality: 1,
    remodel: false,
  });

  const [currentResult, setCurrentResult] = useState(null);
  const [savedScenarios, setSavedScenarios] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [insights, setInsights] = useState(null);

  const runSimulation = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await api.whatIf(form, mods);
      setCurrentResult(res.simulation);
      if (res.location_impact || res.feature_importance) {
        setInsights({
          locationImpact: res.location_impact,
          featureImportance: res.feature_importance,
        });
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // Run an initial simulation when tab opens if none is run yet
    if (!currentResult && !loading) {
      runSimulation();
    }
  }, [form]);

  const saveCurrentScenario = () => {
    if (!currentResult) return;
    const name = `Option ${savedScenarios.length + 1}: ${[
      mods.add_sqft ? `+${mods.add_sqft} sq ft` : null,
      mods.upgrade_quality ? `+${mods.upgrade_quality} quality` : null,
      mods.remodel ? "Remodel" : null,
    ]
      .filter(Boolean)
      .join(", ") || "Standard"}`;

    const newScenario = {
      id: Date.now(),
      name,
      modifications: { ...mods },
      simulation: currentResult,
      modifiedFeatures: currentResult.modified_features,
    };

    setSavedScenarios((prev) => [newScenario, ...prev]);
  };

  const removeScenario = (id) => {
    setSavedScenarios((prev) => prev.filter((s) => s.id !== id));
  };

  const loc = insights?.locationImpact;
  const feat = insights?.featureImportance;
  const marginal = feat?.marginal_estimates || {};

  return (
    <div className="tab-pane what-if-pane">
      {isOutOfDate && <OutOfDateNotice onReRun={runSimulation} isRunning={loading} />}

      <section className="what-if-controls-panel">
        <div className="panel-title-row">
          <h3>Simulate property improvements</h3>
          <p className="panel-sub">
            Test how adding square footage, upgrading construction quality, or remodeling shifts estimated market value.
          </p>
        </div>

        <div className="what-if-inputs-row">
          <div className="input-field">
            <label htmlFor="mod-sqft">Add living area (sq ft)</label>
            <input
              id="mod-sqft"
              type="number"
              step="50"
              min="0"
              max="2000"
              value={mods.add_sqft}
              onChange={(e) =>
                setMods({ ...mods, add_sqft: Number(e.target.value) })
              }
            />
          </div>

          <div className="input-field">
            <label htmlFor="mod-qual">Raise quality by (points)</label>
            <input
              id="mod-qual"
              type="number"
              min="0"
              max="3"
              value={mods.upgrade_quality}
              onChange={(e) =>
                setMods({ ...mods, upgrade_quality: Number(e.target.value) })
              }
            />
          </div>

          <div className="input-field checkbox-field">
            <label className="checkbox-label" htmlFor="mod-remodel">
              <input
                id="mod-remodel"
                type="checkbox"
                checked={mods.remodel}
                onChange={(e) =>
                  setMods({ ...mods, remodel: e.target.checked })
                }
              />
              <span>Full interior remodel</span>
            </label>
          </div>

          <div className="what-if-btn-group">
            <button
              type="button"
              className="primary-button"
              onClick={runSimulation}
              disabled={loading}
            >
              {loading ? "Calculating…" : "Calculate gain"}
            </button>
            {currentResult && (
              <button
                type="button"
                className="ghost-button"
                onClick={saveCurrentScenario}
              >
                Save for comparison
              </button>
            )}
          </div>
        </div>

        {error && <ErrorState error={error} onRetry={runSimulation} />}

        {currentResult && !loading && (
          <div className="simulation-live-result">
            <div className="sim-price-flow">
              <div className="sim-block">
                <span className="sim-label">Baseline valuation</span>
                <span className="sim-val">{usd(currentResult.baseline_price)}</span>
              </div>
              <span className="sim-arrow" aria-hidden="true">→</span>
              <div className="sim-block">
                <span className="sim-label">Projected valuation</span>
                <span className="sim-val highlight">{usd(currentResult.modified_price)}</span>
              </div>
            </div>

            <div className="sim-delta-badge">
              <span className={currentResult.dollar_value_added >= 0 ? "gain-up" : "gain-down"}>
                {currentResult.dollar_value_added >= 0 ? "+" : ""}
                {usd(currentResult.dollar_value_added)} ({pct(currentResult.percentage_increase)})
              </span>
            </div>

            <button
              type="button"
              className="primary-button btn-small apply-revalue-btn"
              onClick={() => onApplyAndRevalue(currentResult.modified_features)}
            >
              Apply these changes and re-value
            </button>
          </div>
        )}
      </section>

      {savedScenarios.length > 0 && (
        <section className="saved-scenarios-section">
          <h3>Saved comparison scenarios</h3>
          <div className="scenarios-grid">
            {savedScenarios.map((sc) => (
              <div key={sc.id} className="scenario-card">
                <div className="sc-header">
                  <h4>{sc.name}</h4>
                  <button
                    type="button"
                    className="delete-btn"
                    onClick={() => removeScenario(sc.id)}
                    aria-label={`Remove ${sc.name}`}
                  >
                    ×
                  </button>
                </div>
                <div className="sc-body">
                  <p className="sc-pricing">
                    <span>{usd(sc.simulation.baseline_price)}</span> →{" "}
                    <strong>{usd(sc.simulation.modified_price)}</strong>
                  </p>
                  <p className="sc-gain">
                    Estimated gain:{" "}
                    <span className={sc.simulation.dollar_value_added >= 0 ? "gain-up" : "gain-down"}>
                      {sc.simulation.dollar_value_added >= 0 ? "+" : ""}
                      {usd(sc.simulation.dollar_value_added)} ({pct(sc.simulation.percentage_increase)})
                    </span>
                  </p>
                </div>
                <button
                  type="button"
                  className="ghost-button btn-small sc-apply-btn"
                  onClick={() => onApplyAndRevalue(sc.modifiedFeatures)}
                >
                  Apply these changes and re-value
                </button>
              </div>
            ))}
          </div>
        </section>
      )}

      {insights && (
        <section className="market-drivers-section">
          <div className="driver-column">
            <h3>Neighborhood premium</h3>
            {loc?.found ? (
              <div className="insight-card">
                <p>
                  Homes in <strong>{loc.neighborhood}</strong> carry an estimated{" "}
                  <strong>{pct(loc.location_premium_pct)}</strong>{" "}
                  ({usd(loc.dollar_premium_vs_city)}) premium compared to the Ames city median.
                </p>
                <p className="insight-sub">
                  Neighborhood median sales: {usd(loc.median_neighborhood_price)} · Category: {loc.tier}
                </p>
              </div>
            ) : (
              <p className="muted-text">Standard city baseline neighborhood.</p>
            )}
          </div>

          <div className="driver-column">
            <h3>What moves the price most</h3>
            <ul className="drivers-list">
              <li>
                <strong>Architectural quality:</strong> Approx.{" "}
                {usd(marginal.per_quality_point || 24000)} per quality point
              </li>
              <li>
                <strong>Modernization:</strong> Approx.{" "}
                {usd(marginal.remodel_modernization_uplift || 18000)} uplift for remodel
              </li>
              <li>
                <strong>Garage capacity:</strong> Approx.{" "}
                {usd(marginal.per_garage_car_stall || 12500)} per extra car stall
              </li>
              <li>
                <strong>Bathrooms:</strong> Approx.{" "}
                {usd(marginal.per_full_bathroom || 8500)} per additional full bath
              </li>
              <li>
                <strong>Living space:</strong> Approx.{" "}
                {usd(marginal.per_sqft_living_area || 135)} per square foot
              </li>
            </ul>
          </div>
        </section>
      )}
    </div>
  );
}
