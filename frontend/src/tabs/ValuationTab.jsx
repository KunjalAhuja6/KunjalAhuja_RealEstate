import { usd } from "../api";
import RangeBar from "../components/RangeBar";
import OutOfDateNotice from "../components/OutOfDateNotice";
import LoadingState from "../components/LoadingState";
import EmptyState from "../components/EmptyState";
import ErrorState from "../components/ErrorState";

function buildWhyThisValue(data) {
  const bv = data?.basic_valuation || {};
  const ml = data?.ml_prediction || {};
  const medianComp = bv.median_comparable_price != null ? Number(bv.median_comparable_price) : null;
  const mlEst = ml.predicted_price != null ? Number(ml.predicted_price) : null;
  const fmv = data?.reconciled_fair_market_value != null
    ? Number(data.reconciled_fair_market_value)
    : bv.estimated_fair_market_value != null
    ? Number(bv.estimated_fair_market_value)
    : null;

  // Sentence 1: What similar homes sold for and what the price model estimates
  let s1 = "";
  if (medianComp != null && mlEst != null) {
    s1 = `Similar homes in the neighborhood recently sold for a median of ${usd(medianComp)}, while the price model estimates ${usd(mlEst)}.`;
  } else if (medianComp != null) {
    s1 = `Similar homes in the neighborhood recently sold for a median of ${usd(medianComp)}.`;
  } else if (mlEst != null) {
    s1 = `The price model estimates the property at ${usd(mlEst)}.`;
  } else {
    s1 = `This valuation is grounded in neighborhood sales data and subject property condition features.`;
  }

  // Sentence 2: Fair value relationship (sits between, above, below, or equal)
  let s2 = "";
  if (fmv != null && medianComp != null && mlEst != null) {
    const minVal = Math.min(medianComp, mlEst);
    const maxVal = Math.max(medianComp, mlEst);

    if (fmv > minVal && fmv < maxVal) {
      s2 = `The fair value of ${usd(fmv)} sits between them to reflect the home's specific condition and location.`;
    } else if (fmv > maxVal) {
      s2 = `The fair value of ${usd(fmv)} sits above both benchmarks to reflect the property's superior quality and attributes.`;
    } else if (fmv < minVal) {
      s2 = `The fair value of ${usd(fmv)} sits below both benchmarks to account for property age and physical depreciation.`;
    } else {
      s2 = `The fair value of ${usd(fmv)} aligns directly with these market indications.`;
    }
  } else if (fmv != null) {
    s2 = `The fair value of ${usd(fmv)} reflects the property's overall size, age, and architectural quality.`;
  }

  return `${s1} ${s2}`.trim();
}

function getCheckerWarning(data) {
  const judge = data?.judge_audit || {};
  const verdict = String(judge.verdict || "APPROVED").toUpperCase();
  if (verdict === "APPROVED") return null;

  if (verdict.includes("OVERVALUED")) {
    return "The checker thinks this may be high, so review it before quoting.";
  }
  if (verdict.includes("UNDERVALUED")) {
    return "The checker thinks this may be low, so review it before quoting.";
  }
  return "The checker suggests reviewing this estimate before quoting.";
}

export default function ValuationTab({
  data,
  loading,
  error,
  isOutOfDate,
  onReRun,
  onNavigateTab,
}) {
  if (loading) {
    return <LoadingState message="Working on your valuation…" />;
  }

  if (error) {
    return <ErrorState error={error} onRetry={onReRun} />;
  }

  if (!data) {
    return (
      <EmptyState
        title="Ready to value this property"
        message="Review the property features on the left and click 'Run valuation' to generate an appraisal grounded in local market sales."
      />
    );
  }

  const ci = data.confidence_interval || {};
  const bv = data.basic_valuation || {};
  const cls = data.classification || {};
  const ml = data.ml_prediction || {};

  const lowBound = ci.low ?? bv.valuation_range_low ?? 150000;
  const highBound = ci.high ?? bv.valuation_range_high ?? 250000;
  const fmv = data.reconciled_fair_market_value ?? bv.estimated_fair_market_value ?? 200000;
  const mlPrice = ml.predicted_price ?? null;
  const medianPrice = bv.median_comparable_price ?? null;

  const whyThisValue = buildWhyThisValue(data);
  const checkerWarning = getCheckerWarning(data);

  return (
    <div className="tab-pane valuation-pane">
      {isOutOfDate && <OutOfDateNotice onReRun={onReRun} isRunning={loading} />}

      <section className="hero-valuation">
        <p className="kicker">Fair market value</p>
        <div className="fmv-number">{usd(fmv)}</div>

        <p className="valuation-subline">
          Estimated range: <strong>{usd(lowBound)}</strong> to{" "}
          <strong>{usd(highBound)}</strong>
          {bv.implied_price_per_sqft && (
            <span> · {usd(bv.implied_price_per_sqft)}/sq ft</span>
          )}
          {bv.confidence_level && <span> · {bv.confidence_level} confidence</span>}
        </p>

        <RangeBar
          low={lowBound}
          high={highBound}
          fmv={fmv}
          ml={mlPrice}
          median={medianPrice}
        />

        <div className="why-value-container">
          <p className="why-this-value-paragraph">{whyThisValue}</p>

          {checkerWarning && (
            <div className="checker-flag-box">
              <span className="checker-warning-sentence">{checkerWarning}</span>
              <button
                type="button"
                className="text-link-button"
                onClick={() => onNavigateTab("risk")}
              >
                Review the checker's concerns
              </button>
            </div>
          )}
        </div>
      </section>

      <section className="property-classification-panel">
        <h3>Property classification</h3>
        <dl className="classification-grid">
          <div>
            <dt>Market tier</dt>
            <dd>{cls.market_tier || "Standard Residential"}</dd>
          </div>
          <div>
            <dt>Type</dt>
            <dd>{cls.property_typology || "Single-Family Detached"}</dd>
          </div>
          <div>
            <dt>Condition grade</dt>
            <dd>{cls.condition_grade || "Good"}</dd>
          </div>
          <div>
            <dt>Investment profile</dt>
            <dd>{cls.investment_profile || "Primary Residence"}</dd>
          </div>
        </dl>
      </section>

      <div className="action-buttons-row">
        <button
          type="button"
          className="ghost-button"
          onClick={() => onNavigateTab("comps")}
        >
          See the sales behind this value
        </button>
        <button
          type="button"
          className="ghost-button"
          onClick={() => onNavigateTab("whatif")}
        >
          See what could raise it
        </button>
      </div>
    </div>
  );
}
