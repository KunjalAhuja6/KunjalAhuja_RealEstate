import { useState, useEffect } from "react";
import { usd, api, toPayload } from "../api";
import ScoreBars from "../components/ScoreBars";
import OutOfDateNotice from "../components/OutOfDateNotice";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";

const VERDICT_MAP = {
  APPROVED: { label: "Approved", tone: "ok" },
  FLAGGED_OVERVALUED: { label: "Looks too high", tone: "bad" },
  FLAGGED_UNDERVALUED: { label: "Looks too low", tone: "warn" },
  NEEDS_REVIEW: { label: "Needs review", tone: "warn" },
};

export default function RiskAndAuditTab({
  form,
  valuationData,
  isOutOfDate,
  onReRun,
  onNavigateTab,
}) {
  const [riskData, setRiskData] = useState(null);
  const [judgeData, setJudgeData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Custom price validation box
  const [userQuote, setUserQuote] = useState("");
  const [customAudit, setCustomAudit] = useState(null);
  const [auditingQuote, setAuditingQuote] = useState(false);
  const [quoteError, setQuoteError] = useState("");

  const loadRiskAndAudit = async () => {
    setLoading(true);
    setError("");
    try {
      const riskRes = await api.predictRisk(form);
      setRiskData(riskRes);

      if (valuationData?.judge_audit) {
        setJudgeData(valuationData.judge_audit);
      } else {
        // Fallback default audit validation
        const fmv = valuationData?.reconciled_fair_market_value || 200000;
        const mlPrice = valuationData?.ml_prediction?.predicted_price || fmv;
        const compMed = valuationData?.basic_valuation?.median_comparable_price || fmv;

        const auditRes = await api.judgeValidate({
          property_features: toPayload(form),
          recommended_price: fmv,
          ml_predicted_price: mlPrice,
          comp_median_price: compMed,
          comp_count: valuationData?.basic_valuation?.comparable_count_used || 5,
          risk_score: riskRes.overall_risk_score || 30.0,
          valuation_explanation: "Standard appraisal audit.",
        });
        setJudgeData(auditRes);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (valuationData) {
      if (valuationData.risk_assessment) setRiskData(valuationData.risk_assessment);
      if (valuationData.judge_audit) setJudgeData(valuationData.judge_audit);
    }
    if (!riskData && !loading) {
      loadRiskAndAudit();
    }
  }, [valuationData]);

  const handleAuditCustomPrice = async (e) => {
    e.preventDefault();
    if (!userQuote || isNaN(userQuote)) return;
    setAuditingQuote(true);
    setQuoteError("");
    try {
      const quotedVal = Number(userQuote);
      const mlPrice = valuationData?.ml_prediction?.predicted_price || quotedVal;
      const compMed = valuationData?.basic_valuation?.median_comparable_price || quotedVal;
      const rScore = riskData?.overall_risk_score || 30.0;

      const res = await api.judgeValidate({
        property_features: toPayload(form),
        recommended_price: quotedVal,
        ml_predicted_price: mlPrice,
        comp_median_price: compMed,
        comp_count: valuationData?.basic_valuation?.comparable_count_used || 5,
        risk_score: rScore,
        valuation_explanation: `Appraiser manual target quote of ${usd(quotedVal)}.`,
      });
      setCustomAudit(res);
    } catch (err) {
      setQuoteError(err.message);
    } finally {
      setAuditingQuote(false);
    }
  };

  if (loading) {
    return <LoadingState message="Evaluating property risk and appraisal audit…" />;
  }

  if (error) {
    return <ErrorState error={error} onRetry={loadRiskAndAudit} />;
  }

  const activeAudit = customAudit || judgeData;
  const verdictObj = VERDICT_MAP[activeAudit?.verdict] || {
    label: activeAudit?.verdict || "Under Review",
    tone: "warn",
  };

  const riskCat = riskData?.risk_category || "Low";
  const riskTone =
    riskCat.toLowerCase().includes("low")
      ? "ok"
      : riskCat.toLowerCase().includes("mod")
      ? "warn"
      : "bad";

  return (
    <div className="tab-pane risk-audit-pane">
      {isOutOfDate && <OutOfDateNotice onReRun={loadRiskAndAudit} isRunning={loading} />}

      <div className="risk-audit-two-col">
        {/* Risk Column */}
        <section className="risk-card">
          <div className="card-top-row">
            <h3>Property risk assessment</h3>
            <span className={`pill-badge ${riskTone}`}>
              {riskCat} risk · {Math.round(riskData?.overall_risk_score || 0)}/100
            </span>
          </div>

          <p className="summary-text">{riskData?.summary}</p>

          <div className="dimension-bars-wrap">
            <h4>Risk dimensions</h4>
            <ScoreBars data={riskData?.dimension_scores} />
          </div>

          {riskData?.risk_factors?.length > 0 && (
            <div className="risk-reasons-block">
              <h4>Main risk reasons</h4>
              <ul className="bullet-list warning-bullets">
                {riskData.risk_factors.map((rf, i) => (
                  <li key={i}>{rf}</li>
                ))}
              </ul>
            </div>
          )}

          {riskData?.mitigation_recommendations?.length > 0 && (
            <div className="risk-mitigation-block">
              <h4>How to reduce risk</h4>
              <ul className="bullet-list success-bullets">
                {riskData.mitigation_recommendations.map((rec, i) => (
                  <li key={i}>{rec}</li>
                ))}
              </ul>
            </div>
          )}
        </section>

        {/* Audit Column */}
        <section className="audit-card">
          <div className="card-top-row">
            <h3>Appraisal audit check</h3>
            <span className={`pill-badge ${verdictObj.tone}`}>
              {verdictObj.label}
            </span>
          </div>

          {activeAudit && (
            <>
              <p className="summary-text">{activeAudit.audit_commentary}</p>

              {activeAudit.recommended_adjustment != null &&
                activeAudit.recommended_adjustment !== 0 && (
                  <div className="adjustment-callout">
                    <span>Suggested adjustment:</span>
                    <strong>{usd(activeAudit.recommended_adjustment)}</strong>
                  </div>
                )}

              {activeAudit.flags?.length > 0 && (
                <div className="audit-flags-block">
                  <h4>Audit flags</h4>
                  <ul className="bullet-list warning-bullets">
                    {activeAudit.flags.map((flag, i) => (
                      <li key={i}>{flag}</li>
                    ))}
                  </ul>
                </div>
              )}
            </>
          )}

          {/* User Price Validation Box */}
          <div className="custom-quote-box">
            <h4>Check your own quoted price</h4>
            <p className="field-subtext">
              Have the system audit a specific price you plan to quote to the client.
            </p>
            <form onSubmit={handleAuditCustomPrice} className="quote-form-inline">
              <input
                type="number"
                step="1000"
                placeholder="e.g. 215000"
                value={userQuote}
                onChange={(e) => setUserQuote(e.target.value)}
                required
              />
              <button
                type="submit"
                className="primary-button btn-small"
                disabled={auditingQuote}
              >
                {auditingQuote ? "Checking…" : "Check price"}
              </button>
            </form>
            {quoteError && <p className="field-error">{quoteError}</p>}
          </div>
        </section>
      </div>

      <div className="action-buttons-row">
        <button
          type="button"
          className="ghost-button"
          onClick={() => onNavigateTab("feedback")}
        >
          Log the final agreed price
        </button>
      </div>
    </div>
  );
}
