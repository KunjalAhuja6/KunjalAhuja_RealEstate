import { useState, useEffect } from "react";
import { usd, api, toPayload } from "../api";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";

export default function FeedbackTab({ form, valuationData }) {
  const [rating, setRating] = useState(5);
  const [actualVal, setActualVal] = useState("");
  const [comments, setComments] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [submitSuccess, setSubmitSuccess] = useState("");
  const [submitError, setSubmitError] = useState("");

  const [summary, setSummary] = useState(null);
  const [loadingSummary, setLoadingSummary] = useState(false);

  const loadSummary = async () => {
    setLoadingSummary(true);
    try {
      const res = await api.getFeedbackSummary();
      setSummary(res);
    } catch {
      // Non-fatal if summary cannot load
    } finally {
      setLoadingSummary(false);
    }
  };

  useEffect(() => {
    loadSummary();
    if (valuationData?.reconciled_fair_market_value && !actualVal) {
      setActualVal(String(Math.round(valuationData.reconciled_fair_market_value)));
    }
  }, [valuationData]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!actualVal || isNaN(actualVal)) return;

    setSubmitting(true);
    setSubmitSuccess("");
    setSubmitError("");

    try {
      const recVal = valuationData?.reconciled_fair_market_value || 200000;
      const calFactor = valuationData?.applied_calibration_factor || 1.0;

      await api.submitFeedback({
        property_id: "CUSTOM-APPRAISAL",
        recommended_value: recVal,
        actual_valuation: Number(actualVal),
        user_rating: rating,
        appraiser_comments: comments || "Closing price logged.",
        property_features: toPayload(form),
        professional_role: "Real Estate Agent",
        applied_calibration_factor: calFactor,
      });

      setSubmitSuccess("Your appraisal outcome was successfully logged and will calibrate future valuations.");
      setComments("");
      await loadSummary();
    } catch (err) {
      setSubmitError(err.message);
    } finally {
      setSubmitting(false);
    }
  };

  const biasNum = summary?.systematic_bias_pct ?? 0;
  const biasFormatted = `${biasNum >= 0 ? "+" : ""}${Number(biasNum).toFixed(1)}%`;
  const recentRecords = summary?.recent_records || [];

  return (
    <div className="tab-pane feedback-pane">
      <section className="feedback-form-card">
        <h3>Log agreed appraisal outcome</h3>
        <p className="field-subtext">
          Record verified sales or certified appraisal outcomes to continuously calibrate valuation accuracy.
        </p>

        <form onSubmit={handleSubmit} className="feedback-form">
          <div className="form-group">
            <label>Rating</label>
            <div className="stars-input-row" role="radiogroup" aria-label="Rating 1 to 5 stars">
              {[1, 2, 3, 4, 5].map((star) => (
                <button
                  type="button"
                  key={star}
                  className={`star-btn ${star <= rating ? "active" : ""}`}
                  onClick={() => setRating(star)}
                  aria-label={`${star} star${star > 1 ? "s" : ""}`}
                  aria-checked={rating === star}
                >
                  ★
                </button>
              ))}
            </div>
          </div>

          <div className="form-grid-2">
            <div className="form-group">
              <label htmlFor="fb-recommended">System estimated value</label>
              <input
                id="fb-recommended"
                type="text"
                disabled
                value={usd(valuationData?.reconciled_fair_market_value || 200000)}
              />
            </div>

            <div className="form-group">
              <label htmlFor="fb-actual">Actual agreed / closing price ($)</label>
              <input
                id="fb-actual"
                type="number"
                step="1000"
                required
                placeholder="e.g. 215000"
                value={actualVal}
                onChange={(e) => setActualVal(e.target.value)}
              />
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="fb-comments">Observations or notes</label>
            <textarea
              id="fb-comments"
              rows="3"
              placeholder="e.g. Final transaction was 3% over estimate due to competitive bidding on cul-de-sac."
              value={comments}
              onChange={(e) => setComments(e.target.value)}
            />
          </div>

          <button
            type="submit"
            className="primary-button"
            disabled={submitting || !actualVal}
          >
            {submitting ? "Logging outcome…" : "Save appraisal outcome"}
          </button>

          {submitSuccess && <p className="success-message">{submitSuccess}</p>}
          {submitError && <ErrorState error={submitError} />}
        </form>
      </section>

      <section className="feedback-history-card">
        <div className="calibration-banner">
          <p className="calibration-plain-line">
            Your appraisals are adjusting estimates by about <strong>{biasFormatted}</strong>.
          </p>
        </div>

        <h3>Recent appraisal history</h3>
        {loadingSummary && <LoadingState message="Loading appraisal history…" />}

        {!loadingSummary && recentRecords.length === 0 && (
          <p className="muted-text">No previous appraisal outcomes logged yet.</p>
        )}

        {!loadingSummary && recentRecords.length > 0 && (
          <div className="table-responsive">
            <table className="data-table">
              <thead>
                <tr>
                  <th scope="col">Date</th>
                  <th scope="col">Role</th>
                  <th scope="col">Estimated</th>
                  <th scope="col">Agreed price</th>
                  <th scope="col">Rating</th>
                  <th scope="col">Notes</th>
                </tr>
              </thead>
              <tbody>
                {recentRecords.slice(-8).reverse().map((rec, i) => (
                  <tr key={rec.feedback_id || i}>
                    <td>
                      {rec.timestamp
                        ? new Date(rec.timestamp).toLocaleDateString("en-US", {
                            month: "short",
                            day: "numeric",
                            year: "numeric",
                          })
                        : "–"}
                    </td>
                    <td>{rec.professional_role || "Appraiser"}</td>
                    <td>{usd(rec.recommended_value)}</td>
                    <td>
                      <strong>{usd(rec.actual_valuation)}</strong>
                    </td>
                    <td>
                      <span className="rating-display">
                        {"★".repeat(rec.user_rating || 5)}
                      </span>
                    </td>
                    <td className="note-cell">
                      {rec.appraiser_comments || "No comments"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
