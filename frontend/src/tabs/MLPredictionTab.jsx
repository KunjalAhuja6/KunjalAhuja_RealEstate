import { useState, useEffect } from "react";
import { usd, api } from "../api";
import OutOfDateNotice from "../components/OutOfDateNotice";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";

export default function MLPredictionTab({
  form,
  valuationData,
  isOutOfDate,
  onReRun,
}) {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const loadPrediction = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await api.predictPrice(form);
      setPrediction(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (valuationData?.ml_prediction) {
      setPrediction(valuationData.ml_prediction);
    } else if (!prediction && !loading) {
      loadPrediction();
    }
  }, [valuationData]);

  if (loading) {
    return <LoadingState message="Calculating statistical price estimate…" />;
  }

  if (error) {
    return <ErrorState error={error} onRetry={loadPrediction} />;
  }

  const estPrice = prediction?.predicted_price;
  const low = prediction?.lower_bound_90pct;
  const high = prediction?.upper_bound_90pct;
  const ppsf = prediction?.price_per_sqft;

  return (
    <div className="tab-pane ml-prediction-pane">
      {isOutOfDate && <OutOfDateNotice onReRun={loadPrediction} isRunning={loading} />}

      <section className="hero-valuation">
        <p className="kicker">Statistical price estimate</p>
        <div className="fmv-number">{usd(estPrice)}</div>

        <p className="prediction-bounds-line">
          Likely between <strong>{usd(low)}</strong> and{" "}
          <strong>{usd(high)}</strong>
        </p>

        {ppsf != null && (
          <p className="prediction-unit-cost">
            Implied price: <strong>{usd(ppsf)} / sq ft</strong>
          </p>
        )}
      </section>
    </div>
  );
}
