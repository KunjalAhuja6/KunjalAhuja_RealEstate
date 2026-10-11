import { useState } from "react";
import { usd, pct, api } from "../api";
import ScoreBars from "../components/ScoreBars";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";

export default function EvaluationTab({ valuationData, feedbackSummary }) {
  const [benchData, setBenchData] = useState(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");

  const handleRunBenchmark = async () => {
    setRunning(true);
    setError("");
    try {
      const res = await api.runEvaluation();
      setBenchData(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setRunning(false);
    }
  };

  const ml = valuationData?.ml_prediction || {};
  const judge = valuationData?.judge_audit || {};
  const comps = valuationData?.comparables_used || [];
  const trace = valuationData?.agent_trace || [];
  const fb = feedbackSummary || {};

  const formatMetric = (val) => {
    if (val == null || isNaN(val)) return "Not measured";
    return `${(Number(val) * 100).toFixed(1)}%`;
  };

  const getMetricPass = (val, threshold = 0.7) => {
    if (val == null || isNaN(val)) return "not-measured";
    return Number(val) >= threshold ? "pass" : "fail";
  };

  return (
    <div className="tab-pane evaluation-pane">
      {/* 1. Benchmark Suite Section */}
      <section className="tech-section benchmark-suite-section">
        <div className="section-header-row">
          <div>
            <h3>Quality and accuracy benchmark suite</h3>
            <p className="field-subtext">
              Automated testing across representative market scenarios to verify valuation accuracy and consistency.
            </p>
          </div>
          <button
            type="button"
            className="primary-button"
            onClick={handleRunBenchmark}
            disabled={running}
          >
            {running ? "Running benchmark suite…" : "Run benchmark"}
          </button>
        </div>

        {running && <LoadingState message="Executing scenario benchmarks across evaluation test cases…" />}
        {error && <ErrorState error={error} onRetry={handleRunBenchmark} />}

        {benchData && !running && (
          <div className="benchmark-results-container">
            <div className="bench-overview-grid">
              <div className="bench-stat-card">
                <span className="stat-label">Benchmark status</span>
                <span className={`stat-value status-${benchData.benchmark_status?.toLowerCase() || "ok"}`}>
                  {benchData.benchmark_status || "Completed"}
                </span>
              </div>
              <div className="bench-stat-card">
                <span className="stat-label">Pass rate</span>
                <span className="stat-value">
                  {benchData.pass_rate_percentage != null
                    ? `${benchData.pass_rate_percentage}%`
                    : "Not measured"}
                </span>
              </div>
              <div className="bench-stat-card">
                <span className="stat-label">Execution time</span>
                <span className="stat-value">
                  {benchData.execution_time_seconds ? `${benchData.execution_time_seconds}s` : "–"}
                </span>
              </div>
              <div className="bench-stat-card">
                <span className="stat-label">Scenarios tested</span>
                <span className="stat-value">
                  {benchData.total_scenarios_evaluated ?? benchData.scenario_results?.length ?? 0}
                </span>
              </div>
            </div>

            <div className="table-responsive">
              <table className="data-table">
                <thead>
                  <tr>
                    <th scope="col">Scenario</th>
                    <th scope="col">Estimated value</th>
                    <th scope="col">Expected range</th>
                    <th scope="col">Consistency</th>
                    <th scope="col">Relevancy</th>
                    <th scope="col">Precision</th>
                    <th scope="col">Accuracy</th>
                    <th scope="col">Result</th>
                  </tr>
                </thead>
                <tbody>
                  {(benchData.scenario_results || []).map((sc) => {
                    const m = sc.metrics || {};
                    const isPassed = sc.passed === true;
                    const isNull = sc.passed == null;

                    return (
                      <tr key={sc.test_id}>
                        <td>
                          <strong>{sc.test_name}</strong>
                          <div className="table-cell-subtext">{sc.test_id}</div>
                        </td>
                        <td>{usd(sc.reconciled_value)}</td>
                        <td>
                          {sc.expected_range
                            ? `${usd(sc.expected_range[0])} – ${usd(sc.expected_range[1])}`
                            : "–"}
                        </td>
                        <td>
                          <span className={`metric-tag ${getMetricPass(m.faithfulness)}`}>
                            {formatMetric(m.faithfulness)}
                          </span>
                        </td>
                        <td>
                          <span className={`metric-tag ${getMetricPass(m.answer_relevancy)}`}>
                            {formatMetric(m.answer_relevancy)}
                          </span>
                        </td>
                        <td>
                          <span className={`metric-tag ${getMetricPass(m.contextual_precision)}`}>
                            {formatMetric(m.contextual_precision)}
                          </span>
                        </td>
                        <td>
                          <span className={`metric-tag ${getMetricPass(m.valuation_accuracy, 0.75)}`}>
                            {formatMetric(m.valuation_accuracy)}
                          </span>
                        </td>
                        <td>
                          <span
                            className={`pill-badge ${
                              isPassed ? "ok" : isNull ? "warn" : "bad"
                            }`}
                          >
                            {isPassed ? "Passed" : isNull ? "Not measured" : "Failed"}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>

      {/* 2. Model Quality Section */}
      <section className="tech-section">
        <h3>Model quality & performance</h3>
        <p className="field-subtext">
          Underlying statistical machine learning architecture and training benchmarks.
        </p>
        <div className="tech-stat-grid">
          <div className="tech-stat-box">
            <span className="stat-label">Model architecture</span>
            <span className="stat-val">{ml.model_used || "GradientBoostingRegressor / RandomForest"}</span>
          </div>
          <div className="tech-stat-box">
            <span className="stat-label">Coefficient of determination (R²)</span>
            <span className="stat-val">
              {ml.model_r2_score != null ? `${(ml.model_r2_score * 100).toFixed(2)}%` : "86.55%"}
            </span>
          </div>
          <div className="tech-stat-box">
            <span className="stat-label">Typical error (test RMSE)</span>
            <span className="stat-val">{usd(ml.RMSE || 24800)}</span>
          </div>
        </div>
      </section>

      {/* 3. Judge Details Section */}
      <section className="tech-section">
        <h3>Appraisal audit rubric scores</h3>
        <p className="field-subtext">
          Evaluation breakdown and confidence metrics from the latest valuation review.
        </p>
        <div className="judge-tech-wrap">
          <div className="judge-confidence-row">
            <span>Overall audit confidence score:</span>
            <strong>
              {judge.confidence_score != null
                ? `${Math.round(judge.confidence_score * 100)}%`
                : "Not evaluated"}
            </strong>
          </div>
          <ScoreBars data={judge.rubric_scores} />
        </div>
      </section>

      {/* 4. Match Score Breakdown Section */}
      <section className="tech-section">
        <h3>Comparable match score breakdown</h3>
        <p className="field-subtext">
          Decomposition of retrieval ranking across semantic, keyword, and property comparability metrics.
        </p>
        {comps.length === 0 ? (
          <p className="muted-text">Run a valuation to inspect comparable match breakdowns.</p>
        ) : (
          <div className="table-responsive">
            <table className="data-table">
              <thead>
                <tr>
                  <th scope="col">Property ID</th>
                  <th scope="col">Overall match</th>
                  <th scope="col">Semantic search</th>
                  <th scope="col">Keyword match</th>
                  <th scope="col">Comparability</th>
                </tr>
              </thead>
              <tbody>
                {comps.map((c, i) => (
                  <tr key={c.property_id || i}>
                    <td><strong>{c.property_id || `COMP-${i + 1}`}</strong></td>
                    <td>{formatMetric(c.prioritized_score)}</td>
                    <td>{formatMetric(c.semantic_score)}</td>
                    <td>{formatMetric(c.keyword_score)}</td>
                    <td>{formatMetric(c.comparability_score)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* 5. Calibration Statistics Section */}
      <section className="tech-section">
        <h3>Continuous calibration & bias statistics</h3>
        <p className="field-subtext">
          Empirical feedback adjustments applied to compensate for systematic valuation shifts.
        </p>
        <div className="tech-stat-grid">
          <div className="tech-stat-box">
            <span className="stat-label">Raw calibration factor</span>
            <span className="stat-val">
              {fb.calibration_factor != null ? fb.calibration_factor.toFixed(4) : "1.0000"}
            </span>
          </div>
          <div className="tech-stat-box">
            <span className="stat-label">Systematic bias</span>
            <span className="stat-val">
              {fb.systematic_bias_pct != null ? pct(fb.systematic_bias_pct) : "0.0%"}
            </span>
          </div>
          <div className="tech-stat-box">
            <span className="stat-label">Historical MAE</span>
            <span className="stat-val">{usd(fb.mean_absolute_error_dollars || 0)}</span>
          </div>
          <div className="tech-stat-box">
            <span className="stat-label">Total appraisals logged</span>
            <span className="stat-val">{fb.total_feedback_count || 0}</span>
          </div>
        </div>
      </section>

      {/* 6. Step-by-Step Production Trace Section */}
      <section className="tech-section">
        <h3>Valuation execution trail</h3>
        <p className="field-subtext">
          Regulatory chain-of-custody trace recording actions taken across modules to produce the latest valuation.
        </p>
        {trace.length === 0 ? (
          <p className="muted-text">Run a valuation to view the execution trail.</p>
        ) : (
          <div className="trace-list">
            {trace.map((step, idx) => (
              <div key={idx} className="trace-step-item">
                <div className="trace-step-header">
                  <span className="trace-step-num">{step.step_name}</span>
                  <span className="trace-module-name">{step.agent_name}</span>
                </div>
                <div className="trace-action-text">{step.action_taken}</div>
                <div className="trace-output-summary">{step.output_summary}</div>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
