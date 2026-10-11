import pytest
from src.agents.classification_agent import classification_agent
from src.agents.valuation_agent import valuation_agent
from src.agents.risk_agent import risk_agent
from src.agents.judge_agent import judge_agent
from src.agents.orchestrator import orchestrator
from src.rag.hybrid_retriever import hybrid_retriever
from src.services.feedback_service import feedback_service

def test_classification_agent():
    features = {
        "gr_liv_area": 2800,
        "overall_qual": 9,
        "overall_cond": 7,
        "year_built": 2010,
        "neighborhood": "Stone Brook",
    }
    res = classification_agent.classify_property(features)
    assert "Luxury" in res.market_tier
    assert res.condition_grade != ""
    assert res.property_typology != ""

def test_risk_agent():
    high_risk_features = {
        "gr_liv_area": 1400,
        "overall_qual": 8,
        "overall_cond": 3,  # Severe quality-condition gap
        "year_built": 1940,
        "year_remod_add": 1940,  # 86 years without remodel
        "bedrooms": 4,
    }
    risk_res = risk_agent.predict_risk(high_risk_features)
    assert risk_res.overall_risk_score > 50.0
    assert len(risk_res.risk_factors) > 0
    assert len(risk_res.mitigation_recommendations) > 0

def test_judge_agent():
    target = {"gr_liv_area": 1600, "overall_qual": 6}
    # Aligned valuation test
    audit = judge_agent.validate_valuation(
        target_features=target,
        recommended_price=180000.0,
        ml_predicted_price=182000.0,
        comp_median_price=178000.0,
        comp_count=5,
        risk_score=25.0,
        valuation_explanation="Consistent market evidence.",
    )
    assert audit.verdict in ["APPROVED", "NEEDS_REVIEW", "FLAGGED_OVERVALUED", "FLAGGED_UNDERVALUED"]
    assert audit.confidence_score > 0.60
    assert len(audit.rubric_scores) >= 4

def test_multi_agent_orchestrator_and_calibration():
    sample = {
        "neighborhood": "North Ames",
        "gr_liv_area": 1650,
        "overall_qual": 6,
        "overall_cond": 5,
        "year_built": 1975,
        "bedrooms": 3,
        "bathrooms": 2,
    }
    fb_sum = feedback_service.get_feedback_summary()
    expected_cal = fb_sum.get("calibration_factor", 1.0)

    dossier = orchestrator.run_collaborative_valuation(sample)
    assert dossier.is_valid is True
    assert dossier.reconciled_fair_market_value > 0
    assert len(dossier.agent_trace) >= 8

    # Verify calibration step is recorded in trace
    cal_trace = [s for s in dossier.agent_trace if "Calibration" in s.step_name]
    assert len(cal_trace) > 0
    assert f"{expected_cal:.4f}" in cal_trace[0].output_summary

def test_orchestrator_when_filter_matches_nothing():
    sample = {
        "neighborhood": "North Ames",
        "gr_liv_area": 1650,
        "overall_qual": 6,
        "year_built": 1975,
        "bedrooms": 3,
        "bathrooms": 2,
    }
    # Strict filter that matches nothing
    strict_filters = {"neighborhood": "NonExistentPlaceXYZ", "min_bedrooms": 15}
    dossier = orchestrator.run_collaborative_valuation(sample, filters=strict_filters)
    assert dossier.is_valid is True
    assert len(dossier.comparables_used) == 0
    assert dossier.basic_valuation.comparable_count_used == 0
    assert dossier.reconciled_fair_market_value > 0  # Reconciled via ML prediction


def test_calibration_factor_does_not_drift():
    """
    Item 2: Verifies that submitting feedback where the recommended_value was
    already calibrated does NOT drive the calibration factor back towards 1.0.

    Scenario:
    - The model's uncalibrated prediction was 100_000.
    - The system applied calibration_factor = 1.10, so it showed 110_000.
    - The actual sale was 112_000 (model actually under-predicted by 12%).
    - If we naively measured error against 110_000 (calibrated), we'd see
      only +1.8% bias and the factor would trend back toward 1.0.
    - If we correctly measure against the uncalibrated 100_000, we see
      +12% bias and the factor stays elevated, accurately tracking the
      model's raw systematic error.
    """
    from src.services.feedback_service import FeedbackService, FeedbackSubmission
    import tempfile
    import json
    from pathlib import Path

    # Isolated in-memory feedback store so the test doesn't touch the real file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump([], f)
        tmp_path = Path(f.name)

    svc = FeedbackService(feedback_path=tmp_path)

    uncalibrated_prediction = 100_000.0
    calibration_factor_applied = 1.10
    calibrated_shown_to_user = uncalibrated_prediction * calibration_factor_applied  # 110_000
    actual_sale = 112_000.0

    svc.submit_feedback(
        FeedbackSubmission(
            property_id="TEST-DRIFT",
            recommended_value=calibrated_shown_to_user,
            actual_valuation=actual_sale,
            user_rating=4,
            applied_calibration_factor=calibration_factor_applied,
        )
    )

    summary = svc.get_feedback_summary()
    new_factor = summary["calibration_factor"]

    # Systematic bias measured against uncalibrated value (100_000):
    #   actual - uncal = 112_000 - 100_000 = +12_000 (+12%)
    #   calibration_factor = 1 + 0.12 = 1.12
    assert abs(new_factor - 1.12) < 0.01, (
        f"Expected factor ~1.12 (measured against uncalibrated value), got {new_factor}. "
        "Factor appears to have been measured against calibrated value, causing drift."
    )

    # Crucially: the factor must NOT drift back toward 1.0 when calibrated
    # feedback is submitted with a known positive bias.
    assert new_factor > 1.05, (
        f"Calibration factor ({new_factor}) drifted too close to 1.0 after "
        "calibrated feedback — bias was measured against the already-calibrated value."
    )

    # Cleanup
    tmp_path.unlink(missing_ok=True)


def test_briardale_search_widening_and_reconciliation():
    """
    Tests that a property with no direct matching comps in its small neighborhood
    (e.g., Briardale: 2100 sq ft, quality 8) triggers search widening:
    1. comps_widened is True.
    2. comp_fit is high (>= 0.8).
    3. Reconciled value is well-aligned with ML and comps median.
    4. Judge verdict is APPROVED (not FLAGGED_OVERVALUED or NEEDS_REVIEW).
    """
    from src.agents.orchestrator import orchestrator
    subject = {
        "neighborhood": "Briardale",
        "gr_liv_area": 2100,
        "overall_qual": 8,
        "overall_cond": 5,
        "year_built": 1995,
        "total_bsmt_sf": 1000,
        "garage_cars": 2,
        "full_bath": 2,
        "bedroom_abv_gr": 3,
        "lot_area": 8000,
    }
    dossier = orchestrator.run_collaborative_valuation(subject, top_k_comps=5)
    assert dossier.comps_widened is True, "Expected comps_widened to be True for 2100 sq ft Qual 8 in Briardale"
    assert dossier.comp_fit >= 0.8, f"Expected comp_fit >= 0.8, got {dossier.comp_fit}"
    assert dossier.judge_audit.verdict == "APPROVED", f"Expected APPROVED verdict, got {dossier.judge_audit.verdict}"

    ml_p = dossier.ml_prediction["predicted_price"]
    comp_med = dossier.basic_valuation.median_comparable_price
    # Comps median should be within 20% of ML prediction after widening and size/location adjustment
    assert abs(comp_med - ml_p) / ml_p < 0.20, f"Comp median ({comp_med}) diverged from ML ({ml_p})"


def test_north_ames_does_not_widen_search():
    """
    Tests that a typical property in a large neighborhood (North Ames: 1600 sq ft, quality 6)
    finds sufficient local comps and does NOT widen search.
    """
    from src.agents.orchestrator import orchestrator
    subject = {
        "neighborhood": "North Ames",
        "gr_liv_area": 1600,
        "overall_qual": 6,
        "overall_cond": 5,
        "year_built": 1970,
        "total_bsmt_sf": 1000,
        "garage_cars": 2,
        "full_bath": 2,
        "bedroom_abv_gr": 3,
        "lot_area": 9000,
    }
    dossier = orchestrator.run_collaborative_valuation(subject, top_k_comps=5)
    assert dossier.comps_widened is False, "North Ames should find local comps without widening"
    for comp in dossier.basic_valuation.comparables_summary:
        assert comp["neighborhood"] == "North Ames", f"Comp neighborhood {comp['neighborhood']} != North Ames"


def test_comp_fit_dynamic_weighting_bounds():
    """
    Tests dynamic weighting formula:
      comp_weight = clip(0.5 * comp_fit + 0.1, 0.1, 0.5)
      ml_weight = 1 - comp_weight
    """
    import numpy as np
    def compute_weights(fit):
        cw = float(np.clip(0.5 * fit + 0.1, 0.1, 0.5))
        return cw, 1.0 - cw

    # Perfect fit -> 50% comp, 50% ML
    cw_1, mw_1 = compute_weights(1.0)
    assert cw_1 == 0.5
    assert mw_1 == 0.5

    # Zero fit -> 10% comp, 90% ML
    cw_0, mw_0 = compute_weights(0.0)
    assert cw_0 == 0.1
    assert mw_0 == 0.9

    # Half fit -> 35% comp, 65% ML
    cw_half, mw_half = compute_weights(0.5)
    assert round(cw_half, 2) == 0.35
    assert round(mw_half, 2) == 0.65


