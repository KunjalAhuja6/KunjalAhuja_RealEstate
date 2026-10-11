from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import numpy as np

from src.utils.logger import get_logger
from src.rag.guardrails import validate_property_input, validate_natural_language_query
from src.rag.hybrid_retriever import hybrid_retriever
from src.agents.classification_agent import classification_agent, ClassificationResult
from src.agents.valuation_agent import valuation_agent, BasicValuationRecommendation
from src.agents.risk_agent import risk_agent, RiskPredictionResult
from src.agents.judge_agent import judge_agent, JudgeValidationResult
from src.services.ml_prediction import price_predictor
from src.services.feature_impact import feature_impact_service
from src.services.feedback_service import feedback_service

logger = get_logger("MultiAgentOrchestrator")

class AgentTraceStep(BaseModel):
    step_name: str
    agent_name: str
    action_taken: str
    output_summary: str

class ValuationDossier(BaseModel):
    is_valid: bool
    reconciled_fair_market_value: float
    confidence_interval: Dict[str, float]
    classification: ClassificationResult
    basic_valuation: BasicValuationRecommendation
    ml_prediction: Dict[str, Any]
    risk_assessment: RiskPredictionResult
    location_impact: Dict[str, Any]
    judge_audit: JudgeValidationResult
    comparables_used: List[Dict[str, Any]]
    agent_trace: List[AgentTraceStep]
    final_executive_summary: str
    applied_calibration_factor: float = 1.0
    comps_widened: bool = False
    comp_fit: float = 1.0

class MultiAgentValuationOrchestrator:
    """
    Coordinates multi-agent collaboration across Classification, Valuation,
    ML Prediction, Risk Analysis, and LLM Judge Validation.
    Applies the continuous feedback calibration factor in reconciliation.
    """
    def __init__(self):
        pass

    def run_collaborative_valuation(
        self,
        property_features: Dict[str, Any],
        query: Optional[str] = None,
        top_k_comps: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> ValuationDossier:
        """
        Executes end-to-end multi-agent collaborative valuation workflow.
        """
        trace: List[AgentTraceStep] = []

        # Step 1: Input Validation Guardrails
        val_res = validate_property_input(property_features)
        if not val_res.is_valid:
            logger.warning(f"Guardrail errors: {val_res.errors}")
            trace.append(AgentTraceStep(
                step_name="1. Input Guardrails",
                agent_name="Guardrail Validator",
                action_taken="Rejected input features",
                output_summary=f"Validation failed: {', '.join(val_res.errors)}",
            ))
            raise ValueError(f"Input validation failed: {'; '.join(val_res.errors)}")

        cleaned_features = val_res.sanitized_input
        trace.append(AgentTraceStep(
            step_name="1. Input Guardrails",
            agent_name="Guardrail Validator",
            action_taken="Sanitized and validated property attributes across all fields",
            output_summary=f"Features verified across {len(cleaned_features)} dimensions.",
        ))

        # Natural language query validation if provided
        search_query = query
        if search_query:
            q_res = validate_natural_language_query(search_query)
            if not q_res.is_valid:
                raise ValueError(f"Query validation failed: {'; '.join(q_res.errors)}")
        else:
            neigh = cleaned_features.get("neighborhood", "Ames")
            sqft = int(cleaned_features.get("gr_liv_area", 1600))
            beds = int(cleaned_features.get("bedrooms", 3))
            search_query = f"{beds} bedroom home in {neigh} with {sqft} sq ft"

        # Step 2: Hybrid Retrieval & Comparable-Property Prioritization
        merged_filters = dict(filters) if filters is not None else None
        if merged_filters is None and "neighborhood" in cleaned_features:
            merged_filters = {"neighborhood": cleaned_features["neighborhood"]}

        retrieved_comps = hybrid_retriever.retrieve(
            query=search_query,
            top_k=top_k_comps,
            filters=merged_filters,
            target_property=cleaned_features,
            include_synthetic=False,
        )

        target_sqft = float(cleaned_features.get("gr_liv_area") or cleaned_features.get("GrLivArea") or 1600)
        target_qual = float(cleaned_features.get("overall_qual") or cleaned_features.get("OverallQual") or 6)

        def _count_qualified_comps(comps: List[Dict[str, Any]], sqft_tol: float, qual_tol: float) -> int:
            cnt = 0
            for c in comps:
                c_data = c.get("data", {})
                c_sqft = float(c_data.get("GrLivArea", 0))
                c_qual = float(c_data.get("OverallQual", 0))
                if (abs(c_sqft - target_sqft) / max(target_sqft, 1.0) <= sqft_tol) and (abs(c_qual - target_qual) <= qual_tol):
                    cnt += 1
            return cnt

        comps_widened = False
        comps_relaxed = False

        # Widen search when needed:
        # If fewer than 3 comps in neighborhood have living area within +/-25% and quality within +/-1
        if _count_qualified_comps(retrieved_comps, 0.25, 1.0) < 3:
            comps_widened = True
            widen_filters = {k: v for k, v in merged_filters.items() if k not in ("neighborhood", "location")} if merged_filters else {}
            widen_filters["min_sqft"] = 0.75 * target_sqft
            widen_filters["max_sqft"] = 1.25 * target_sqft
            widen_filters["min_quality"] = target_qual - 1
            widen_filters["max_quality"] = target_qual + 1

            widened_comps = hybrid_retriever.retrieve(
                query=search_query,
                top_k=top_k_comps,
                filters=widen_filters,
                target_property=cleaned_features,
                include_synthetic=False,
            )

            # If still fewer than 3, relax to +/-40% and +/-2
            if len(widened_comps) < 3:
                comps_relaxed = True
                relax_filters = {k: v for k, v in merged_filters.items() if k not in ("neighborhood", "location")} if merged_filters else {}
                relax_filters["min_sqft"] = 0.60 * target_sqft
                relax_filters["max_sqft"] = 1.40 * target_sqft
                relax_filters["min_quality"] = max(1.0, target_qual - 2)
                relax_filters["max_quality"] = min(10.0, target_qual + 2)

                retrieved_comps = hybrid_retriever.retrieve(
                    query=search_query,
                    top_k=top_k_comps,
                    filters=relax_filters,
                    target_property=cleaned_features,
                    include_synthetic=False,
                )
            else:
                retrieved_comps = widened_comps

        widened_note = " (widened city-wide)" if comps_widened else ""
        trace.append(AgentTraceStep(
            step_name="2. Hybrid Retrieval & Prioritization",
            agent_name="Hybrid Search Engine",
            action_taken="Executed TF-IDF keyword + semantic vector retrieval with comparability ranking" + (" with city-wide widening" if comps_widened else ""),
            output_summary=f"Retrieved {len(retrieved_comps)} prioritized comparable properties{widened_note} (top comp: {retrieved_comps[0]['property_id'] if retrieved_comps else 'None'}).",
        ))

        # Step 3: Classification Agent
        classification = classification_agent.classify_property(
            features=cleaned_features,
            query=search_query,
        )
        trace.append(AgentTraceStep(
            step_name="3. Market Classification",
            agent_name="Classification Agent",
            action_taken="Segmented property into market tier, typology, and investment profile",
            output_summary=f"Classified as {classification.market_tier} ({classification.property_typology}), Condition: {classification.condition_grade}.",
        ))

        # Step 4: ML Price Prediction
        ml_result = price_predictor.predict(cleaned_features)
        trace.append(AgentTraceStep(
            step_name="4. Supervised ML Prediction",
            agent_name="Predictive ML Service",
            action_taken="Inference via trained Gradient Boosting regression model",
            output_summary=f"ML Point Estimate: ${ml_result['predicted_price']:,.2f} (R2: {ml_result['model_r2_score']:.4f}).",
        ))

        # Step 5: Basic Valuation Agent
        basic_val = valuation_agent.recommend_valuation(
            target_features=cleaned_features,
            comparables=retrieved_comps,
            user_query=search_query,
        )
        trace.append(AgentTraceStep(
            step_name="5. Comparable Valuation",
            agent_name="Valuation Agent",
            action_taken="Calculated comp-median and PPSF-weighted market estimate",
            output_summary=f"Comp-adjusted FMV: ${basic_val.estimated_fair_market_value:,.2f} (Range: ${basic_val.valuation_range_low:,.2f} - ${basic_val.valuation_range_high:,.2f}).",
        ))

        # Step 6: Risk Assessment Agent
        risk_result = risk_agent.predict_risk(
            features=cleaned_features,
            comparables=retrieved_comps,
        )
        trace.append(AgentTraceStep(
            step_name="6. Property Value Risk Analysis",
            agent_name="Risk Agent",
            action_taken="Assessed physical depreciation, condition discrepancy, and market volatility",
            output_summary=f"Assessed {risk_result.risk_category} (Score: {risk_result.overall_risk_score}/100) with {len(risk_result.risk_factors)} factors.",
        ))

        # Step 7: Location & Feature Impact
        neigh_name = cleaned_features.get("neighborhood", "North Ames")
        loc_impact = feature_impact_service.get_location_impact(neigh_name)
        trace.append(AgentTraceStep(
            step_name="7. Location Impact Analysis",
            agent_name="Feature Impact Service",
            action_taken="Calculated neighborhood premium and price elasticity",
            output_summary=f"Neighborhood '{neigh_name}' Premium: {loc_impact.get('location_premium_pct', 0.0)}% vs City Baseline.",
        ))

        # Step 8: Valuation Reconciliation Synthesis & Continuous Feedback Calibration
        comp_fmv = basic_val.estimated_fair_market_value
        ml_price = ml_result["predicted_price"]

        # comp_fit in [0,1] is the share of comps within +/-25% size and +/-1 quality, scaled by min(1, comps_found / 5).
        # comp_weight = clip(0.5 * comp_fit + 0.1, 0.1, 0.5); ml_weight = 1 - comp_weight.
        comps_found = len(retrieved_comps)
        if comps_found > 0:
            matching_share = _count_qualified_comps(retrieved_comps, 0.25, 1.0) / comps_found
            comp_fit = float(matching_share * min(1.0, comps_found / 5.0))
            comp_fit = float(np.clip(comp_fit, 0.0, 1.0))
        else:
            comp_fit = 0.0

        comp_weight = float(np.clip(0.5 * comp_fit + 0.1, 0.1, 0.5))
        ml_weight = float(1.0 - comp_weight)

        if comps_found > 0:
            reconciled_price = (comp_weight * comp_fmv) + (ml_weight * ml_price)
        else:
            reconciled_price = ml_price

        # Apply risk haircut if risk is elevated
        if risk_result.overall_risk_score > 60:
            reconciled_price *= 0.96  # 4% risk buffer
        elif risk_result.overall_risk_score > 45:
            reconciled_price *= 0.98  # 2% risk buffer

        # Apply calibration_factor from continuous feedback loop
        fb_summary = feedback_service.get_feedback_summary()
        calibration_factor = float(fb_summary.get("calibration_factor", 1.0))
        uncalibrated_price = reconciled_price
        reconciled_price = round(reconciled_price * calibration_factor, 2)

        trace.append(AgentTraceStep(
            step_name="8. Calibration & Reconciliation",
            agent_name="Reconciliation Synthesizer",
            action_taken=f"Reconciled comp FMV and ML prediction using dynamic comp_fit weighting ({comp_weight:.2f}/{ml_weight:.2f}); applied continuous feedback calibration factor ({calibration_factor:.4f})",
            output_summary=f"Reconciled FMV calibrated from ${uncalibrated_price:,.2f} to ${reconciled_price:,.2f} (Calibration factor: {calibration_factor:.4f}, comp_fit: {comp_fit:.2f}, comp_weight: {comp_weight:.2f}, ml_weight: {ml_weight:.2f}).",
        ))

        # Compute confidence interval from reconciled value with existing method, scaled by comp_fit
        # Documented constant: COMP_FIT_CI_SCALING = 0.30 adjusts interval width (narrower when comp_fit high, wider when low)
        COMP_FIT_CI_SCALING = 0.30
        raw_low = min(basic_val.valuation_range_low, ml_result["lower_bound_90pct"])
        raw_high = max(basic_val.valuation_range_high, ml_result["upper_bound_90pct"])
        base_half_width = max(reconciled_price - raw_low, raw_high - reconciled_price)
        ci_scale = 1.0 - ((comp_fit - 0.5) * COMP_FIT_CI_SCALING)
        ci_low = max(20000.0, round(reconciled_price - (base_half_width * ci_scale), 2))
        ci_high = round(reconciled_price + (base_half_width * ci_scale), 2)

        # Step 9: LLM-as-Judge Agent Validation
        judge_res = judge_agent.validate_valuation(
            target_features=cleaned_features,
            recommended_price=reconciled_price,
            ml_predicted_price=ml_price,
            comp_median_price=basic_val.median_comparable_price,
            comp_count=len(retrieved_comps),
            risk_score=risk_result.overall_risk_score,
            valuation_explanation=basic_val.valuation_explanation,
        )
        trace.append(AgentTraceStep(
            step_name="9. Independent LLM Judge Audit",
            agent_name="Appraisal Judge Agent",
            action_taken="Conducted multi-criterion audit across comps, ML consistency, and risk",
            output_summary=f"Verdict: {judge_res.verdict} (Confidence: {judge_res.confidence_score*100:.1f}%).",
        ))

        # Final Executive Summary
        exec_summary = (
            f"MULTI-AGENT VALUATION DOSSIER:\n"
            f"• Reconciled Fair Market Value: ${reconciled_price:,.2f} "
            f"(${reconciled_price / max(float(cleaned_features.get('gr_liv_area', 1600)), 1):.2f}/sq ft).\n"
            f"• Calibration Factor Applied: {calibration_factor:.4f} (Derived from historical feedback loop).\n"
            f"• Recommended Valuation Range: ${ci_low:,.2f} – ${ci_high:,.2f}.\n"
            f"• Market Segment: {classification.market_tier} ({classification.property_typology}).\n"
            f"• Statistical Alignment: ML estimate is ${ml_price:,.2f}; Comp median is ${basic_val.median_comparable_price:,.2f}.\n"
            f"• Risk Profile: {risk_result.risk_category} (Score: {risk_result.overall_risk_score}/100).\n"
            f"• Audit Status: {judge_res.verdict} (Confidence: {judge_res.confidence_score*100:.0f}%)."
        )

        return ValuationDossier(
            is_valid=True,
            reconciled_fair_market_value=reconciled_price,
            confidence_interval={"low": round(ci_low, 2), "high": round(ci_high, 2)},
            classification=classification,
            basic_valuation=basic_val,
            ml_prediction=ml_result,
            risk_assessment=risk_result,
            location_impact=loc_impact,
            judge_audit=judge_res,
            comparables_used=retrieved_comps,
            agent_trace=trace,
            final_executive_summary=exec_summary,
            applied_calibration_factor=calibration_factor,
            comps_widened=comps_widened,
            comp_fit=round(comp_fit, 4),
        )

orchestrator = MultiAgentValuationOrchestrator()
