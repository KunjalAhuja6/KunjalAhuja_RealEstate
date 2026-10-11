import json
import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from src.utils.logger import get_logger
from src.utils.llm_client import llm_client

logger = get_logger("JudgeAgent")

class JudgeValidationResult(BaseModel):
    verdict: str  # APPROVED, FLAGGED_OVERVALUED, FLAGGED_UNDERVALUED, NEEDS_REVIEW
    confidence_score: float  # 0.0 to 1.0
    rubric_scores: Dict[str, float]  # Market Alignment, ML Consistency, Risk Coherence, Logical Integrity
    audit_commentary: str
    recommended_adjustment: Optional[float] = None
    flags: List[str] = Field(default_factory=list)

class ValuationJudgeAgent:
    """
    LLM-as-Judge agent that audits and validates proposed real estate valuations
    against comparable sales, machine learning predictions, and property risk factors.
    Calls live LLM when API key is present, preserving a deterministic rule-based evaluation as fallback.
    """
    def __init__(self):
        pass

    def validate_valuation(
        self,
        target_features: Dict[str, Any],
        recommended_price: float,
        ml_predicted_price: float,
        comp_median_price: float,
        comp_count: int,
        risk_score: float,
        valuation_explanation: str,
    ) -> JudgeValidationResult:
        """
        Conducts a multi-criterion appraisal audit on the proposed valuation.
        """
        # Always compute the deterministic rule-based audit first as reliable baseline / fallback
        rule_result = self._rule_based_audit(
            target_features=target_features,
            recommended_price=recommended_price,
            ml_predicted_price=ml_predicted_price,
            comp_median_price=comp_median_price,
            comp_count=comp_count,
            risk_score=risk_score,
            valuation_explanation=valuation_explanation,
        )

        # If LLM key is present, invoke live LLM evaluation
        if llm_client.is_live:
            try:
                system_prompt = (
                    "You are a Senior Certified Real Estate Appraiser & Audit Judge. "
                    "You independently audit proposed property valuations against statistical ML predictions, "
                    "comparable sales, and property risk scores. Output your response as a valid JSON object only."
                )
                user_prompt = (
                    f"Audit the following proposed real estate valuation:\n"
                    f"Subject Property Attributes: {json.dumps(target_features)}\n"
                    f"Proposed Recommended Valuation: ${recommended_price:,.2f}\n"
                    f"Supervised ML Point Prediction: ${ml_predicted_price:,.2f}\n"
                    f"Comparable Sales Median: ${comp_median_price:,.2f} (from {comp_count} comps)\n"
                    f"Property Risk Score: {risk_score:.1f}/100\n"
                    f"Valuation Explanation: {valuation_explanation}\n\n"
                    f"Evaluate across: (1) Market Comp Alignment, (2) ML Model Consistency, "
                    f"(3) Risk Coherence, and (4) Logical Integrity.\n"
                    f"Return a JSON object with keys:\n"
                    f'- "verdict": one of ["APPROVED", "FLAGGED_OVERVALUED", "FLAGGED_UNDERVALUED", "NEEDS_REVIEW"]\n'
                    f'- "confidence_score": float between 0.0 and 1.0\n'
                    f'- "rubric_scores": dict with numeric scores (0.0-1.0) for "market_comp_alignment", "ml_model_consistency", "risk_coherence", "logical_integrity"\n'
                    f'- "audit_commentary": a 2-3 sentence professional appraisal commentary\n'
                    f'- "recommended_adjustment": null or dollar amount adjustment\n'
                    f'- "flags": list of warning strings\n'
                    f"JSON Output:"
                )
                raw_response = llm_client.generate(prompt=user_prompt, system_prompt=system_prompt, temperature=0.1)
                
                # Extract JSON block if wrapped in markdown code fence
                json_str = raw_response.strip()
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0].strip()

                parsed = json.loads(json_str)
                verdict = str(parsed.get("verdict", rule_result.verdict)).upper()
                if verdict not in ["APPROVED", "FLAGGED_OVERVALUED", "FLAGGED_UNDERVALUED", "NEEDS_REVIEW"]:
                    verdict = rule_result.verdict

                conf = float(parsed.get("confidence_score", rule_result.confidence_score))
                conf = max(0.0, min(1.0, conf))

                rubric = parsed.get("rubric_scores", rule_result.rubric_scores)
                commentary = str(parsed.get("audit_commentary", rule_result.audit_commentary))
                adj = parsed.get("recommended_adjustment", rule_result.recommended_adjustment)
                if adj is not None:
                    try:
                        adj = float(adj)
                    except (ValueError, TypeError):
                        adj = None
                flags = parsed.get("flags", rule_result.flags)

                logger.info(f"LLM-as-Judge live audit completed: {verdict} (Confidence: {conf*100:.1f}%)")
                return JudgeValidationResult(
                    verdict=verdict,
                    confidence_score=round(conf, 3),
                    rubric_scores=rubric,
                    audit_commentary=commentary,
                    recommended_adjustment=adj,
                    flags=flags,
                )
            except Exception as e:
                logger.warning(f"Error during live LLM-as-Judge call ({e}). Smoothly fell back to rule-based path.")

        return rule_result

    def _rule_based_audit(
        self,
        target_features: Dict[str, Any],
        recommended_price: float,
        ml_predicted_price: float,
        comp_median_price: float,
        comp_count: int,
        risk_score: float,
        valuation_explanation: str,
    ) -> JudgeValidationResult:
        """Deterministic, grounded appraisal audit fallback."""
        flags = []
        rubric_scores = {}

        # 1. Market Alignment (Check against comp median)
        if comp_count > 0 and comp_median_price > 0:
            comp_diff_pct = (recommended_price - comp_median_price) / comp_median_price
            if abs(comp_diff_pct) <= 0.08:
                rubric_scores["market_comp_alignment"] = 1.0
            elif abs(comp_diff_pct) <= 0.18:
                rubric_scores["market_comp_alignment"] = 0.80
            else:
                rubric_scores["market_comp_alignment"] = 0.50
                if comp_diff_pct > 0.18:
                    flags.append(f"Valuation is {comp_diff_pct*100:.1f}% above comparable sales median (${comp_median_price:,.0f}).")
                else:
                    flags.append(f"Valuation is {abs(comp_diff_pct)*100:.1f}% below comparable sales median (${comp_median_price:,.0f}).")
        else:
            rubric_scores["market_comp_alignment"] = 0.60
            flags.append("Zero comparable sales available for market calibration; relying primarily on statistical regression.")

        # 2. ML Consistency
        ml_diff_pct = (recommended_price - ml_predicted_price) / max(ml_predicted_price, 1.0)
        if abs(ml_diff_pct) <= 0.07:
            rubric_scores["ml_model_consistency"] = 1.0
        elif abs(ml_diff_pct) <= 0.15:
            rubric_scores["ml_model_consistency"] = 0.85
        else:
            rubric_scores["ml_model_consistency"] = 0.55
            flags.append(f"Valuation diverges from ML regression estimate (${ml_predicted_price:,.0f}) by {ml_diff_pct*100:+.1f}%.")

        # 3. Risk Coherence
        if risk_score > 60 and recommended_price > ml_predicted_price:
            rubric_scores["risk_coherence"] = 0.50
            flags.append(f"Elevated property risk score ({risk_score}/100) not adequately discounted in proposed pricing.")
        elif risk_score > 40:
            rubric_scores["risk_coherence"] = 0.80
        else:
            rubric_scores["risk_coherence"] = 0.95

        # 4. Mathematical & Logical Integrity
        rubric_scores["logical_integrity"] = 0.95

        # Weighted judge score
        composite_judge_score = (
            (0.35 * rubric_scores["market_comp_alignment"])
            + (0.35 * rubric_scores["ml_model_consistency"])
            + (0.15 * rubric_scores["risk_coherence"])
            + (0.15 * rubric_scores["logical_integrity"])
        )
        composite_judge_score = round(composite_judge_score, 3)

        # Verdict Determination
        rec_adj = None
        if ml_diff_pct > 0.15 and (comp_count == 0 or comp_diff_pct > 0.15):
            verdict = "FLAGGED_OVERVALUED"
            rec_adj = round((ml_predicted_price + comp_median_price) / 2.0 - recommended_price, 2)
            commentary = (
                f"AUDIT VERDICT: FLAGGED OVERVALUED. The proposed valuation of ${recommended_price:,.2f} "
                f"exceeds both the ML statistical estimate (${ml_predicted_price:,.2f}) and market comps. "
                f"Recommended downward correction of approximately ${abs(rec_adj):,.2f}."
            )
        elif ml_diff_pct < -0.15 and (comp_count == 0 or comp_diff_pct < -0.15):
            verdict = "FLAGGED_UNDERVALUED"
            rec_adj = round((ml_predicted_price + comp_median_price) / 2.0 - recommended_price, 2)
            commentary = (
                f"AUDIT VERDICT: FLAGGED UNDERVALUED. The proposed valuation of ${recommended_price:,.2f} "
                f"is significantly lower than benchmark indicators. Upward correction of approximately ${rec_adj:,.2f} advised."
            )
        elif comp_count < 2 or (comp_count > 0 and abs(comp_diff_pct) > 0.25):
            verdict = "NEEDS_REVIEW"
            commentary = (
                f"AUDIT VERDICT: NEEDS APPRAISAL REVIEW. High divergence between comps and model or low comp sample count. "
                f"Proposed valuation of ${recommended_price:,.2f} requires verification by certified real estate appraiser."
            )
        else:
            verdict = "APPROVED"
            commentary = (
                f"AUDIT VERDICT: APPROVED. Proposed valuation of ${recommended_price:,.2f} is well-grounded "
                f"in market comps (median: ${comp_median_price:,.2f}) and conforms within {abs(ml_diff_pct)*100:.1f}% "
                f"of ML predictive regression (${ml_predicted_price:,.2f})."
            )

        return JudgeValidationResult(
            verdict=verdict,
            confidence_score=composite_judge_score,
            rubric_scores=rubric_scores,
            audit_commentary=commentary,
            recommended_adjustment=rec_adj,
            flags=flags,
        )

judge_agent = ValuationJudgeAgent()
