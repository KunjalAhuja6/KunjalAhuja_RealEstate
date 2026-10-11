from typing import Dict, Any, List, Optional
import numpy as np
from pydantic import BaseModel, Field
from src.utils.logger import get_logger
from src.utils.llm_client import llm_client

logger = get_logger("ValuationAgent")

class BasicValuationRecommendation(BaseModel):
    estimated_fair_market_value: float
    valuation_range_low: float
    valuation_range_high: float
    median_comparable_price: float
    mean_comparable_price: float
    implied_price_per_sqft: float
    comparable_count_used: int
    confidence_level: str
    valuation_explanation: str
    comparables_summary: List[Dict[str, Any]]

class BasicValuationAgent:
    """
    Computes empirical valuation recommendations from retrieved comparable properties,
    applying price-per-square-foot weighting and quality adjustments.
    Calls live LLM for rich appraisal explanation when key is present, preserving rule-based fallback.
    """
    def __init__(self):
        self._neigh_ppsfs: Optional[Dict[str, float]] = None

    def _get_neigh_ppsfs(self) -> Dict[str, float]:
        """Calculates and caches median price per sq ft per neighborhood on real sales only."""
        if self._neigh_ppsfs is None:
            try:
                from src.rag.data_pipeline import load_cleaned_data
                df = load_cleaned_data()
                if "is_synthetic" in df.columns:
                    real_df = df[~df["is_synthetic"].astype(bool)].copy()
                else:
                    real_df = df.copy()
                real_df["ppsf"] = real_df["SalePrice"] / np.maximum(real_df["GrLivArea"], 1.0)
                medians = real_df.groupby("Neighborhood")["ppsf"].median().to_dict()
                self._neigh_ppsfs = {str(k).strip().lower(): float(v) for k, v in medians.items()}
            except Exception as e:
                logger.warning(f"Error computing neighborhood median PPSFs ({e}). Using empty map.")
                self._neigh_ppsfs = {}
        return self._neigh_ppsfs

    def recommend_valuation(
        self,
        target_features: Dict[str, Any],
        comparables: List[Dict[str, Any]],
        user_query: Optional[str] = None,
    ) -> BasicValuationRecommendation:
        """
        Derives valuation recommendation based on retrieved comparable properties.
        """
        target_sqft = float(target_features.get("gr_liv_area") or target_features.get("GrLivArea") or 1600)
        target_qual = float(target_features.get("overall_qual") or target_features.get("OverallQual") or 6)
        target_neigh = str(target_features.get("neighborhood") or target_features.get("location") or "").strip()

        if not comparables:
            # Fallback when no comps are available (e.g. strict filters matched 0 properties)
            base_val = round(target_sqft * 125.0 * (target_qual / 6.0), 2)
            range_low = round(base_val * 0.88, 2)
            range_high = round(base_val * 1.12, 2)
            implied_ppsf = round(base_val / max(target_sqft, 1.0), 2)

            explanation = (
                f"No comparable sales matched the applied search criteria. Baseline heuristic valuation "
                f"estimated at ${base_val:,.2f} (${implied_ppsf:.2f}/sq ft) with a recommended valuation range of "
                f"${range_low:,.2f} to ${range_high:,.2f} reflecting {target_sqft:.0f} sq ft and quality grade {target_qual:.0f}/10."
            )

            if llm_client.is_live:
                try:
                    user_prompt = (
                        f"Subject Property: {target_sqft:.0f} sq ft, Quality {target_qual:.0f}/10 in {target_features.get('neighborhood', 'Ames')}.\n"
                        f"No direct comps were found matching strict filters.\n"
                        f"Estimated Baseline Fair Market Value: ${base_val:,.2f} (Range: ${range_low:,.2f} - ${range_high:,.2f}).\n"
                        f"Provide a concise, professional 2-sentence appraisal explanation noting that no direct comps matched and baseline valuation was used."
                    )
                    system_prompt = "You are a Senior Certified Real Estate Appraiser."
                    llm_text = llm_client.generate(prompt=user_prompt, system_prompt=system_prompt)
                    if llm_text and len(llm_text.strip()) > 20:
                        explanation = llm_text.strip()
                except Exception as e:
                    logger.warning(f"Live LLM explanation call failed ({e}). Kept rule-based fallback.")

            return BasicValuationRecommendation(
                estimated_fair_market_value=base_val,
                valuation_range_low=range_low,
                valuation_range_high=range_high,
                median_comparable_price=base_val,
                mean_comparable_price=base_val,
                implied_price_per_sqft=implied_ppsf,
                comparable_count_used=0,
                confidence_level="Low (0 comps matched filters)",
                valuation_explanation=explanation,
                comparables_summary=[],
            )

        neigh_ppsfs = self._get_neigh_ppsfs()
        comp_prices = []
        comp_ppsfs = []
        comp_summaries = []

        for comp in comparables:
            data = comp.get("data", {})
            price = float(data.get("SalePrice", 0))
            sqft = float(data.get("GrLivArea", 1))
            qual = float(data.get("OverallQual", 6))
            comp_ppsf = price / max(sqft, 1.0)

            # 1. Quality adjustment factor: adjust comp price for quality differential (5% per grade)
            qual_adjustment_pct = (target_qual - qual) * 0.05
            qual_adj = price * qual_adjustment_pct

            # 2. Size adjustment: 50% of comp's price per sq ft times area difference
            size_adj = 0.50 * comp_ppsf * (target_sqft - sqft)

            # Base adjusted price
            base_adjusted_price = price + qual_adj + size_adj

            # 3. Cross-neighborhood location scaling: ratio of target to comp neighborhood median ppsf (real rows only, clipped to 0.7-1.4)
            comp_neigh = str(data.get("Neighborhood", "")).strip()
            if target_neigh and comp_neigh and target_neigh.lower() != comp_neigh.lower():
                t_ppsf = neigh_ppsfs.get(target_neigh.lower())
                c_ppsf = neigh_ppsfs.get(comp_neigh.lower())
                if t_ppsf and c_ppsf and c_ppsf > 0:
                    loc_ratio = float(np.clip(t_ppsf / c_ppsf, 0.7, 1.4))
                else:
                    loc_ratio = 1.0
            else:
                loc_ratio = 1.0

            adjusted_price = base_adjusted_price * loc_ratio

            comp_prices.append(adjusted_price)
            comp_ppsfs.append(adjusted_price / max(target_sqft, 1.0))

            comp_summaries.append({
                "property_id": comp.get("property_id"),
                "neighborhood": data.get("Neighborhood"),
                "unadjusted_sale_price": price,
                "adjusted_sale_price": round(adjusted_price, 2),
                "living_area_sqft": int(sqft),
                "overall_quality": int(qual),
                "year_built": int(data.get("YearBuilt", 0)),
                "comparability_score": comp.get("comparability_score", 0.8),
            })

        comp_prices = np.array(comp_prices)
        comp_ppsfs = np.array(comp_ppsfs)

        median_price = float(np.median(comp_prices))
        mean_price = float(np.mean(comp_prices))
        median_ppsf = float(np.median(comp_ppsfs))

        # Size-weighted valuation: PPSF applied to target living area
        sqft_weighted_val = target_sqft * median_ppsf

        # Composite fair market value (50% comp median, 50% sqft-weighted)
        fmv = round((0.50 * median_price) + (0.50 * sqft_weighted_val), 2)

        # Standard deviation for range
        std_price = float(np.std(comp_prices)) if len(comp_prices) > 1 else fmv * 0.06
        range_low = round(max(20000.0, fmv - (1.28 * std_price)), 2)
        range_high = round(fmv + (1.28 * std_price), 2)

        # Confidence assessment
        avg_comp_score = np.mean([c.get("comparability_score", 0.7) for c in comparables])
        if avg_comp_score >= 0.80 and len(comparables) >= 3:
            confidence = "High (Strong comparable alignment)"
        elif avg_comp_score >= 0.65:
            confidence = "Moderate (Acceptable comparable alignment)"
        else:
            confidence = "Fair (Wide variance in comparable matches)"

        implied_ppsf = round(fmv / max(target_sqft, 1.0), 2)

        # Deterministic rule-based explanation
        explanation = (
            f"Based on {len(comparables)} prioritized comparable sales, the estimated fair market value "
            f"is ${fmv:,.2f} (${implied_ppsf:.2f}/sq ft), within a recommended valuation range of "
            f"${range_low:,.2f} to ${range_high:,.2f}. The median quality-adjusted comp sale price is "
            f"${median_price:,.2f}. The valuation reflects the subject property's {target_sqft:.0f} sq ft "
            f"living area and quality score of {target_qual:.0f}/10."
        )

        # If LLM key is present, generate expert narrative explanation
        if llm_client.is_live:
            try:
                system_prompt = (
                    "You are a Senior Certified Real Estate Appraiser. Provide a concise, authoritative "
                    "2-3 sentence valuation explanation justifying the fair market value based on comparable sales."
                )
                user_prompt = (
                    f"Subject Property: {target_sqft:.0f} sq ft, {target_qual:.0f}/10 quality rating in {target_features.get('neighborhood', 'Ames')}.\n"
                    f"Prioritized Comparable Sales Count: {len(comparables)}\n"
                    f"Adjusted Comp Median Sale Price: ${median_price:,.2f}\n"
                    f"Estimated Fair Market Value: ${fmv:,.2f} (${implied_ppsf:.2f}/sq ft)\n"
                    f"Valuation Range: ${range_low:,.2f} – ${range_high:,.2f}\n"
                    f"Write a concise professional explanation for the client valuation report."
                )
                llm_exp = llm_client.generate(prompt=user_prompt, system_prompt=system_prompt, temperature=0.2)
                if llm_exp and len(llm_exp.strip()) > 20:
                    explanation = llm_exp.strip()
                    logger.info("Generated live LLM valuation explanation.")
            except Exception as e:
                logger.warning(f"Error calling live LLM for explanation ({e}). Kept rule-based fallback.")

        return BasicValuationRecommendation(
            estimated_fair_market_value=fmv,
            valuation_range_low=range_low,
            valuation_range_high=range_high,
            median_comparable_price=round(median_price, 2),
            mean_comparable_price=round(mean_price, 2),
            implied_price_per_sqft=implied_ppsf,
            comparable_count_used=len(comparables),
            confidence_level=confidence,
            valuation_explanation=explanation,
            comparables_summary=comp_summaries,
        )

valuation_agent = BasicValuationAgent()
