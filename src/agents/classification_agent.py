from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from src.utils.logger import get_logger

logger = get_logger("ClassificationAgent")

class ClassificationResult(BaseModel):
    market_tier: str
    property_typology: str
    condition_grade: str
    investment_profile: str
    query_intent: str
    classification_rationale: str
    target_demographic: str

class PropertyClassificationAgent:
    """
    Agent responsible for classifying property market segment, physical typology,
    preservation grade, investment profile, and user query intent.
    """
    def __init__(self):
        pass

    def classify_query_intent(self, query: str) -> str:
        """Determines the user's primary analytical intent from natural language query."""
        q_lower = query.lower()
        if any(term in q_lower for term in ["why differ", "different value", "explain the difference", "discrepancy", "compare two"]):
            return "price_discrepancy_analysis"
        elif any(term in q_lower for term in ["what is the value", "estimate", "expected property value", "worth", "valuation", "how much"]):
            return "property_valuation_request"
        elif any(term in q_lower for term in ["similar", "comparable", "comps", "find properties", "sold previously"]):
            return "comparable_property_search"
        elif any(term in q_lower for term in ["trend", "neighborhood", "market", "average price"]):
            return "market_segment_inquiry"
        return "general_property_inquiry"

    def classify_property(
        self,
        features: Dict[str, Any],
        query: Optional[str] = None,
        estimated_price: Optional[float] = None,
    ) -> ClassificationResult:
        """
        Classifies the property along market, architectural, and financial dimensions.
        """
        sqft = float(features.get("gr_liv_area") or features.get("GrLivArea") or 1600)
        qual = int(features.get("overall_qual") or features.get("OverallQual") or 6)
        cond = int(features.get("overall_cond") or features.get("OverallCond") or 5)
        year_built = int(features.get("year_built") or features.get("YearBuilt") or 1990)
        year_remod = int(features.get("year_remod_add") or features.get("YearRemodAdd") or year_built)
        bedrooms = int(features.get("bedrooms") or features.get("BedroomAbvGr") or 3)

        # 1. Market Tier
        price_ref = estimated_price if estimated_price is not None else (sqft * 125 * (qual / 6.0))
        if price_ref >= 300000 or qual >= 8:
            market_tier = "Luxury / High-End"
            target_demo = "Affluent buyers, move-up executives seeking premium finishes."
        elif price_ref >= 215000:
            market_tier = "Upper-Middle Market"
            target_demo = "Established families seeking suburban comfort and updated amenities."
        elif price_ref >= 150000:
            market_tier = "Mid-Market Standard"
            target_demo = "First-time family buyers and relocations seeking value."
        else:
            market_tier = "Affordable / Entry-Level"
            target_demo = "First-time homeowners, budget buyers, or long-term rental investors."

        # 2. Typology
        if sqft >= 2400 and qual >= 7:
            typology = "Executive Estate / Sprawling Residence"
        elif sqft >= 1500 and bedrooms >= 3:
            typology = "Suburban Single-Family Family Residence"
        elif sqft < 1200 and bedrooms <= 2:
            typology = "Starter Home / Compact Urban Residence"
        elif year_built < 1970 and sqft < 1600:
            typology = "Classic Mid-Century Ranch"
        else:
            typology = "Traditional Multi-Room Residential"

        # 3. Condition Grade
        effective_age = 2026 - max(year_built, year_remod)
        if effective_age <= 15 and qual >= 7 and cond >= 6:
            condition_grade = "Grade A (Pristine / Turnkey Modern)"
        elif effective_age <= 30 and cond >= 5:
            condition_grade = "Grade B (Well-Maintained Standard)"
        elif qual <= 4 or cond <= 4 or (effective_age > 50 and year_remod == year_built):
            condition_grade = "Grade D (Value-Add / Renovation Candidate)"
        else:
            condition_grade = "Grade C (Functional Average)"

        # 4. Investment Profile
        if "Grade D" in condition_grade:
            investment_profile = "Value-Add / Fix-and-Flip Opportunity"
        elif "Luxury" in market_tier:
            investment_profile = "High-Equity Core Wealth Preservation"
        elif "Affordable" in market_tier:
            investment_profile = "High Cash-Flow Rental Investment"
        else:
            investment_profile = "Core Suburban Appreciation"

        # 5. Query Intent
        intent = self.classify_query_intent(query) if query else "property_valuation_request"

        rationale = (
            f"Classified as {market_tier} ({typology}) based on {sqft:.0f} sq ft living area, "
            f"quality score {qual}/10, built in {year_built} (remodeled {year_remod}). "
            f"Condition assessed at {condition_grade} indicating {investment_profile} profile."
        )

        return ClassificationResult(
            market_tier=market_tier,
            property_typology=typology,
            condition_grade=condition_grade,
            investment_profile=investment_profile,
            query_intent=intent,
            classification_rationale=rationale,
            target_demographic=target_demo,
        )

classification_agent = PropertyClassificationAgent()
