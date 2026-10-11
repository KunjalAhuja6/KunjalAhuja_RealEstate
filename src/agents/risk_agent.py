from typing import Dict, Any, List
from pydantic import BaseModel, Field
from src.utils.logger import get_logger
from src.rag.data_pipeline import load_cleaned_data

logger = get_logger("RiskAgent")

class RiskPredictionResult(BaseModel):
    overall_risk_score: float  # 0 to 100
    risk_category: str  # Low, Moderate, High, Severe
    risk_factors: List[str]
    dimension_scores: Dict[str, float]
    mitigation_recommendations: List[str]
    summary: str

class PropertyValueRiskAgent:
    """
    Evaluates physical, age, market volatility, and layout risks that threaten
    property valuation accuracy and asset price stability.
    """
    def __init__(self):
        self.df = load_cleaned_data()
        self.neigh_volatility = self._compute_neighborhood_volatility()

    def _compute_neighborhood_volatility(self) -> Dict[str, float]:
        """Calculates normalized coefficient of variation for sales prices per neighborhood."""
        volatility = {}
        grouped = self.df.groupby("Neighborhood")["SalePrice"]
        for neigh, prices in grouped:
            mean = prices.mean()
            std = prices.std()
            cv = (std / mean) if mean > 0 else 0.25
            volatility[neigh] = float(cv)
        return volatility

    def predict_risk(
        self,
        features: Dict[str, Any],
        comparables: List[Dict[str, Any]] = None,
    ) -> RiskPredictionResult:
        """
        Calculates multi-dimensional property value risk score and flags.
        """
        sqft = float(features.get("gr_liv_area") or features.get("GrLivArea") or 1600)
        qual = int(features.get("overall_qual") or features.get("OverallQual") or 6)
        cond = int(features.get("overall_cond") or features.get("OverallCond") or 5)
        year_built = int(features.get("year_built") or features.get("YearBuilt") or 1990)
        year_remod = int(features.get("year_remod_add") or features.get("YearRemodAdd") or year_built)
        bedrooms = int(features.get("bedrooms") or features.get("BedroomAbvGr") or 3)
        neighborhood = str(features.get("neighborhood") or features.get("Neighborhood") or "").strip()

        risk_factors = []
        mitigations = []

        # 1. Age & Physical Depreciation Risk (Weight: 30%)
        current_year = 2026
        physical_age = current_year - year_built
        years_since_remod = current_year - year_remod

        if years_since_remod > 35:
            age_risk = 75.0
            risk_factors.append(f"Aging infrastructure: {years_since_remod} years since last remodel/major system update.")
            mitigations.append("Conduct comprehensive inspection of roof, HVAC, electrical wiring, and plumbing.")
        elif years_since_remod > 20:
            age_risk = 45.0
            risk_factors.append(f"Moderate mechanical wear: Property has not been modernized in {years_since_remod} years.")
            mitigations.append("Budget reserves for upcoming appliance and mechanical renewals.")
        else:
            age_risk = 15.0

        # 2. Quality-Condition Discrepancy Risk (Weight: 25%)
        # Large gap between architectural quality and physical condition
        gap = qual - cond
        if gap >= 3:
            qc_risk = 80.0
            risk_factors.append(f"Deferred maintenance alert: High architectural quality ({qual}/10) coupled with degraded condition ({cond}/10).")
            mitigations.append("Appraise immediate cosmetic and structural repair costs to avoid further value erosion.")
        elif cond <= 3:
            qc_risk = 75.0
            risk_factors.append(f"Substandard condition rating ({cond}/10) severely depresses buyer demand.")
            mitigations.append("Structure contingency clauses in financing reflecting poor condition.")
        elif gap <= -2:
            qc_risk = 40.0
            risk_factors.append(f"Over-improvement hazard: High condition ({cond}/10) on modest base quality ({qual}/10) may limit resale premium.")
            mitigations.append("Verify local ceiling prices to avoid over-capitalizing on finishes.")
        else:
            qc_risk = 20.0

        # 3. Neighborhood Price Volatility Risk (Weight: 25%)
        cv = self.neigh_volatility.get(neighborhood, 0.30)
        # CV typically ranges 0.15 to 0.50
        market_risk = min(100.0, max(10.0, (cv / 0.45) * 60.0))
        if market_risk > 50:
            risk_factors.append(f"Market dispersion: Neighborhood '{neighborhood}' exhibits high price volatility (Coeff. Var: {cv:.2f}).")
            mitigations.append("Rely on closely matched comps sold within the last 6 months to establish firm pricing.")

        # 4. Outlier Layout & Liquidity Risk (Weight: 20%)
        layout_risk = 15.0
        if bedrooms > 0 and (sqft / bedrooms) < 320:
            layout_risk = 65.0
            risk_factors.append(f"Cramped configuration: {bedrooms} bedrooms within {sqft:.0f} sq ft ({sqft/bedrooms:.0f} sq ft/bed).")
            mitigations.append("Assess functional utility and buyer willingness for compact layouts.")
        elif bedrooms <= 1 and sqft > 2000:
            layout_risk = 60.0
            risk_factors.append(f"Atypical layout: Only {bedrooms} bedroom in a large {sqft:.0f} sq ft property.")
            mitigations.append("Evaluate potential to partition space into standard multi-bedroom configuration.")

        # Overall Weighted Risk Score (0 - 100)
        overall_score = (
            (0.30 * age_risk)
            + (0.25 * qc_risk)
            + (0.25 * market_risk)
            + (0.20 * layout_risk)
        )
        overall_score = round(float(overall_score), 1)

        # Categorization
        if overall_score <= 30.0:
            risk_category = "Low Risk"
        elif overall_score <= 55.0:
            risk_category = "Moderate Risk"
        elif overall_score <= 75.0:
            risk_category = "High Risk"
        else:
            risk_category = "Severe Risk"

        if not risk_factors:
            risk_factors.append("No critical structural, age, or volatility vulnerabilities detected.")
        if not mitigations:
            mitigations.append("Standard property condition disclosure and routine pre-sale walk-through recommended.")

        summary = (
            f"Overall Risk Profile assessed as '{risk_category}' (Score: {overall_score}/100). "
            f"Key risk drivers: Age Depreciation ({age_risk:.0f}/100), Condition Gap ({qc_risk:.0f}/100), "
            f"and Market Volatility ({market_risk:.0f}/100)."
        )

        return RiskPredictionResult(
            overall_risk_score=overall_score,
            risk_category=risk_category,
            risk_factors=risk_factors,
            dimension_scores={
                "age_depreciation_risk": round(age_risk, 1),
                "condition_discrepancy_risk": round(qc_risk, 1),
                "market_volatility_risk": round(market_risk, 1),
                "layout_liquidity_risk": round(layout_risk, 1),
            },
            mitigation_recommendations=mitigations,
            summary=summary,
        )

risk_agent = PropertyValueRiskAgent()
