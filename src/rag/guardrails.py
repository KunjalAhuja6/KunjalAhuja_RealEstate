import re
from typing import Dict, Any, Optional, Tuple, List
from pydantic import BaseModel, Field

class ValidationResult(BaseModel):
    is_valid: bool
    sanitized_input: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

INJECTION_PATTERNS = [
    r"ignore (all )?previous instructions",
    r"disregard (all )?(prior|previous)",
    r"you are now (an? )?(unrestricted|dan|evil)",
    r"jailbreak",
    r"system prompt",
    r"<script.*?>",
    r"drop table",
    r"delete from",
    r"base64_decode",
]

OFF_TOPIC_PATTERNS = [
    r"\b(write a poem|poem about|write code|python script|recipe for|bake a cake)\b",
    r"\b(who won the world cup|capital of france|write an essay)\b",
]

REAL_ESTATE_KEYWORDS = [
    "property", "house", "home", "condo", "townhouse", "sq ft", "square feet",
    "bedroom", "bed", "bath", "bathroom", "valuation", "value", "price", "sale",
    "neighborhood", "garage", "lot", "remodel", "built", "ames", "invest",
    "comp", "comparable", "market", "assessment", "appraisal", "qual", "cond"
]

def validate_property_input(data: Dict[str, Any]) -> ValidationResult:
    """
    Validates structured property features against physical and operational constraints.
    Passes every recognized field through explicit guardrail boundaries and sanitization.
    """
    errors = []
    warnings = []
    sanitized = {}

    # Normalize key lookup
    normalized_keys = {k.replace("_", "").lower(): (k, v) for k, v in data.items()}

    def get_val(canonical_name: str):
        lookup = canonical_name.replace("_", "").lower()
        if lookup in normalized_keys:
            return normalized_keys[lookup][1]
        return None

    # 1. Living Area (GrLivArea)
    raw_sqft = get_val("gr_liv_area")
    if raw_sqft is not None:
        try:
            val = float(raw_sqft)
            if val < 100 or val > 15000:
                errors.append(f"Living area ({val:.0f} sq ft) is outside realistic bounds [100 - 15,000 sq ft].")
            else:
                sanitized["gr_liv_area"] = val
                sanitized["GrLivArea"] = val
        except (ValueError, TypeError):
            errors.append("Living area must be a valid numeric value.")

    # 2. Bedrooms (BedroomAbvGr)
    raw_beds = get_val("bedrooms")
    if raw_beds is not None:
        try:
            val = int(raw_beds)
            if val < 0 or val > 15:
                errors.append(f"Bedrooms ({val}) must be between 0 and 15.")
            else:
                sanitized["bedrooms"] = val
                sanitized["BedroomAbvGr"] = val
        except (ValueError, TypeError):
            errors.append("Bedrooms count must be an integer.")

    # 3. Bathrooms (FullBath)
    raw_baths = get_val("bathrooms")
    if raw_baths is not None:
        try:
            val = float(raw_baths)
            if val < 0 or val > 10:
                errors.append(f"Bathrooms ({val}) must be between 0 and 10.")
            else:
                sanitized["bathrooms"] = val
                sanitized["FullBath"] = val
        except (ValueError, TypeError):
            errors.append("Bathrooms count must be a numeric value.")

    # 4. Year Built (YearBuilt)
    raw_yb = get_val("year_built")
    if raw_yb is not None:
        try:
            val = int(raw_yb)
            if val < 1850 or val > 2026:
                errors.append(f"Year built ({val}) must be between 1850 and 2026.")
            else:
                sanitized["year_built"] = val
                sanitized["YearBuilt"] = val
        except (ValueError, TypeError):
            errors.append("Year built must be a valid 4-digit year.")

    # 5. Year Remodeled (YearRemodAdd)
    raw_yra = get_val("year_remod_add")
    if raw_yra is not None:
        try:
            val = int(raw_yra)
            if val < 1850 or val > 2026:
                errors.append(f"Year remodeled ({val}) must be between 1850 and 2026.")
            else:
                if "year_built" in sanitized and val < sanitized["year_built"]:
                    warnings.append(f"Year remodeled ({val}) preceded year built ({sanitized['year_built']}); adjusted to year built.")
                    val = sanitized["year_built"]
                sanitized["year_remod_add"] = val
                sanitized["YearRemodAdd"] = val
        except (ValueError, TypeError):
            errors.append("Year remodeled must be a valid 4-digit year.")

    # 6. Overall Quality (1-10)
    raw_qual = get_val("overall_qual")
    if raw_qual is not None:
        try:
            val = int(raw_qual)
            if val < 1 or val > 10:
                errors.append(f"Overall quality rating ({val}) must be on a 1-10 scale.")
            else:
                sanitized["overall_qual"] = val
                sanitized["OverallQual"] = val
        except (ValueError, TypeError):
            errors.append("Overall quality must be an integer between 1 and 10.")

    # 7. Overall Condition (1-10)
    raw_cond = get_val("overall_cond")
    if raw_cond is not None:
        try:
            val = int(raw_cond)
            if val < 1 or val > 10:
                errors.append(f"Overall condition rating ({val}) must be on a 1-10 scale.")
            else:
                sanitized["overall_cond"] = val
                sanitized["OverallCond"] = val
        except (ValueError, TypeError):
            errors.append("Overall condition must be an integer between 1 and 10.")

    # 8. Garage Cars (GarageCars)
    raw_gc = get_val("garage_cars")
    if raw_gc is not None:
        try:
            val = int(raw_gc)
            if val < 0 or val > 6:
                errors.append(f"Garage capacity ({val} cars) must be between 0 and 6.")
            else:
                sanitized["garage_cars"] = val
                sanitized["GarageCars"] = val
        except (ValueError, TypeError):
            errors.append("Garage cars must be an integer.")

    # 9. Garage Area (GarageArea)
    raw_ga = get_val("garage_area")
    if raw_ga is not None:
        try:
            val = float(raw_ga)
            if val < 0 or val > 3000:
                errors.append(f"Garage area ({val:.0f} sq ft) must be between 0 and 3,000 sq ft.")
            else:
                sanitized["garage_area"] = val
                sanitized["GarageArea"] = val
        except (ValueError, TypeError):
            errors.append("Garage area must be a numeric value.")

    # 10. Total Basement SF (TotalBsmtSF)
    raw_bsmt = get_val("total_bsmt_sf")
    if raw_bsmt is not None:
        try:
            val = float(raw_bsmt)
            if val < 0 or val > 10000:
                errors.append(f"Basement area ({val:.0f} sq ft) must be between 0 and 10,000 sq ft.")
            else:
                sanitized["total_bsmt_sf"] = val
                sanitized["TotalBsmtSF"] = val
        except (ValueError, TypeError):
            errors.append("Basement area must be a numeric value.")

    # 11. Lot Area (LotArea)
    raw_lot = get_val("lot_area")
    if raw_lot is not None:
        try:
            val = float(raw_lot)
            if val < 0 or val > 500000:
                errors.append(f"Lot area ({val:.0f} sq ft) must be between 0 and 500,000 sq ft.")
            else:
                sanitized["lot_area"] = val
                sanitized["LotArea"] = val
        except (ValueError, TypeError):
            errors.append("Lot area must be a numeric value.")

    # 12. Neighborhood
    raw_neigh = get_val("neighborhood") or get_val("location")
    if raw_neigh:
        neigh_clean = re.sub(r"[^a-zA-Z0-9\s_-]", "", str(raw_neigh)).strip()
        if len(neigh_clean) == 0:
            errors.append("Neighborhood name contains invalid characters.")
        else:
            sanitized["neighborhood"] = neigh_clean
            sanitized["Neighborhood"] = neigh_clean

    # 13. Total Rooms Above Grade (TotRmsAbvGrd) — optional
    raw_rooms = get_val("total_rooms")
    if raw_rooms is not None:
        try:
            val = int(raw_rooms)
            if val < 1 or val > 30:
                errors.append(f"Total rooms ({val}) must be between 1 and 30.")
            else:
                sanitized["total_rooms"] = val
                sanitized["TotRmsAbvGrd"] = val
        except (ValueError, TypeError):
            errors.append("Total rooms must be an integer.")

    # 14. Garage Year Built (GarageYrBlt) — optional
    raw_gyrb = get_val("garage_yr_blt")
    if raw_gyrb is not None:
        try:
            val = int(raw_gyrb)
            if val < 1850 or val > 2026:
                errors.append(f"Garage year built ({val}) must be between 1850 and 2026.")
            else:
                sanitized["garage_yr_blt"] = val
                sanitized["GarageYrBlt"] = val
        except (ValueError, TypeError):
            errors.append("Garage year built must be a valid 4-digit year.")

    return ValidationResult(
        is_valid=len(errors) == 0,
        sanitized_input=sanitized,
        errors=errors,
        warnings=warnings,
    )

def validate_natural_language_query(query: str) -> ValidationResult:
    """
    Validates natural-language user queries against prompt injections, off-topic prompts,
    and minimal informative content.
    """
    errors = []
    warnings = []

    if not query or len(query.strip()) < 3:
        return ValidationResult(
            is_valid=False,
            errors=["Query is too short. Please provide details about the property or market request."],
        )

    q_lower = query.lower().strip()

    # Injection checks
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, q_lower, re.IGNORECASE):
            return ValidationResult(
                is_valid=False,
                errors=["Security guardrail triggered: Query contains prohibited adversarial or injection pattern."],
            )

    # Off-topic checks
    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, q_lower, re.IGNORECASE):
            return ValidationResult(
                is_valid=False,
                errors=["Relevance guardrail triggered: Assistant only answers real-estate valuation and property queries."],
            )

    has_keyword = any(kw in q_lower for kw in REAL_ESTATE_KEYWORDS) or any(char.isdigit() for char in q_lower)
    if not has_keyword and len(query.split()) > 4:
        warnings.append("Query may not contain typical real estate keywords; retrieval will proceed using semantic matching.")

    return ValidationResult(
        is_valid=True,
        sanitized_input={"query": query.strip()},
        errors=errors,
        warnings=warnings,
    )
