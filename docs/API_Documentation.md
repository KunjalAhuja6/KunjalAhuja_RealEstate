# API Documentation: AI-Powered Real Estate Property Valuation Assistant

Base URL: `http://localhost:8000`  
API Prefix: `/api/v1`  
Interactive OpenAPI UI: `http://localhost:8000/docs`

---

## 1. System Health Check
- **Endpoint**: `/health`
- **HTTP Method**: `GET`
- **Description**: Returns microservice operational status.
- **Request Format**: None
- **Sample Request**:
```bash
curl -X GET http://localhost:8000/health
```
- **Sample Response**:
```json
{
  "status": "healthy",
  "service": "AI-Powered Real Estate Property Valuation Assistant",
  "version": "1.0.0"
}
```

---

## 2. Input Validation Guardrails
- **Endpoint**: `/api/v1/guardrails/validate`
- **HTTP Method**: `POST`
- **Description**: Validates structured property features and natural-language user queries against physical boundaries, out-of-bounds parameters, and prompt injection attacks.
- **Request Format**:
```json
{
  "features": {
    "gr_liv_area": 1800,
    "bedrooms": 3,
    "bathrooms": 2,
    "year_built": 1995,
    "overall_qual": 7
  },
  "query": "3 bedroom house in North Ames"
}
```
- **Sample Response**:
```json
{
  "is_valid": true,
  "sanitized_input": {
    "gr_liv_area": 1800.0,
    "bedrooms": 3,
    "bathrooms": 2.0,
    "year_built": 1995,
    "overall_qual": 7
  },
  "errors": [],
  "warnings": []
}
```

---

## 3. Hybrid Search & Comparable-Property Prioritization
- **Endpoint**: `/api/v1/search`
- **HTTP Method**: `POST`
- **Description**: Executes true hybrid retrieval combining dense semantic embeddings (`all-MiniLM-L6-v2`) and lexical TF-IDF keyword search over the full filtered candidate set. Supports metadata filtering (including `min_total_rooms` and `max_total_rooms`) and multi-factor comparable-property prioritization ranking.
- **Empty Filter Behavior**: When metadata filters match 0 properties, returns an empty list (`[]`) with `count: 0` instead of falling back to all properties.
- **Validation**: Rejects invalid natural language queries or invalid `target_property` features with HTTP `400 Bad Request`.
- **Request Format**:
```json
{
  "query": "3 bedroom single family home in North Ames built after 1990",
  "top_k": 3,
  "filters": {
    "neighborhood": "North Ames",
    "min_bedrooms": 3,
    "min_total_rooms": 6,
    "max_total_rooms": 10
  },
  "target_property": {
    "neighborhood": "North Ames",
    "gr_liv_area": 1800,
    "bedrooms": 3,
    "bathrooms": 2,
    "year_built": 1995,
    "overall_qual": 7,
    "total_rooms": 7,
    "garage_yr_blt": 1995
  }
}
```
- **Sample Response**:
```json
{
  "query": "3 bedroom single family home in North Ames built after 1990",
  "count": 3,
  "filters_applied": {
    "neighborhood": "North Ames",
    "min_bedrooms": 3
  },
  "comparables": [
    {
      "property_id": "PROP-0604",
      "hybrid_score": 0.8945,
      "semantic_score": 0.8872,
      "keyword_score": 0.9034,
      "comparability_score": 0.8286,
      "prioritized_score": 0.8615,
      "comparability_details": {
        "location_similarity": 1.0,
        "sqft_similarity": 0.89,
        "quality_similarity": 0.89,
        "age_similarity": 0.82,
        "room_similarity": 0.88,
        "sqft_delta": -144.0,
        "price_per_sqft": 107.26
      },
      "data": {
        "Property_ID": "PROP-0604",
        "SalePrice": 177625.0,
        "OverallQual": 6,
        "OverallCond": 6,
        "GrLivArea": 1656.0,
        "YearBuilt": 1960,
        "FullBath": 2,
        "BedroomAbvGr": 3,
        "Neighborhood": "North Ames"
      },
      "text": "Property PROP-0604 located in North Ames neighborhood. Living area: 1656 sq ft..."
    }
  ]
}
```

---

## 4. Property Classification Agent
- **Endpoint**: `/api/v1/classify`
- **HTTP Method**: `POST`
- **Description**: Classifies properties into Market Tiers, Architectural Typologies, Condition Grades, and Investment Profiles.
- **Request Format**:
```json
{
  "property_features": {
    "neighborhood": "Stone Brook",
    "gr_liv_area": 2600,
    "bedrooms": 4,
    "bathrooms": 3,
    "year_built": 2005,
    "overall_qual": 9,
    "overall_cond": 7
  },
  "query": "What is the market tier for this luxury home?"
}
```
- **Sample Response**:
```json
{
  "market_tier": "Luxury / High-End",
  "property_typology": "Executive Estate / Sprawling Residence",
  "condition_grade": "Grade A (Pristine / Turnkey Modern)",
  "investment_profile": "High-Equity Core Wealth Preservation",
  "query_intent": "property_valuation_request",
  "classification_rationale": "Classified as Luxury / High-End (Executive Estate / Sprawling Residence) based on 2600 sq ft living area, quality score 9/10, built in 2005...",
  "target_demographic": "Affluent buyers, move-up executives seeking premium finishes."
}
```

---

## 5. Basic Property Valuation Recommendation
- **Endpoint**: `/api/v1/valuation/basic`
- **HTTP Method**: `POST`
- **Description**: Calculates comp-adjusted fair market value, median and mean comparable sale prices, size-weighted PPSF estimate, and valuation confidence range.
- **Request Format**:
```json
{
  "property_features": {
    "neighborhood": "North Ames",
    "gr_liv_area": 1800,
    "bedrooms": 3,
    "bathrooms": 2,
    "year_built": 1995,
    "overall_qual": 7
  },
  "top_k_comps": 5
}
```
- **Sample Response**:
```json
{
  "estimated_fair_market_value": 174250.0,
  "valuation_range_low": 142100.0,
  "valuation_range_high": 206400.0,
  "median_comparable_price": 172000.0,
  "mean_comparable_price": 175300.0,
  "implied_price_per_sqft": 96.81,
  "comparable_count_used": 5,
  "confidence_level": "High (Strong comparable alignment)",
  "valuation_explanation": "Based on 5 prioritized comparable sales, the estimated fair market value is $174,250.00...",
  "comparables_summary": [
    {
      "property_id": "PROP-0604",
      "neighborhood": "North Ames",
      "unadjusted_sale_price": 177625.0,
      "adjusted_sale_price": 186506.25,
      "living_area_sqft": 1656,
      "overall_quality": 6,
      "year_built": 1960,
      "comparability_score": 0.8286
    }
  ]
}
```

---

## 6. Supervised ML Price Prediction
- **Endpoint**: `/api/v1/valuation/predict-price`
- **HTTP Method**: `POST`
- **Description**: Computes statistical price prediction using a trained Gradient Boosting regressor with 90% confidence interval.
- **Request Format**:
```json
{
  "property_features": {
    "neighborhood": "North Ames",
    "gr_liv_area": 1800,
    "bedrooms": 3,
    "bathrooms": 2,
    "year_built": 1995,
    "overall_qual": 7,
    "overall_cond": 6,
    "garage_cars": 2,
    "total_bsmt_sf": 1000
  }
}
```
- **Sample Response**:
```json
{
  "predicted_price": 206940.44,
  "lower_bound_90pct": 161093.65,
  "upper_bound_90pct": 252787.23,
  "price_per_sqft": 114.97,
  "model_r2_score": 0.9031,
  "model_used": "GradientBoosting",
  "input_features": {
    "OverallQual": 7.0,
    "OverallCond": 6.0,
    "GrLivArea": 1800.0,
    "Neighborhood": "North Ames"
  }
}
```

---

## 7. Property Value Risk Prediction
- **Endpoint**: `/api/v1/valuation/predict-risk`
- **HTTP Method**: `POST`
- **Description**: Evaluates multi-dimensional property value risk score (0-100) across physical age depreciation, quality-condition discrepancy, market volatility, and layout obsolescence.
- **Request Format**:
```json
{
  "property_features": {
    "neighborhood": "North Ames",
    "gr_liv_area": 1800,
    "bedrooms": 3,
    "year_built": 1955,
    "year_remod_add": 1955,
    "overall_qual": 7,
    "overall_cond": 4
  }
}
```
- **Sample Response**:
```json
{
  "overall_risk_score": 62.4,
  "risk_category": "High Risk",
  "risk_factors": [
    "Aging infrastructure: 71 years since last remodel/major system update.",
    "Deferred maintenance alert: High architectural quality (7/10) coupled with degraded condition (4/10)."
  ],
  "dimension_scores": {
    "age_depreciation_risk": 75.0,
    "condition_discrepancy_risk": 80.0,
    "market_volatility_risk": 42.0,
    "layout_liquidity_risk": 15.0
  },
  "mitigation_recommendations": [
    "Conduct comprehensive inspection of roof, HVAC, electrical wiring, and plumbing.",
    "Appraise immediate cosmetic and structural repair costs to avoid further value erosion."
  ],
  "summary": "Overall Risk Profile assessed as 'High Risk' (Score: 62.4/100)..."
}
```

---

## 8. Feature & Location Impact Analysis
- **Endpoint**: `/api/v1/valuation/impact-analysis`
- **HTTP Method**: `POST`
- **Description**: Analyzes neighborhood price premiums and simulates what-if renovation/expansion scenarios.
- **Request Format**:
```json
{
  "base_features": {
    "neighborhood": "North Ames",
    "gr_liv_area": 1600,
    "bedrooms": 3,
    "overall_qual": 6,
    "year_built": 1990
  },
  "modifications": {
    "add_sqft": 300,
    "upgrade_quality": 1,
    "remodel": true
  }
}
```
- **Sample Response**:
```json
{
  "simulation": {
    "baseline_price": 172400.0,
    "modified_price": 218500.0,
    "dollar_value_added": 46100.0,
    "percentage_increase": 26.74,
    "modifications_applied": {
      "add_sqft": 300,
      "upgrade_quality": 1,
      "remodel": true
    }
  },
  "location_impact": {
    "neighborhood": "North Ames",
    "found": true,
    "location_premium_pct": -12.5,
    "dollar_premium_vs_city": -20000.0,
    "median_neighborhood_price": 140000.0,
    "city_baseline_median": 160000.0
  },
  "feature_importance": {
    "top_drivers": {
      "OverallQual": 0.54,
      "GrLivArea": 0.22,
      "TotalBsmtSF": 0.08
    }
  }
}
```

---

## 9. LLM-as-Judge Valuation Validation
- **Endpoint**: `/api/v1/valuation/judge-validate`
- **HTTP Method**: `POST`
- **Description**: Independent audit of valuation by appraisal judge agent evaluating market alignment, ML consistency, and risk coherence. Uses live LLM client when `OPENAI_API_KEY` is present, with deterministic rubric evaluation as fallback.
- **Validation**: Enforces strict input validation on `property_features` (HTTP `400 Bad Request` on invalid/out-of-bounds fields).
- **Request Format**:
```json
{
  "property_features": {
    "neighborhood": "North Ames",
    "gr_liv_area": 1800,
    "overall_qual": 7,
    "total_rooms": 7,
    "garage_yr_blt": 1995
  },
  "recommended_price": 185000.0,
  "ml_predicted_price": 190000.0,
  "comp_median_price": 182000.0,
  "comp_count": 5,
  "risk_score": 28.0,
  "valuation_explanation": "Comp-adjusted market estimate."
}
```
- **Sample Response**:
```json
{
  "verdict": "APPROVED",
  "confidence_score": 0.945,
  "rubric_scores": {
    "market_comp_alignment": 1.0,
    "ml_model_consistency": 1.0,
    "risk_coherence": 0.95,
    "logical_integrity": 0.95
  },
  "audit_commentary": "AUDIT VERDICT: APPROVED. Proposed valuation of $185,000.00 is well-grounded in market comps (median: $182,000.00) and conforms within 2.6% of ML predictive regression ($190,000.00).",
  "recommended_adjustment": null,
  "flags": []
}
```

---

## 10. Multi-Agent Valuation Collaboration
- **Endpoint**: `/api/v1/valuation/collaborate`
- **HTTP Method**: `POST`
- **Description**: Orchestrates the complete collaborative workflow across all agents and services into a unified Valuation Dossier. Employs real sales only for comparables, automatically widens search city-wide when fewer than 3 neighborhood comps match size/quality bounds (`comps_widened`), and applies dynamic `comp_fit` weighting for reconciliation.
- **New Fields**:
  - `comps_widened` (bool): `true` if search was automatically widened across Ames due to low neighborhood comp sample size/quality alignment.
  - `comp_fit` (float, 0.0 to 1.0): Measure of comparable physical fit alignment against the subject property, scaling comp weight between 0.10 and 0.50.
- **Validation**: Rejects invalid property features with HTTP `400 Bad Request`.
- **Request Format**:
```json
{
  "property_features": {
    "neighborhood": "North Ames",
    "gr_liv_area": 1800,
    "bedrooms": 3,
    "bathrooms": 2,
    "year_built": 1995,
    "overall_qual": 7,
    "overall_cond": 6,
    "garage_cars": 2,
    "total_rooms": 7,
    "garage_yr_blt": 1995
  },
  "query": "Valuation of 3 bedroom home in North Ames built 1995",
  "top_k_comps": 5
}
```
- **Sample Response**:
```json
{
  "is_valid": true,
  "reconciled_fair_market_value": 187552.45,
  "applied_calibration_factor": 1.0045,
  "comps_widened": false,
  "comp_fit": 1.0,
  "confidence_interval": {
    "low": 128363.13,
    "high": 252787.23
  },
  "classification": {
    "market_tier": "Upper-Middle Market",
    "property_typology": "Suburban Single-Family Family Residence",
    "condition_grade": "Grade C (Functional Average)",
    "investment_profile": "Core Suburban Appreciation"
  },
  "ml_prediction": {
    "predicted_price": 206940.44,
    "model_r2_score": 0.9031
  },
  "basic_valuation": {
    "estimated_fair_market_value": 168164.46,
    "median_comparable_price": 164000.0
  },
  "risk_assessment": {
    "overall_risk_score": 28.8,
    "risk_category": "Low Risk"
  },
  "judge_audit": {
    "verdict": "APPROVED",
    "confidence_score": 0.88
  },
  "agent_trace": [
    {
      "step_name": "1. Input Guardrails",
      "agent_name": "Guardrail Validator",
      "action_taken": "Sanitized and validated property attributes across all fields",
      "output_summary": "Features verified across 11 dimensions."
    }
  ],
  "final_executive_summary": "MULTI-AGENT VALUATION DOSSIER: Reconciled FMV: $187,552.45..."
}
```

---

## 11. Feedback Loop: Submit Appraisal Outcome
- **Endpoint**: `/api/v1/feedback/submit`
- **HTTP Method**: `POST`
- **Description**: Submits professional appraisal feedback, verified transaction prices, ratings, and optional `applied_calibration_factor` (default 1.0) so systematic error is evaluated against the uncalibrated model estimate without drifting.
- **Request Format**:
```json
{
  "property_id": "PROP-0604",
  "recommended_value": 172000.0,
  "actual_valuation": 177625.0,
  "applied_calibration_factor": 1.0,
  "user_rating": 5,
  "appraiser_comments": "Close alignment with closing price; minor kitchen update accounted for difference.",
  "professional_role": "Certified Appraiser"
}
```
- **Sample Response**:
```json
{
  "feedback_id": "FB-004",
  "timestamp": "2026-10-06T22:15:00",
  "property_id": "PROP-0604",
  "recommended_value": 172000.0,
  "actual_valuation": 177625.0,
  "applied_calibration_factor": 1.0,
  "uncalibrated_recommended_value": 172000.0,
  "error_dollars": 5625.0,
  "error_pct": 3.27,
  "user_rating": 5,
  "appraiser_comments": "Close alignment with closing price...",
  "professional_role": "Certified Appraiser"
}
```

---

## 12. Feedback Loop: Summary & Calibration Bias
- **Endpoint**: `/api/v1/feedback/summary`
- **HTTP Method**: `GET`
- **Description**: Returns continuous improvement learning metrics, historical MAE, MAPE, and systematic bias calibration factor.
- **Request Format**: None
- **Sample Response**:
```json
{
  "total_feedback_count": 4,
  "average_rating": 4.75,
  "mean_absolute_error_dollars": 5031.25,
  "mean_absolute_percentage_error": 2.76,
  "systematic_bias_pct": 0.45,
  "calibration_factor": 1.0045,
  "recent_records": [...]
}
```

---

## 13. DeepEval Quality Benchmark Execution
- **Endpoint**: `/api/v1/evaluation/run`
- **HTTP Method**: `GET`
- **Description**: Executes the DeepEval recommendation quality benchmark test suite across 5 curated test scenarios. Response depends on whether `OPENAI_API_KEY` is configured.

**Offline mode (no API key) — `evaluation_mode = "offline_no_llm"`**:
```json
{
  "evaluation_framework": "DeepEval (v4.2.8)",
  "evaluation_mode": "offline_no_llm",
  "execution_time_seconds": 4.1,
  "total_scenarios_evaluated": 5,
  "passed_scenarios": null,
  "pass_rate_percentage": null,
  "mean_scores": {
    "mean_faithfulness": null,
    "mean_answer_relevancy": null,
    "mean_contextual_precision": null,
    "mean_contextual_recall": null,
    "mean_valuation_accuracy": 0.893
  },
  "benchmark_status": "NOT_EVALUATED"
}
```
The four LLM-backed DeepEval metrics are skipped (`deepeval_metric_status: "SKIPPED_NO_LLM_KEY"` per scenario).

**Live mode (`OPENAI_API_KEY` set) — `evaluation_mode = "live_llm_judge"`**:
```json
{
  "evaluation_framework": "DeepEval (v4.2.8)",
  "evaluation_mode": "live_llm_judge",
  "execution_time_seconds": 18.4,
  "total_scenarios_evaluated": 5,
  "passed_scenarios": 4,
  "pass_rate_percentage": 80.0,
  "mean_scores": {
    "mean_faithfulness": 0.82,
    "mean_answer_relevancy": 0.79,
    "mean_contextual_precision": 0.76,
    "mean_contextual_recall": 0.73,
    "mean_valuation_accuracy": 0.893
  },
  "benchmark_status": "PASSED"
}
```
> Note: Live score values are illustrative. Actual scores depend on the LLM provider and run-time.

---

## 14. Dataset Information & Summary
- **Endpoint**: `/api/v1/dataset/info`
- **HTTP Method**: `GET`
- **Description**: Returns dataset documentation and statistical properties for the Ames Housing dataset.
- **Request Format**: None
- **Sample Response**:
```json
{
  "dataset_name": "Ames Housing Dataset",
  "source": "OpenML Data ID 43926 / Dean De Cock (2011)",
  "total_records": 2930,
  "key_fields": [
    "SalePrice", "OverallQual", "OverallCond", "GrLivArea",
    "TotalBsmtSF", "GarageCars", "GarageArea", "YearBuilt",
    "YearRemodAdd", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd",
    "Neighborhood", "LotArea", "GarageYrBlt"
  ],
  "neighborhoods_count": 28,
  "price_statistics": {
    "min": 12789.0,
    "median": 160000.0,
    "mean": 180796.06,
    "max": 755000.0
  }
}
```
