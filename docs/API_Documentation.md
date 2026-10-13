# API Documentation: AI-Powered Real Estate Property Valuation Assistant

**Author / Presenter**: Kunjal Ahuja  
**Domain**: Real Estate & PropTech Valuation Intelligence  
**Capstone**: Prodapt AI/ML Engineering Capstone  

Base URL: `http://localhost:8000`  
API Prefix: `/api/v1`  
Interactive OpenAPI UI: `http://localhost:8000/docs`  
Total Endpoints: **16 REST Endpoints**  

---

## 1. Root System Status
- **Endpoint**: `/`
- **HTTP Method**: `GET`
- **Description**: Returns microservice identity, operational status, and interactive documentation link.
- **Request Format**: None
- **Sample Request**:
```bash
curl -X GET http://localhost:8000/
```
- **Sample Response**:
```json
{
  "status": "online",
  "service": "AI-Powered Real Estate Property Valuation Assistant",
  "docs": "/docs"
}
```

---

## 2. System Health Check
- **Endpoint**: `/health`
- **HTTP Method**: `GET`
- **Description**: Returns microservice operational health status and version identifier.
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

## 3. Input Validation Guardrails
- **Endpoint**: `/api/v1/guardrails/validate`
- **HTTP Method**: `POST`
- **Description**: Validates structured property features and natural-language user queries against physical boundaries (14 fields: living area 100–15,000 sq ft, quality 1–10, etc.) and detects prompt injection attacks.
- **Request Format**:
```json
{
  "features": {
    "neighborhood": "North Ames",
    "gr_liv_area": 1800,
    "bedrooms": 3,
    "bathrooms": 2,
    "year_built": 1995,
    "overall_qual": 7,
    "overall_cond": 6,
    "total_rooms": 7,
    "garage_yr_blt": 1995
  },
  "query": "3 bedroom house in North Ames"
}
```
- **Sample Response**:
```json
{
  "is_valid": true,
  "sanitized_input": {
    "neighborhood": "North Ames",
    "gr_liv_area": 1800.0,
    "bedrooms": 3,
    "bathrooms": 2.0,
    "year_built": 1995,
    "overall_qual": 7,
    "overall_cond": 6,
    "total_rooms": 7,
    "garage_yr_blt": 1995
  },
  "errors": [],
  "warnings": []
}
```

---

## 4. Hybrid Search & Comparable Prioritization
- **Endpoint**: `/api/v1/search`
- **HTTP Method**: `POST`
- **Description**: Executes true hybrid retrieval combining dense semantic embeddings (`all-MiniLM-L6-v2`, 384 dimensions) and lexical TF-IDF keyword search over the full candidate set ($\alpha = 0.55$). Supports metadata filtering, real-sales isolation (`include_synthetic=False` by default), and dynamic neighborhood widening.
- **Empty Filter Behavior**: When metadata filters match 0 properties, returns an empty list (`[]`) with `count: 0` instead of falling back to all properties.
- **Request Format**:
```json
{
  "query": "3 bedroom single family home in North Ames built after 1990",
  "top_k": 3,
  "filters": {
    "neighborhood": "North Ames",
    "min_bedrooms": 3,
    "min_year_built": 1990
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
    "min_bedrooms": 3,
    "min_year_built": 1990
  },
  "comparables": [
    {
      "property_id": "PROP-2619",
      "neighborhood": "North Ames",
      "sale_price": 172000.0,
      "living_area": 1656,
      "bedrooms": 3,
      "bathrooms": 2.0,
      "year_built": 1996,
      "overall_quality": 6,
      "hybrid_score": 0.8124,
      "comparability_score": 0.8841,
      "final_score": 0.8483
    }
  ]
}
```

---

## 5. Property Classification Agent
- **Endpoint**: `/api/v1/classify`
- **HTTP Method**: `POST`
- **Description**: Segments property profiles into Market Tiers, Architectural Typologies, Condition Preservation Grades, and Target Demographics.
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
    "overall_cond": 6
  }
}
```
- **Sample Response**:
```json
{
  "market_tier": "Upper-Middle Market",
  "typology": "Suburban Single-Family",
  "preservation_grade": "Grade B (Standard Modern)",
  "target_buyer_demographic": "Move-up families and professional households",
  "liquidity_assessment": "High liquidity (core suburban segment with active demand)"
}
```

---

## 6. Basic Valuation Recommendation Agent
- **Endpoint**: `/api/v1/valuation/basic`
- **HTTP Method**: `POST`
- **Description**: Calculates comp-adjusted median, mean, and size-weighted PPSF fair market value using retrieved historical comps with quality delta adjustments.
- **Request Format**:
```json
{
  "property_features": {
    "neighborhood": "North Ames",
    "gr_liv_area": 1600,
    "bedrooms": 3,
    "bathrooms": 2,
    "year_built": 1970,
    "overall_qual": 6,
    "overall_cond": 5
  },
  "top_k_comps": 5
}
```
- **Sample Response**:
```json
{
  "estimated_fair_market_value": 171200.0,
  "valuation_range_low": 136960.0,
  "valuation_range_high": 205440.0,
  "median_comparable_price": 171200.0,
  "mean_comparable_price": 170733.33,
  "implied_price_per_sqft": 107.0,
  "comparable_count_used": 3,
  "confidence_level": "High (Strong comparable alignment)",
  "valuation_explanation": "Based on 3 prioritized comparable sales in North Ames, the median comparable value is $171,200.00."
}
```

---

## 7. Supervised ML Price Prediction
- **Endpoint**: `/api/v1/valuation/predict-price`
- **HTTP Method**: `POST`
- **Description**: Computes statistical price prediction using a trained `GradientBoostingRegressor` ($R^2 = 0.9198$, $\text{RMSE} = \$25,361.84$) with 90% empirical prediction intervals.
- **Request Format**:
```json
{
  "property_features": {
    "neighborhood": "North Ames",
    "gr_liv_area": 1600,
    "bedrooms": 3,
    "bathrooms": 2,
    "year_built": 1970,
    "overall_qual": 6,
    "overall_cond": 5,
    "garage_cars": 2,
    "total_bsmt_sf": 1000
  }
}
```
- **Sample Response**:
```json
{
  "predicted_price": 172374.0,
  "lower_bound_90pct": 130689.86,
  "upper_bound_90pct": 214058.14,
  "price_per_sqft": 107.73,
  "model_r2_score": 0.9198,
  "model_used": "GradientBoostingRegressor",
  "input_features": {
    "OverallQual": 6.0,
    "OverallCond": 5.0,
    "GrLivArea": 1600.0,
    "TotalBsmtSF": 1000.0,
    "GarageCars": 2.0,
    "YearBuilt": 1970.0
  }
}
```

---

## 8. Property Value Risk Assessment
- **Endpoint**: `/api/v1/valuation/predict-risk`
- **HTTP Method**: `POST`
- **Description**: Assesses risk (0–100 index) across Physical Age Depreciation (30%), Quality-Condition Discrepancy (25%), Neighborhood Price Volatility (25%), and Layout Obsolescence (20%).
- **Request Format**:
```json
{
  "property_features": {
    "neighborhood": "North Ames",
    "gr_liv_area": 1600,
    "bedrooms": 3,
    "bathrooms": 2,
    "year_built": 1970,
    "overall_qual": 6,
    "overall_cond": 5
  }
}
```
- **Sample Response**:
```json
{
  "composite_risk_score": 41.3,
  "risk_tier": "Moderate Risk",
  "breakdown": {
    "physical_depreciation_risk": 56.0,
    "condition_gap_risk": 20.0,
    "market_volatility_risk": 35.0,
    "liquidity_risk": 40.0
  },
  "key_risk_drivers": [
    "Structure is 56 years old; mechanical systems may require capital reserves."
  ]
}
```

---

## 9. Feature Impact & What-If Renovation Analysis
- **Endpoint**: `/api/v1/valuation/impact-analysis`
- **HTTP Method**: `POST`
- **Description**: Computes neighborhood price premiums relative to the city median ($160,000) and calculates real-time valuation changes from planned renovations.
- **Request Format**:
```json
{
  "neighborhood": "North Ames",
  "base_features": {
    "neighborhood": "North Ames",
    "gr_liv_area": 1600,
    "bedrooms": 3,
    "bathrooms": 2,
    "year_built": 1970,
    "overall_qual": 6,
    "overall_cond": 5
  },
  "renovation_adjustments": {
    "added_sqft": 300,
    "quality_increase": 1
  }
}
```
- **Sample Response**:
```json
{
  "neighborhood_analysis": {
    "neighborhood": "North Ames",
    "median_price": 140000.0,
    "city_median_price": 160000.0,
    "premium_or_discount_pct": -12.5,
    "tier": "Mid-Market Value"
  },
  "what_if_simulation": {
    "baseline_valuation": 172374.0,
    "renovated_valuation": 205840.0,
    "value_lift_dollars": 33466.0,
    "value_lift_percentage": 19.42
  }
}
```

---

## 10. LLM-as-Judge Valuation Audit
- **Endpoint**: `/api/v1/valuation/judge-validate`
- **HTTP Method**: `POST`
- **Description**: Audits valuation consistency against historical comps and ML models, returning structured verdicts (`APPROVED`, `FLAGGED_OVERVALUED`, `FLAGGED_UNDERVALUED`, `NEEDS_REVIEW`). Rejects invalid inputs with HTTP 400.
- **Request Format**:
```json
{
  "reconciled_fmv": 171631.12,
  "comp_median": 171200.0,
  "ml_prediction": 172374.0,
  "confidence_interval": {
    "low": 131616.53,
    "high": 211645.71
  },
  "property_features": {
    "neighborhood": "North Ames",
    "gr_liv_area": 1600,
    "overall_qual": 6
  }
}
```
- **Sample Response**:
```json
{
  "verdict": "APPROVED",
  "audit_passed": true,
  "confidence_score": 0.95,
  "discrepancy_percentage": 0.43,
  "audit_rationale": "The reconciled valuation of $171,631.12 aligns closely with both the comp median ($171,200.00) and ML point estimate ($172,374.00), well within the 90% confidence interval."
}
```

---

## 11. Multi-Agent Valuation Orchestrator
- **Endpoint**: `/api/v1/valuation/collaborate`
- **HTTP Method**: `POST`
- **Description**: Orchestrates the complete multi-agent workflow: guardrails, classification, hybrid retrieval, dynamic `comp_fit` reconciliation, risk scoring, feature analysis, and LLM-as-Judge auditing.
- **Request Format**:
```json
{
  "property_features": {
    "neighborhood": "North Ames",
    "gr_liv_area": 1600,
    "bedrooms": 3,
    "bathrooms": 2,
    "year_built": 1970,
    "overall_qual": 6,
    "overall_cond": 5,
    "garage_cars": 2,
    "total_rooms": 7,
    "garage_yr_blt": 1970
  },
  "query": "Standard 3 bedroom home in North Ames built in 1970",
  "top_k_comps": 5
}
```
- **Sample Response**:
```json
{
  "reconciled_fair_market_value": 171631.12,
  "confidence_interval": {
    "low": 131616.53,
    "high": 211645.71
  },
  "comp_fit": 1.0,
  "comp_weight": 0.5,
  "ml_weight": 0.5,
  "widened_search_used": false,
  "classification": {
    "market_tier": "Mid-Market Standard",
    "typology": "Suburban Single-Family"
  },
  "basic_valuation": {
    "median_comparable_price": 171200.0,
    "comparables_summary": [
      {"property_id": "PROP-2619", "sale_price": 172000.0},
      {"property_id": "PROP-1613", "sale_price": 171200.0},
      {"property_id": "PROP-1928", "sale_price": 169000.0}
    ]
  },
  "ml_prediction": {
    "predicted_price": 172374.0,
    "model_r2_score": 0.9198
  },
  "risk_assessment": {
    "composite_risk_score": 41.3,
    "risk_tier": "Moderate Risk"
  },
  "judge_audit": {
    "verdict": "APPROVED",
    "audit_passed": true
  }
}
```

---

## 12. Continuous Improvement: Feedback Submission
- **Endpoint**: `/api/v1/feedback/submit`
- **HTTP Method**: `POST`
- **Description**: Captures appraiser evaluations, actual verified sales prices, user ratings (1–5 stars), and optional applied calibration factor to update model calibration metrics.
- **Request Format**:
```json
{
  "property_id": "PROP-2619",
  "recommended_value": 171631.12,
  "actual_sale_price": 172000.0,
  "user_rating": 5,
  "applied_calibration_factor": 1.0,
  "appraiser_comments": "Valuation aligned closely with executed purchase contract.",
  "professional_role": "Certified General Appraiser"
}
```
- **Sample Response**:
```json
{
  "feedback_id": "FB-20261011-849201",
  "status": "recorded",
  "error_dollars": 368.88,
  "error_pct": 0.21,
  "calibration_applied": 1.0
}
```

---

## 13. Continuous Improvement: Summary & Calibration Bias
- **Endpoint**: `/api/v1/feedback/summary`
- **HTTP Method**: `GET`
- **Description**: Returns aggregate feedback learning metrics, MAE, MAPE, uncalibrated systematic bias, and active dynamic calibration factor.
- **Request Format**: None
- **Sample Response**:
```json
{
  "total_feedback_count": 5,
  "average_rating": 4.8,
  "mean_absolute_error_dollars": 3412.5,
  "mean_absolute_percentage_error": 1.95,
  "systematic_bias_pct": 0.32,
  "calibration_factor": 1.0032,
  "recent_records": [...]
}
```

---

## 14. DeepEval Quality Benchmark Execution
- **Endpoint**: `/api/v1/evaluation/run`
- **HTTP Method**: `GET`
- **Description**: Executes the DeepEval recommendation quality benchmark test suite across 5 curated test scenarios (`EVAL-01` through `EVAL-05`). Operates honestly in dual modes. Automatically persists the run to `data/eval_history.json` (retaining the 3 most recent runs) and returns a `run_at` ISO timestamp in the response.
- **Request Format**: None

**Live LLM Mode Output (Measured from live execution)**:
```json
{
  "evaluation_framework": "DeepEval (v4.2.8)",
  "evaluation_mode": "live_llm_judge",
  "execution_time_seconds": 61.98,
  "total_scenarios_evaluated": 5,
  "passed_scenarios": 5,
  "pass_rate_percentage": 100.0,
  "mean_scores": {
    "mean_faithfulness": 0.900,
    "mean_answer_relevancy": 0.880,
    "mean_contextual_precision": 0.850,
    "mean_contextual_recall": 0.850,
    "mean_valuation_accuracy": 1.000
  },
  "benchmark_status": "PASSED",
  "run_at": "2026-10-12T12:00:00"
}
```

**Offline Mode Output (Without `OPENAI_API_KEY`)**:
```json
{
  "evaluation_framework": "DeepEval (v4.2.8)",
  "evaluation_mode": "offline_no_llm",
  "execution_time_seconds": 4.2,
  "total_scenarios_evaluated": 5,
  "passed_scenarios": null,
  "pass_rate_percentage": null,
  "mean_scores": {
    "mean_faithfulness": null,
    "mean_answer_relevancy": null,
    "mean_contextual_precision": null,
    "mean_contextual_recall": null,
    "mean_valuation_accuracy": 1.000
  },
  "benchmark_status": "NOT_EVALUATED",
  "run_at": "2026-10-12T12:00:00"
}
```

---

## 15. DeepEval Benchmark History
- **Endpoint**: `/api/v1/evaluation/history`
- **HTTP Method**: `GET`
- **Description**: Returns the last 3 saved DeepEval benchmark runs, ordered newest first. Returns an empty array `[]` if no runs have been executed yet or if history is missing.
- **Request Format**: None
- **Sample Request**:
```bash
curl -X GET http://localhost:8000/api/v1/evaluation/history
```
- **Sample Response**:
```json
[
  {
    "evaluation_framework": "DeepEval (v4.2.8)",
    "evaluation_mode": "live_llm_judge",
    "execution_time_seconds": 61.98,
    "total_scenarios_evaluated": 5,
    "passed_scenarios": 5,
    "pass_rate_percentage": 100.0,
    "mean_scores": {
      "mean_faithfulness": 0.9,
      "mean_answer_relevancy": 0.88,
      "mean_contextual_precision": 0.85,
      "mean_contextual_recall": 0.85,
      "mean_valuation_accuracy": 1.0
    },
    "benchmark_status": "PASSED",
    "scenario_results": [...],
    "run_at": "2026-10-12T12:00:00"
  }
]
```

---

## 16. Dataset Information & Summary
- **Endpoint**: `/api/v1/dataset/info`
- **HTTP Method**: `GET`
- **Description**: Returns dataset documentation, record counts (including synthetic augmentation), key fields, and statistical summary metrics.
- **Request Format**: None
- **Sample Request**:
```bash
curl -X GET http://localhost:8000/api/v1/dataset/info
```
- **Sample Response**:
```json
{
  "dataset_name": "Ames Housing Dataset",
  "source": "OpenML Data ID 43926 / Dean De Cock (2011)",
  "total_records": 10000,
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
    "mean": 180406.95,
    "max": 755000.0
  }
}
```
