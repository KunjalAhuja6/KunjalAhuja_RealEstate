from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.rag.guardrails import validate_property_input, validate_natural_language_query, ValidationResult
from src.rag.hybrid_retriever import hybrid_retriever
from src.rag.data_pipeline import load_cleaned_data
from src.agents.classification_agent import classification_agent, ClassificationResult
from src.agents.valuation_agent import valuation_agent, BasicValuationRecommendation
from src.agents.risk_agent import risk_agent, RiskPredictionResult
from src.agents.judge_agent import judge_agent, JudgeValidationResult
from src.agents.orchestrator import orchestrator, ValuationDossier
from src.services.ml_prediction import price_predictor
from src.services.feature_impact import feature_impact_service
from src.services.feedback_service import feedback_service, FeedbackSubmission
from src.services.evaluation_service import evaluation_runner, load_eval_history, save_eval_run

router = APIRouter(prefix="/api/v1", tags=["Real Estate Valuation Assistant"])

# Request Schemas
class PropertyFeaturesPayload(BaseModel):
    neighborhood: Optional[str] = Field("North Ames", description="Neighborhood name")
    gr_liv_area: Optional[float] = Field(1600.0, description="Above ground living area in sq ft")
    bedrooms: Optional[int] = Field(3, description="Number of bedrooms above ground")
    bathrooms: Optional[float] = Field(2.0, description="Number of full bathrooms")
    year_built: Optional[int] = Field(1995, description="Year built")
    year_remod_add: Optional[int] = Field(1995, description="Year remodeled")
    overall_qual: Optional[int] = Field(6, description="Overall architectural quality rating (1-10)")
    overall_cond: Optional[int] = Field(5, description="Overall physical condition rating (1-10)")
    total_bsmt_sf: Optional[float] = Field(1000.0, description="Total basement square footage")
    garage_cars: Optional[int] = Field(2, description="Garage car capacity")
    garage_area: Optional[float] = Field(480.0, description="Garage square footage")
    lot_area: Optional[float] = Field(9000.0, description="Lot size in square feet")
    # Item 3: new optional key-field inputs
    total_rooms: Optional[int] = Field(None, description="Total rooms above grade (TotRmsAbvGrd)")
    garage_yr_blt: Optional[int] = Field(None, description="Year garage was built (GarageYrBlt)")

class GuardrailsRequest(BaseModel):
    features: Optional[Dict[str, Any]] = None
    query: Optional[str] = None

class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural language property query")
    top_k: int = Field(5, ge=1, le=50, description="Number of comparable properties to retrieve")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadata filters (location, age, rooms, sqft)")
    target_property: Optional[Dict[str, Any]] = None

class ClassifyRequest(BaseModel):
    property_features: PropertyFeaturesPayload
    query: Optional[str] = None

class BasicValuationRequest(BaseModel):
    property_features: PropertyFeaturesPayload
    query: Optional[str] = None
    top_k_comps: int = Field(5, ge=1, le=20)

class PricePredictionRequest(BaseModel):
    property_features: PropertyFeaturesPayload

class RiskPredictionRequest(BaseModel):
    property_features: PropertyFeaturesPayload

class WhatIfSimulationRequest(BaseModel):
    base_features: PropertyFeaturesPayload
    modifications: Dict[str, Any] = Field(..., description="e.g. {'add_sqft': 300, 'upgrade_quality': 1, 'remodel': True}")

class JudgeValidationRequest(BaseModel):
    property_features: PropertyFeaturesPayload
    recommended_price: float
    ml_predicted_price: float
    comp_median_price: float
    comp_count: int = 5
    risk_score: float = 30.0
    valuation_explanation: str = "Empirical comp-adjusted fair market valuation."

class CollaborationRequest(BaseModel):
    property_features: PropertyFeaturesPayload
    query: Optional[str] = None
    top_k_comps: int = Field(5, ge=1, le=20)
    filters: Optional[Dict[str, Any]] = None

# --- Endpoints ---

@router.post("/guardrails/validate", response_model=ValidationResult)
def validate_input(req: GuardrailsRequest):
    """
    Requirement 1.6: Input validation guardrails for property boundaries, injection, and relevance.
    """
    if req.query:
        q_res = validate_natural_language_query(req.query)
        if not q_res.is_valid:
            return q_res

    if req.features:
        return validate_property_input(req.features)

    return ValidationResult(is_valid=True)

@router.post("/search")
def search_comparables(req: SearchRequest):
    """
    Requirements 1.1, 1.2, 1.4, 1.7:
    Hybrid search (keywords + semantic) with metadata filtering and comparable-property prioritization.
    Respects filters that match nothing by returning an empty list.
    """
    # Guardrail check on query
    q_val = validate_natural_language_query(req.query)
    if not q_val.is_valid:
        raise HTTPException(status_code=400, detail="; ".join(q_val.errors))

    clean_target = None
    if req.target_property:
        t_val = validate_property_input(req.target_property)
        if not t_val.is_valid:
            raise HTTPException(status_code=400, detail="; ".join(t_val.errors))
        clean_target = t_val.sanitized_input

    results = hybrid_retriever.retrieve(
        query=req.query,
        top_k=req.top_k,
        filters=req.filters,
        target_property=clean_target,
    )
    return {
        "query": req.query,
        "count": len(results),
        "filters_applied": req.filters,
        "comparables": results,
    }

@router.post("/classify", response_model=ClassificationResult)
def classify_property(req: ClassifyRequest):
    """
    Requirement 1.3: Property classification agent.
    """
    feat_dict = req.property_features.model_dump()
    val_res = validate_property_input(feat_dict)
    if not val_res.is_valid:
        raise HTTPException(status_code=400, detail="; ".join(val_res.errors))

    return classification_agent.classify_property(
        features=val_res.sanitized_input,
        query=req.query,
    )

@router.post("/valuation/basic", response_model=BasicValuationRecommendation)
def basic_valuation(req: BasicValuationRequest):
    """
    Requirement 1.5: Basic property valuation recommendation.
    """
    feat_dict = req.property_features.model_dump()
    val_res = validate_property_input(feat_dict)
    if not val_res.is_valid:
        raise HTTPException(status_code=400, detail="; ".join(val_res.errors))

    query = req.query or f"{feat_dict.get('bedrooms', 3)} bed home in {feat_dict.get('neighborhood', 'Ames')}"
    comps = hybrid_retriever.retrieve(
        query=query,
        top_k=req.top_k_comps,
        target_property=val_res.sanitized_input,
    )

    return valuation_agent.recommend_valuation(
        target_features=val_res.sanitized_input,
        comparables=comps,
        user_query=query,
    )

@router.post("/valuation/predict-price")
def predict_property_price(req: PricePredictionRequest):
    """
    Requirement 2.2: Property price prediction via supervised regression models.
    """
    feat_dict = req.property_features.model_dump()
    val_res = validate_property_input(feat_dict)
    if not val_res.is_valid:
        raise HTTPException(status_code=400, detail="; ".join(val_res.errors))

    return price_predictor.predict(val_res.sanitized_input)

@router.post("/valuation/predict-risk", response_model=RiskPredictionResult)
def predict_property_risk(req: RiskPredictionRequest):
    """
    Requirement 2.5: Property value risk prediction across physical, age, and market volatility dimensions.
    """
    feat_dict = req.property_features.model_dump()
    val_res = validate_property_input(feat_dict)
    if not val_res.is_valid:
        raise HTTPException(status_code=400, detail="; ".join(val_res.errors))

    return risk_agent.predict_risk(val_res.sanitized_input)

@router.post("/valuation/impact-analysis")
def feature_and_location_impact(req: WhatIfSimulationRequest):
    """
    Requirement 2.6: Property feature and location impact analysis with what-if scenario simulation.
    """
    feat_dict = req.base_features.model_dump()
    val_res = validate_property_input(feat_dict)
    if not val_res.is_valid:
        raise HTTPException(status_code=400, detail="; ".join(val_res.errors))

    simulation = feature_impact_service.simulate_what_if_scenario(
        base_features=val_res.sanitized_input,
        modifications=req.modifications,
    )
    loc_impact = feature_impact_service.get_location_impact(feat_dict.get("neighborhood", "North Ames"))
    feat_importance = feature_impact_service.get_feature_importances()

    return {
        "simulation": simulation,
        "location_impact": loc_impact,
        "feature_importance": feat_importance,
    }

@router.post("/valuation/judge-validate", response_model=JudgeValidationResult)
def judge_validate(req: JudgeValidationRequest):
    """
    Requirement 2.3: LLM-as-judge for valuation validation.
    """
    feat_dict = req.property_features.model_dump()
    val_res = validate_property_input(feat_dict)
    if not val_res.is_valid:
        raise HTTPException(status_code=400, detail="; ".join(val_res.errors))

    return judge_agent.validate_valuation(
        target_features=val_res.sanitized_input,
        recommended_price=req.recommended_price,
        ml_predicted_price=req.ml_predicted_price,
        comp_median_price=req.comp_median_price,
        comp_count=req.comp_count,
        risk_score=req.risk_score,
        valuation_explanation=req.valuation_explanation,
    )

@router.post("/valuation/collaborate", response_model=ValuationDossier)
def multi_agent_collaborate(req: CollaborationRequest):
    """
    Requirement 2.4: Multi-agent property valuation collaboration.
    """
    feat_dict = req.property_features.model_dump()
    val_res = validate_property_input(feat_dict)
    if not val_res.is_valid:
        raise HTTPException(status_code=400, detail="; ".join(val_res.errors))

    try:
        return orchestrator.run_collaborative_valuation(
            property_features=val_res.sanitized_input,
            query=req.query,
            top_k_comps=req.top_k_comps,
            filters=req.filters,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/feedback/submit")
def submit_appraisal_feedback(feedback: FeedbackSubmission):
    """
    Requirement 2.7: Feedback loop for continuous improvement (submission).
    """
    return feedback_service.submit_feedback(feedback)

@router.get("/feedback/summary")
def get_feedback_summary():
    """
    Requirement 2.7: Feedback loop for continuous improvement (metrics & calibration factor).
    """
    return feedback_service.get_feedback_summary()

@router.get("/evaluation/run")
def run_evaluation_benchmark():
    """
    Requirement 2.1: DeepEval for recommendation quality evaluation.
    """
    return save_eval_run(evaluation_runner.run_benchmark())

@router.get("/evaluation/history")
def get_evaluation_history():
    """
    Returns the last 3 DeepEval evaluation runs, newest first.
    """
    return load_eval_history()

@router.get("/dataset/info")
def get_dataset_info():
    """
    Information and summary metadata for Ames Housing dataset.
    """
    df = load_cleaned_data()
    return {
        "dataset_name": "Ames Housing Dataset",
        "source": "OpenML Data ID 43926 / Dean De Cock (2011)",
        "total_records": len(df),
        "key_fields": [
            "SalePrice", "OverallQual", "OverallCond", "GrLivArea",
            "TotalBsmtSF", "GarageCars", "GarageArea", "YearBuilt",
            "YearRemodAdd", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd",
            "Neighborhood", "LotArea", "GarageYrBlt"
        ],
        "neighborhoods_count": int(df["Neighborhood"].nunique()),
        "price_statistics": {
            "min": float(df["SalePrice"].min()),
            "median": float(df["SalePrice"].median()),
            "mean": round(float(df["SalePrice"].mean()), 2),
            "max": float(df["SalePrice"].max()),
        },
    }
