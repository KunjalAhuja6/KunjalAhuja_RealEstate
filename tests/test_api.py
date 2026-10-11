import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

def test_guardrails_api():
    payload = {
        "features": {
            "gr_liv_area": -100,
            "bedrooms": 30,
        }
    }
    res = client.post("/api/v1/guardrails/validate", json=payload)
    assert res.status_code == 200
    assert res.json()["is_valid"] is False

def test_search_api():
    payload = {
        "query": "3 bedroom house in North Ames",
        "top_k": 3,
    }
    res = client.post("/api/v1/search", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["count"] == 3
    assert len(data["comparables"]) == 3

def test_search_api_with_empty_matching_filter():
    payload = {
        "query": "3 bedroom house in North Ames",
        "top_k": 3,
        "filters": {
            "neighborhood": "NonExistentPlaceXYZ",
            "min_bedrooms": 15,
        },
    }
    res = client.post("/api/v1/search", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["count"] == 0
    assert len(data["comparables"]) == 0

def test_classify_api():
    payload = {
        "property_features": {
            "neighborhood": "North Ames",
            "gr_liv_area": 1700,
            "overall_qual": 7,
            "year_built": 1995,
        }
    }
    res = client.post("/api/v1/classify", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "market_tier" in data

def test_predict_price_api():
    payload = {
        "property_features": {
            "neighborhood": "North Ames",
            "gr_liv_area": 1800,
            "overall_qual": 7,
            "year_built": 1995,
        }
    }
    res = client.post("/api/v1/valuation/predict-price", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "predicted_price" in data
    assert data["predicted_price"] > 100000

def test_collaborate_api():
    payload = {
        "property_features": {
            "neighborhood": "North Ames",
            "gr_liv_area": 1800,
            "overall_qual": 7,
            "year_built": 1995,
        },
        "query": "Valuation of 3 bedroom in North Ames",
        "top_k_comps": 3,
    }
    res = client.post("/api/v1/valuation/collaborate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "reconciled_fair_market_value" in data
    assert "judge_audit" in data
    assert "agent_trace" in data

def test_feedback_api():
    submission = {
        "property_id": "TEST-PROP",
        "recommended_value": 200000.0,
        "actual_valuation": 205000.0,
        "user_rating": 5,
        "appraiser_comments": "Test validation comment.",
        "professional_role": "Certified Appraiser",
    }
    post_res = client.post("/api/v1/feedback/submit", json=submission)
    assert post_res.status_code == 200

    get_res = client.get("/api/v1/feedback/summary")
    assert get_res.status_code == 200
    summary = get_res.json()
    assert summary["total_feedback_count"] >= 1
