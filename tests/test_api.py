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

def test_evaluation_history_api(tmp_path, monkeypatch):
    import src.services.evaluation_service as eval_srv
    test_hist_path = tmp_path / "eval_history.json"
    monkeypatch.setattr(eval_srv, "EVAL_HISTORY_PATH", test_hist_path)

    # 1. When history file does not exist, returns empty list
    res = client.get("/api/v1/evaluation/history")
    assert res.status_code == 200
    assert res.json() == []

    # 2. Save runs directly through save_eval_run and test history capping and timestamps
    sample_run1 = {"benchmark_status": "PASSED", "execution_time_seconds": 1.2}
    sample_run2 = {"benchmark_status": "PASSED", "execution_time_seconds": 1.5}
    sample_run3 = {"benchmark_status": "NOT_EVALUATED", "execution_time_seconds": 0.8}
    sample_run4 = {"benchmark_status": "PASSED", "execution_time_seconds": 2.1}

    saved1 = eval_srv.save_eval_run(sample_run1)
    assert "run_at" in saved1
    assert saved1["execution_time_seconds"] == 1.2

    eval_srv.save_eval_run(sample_run2)
    eval_srv.save_eval_run(sample_run3)
    eval_srv.save_eval_run(sample_run4)

    # File should have max 3 entries, newest first (run 4, 3, 2)
    res2 = client.get("/api/v1/evaluation/history")
    assert res2.status_code == 200
    history = res2.json()
    assert len(history) == 3
    assert history[0]["execution_time_seconds"] == 2.1
    assert history[1]["execution_time_seconds"] == 0.8
    assert history[2]["execution_time_seconds"] == 1.5
    for item in history:
        assert "run_at" in item
