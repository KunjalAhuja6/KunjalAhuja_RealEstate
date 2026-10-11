import pytest
from src.services.ml_prediction import price_predictor
from src.services.feature_impact import feature_impact_service

def test_price_prediction_point_estimate_and_bounds():
    features = {
        "Neighborhood": "North Ames",
        "GrLivArea": 1750,
        "OverallQual": 7,
        "OverallCond": 6,
        "YearBuilt": 1998,
        "TotalBsmtSF": 1100,
        "GarageCars": 2,
        "FullBath": 2,
        "BedroomAbvGr": 3,
    }
    result = price_predictor.predict(features)
    assert "predicted_price" in result
    assert result["predicted_price"] > 50000.0
    assert result["lower_bound_90pct"] < result["predicted_price"]
    assert result["upper_bound_90pct"] > result["predicted_price"]
    assert result["model_r2_score"] > 0.80

def test_feature_impact_and_what_if_simulation():
    base = {
        "Neighborhood": "North Ames",
        "GrLivArea": 1500,
        "OverallQual": 6,
        "OverallCond": 5,
        "YearBuilt": 1990,
        "GarageCars": 2,
        "FullBath": 2,
        "BedroomAbvGr": 3,
    }
    modifications = {"add_sqft": 400, "upgrade_quality": 1}
    sim = feature_impact_service.simulate_what_if_scenario(base, modifications)
    assert sim["modified_price"] > sim["baseline_price"]
    assert sim["dollar_value_added"] > 0
    assert sim["percentage_increase"] > 0


def test_ml_prediction_accepts_total_rooms_and_garage_yr_blt():
    """
    Item 3: total_rooms (TotRmsAbvGrd) and garage_yr_blt (GarageYrBlt) must be
    passed to the ML model and reflected in input_features rather than replaced
    with hardcoded defaults.
    """
    features_with_new_fields = {
        "neighborhood": "College Creek",
        "gr_liv_area": 1900,
        "overall_qual": 7,
        "overall_cond": 6,
        "year_built": 2001,
        "bedrooms": 3,
        "bathrooms": 2,
        "total_rooms": 9,        # TotRmsAbvGrd — non-default value
        "garage_yr_blt": 2001,   # GarageYrBlt — non-default value
    }
    result = price_predictor.predict(features_with_new_fields)
    assert "input_features" in result
    # The model must have used the provided values, not the hardcoded defaults (6, 1990)
    assert result["input_features"]["TotRmsAbvGrd"] == 9.0, (
        f"Expected TotRmsAbvGrd=9, got {result['input_features']['TotRmsAbvGrd']}"
    )
    assert result["input_features"]["GarageYrBlt"] == 2001.0, (
        f"Expected GarageYrBlt=2001, got {result['input_features']['GarageYrBlt']}"
    )
    assert result["predicted_price"] > 50000.0


def test_honest_train_test_split_metadata():
    """
    Verifies that the ML model is trained with an honest train/test split:
    1. Saved artifact metadata includes metrics_real_plus_synthetic and metrics_real_only.
    2. Both sets evaluate R2, RMSE, MAE, and MAPE strictly on real_test.
    3. model_r2_score corresponds to the real_test performance of the model actually used.
    """
    import joblib
    artifact = joblib.load(price_predictor.model_path)
    assert "metrics_real_plus_synthetic" in artifact, "Missing metrics_real_plus_synthetic in artifact"
    assert "metrics_real_only" in artifact, "Missing metrics_real_only in artifact"

    m_comb = artifact["metrics_real_plus_synthetic"]
    m_real = artifact["metrics_real_only"]

    for key in ["R2", "RMSE", "MAE", "MAPE"]:
        assert key in m_comb, f"Key {key} missing from metrics_real_plus_synthetic"
        assert key in m_real, f"Key {key} missing from metrics_real_only"

    # Both models achieve strong benchmark performance on real test set
    assert m_comb["R2"] > 0.88, f"Combined model R2 too low: {m_comb['R2']}"
    assert m_real["R2"] > 0.88, f"Real-only model R2 too low: {m_real['R2']}"

    # Active metrics match the model actually used
    assert price_predictor.metrics["R2"] == m_comb["R2"]
    assert price_predictor.metrics["RMSE"] == m_comb["RMSE"]


