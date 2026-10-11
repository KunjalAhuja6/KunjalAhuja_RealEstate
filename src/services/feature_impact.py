from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from src.utils.config import settings
from src.utils.logger import get_logger
from src.rag.data_pipeline import load_cleaned_data
from src.services.ml_prediction import price_predictor

logger = get_logger("FeatureImpactService")

class FeatureAndLocationImpactService:
    """
    Analyzes the quantitative impact of property physical features and geographical location
    on real estate valuation, including location premiums, feature elasticity, and what-if simulations.
    """
    def __init__(self):
        self.df = load_cleaned_data()
        self.city_median_price = float(self.df["SalePrice"].median())
        self.city_mean_price = float(self.df["SalePrice"].mean())
        self.city_mean_ppsf = float(self.df["PricePerSqFt"].mean())
        self.neighborhood_stats = self._compute_neighborhood_stats()

    def _compute_neighborhood_stats(self) -> Dict[str, Dict[str, Any]]:
        """Precomputes baseline metrics, premiums, and discounts by neighborhood."""
        stats = {}
        grouped = self.df.groupby("Neighborhood")
        for neigh, group in grouped:
            med = float(group["SalePrice"].median())
            mean = float(group["SalePrice"].mean())
            mean_ppsf = float(group["PricePerSqFt"].mean())
            count = int(len(group))
            premium_pct = round(((med - self.city_median_price) / self.city_median_price) * 100, 2)
            dollar_premium = round(med - self.city_median_price, 2)

            stats[neigh] = {
                "sample_count": count,
                "median_price": round(med, 2),
                "mean_price": round(mean, 2),
                "mean_price_per_sqft": round(mean_ppsf, 2),
                "location_premium_pct": premium_pct,
                "dollar_premium_vs_city": dollar_premium,
                "tier": "Premium Location" if premium_pct > 20 else ("Affordable / Value" if premium_pct < -15 else "Market Average"),
            }
        return stats

    def get_location_impact(self, neighborhood: str) -> Dict[str, Any]:
        """Returns location impact analysis for a specific neighborhood."""
        neigh_clean = neighborhood.replace("_", " ").strip()
        # Find exact or best match
        matched_key = None
        for k in self.neighborhood_stats.keys():
            if k.lower() == neigh_clean.lower() or neigh_clean.lower() in k.lower():
                matched_key = k
                break

        if not matched_key:
            return {
                "neighborhood": neighborhood,
                "found": False,
                "message": f"Neighborhood '{neighborhood}' not found in Ames historical dataset.",
                "city_baseline_median": self.city_median_price,
            }

        data = self.neighborhood_stats[matched_key]
        return {
            "neighborhood": matched_key,
            "found": True,
            "sample_sales_count": data["sample_count"],
            "median_neighborhood_price": data["median_price"],
            "city_median_price": self.city_median_price,
            "city_baseline_median": self.city_median_price,
            "location_premium_pct": data["location_premium_pct"],
            "dollar_premium_vs_city": data["dollar_premium_vs_city"],
            "tier": data["tier"],
            "mean_price_per_sqft": data["mean_price_per_sqft"],
            "city_mean_price_per_sqft": round(self.city_mean_ppsf, 2),
        }

    def get_feature_importances(self) -> Dict[str, Any]:
        """Returns global feature importance rankings from the predictive model."""
        importances = price_predictor.feature_importance
        return {
            "top_drivers": importances,
            "marginal_estimates": {
                "per_sqft_living_area": round(float(self.df["PricePerSqFt"].median()), 2),
                "per_quality_point": 24000.0,
                "per_garage_car_stall": 12500.0,
                "per_full_bathroom": 8500.0,
                "remodel_modernization_uplift": 18000.0,
            },
        }

    def simulate_what_if_scenario(
        self,
        base_features: Dict[str, Any],
        modifications: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Simulates what-if scenarios (e.g. adding 300 sq ft, upgrading quality rating, remodeling).
        Computes baseline price, modified price, value added, and ROI insight.
        """
        # Baseline prediction
        base_pred = price_predictor.predict(base_features)
        base_price = base_pred["predicted_price"]

        # Apply modifications
        modified_features = dict(base_features)
        if "add_sqft" in modifications:
            cur_sqft = float(modified_features.get("gr_liv_area") or modified_features.get("GrLivArea") or 1500)
            modified_features["GrLivArea"] = cur_sqft + float(modifications["add_sqft"])
            modified_features["gr_liv_area"] = modified_features["GrLivArea"]

        if "upgrade_quality" in modifications:
            cur_qual = int(modified_features.get("overall_qual") or modified_features.get("OverallQual") or 6)
            new_qual = min(10, cur_qual + int(modifications["upgrade_quality"]))
            modified_features["OverallQual"] = new_qual
            modified_features["overall_qual"] = new_qual

        if "add_garage_cars" in modifications:
            cur_cars = int(modified_features.get("garage_cars") or modified_features.get("GarageCars") or 1)
            modified_features["GarageCars"] = min(5, cur_cars + int(modifications["add_garage_cars"]))
            modified_features["garage_cars"] = modified_features["GarageCars"]

        if "add_bathrooms" in modifications:
            cur_baths = float(modified_features.get("bathrooms") or modified_features.get("FullBath") or 1)
            modified_features["FullBath"] = cur_baths + float(modifications["add_bathrooms"])
            modified_features["bathrooms"] = modified_features["FullBath"]

        if modifications.get("remodel", False):
            modified_features["YearRemodAdd"] = 2024
            modified_features["year_remod_add"] = 2024

        modified_pred = price_predictor.predict(modified_features)
        new_price = modified_pred["predicted_price"]
        dollar_gain = new_price - base_price
        pct_gain = round((dollar_gain / base_price) * 100, 2)

        return {
            "baseline_price": base_price,
            "modified_price": new_price,
            "dollar_value_added": round(dollar_gain, 2),
            "percentage_increase": pct_gain,
            "modifications_applied": modifications,
            "base_features": base_features,
            "modified_features": modified_features,
        }

feature_impact_service = FeatureAndLocationImpactService()
