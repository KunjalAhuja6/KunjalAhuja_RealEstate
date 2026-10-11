import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

from src.utils.config import settings
from src.utils.logger import get_logger
from src.rag.data_pipeline import load_cleaned_data

logger = get_logger("MLPredictionService")

NUMERIC_FEATURES = [
    "OverallQual",
    "OverallCond",
    "GrLivArea",
    "TotalBsmtSF",
    "GarageCars",
    "GarageArea",
    "YearBuilt",
    "YearRemodAdd",
    "FullBath",
    "BedroomAbvGr",
    "TotRmsAbvGrd",
    "LotArea",
    "GarageYrBlt",
]

CATEGORICAL_FEATURES = ["Neighborhood"]

class PropertyPricePredictor:
    """
    Supervised Machine Learning regression service for predicting real estate property prices.
    Trains and persists ensemble models with performance evaluation metrics.
    """
    def __init__(self, model_path: Path = settings.ML_MODEL_PATH):
        self.model_path = model_path
        self.pipeline: Optional[Pipeline] = None
        self.metrics: Dict[str, Any] = {}
        self.feature_importance: Dict[str, float] = {}
        self._initialize()

    def _initialize(self):
        """Loads saved model or trains a new model."""
        if os.path.exists(self.model_path):
            try:
                logger.info(f"Loading trained ML model from {self.model_path}...")
                artifact = joblib.load(self.model_path)
                self.pipeline = artifact["pipeline"]
                self.metrics = artifact["metrics"]
                self.feature_importance = artifact.get("feature_importance", {})
                logger.info(f"Model loaded. Benchmark R2: {self.metrics.get('R2', 'N/A'):.4f}")
                return
            except Exception as e:
                logger.warning(f"Error loading model artifact ({e}). Retraining...")

        self.train_and_save()

    def train_and_save(self) -> Dict[str, Any]:
        """Trains candidate models, selects the best, evaluates metrics, and saves artifact."""
        df = load_cleaned_data()
        logger.info(f"Training ML models on {len(df)} records...")

        features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

        # 1. Honest Split: Distinguish real vs. synthetic rows
        if "is_synthetic" in df.columns:
            synth_mask = df["is_synthetic"].apply(
                lambda x: str(x).strip().lower() in ["true", "1"] if pd.notna(x) else False
            )
            real_df = df[~synth_mask].copy()
            synth_df = df[synth_mask].copy()
        else:
            real_df = df.copy()
            synth_df = pd.DataFrame(columns=df.columns)

        if len(synth_df) > 0:
            real_train, real_test = train_test_split(real_df, test_size=0.20, random_state=42)
            train_df = pd.concat([real_train, synth_df], ignore_index=True)
            test_df = real_test
            logger.info(
                f"Honest split: {len(real_train)} real train, {len(synth_df)} synthetic train, {len(real_test)} real test"
            )
        else:
            real_train, real_test = train_test_split(real_df, test_size=0.20, random_state=42)
            train_df = real_train
            test_df = real_test
            logger.info(
                f"Standard 80/20 split: {len(real_train)} real train, 0 synthetic, {len(real_test)} real test"
            )

        X_train = train_df[features].copy()
        y_train = train_df["SalePrice"].copy()
        X_test = test_df[features].copy()
        y_test = test_df["SalePrice"].copy()

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", "passthrough", NUMERIC_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ]
        )

        # Train multiple candidate models to compare
        candidates = {
            "RandomForest": RandomForestRegressor(n_estimators=150, max_depth=16, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=180, learning_rate=0.08, max_depth=5, random_state=42),
            "Ridge": Ridge(alpha=10.0),
        }

        best_name = None
        best_r2 = -float("inf")
        best_pipeline = None
        eval_results = {}
        best_residuals = None

        for name, regressor in candidates.items():
            pipe = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("regressor", regressor),
            ])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)

            r2 = float(r2_score(y_test, preds))
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            mae = float(mean_absolute_error(y_test, preds))
            mape = float(mean_absolute_percentage_error(y_test, preds))

            eval_results[name] = {
                "R2": round(r2, 4),
                "RMSE": round(rmse, 2),
                "MAE": round(mae, 2),
                "MAPE": round(mape, 4),
            }
            logger.info(f"Model {name} (train_combined -> real_test) -> R2: {r2:.4f}, RMSE: ${rmse:,.2f}, MAE: ${mae:,.2f}, MAPE: {mape*100:.2f}%")

            if r2 > best_r2:
                best_r2 = r2
                best_name = name
                best_pipeline = pipe
                best_residuals = y_test - preds

        metrics_real_plus_synthetic = {
            **eval_results[best_name],
            "best_model_name": best_name,
            "comparison": eval_results,
        }

        # 2. Baseline Model on real_train only evaluated on the same real_test
        if len(synth_df) > 0:
            X_real_train = real_train[features].copy()
            y_real_train = real_train["SalePrice"].copy()

            baseline_candidates = {
                "RandomForest": RandomForestRegressor(n_estimators=150, max_depth=16, random_state=42),
                "GradientBoosting": GradientBoostingRegressor(n_estimators=180, learning_rate=0.08, max_depth=5, random_state=42),
                "Ridge": Ridge(alpha=10.0),
            }
            baseline_results = {}
            best_baseline_name = None
            best_baseline_r2 = -float("inf")

            for name, regressor in baseline_candidates.items():
                pipe_b = Pipeline(steps=[
                    ("preprocessor", preprocessor),
                    ("regressor", regressor),
                ])
                pipe_b.fit(X_real_train, y_real_train)
                preds_b = pipe_b.predict(X_test)

                r2_b = float(r2_score(y_test, preds_b))
                rmse_b = float(np.sqrt(mean_squared_error(y_test, preds_b)))
                mae_b = float(mean_absolute_error(y_test, preds_b))
                mape_b = float(mean_absolute_percentage_error(y_test, preds_b))

                baseline_results[name] = {
                    "R2": round(r2_b, 4),
                    "RMSE": round(rmse_b, 2),
                    "MAE": round(mae_b, 2),
                    "MAPE": round(mape_b, 4),
                }
                logger.info(f"Baseline {name} (real_train_only -> real_test) -> R2: {r2_b:.4f}, RMSE: ${rmse_b:,.2f}, MAE: ${mae_b:,.2f}, MAPE: {mape_b*100:.2f}%")

                if r2_b > best_baseline_r2:
                    best_baseline_r2 = r2_b
                    best_baseline_name = name

            metrics_real_only = {
                **baseline_results[best_baseline_name],
                "best_model_name": best_baseline_name,
                "comparison": baseline_results,
            }
        else:
            metrics_real_only = metrics_real_plus_synthetic

        # Compute 90% prediction interval from real_test residuals
        residual_std = float(np.std(best_residuals)) if best_residuals is not None else 25000.0

        self.pipeline = best_pipeline
        self.metrics = eval_results[best_name].copy()
        self.metrics["best_model_name"] = best_name
        self.metrics["comparison"] = eval_results
        self.metrics["residual_std_real_test"] = round(residual_std, 2)
        self.metrics["metrics_real_plus_synthetic"] = metrics_real_plus_synthetic
        self.metrics["metrics_real_only"] = metrics_real_only

        # Extract feature importances
        try:
            reg = self.pipeline.named_steps["regressor"]
            prep = self.pipeline.named_steps["preprocessor"]
            feature_names = NUMERIC_FEATURES + list(prep.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES))
            if hasattr(reg, "feature_importances_"):
                importances = reg.feature_importances_
                imp_dict = {f: float(imp) for f, imp in zip(feature_names, importances)}
                # Sort descending
                self.feature_importance = dict(sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)[:15])
        except Exception as e:
            logger.warning(f"Could not compute feature importances: {e}")

        # Save artifact
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "metrics": self.metrics,
                "feature_importance": self.feature_importance,
                "metrics_real_plus_synthetic": metrics_real_plus_synthetic,
                "metrics_real_only": metrics_real_only,
            },
            self.model_path,
        )
        logger.info(f"Best model ({best_name}) saved to {self.model_path} with real_test R2: {self.metrics['R2']:.4f}")
        return self.metrics

    def predict(self, property_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predicts property price given input characteristics.
        Returns point estimate, prediction interval, and confidence interval.
        """
        if self.pipeline is None:
            self.train_and_save()

        # Map input features into expected dataframe columns
        row = {}
        # Explicit aliases for fields whose snake_case name does not derive from PascalCase
        _explicit_aliases = {
            "TotRmsAbvGrd": ["total_rooms", "totrmsabvgrd"],
            "GarageYrBlt": ["garage_yr_blt", "garageyrblt"],
        }
        for feat in NUMERIC_FEATURES:
            val = property_features.get(feat)
            if val is None:
                # Check known explicit aliases first
                for alias in _explicit_aliases.get(feat, []):
                    if alias in property_features:
                        val = property_features[alias]
                        break
            if val is None:
                # Check general snake_case variant (strip underscores, lowercase)
                snake = feat.lower()
                for k, v in property_features.items():
                    if k.replace("_", "").lower() == snake:
                        val = v
                        break
            # Fallbacks
            if val is None:
                if feat == "OverallQual": val = 6
                elif feat == "OverallCond": val = 5
                elif feat == "GrLivArea": val = 1500
                elif feat == "TotalBsmtSF": val = 1000
                elif feat == "GarageCars": val = 2
                elif feat == "GarageArea": val = 480
                elif feat == "YearBuilt": val = 1990
                elif feat == "YearRemodAdd": val = 1995
                elif feat == "FullBath": val = 2
                elif feat == "BedroomAbvGr": val = 3
                elif feat == "TotRmsAbvGrd": val = 6
                elif feat == "LotArea": val = 9500
                elif feat == "GarageYrBlt": val = 1990
                else: val = 0
            row[feat] = float(val)

        # Categorical neighborhood
        neigh = property_features.get("Neighborhood") or property_features.get("neighborhood") or property_features.get("location")
        if not neigh:
            neigh = "North Ames"
        row["Neighborhood"] = str(neigh).strip()

        df_in = pd.DataFrame([row])
        predicted_price = float(self.pipeline.predict(df_in)[0])

        # Prediction interval (based on test RMSE)
        rmse = self.metrics.get("RMSE", 25000)
        lower_bound = max(20000.0, predicted_price - (1.645 * rmse))  # 90% confidence interval
        upper_bound = predicted_price + (1.645 * rmse)

        return {
            "predicted_price": round(predicted_price, 2),
            "lower_bound_90pct": round(lower_bound, 2),
            "upper_bound_90pct": round(upper_bound, 2),
            "price_per_sqft": round(predicted_price / max(row["GrLivArea"], 1.0), 2),
            "model_r2_score": self.metrics.get("R2", 0.90),
            "model_used": self.metrics.get("best_model_name", "RandomForest"),
            "input_features": row,
        }

price_predictor = PropertyPricePredictor()
