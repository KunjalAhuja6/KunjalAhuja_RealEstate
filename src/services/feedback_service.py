import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
from pydantic import BaseModel, Field

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger("FeedbackService")


class FeedbackSubmission(BaseModel):
    property_id: Optional[str] = "CUSTOM-PROPERTY"
    recommended_value: float
    actual_valuation: float
    user_rating: int = Field(ge=1, le=5, description="1 to 5 stars")
    appraiser_comments: Optional[str] = ""
    property_features: Optional[Dict[str, Any]] = Field(default_factory=dict)
    professional_role: Optional[str] = "Real Estate Appraiser"
    # The calibration factor that was already applied to produce recommended_value.
    # Callers should pass the factor used by the orchestrator so that the feedback
    # service can measure bias against the *uncalibrated* prediction.
    # Defaults to 1.0 (no calibration applied) for legacy / manual submissions.
    applied_calibration_factor: Optional[float] = 1.0


class FeedbackService:
    """
    Manages continuous learning and feedback collection for the valuation system.
    Maintains persistent feedback log, tracks accuracy trends, and computes
    calibration offsets.

    Bias-measurement design:
    - ``recommended_value`` in a submission is the *calibrated* value shown to the
      user.
    - ``applied_calibration_factor`` records the multiplier that produced that value.
    - Error / bias are computed against
      ``uncalibrated_value = recommended_value / applied_calibration_factor``
      so that the calibration factor does not cancel itself in successive rounds.
    """

    def __init__(self, feedback_path: Path = settings.FEEDBACK_DATA_PATH):
        self.feedback_path = feedback_path
        self._ensure_storage()

    def _ensure_storage(self):
        """Creates initial feedback storage if not present, populating with seed data."""
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(self.feedback_path):
            # Seed records: applied_calibration_factor defaults to 1.0 so the
            # uncalibrated value equals the recommended value (backwards-compatible).
            seed_feedback = [
                {
                    "feedback_id": "FB-001",
                    "timestamp": "2026-09-15T10:30:00",
                    "property_id": "PROP-0604",
                    "recommended_value": 172000.0,
                    "actual_valuation": 177625.0,
                    "applied_calibration_factor": 1.0,
                    "uncalibrated_recommended_value": 172000.0,
                    "error_dollars": 5625.0,
                    "error_pct": 3.17,
                    "user_rating": 5,
                    "appraiser_comments": "Close alignment with closing price; minor kitchen update accounted for difference.",
                    "professional_role": "Certified Appraiser",
                },
                {
                    "feedback_id": "FB-002",
                    "timestamp": "2026-09-22T14:15:00",
                    "property_id": "PROP-0678",
                    "recommended_value": 134000.0,
                    "actual_valuation": 128500.0,
                    "applied_calibration_factor": 1.0,
                    "uncalibrated_recommended_value": 134000.0,
                    "error_dollars": -5500.0,
                    "error_pct": -4.28,
                    "user_rating": 4,
                    "appraiser_comments": "Slightly overvalued due to exterior siding wear not fully captured in photos.",
                    "professional_role": "Listing Agent",
                },
                {
                    "feedback_id": "FB-003",
                    "timestamp": "2026-10-01T09:45:00",
                    "property_id": "PROP-0120",
                    "recommended_value": 245000.0,
                    "actual_valuation": 249000.0,
                    "applied_calibration_factor": 1.0,
                    "uncalibrated_recommended_value": 245000.0,
                    "error_dollars": 4000.0,
                    "error_pct": 1.61,
                    "user_rating": 5,
                    "appraiser_comments": "Excellent comparable selection in College Creek.",
                    "professional_role": "Senior Valuer",
                },
            ]
            with open(self.feedback_path, "w") as f:
                json.dump(seed_feedback, f, indent=2)
            logger.info("Initialized feedback repository with seed appraisals.")

    def _read_records(self) -> List[Dict[str, Any]]:
        try:
            with open(self.feedback_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error reading feedback log: {e}")
            return []

    def _write_records(self, records: List[Dict[str, Any]]):
        with open(self.feedback_path, "w") as f:
            json.dump(records, f, indent=2)

    def submit_feedback(self, submission: FeedbackSubmission) -> Dict[str, Any]:
        """Saves a new user / professional evaluation record.

        Error and bias are always measured against the *uncalibrated* prediction
        so that successive calibration rounds converge rather than drift.
        """
        records = self._read_records()
        new_id = f"FB-{len(records) + 1:03d}"

        applied_cf = float(submission.applied_calibration_factor or 1.0)
        # Recover the uncalibrated prediction the model actually made
        uncalibrated = submission.recommended_value / max(applied_cf, 1e-9)

        dollar_err = submission.actual_valuation - uncalibrated
        pct_err = (dollar_err / max(uncalibrated, 1.0)) * 100

        entry = {
            "feedback_id": new_id,
            "timestamp": datetime.now().isoformat(),
            "property_id": submission.property_id,
            "recommended_value": submission.recommended_value,
            "actual_valuation": submission.actual_valuation,
            "applied_calibration_factor": applied_cf,
            "uncalibrated_recommended_value": round(uncalibrated, 2),
            "error_dollars": round(dollar_err, 2),
            "error_pct": round(pct_err, 2),
            "user_rating": submission.user_rating,
            "appraiser_comments": submission.appraiser_comments,
            "professional_role": submission.professional_role,
            "property_features": submission.property_features,
        }
        records.append(entry)
        self._write_records(records)
        logger.info(f"Recorded feedback {new_id} with rating {submission.user_rating}/5 stars.")
        return entry

    def get_feedback_summary(self) -> Dict[str, Any]:
        """
        Computes summary statistics and learning metrics across all logged feedback entries.

        Error and bias are computed from ``uncalibrated_recommended_value`` so that
        the calibration factor reflects true model bias and does not cancel itself.
        """
        records = self._read_records()
        if not records:
            return {
                "total_feedback_count": 0,
                "average_rating": 0.0,
                "mean_absolute_error_dollars": 0.0,
                "mean_absolute_percentage_error": 0.0,
                "systematic_bias_pct": 0.0,
                "calibration_factor": 1.0,
                "recent_records": [],
            }

        ratings = [r.get("user_rating", 5) for r in records]
        abs_errors = [abs(r.get("error_dollars", 0.0)) for r in records]
        pct_errors = [r.get("error_pct", 0.0) for r in records]
        abs_pct_errors = [abs(r.get("error_pct", 0.0)) for r in records]

        mae = float(np.mean(abs_errors))
        mape = float(np.mean(abs_pct_errors))
        bias = float(np.mean(pct_errors))
        avg_rating = float(np.mean(ratings))

        # Calibration multiplier to correct systematic bias measured against
        # uncalibrated predictions — ensures factor converges towards 1.0 only
        # when there is truly no bias.
        calibration_factor = round(1.0 + (bias / 100.0), 4)

        return {
            "total_feedback_count": len(records),
            "average_rating": round(avg_rating, 2),
            "mean_absolute_error_dollars": round(mae, 2),
            "mean_absolute_percentage_error": round(mape, 2),
            "systematic_bias_pct": round(bias, 2),
            "calibration_factor": calibration_factor,
            "recent_records": records[-10:],
        }


feedback_service = FeedbackService()
