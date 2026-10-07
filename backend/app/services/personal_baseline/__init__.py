"""
Milestone 9: Personal Baseline & Temporal Trend Engine Module
"""
from app.services.personal_baseline.models import (
    CANONICAL_FEATURE_REGISTRY,
    FEATURE_NAME_TO_INDEX,
    INDEX_TO_FEATURE_NAME,
    TopDeviationItem,
    PersonalBaselineResult,
    TemporalTrendEvaluationResult
)
from app.services.personal_baseline.exceptions import (
    PersonalBaselineException,
    InsufficientHistoryError,
    UnsupportedFeatureVectorError,
    DataLeakageError,
    UserAccessDeniedError,
    BaselineCalculationError
)
from app.services.personal_baseline.engine import PersonalBaselineEngine
from app.services.personal_baseline.service import PersonalBaselineService

personal_baseline_service = PersonalBaselineService.get_instance()

__all__ = [
    "CANONICAL_FEATURE_REGISTRY",
    "FEATURE_NAME_TO_INDEX",
    "INDEX_TO_FEATURE_NAME",
    "TopDeviationItem",
    "PersonalBaselineResult",
    "TemporalTrendEvaluationResult",
    "PersonalBaselineException",
    "InsufficientHistoryError",
    "UnsupportedFeatureVectorError",
    "DataLeakageError",
    "UserAccessDeniedError",
    "BaselineCalculationError",
    "PersonalBaselineEngine",
    "PersonalBaselineService",
    "personal_baseline_service"
]
