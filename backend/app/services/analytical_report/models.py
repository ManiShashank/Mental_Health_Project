"""
Milestone 10: Domain Models & DTOs for Final Analytical Report Engine
"""
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field

MANDATORY_RESEARCH_DISCLAIMER = (
    "Descriptive within-subject statistical analysis only. "
    "These results describe observed behavioral feature variation relative to the participant's "
    "longitudinal baseline and are not intended for clinical assessment, psychiatric diagnosis, "
    "risk prediction, triage, or treatment recommendation."
)


def get_utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ReportPeriod(BaseModel):
    """Time window defining the report observation scope."""
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None


class ReportMetadata(BaseModel):
    """Header and provenance metadata for an analytical report."""
    report_id: str
    report_type: str = "longitudinal"
    feature_vector_version: str = "v1.0-140d"
    user_id: str
    room_id: Optional[str] = None
    observation_count: int
    baseline_sample_count: int
    baseline_established: bool
    generated_at: datetime = Field(default_factory=get_utc_now)


class BaselineSummary(BaseModel):
    """Longitudinal normative baseline characteristics consumed from Milestone 9."""
    status: str  # "established" | "insufficient_history" | "not_established"
    sample_count: int
    window_type: str = "all_time"
    feature_vector_version: str = "v1.0-140d"
    is_baseline_established: bool
    valid_dimensions_count: int


class ModalityCoverageItem(BaseModel):
    """Coverage statistics for a single behavioral modality."""
    modality: str  # "communication" | "acoustic" | "nlp"
    measured_count: int
    total_count: int
    percentage: float


class ModalityCoverageSummary(BaseModel):
    """Multi-modality data completeness summary."""
    communication: ModalityCoverageItem
    acoustic: ModalityCoverageItem
    nlp: ModalityCoverageItem


class DescriptiveMetricSummary(BaseModel):
    """Four-number descriptive statistical summary (mean, median, min, max, std)."""
    mean: float
    median: float
    minimum: float
    maximum: float
    std: float


class DeviationSummary(BaseModel):
    """Longitudinal aggregation of within-subject z-score deviation metrics."""
    mean_absolute_z: DescriptiveMetricSummary
    rms_z: DescriptiveMetricSummary
    standardized_euclidean_distance: DescriptiveMetricSummary
    statistical_variation_category: str  # "Stable statistical variation" | "Noticeable statistical variation" | "Significant statistical shift"
    category_description: str


class FeatureShiftItem(BaseModel):
    """Aggregated longitudinal shift summary for a single behavioral dimension."""
    feature_index: int
    feature_name: str
    modality: str
    valid_observations: int
    mean_absolute_z: float
    max_absolute_z: float
    latest_signed_z: float
    latest_absolute_z: float
    baseline_mean: float
    baseline_std: float


class TrendFeatureItem(BaseModel):
    """Longitudinal trajectory summary for a single behavioral dimension."""
    feature_index: int
    feature_name: str
    modality: str
    normalized_slope: float
    raw_slope: float
    direction: str  # "increasing" | "decreasing" | "stable"
    latest_sma3: Optional[float] = None
    latest_sma5: Optional[float] = None
    latest_ema3: Optional[float] = None
    latest_ema5: Optional[float] = None
    latest_roc: Optional[float] = None
    valid_observation_count: int


class TrendSummary(BaseModel):
    """Overall distribution and top dimensional temporal trends."""
    increasing_count: int
    decreasing_count: int
    stable_count: int
    top_trends: List[TrendFeatureItem]


class LongitudinalTrajectoryPoint(BaseModel):
    """Single observation point in longitudinal trajectory series."""
    message_id: str
    timestamp: datetime
    mean_absolute_z: float
    rms_z: float
    standardized_euclidean_distance: float
    valid_dimension_count: int
    top_deviation_feature: Optional[str] = None


class TrajectorySummary(BaseModel):
    """Longitudinal time-series representation suitable for frontend charting."""
    total_points: int
    points: List[LongitudinalTrajectoryPoint]
    timestamps: List[str]
    mean_absolute_z_series: List[float]
    rms_z_series: List[float]
    standardized_euclidean_distance_series: List[float]


class AnalyticalReport(BaseModel):
    """Complete, self-contained, reproducible Milestone 10 Analytical Report."""
    metadata: ReportMetadata
    period: ReportPeriod
    baseline: BaselineSummary
    modality_coverage: ModalityCoverageSummary
    deviation_summary: DeviationSummary
    feature_shifts: List[FeatureShiftItem]
    trend_summary: TrendSummary
    trajectory_summary: TrajectorySummary
    limitations: List[str]
    disclaimer: str = MANDATORY_RESEARCH_DISCLAIMER
