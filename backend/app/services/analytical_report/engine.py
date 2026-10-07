"""
Milestone 10: Pure Analytical Report Aggregation Engine
Vectorized, deterministic mathematical aggregation without ORM, DB, or HTTP dependencies.
"""
from typing import List, Dict, Optional, Any, Tuple
import math
import numpy as np
from datetime import datetime

from app.services.personal_baseline.models import (
    CANONICAL_FEATURE_REGISTRY,
    INDEX_TO_FEATURE_NAME,
    FeatureMetadata
)
from app.services.analytical_report.models import (
    ReportPeriod,
    ReportMetadata,
    BaselineSummary,
    ModalityCoverageItem,
    ModalityCoverageSummary,
    DescriptiveMetricSummary,
    DeviationSummary,
    FeatureShiftItem,
    TrendFeatureItem,
    TrendSummary,
    LongitudinalTrajectoryPoint,
    TrajectorySummary,
    AnalyticalReport,
    MANDATORY_RESEARCH_DISCLAIMER,
    get_utc_now
)
from app.services.analytical_report.exceptions import (
    InsufficientReportDataError,
    UnsupportedFeatureVersionError
)


class AnalyticalReportEngine:
    """
    Pure statistical and longitudinal reporting engine.
    Consumes verified Milestone 9 temporal trend records and personal baseline outputs.
    """

    FEATURE_VECTOR_VERSION = "v1.0-140d"

    @staticmethod
    def compute_metric_summary(values: List[float]) -> DescriptiveMetricSummary:
        """Calculates mean, median, min, max, and sample std for a list of floats."""
        if not values:
            return DescriptiveMetricSummary(mean=0.0, median=0.0, minimum=0.0, maximum=0.0, std=0.0)

        arr = np.array(values, dtype=np.float64)
        mean_val = float(np.mean(arr))
        median_val = float(np.median(arr))
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        std_val = float(np.std(arr, ddof=1)) if len(values) > 1 else 0.0

        return DescriptiveMetricSummary(
            mean=round(mean_val, 4),
            median=round(median_val, 4),
            minimum=round(min_val, 4),
            maximum=round(max_val, 4),
            std=round(std_val, 4)
        )

    @staticmethod
    def categorize_statistical_variation(mean_abs_z: float) -> Tuple[str, str]:
        """
        Classifies overall within-subject statistical variation according to PRF-01.
        Strictly descriptive research categorization; never clinical or diagnostic.
        """
        if mean_abs_z < 1.0:
            category = "Stable statistical variation"
            desc = "Observed behavioral dimensions remain within typical longitudinal baseline parameters (mean |Z| < 1.0)."
        elif mean_abs_z < 2.0:
            category = "Noticeable statistical variation"
            desc = "Observed behavioral dimensions exhibit moderate within-subject shift relative to personal baseline (1.0 <= mean |Z| < 2.0)."
        else:
            category = "Significant statistical shift"
            desc = "Observed behavioral dimensions demonstrate pronounced within-subject statistical divergence from personal baseline (mean |Z| >= 2.0)."

        return category, desc

    @staticmethod
    def aggregate_modality_coverage(temporal_records: List[Any]) -> ModalityCoverageSummary:
        """Computes measured vs total observation counts across modalities."""
        total = len(temporal_records)
        if total == 0:
            empty_item = ModalityCoverageItem(modality="none", measured_count=0, total_count=0, percentage=0.0)
            return ModalityCoverageSummary(
                communication=ModalityCoverageItem(modality="communication", measured_count=0, total_count=0, percentage=0.0),
                acoustic=ModalityCoverageItem(modality="acoustic", measured_count=0, total_count=0, percentage=0.0),
                nlp=ModalityCoverageItem(modality="nlp", measured_count=0, total_count=0, percentage=0.0)
            )

        comm_count = sum(1 for r in temporal_records if getattr(r, "modality_coverage", {}).get("has_communication", False))
        acoustic_count = sum(1 for r in temporal_records if getattr(r, "modality_coverage", {}).get("has_acoustic", False))
        nlp_count = sum(1 for r in temporal_records if getattr(r, "modality_coverage", {}).get("has_nlp", False))

        return ModalityCoverageSummary(
            communication=ModalityCoverageItem(
                modality="communication",
                measured_count=comm_count,
                total_count=total,
                percentage=round((comm_count / total) * 100.0, 1)
            ),
            acoustic=ModalityCoverageItem(
                modality="acoustic",
                measured_count=acoustic_count,
                total_count=total,
                percentage=round((acoustic_count / total) * 100.0, 1)
            ),
            nlp=ModalityCoverageItem(
                modality="nlp",
                measured_count=nlp_count,
                total_count=total,
                percentage=round((nlp_count / total) * 100.0, 1)
            )
        )

    @staticmethod
    def aggregate_deviations(temporal_records: List[Any]) -> DeviationSummary:
        """Computes longitudinal descriptive summaries for aggregate deviation metrics."""
        mean_abs_zs = [float(r.mean_absolute_z) for r in temporal_records]
        rms_zs = [float(r.rms_z) for r in temporal_records]
        euc_dists = [float(r.standardized_euclidean_distance) for r in temporal_records]

        mean_abs_summary = AnalyticalReportEngine.compute_metric_summary(mean_abs_zs)
        rms_summary = AnalyticalReportEngine.compute_metric_summary(rms_zs)
        euc_summary = AnalyticalReportEngine.compute_metric_summary(euc_dists)

        category, desc = AnalyticalReportEngine.categorize_statistical_variation(mean_abs_summary.mean)

        return DeviationSummary(
            mean_absolute_z=mean_abs_summary,
            rms_z=rms_summary,
            standardized_euclidean_distance=euc_summary,
            statistical_variation_category=category,
            category_description=desc
        )

    @staticmethod
    def aggregate_feature_shifts(
        temporal_records: List[Any],
        limit: int = 10
    ) -> List[FeatureShiftItem]:
        """
        Aggregates longitudinal deviations per behavioral feature across all observations.
        Sorts deterministically by: 1. descending mean |Z|, 2. descending max |Z|, 3. ascending feature index.
        """
        # Feature index -> list of valid signed Z values
        feature_z_map: Dict[int, List[float]] = {}
        feature_latest_z_map: Dict[int, float] = {}
        feature_latest_baseline_map: Dict[int, Tuple[float, float]] = {}

        for r in temporal_records:
            z_mask = getattr(r, "zscore_validity_mask", [])
            raw_zs = getattr(r, "raw_zscore_vector", [])
            mean_vec = getattr(r, "baseline_snapshot_mean", [])
            std_vec = getattr(r, "baseline_snapshot_std", [])

            for feat_idx in range(min(135, len(z_mask))):  # Exclude mask metadata [135..139]
                if z_mask[feat_idx]:
                    val_z = float(raw_zs[feat_idx])
                    if feat_idx not in feature_z_map:
                        feature_z_map[feat_idx] = []
                    feature_z_map[feat_idx].append(val_z)
                    feature_latest_z_map[feat_idx] = val_z

                    b_mean = float(mean_vec[feat_idx]) if feat_idx < len(mean_vec) else 0.0
                    b_std = float(std_vec[feat_idx]) if feat_idx < len(std_vec) else 0.0
                    feature_latest_baseline_map[feat_idx] = (b_mean, b_std)

        shift_items: List[FeatureShiftItem] = []
        for feat_idx, z_values in feature_z_map.items():
            if not z_values:
                continue

            abs_arr = np.abs(np.array(z_values, dtype=np.float64))
            mean_abs = float(np.mean(abs_arr))
            max_abs = float(np.max(abs_arr))
            latest_z = feature_latest_z_map.get(feat_idx, 0.0)
            b_mean, b_std = feature_latest_baseline_map.get(feat_idx, (0.0, 0.0))

            meta = CANONICAL_FEATURE_REGISTRY[feat_idx]
            shift_items.append(FeatureShiftItem(
                feature_index=feat_idx,
                feature_name=meta.name,
                modality=meta.modality,
                valid_observations=len(z_values),
                mean_absolute_z=round(mean_abs, 4),
                max_absolute_z=round(max_abs, 4),
                latest_signed_z=round(latest_z, 4),
                latest_absolute_z=round(abs(latest_z), 4),
                baseline_mean=round(b_mean, 4),
                baseline_std=round(b_std, 4)
            ))

        # Deterministic sort: mean |Z| desc, max |Z| desc, index asc
        shift_items.sort(key=lambda x: (-x.mean_absolute_z, -x.max_absolute_z, x.feature_index))
        return shift_items[:limit]

    @staticmethod
    def aggregate_temporal_trends(
        temporal_records: List[Any],
        limit: int = 10
    ) -> TrendSummary:
        """
        Aggregates longitudinal temporal dynamics from Milestone 9 records.
        Preserves M9 trend direction rules (> +0.01 increasing, < -0.01 decreasing, else stable).
        """
        if not temporal_records:
            return TrendSummary(increasing_count=0, decreasing_count=0, stable_count=0, top_trends=[])

        # Use latest observation's trend slopes and moving averages
        latest_rec = temporal_records[-1]
        slopes_summary = getattr(latest_rec, "trend_slopes_summary", {}) or {}
        sma_summary = getattr(latest_rec, "moving_averages_summary", {}) or {}
        roc_summary = getattr(latest_rec, "rate_of_change_summary", {}) or {}

        increasing_c = 0
        decreasing_c = 0
        stable_c = 0
        trend_items: List[TrendFeatureItem] = []

        for feat_idx in range(135):  # Exclude mask [135..139]
            meta = CANONICAL_FEATURE_REGISTRY[feat_idx]
            fname = meta.name

            slope_data = slopes_summary.get(fname, {})
            if not slope_data or not slope_data.get("valid", False):
                continue

            direction = slope_data.get("direction", "stable")
            norm_slope = float(slope_data.get("slope_normalized_per_hour", 0.0))
            raw_slope = float(slope_data.get("slope_raw_per_hour", 0.0))

            if direction == "increasing":
                increasing_c += 1
            elif direction == "decreasing":
                decreasing_c += 1
            else:
                stable_c += 1

            ma_data = sma_summary.get(fname, {})
            roc_data = roc_summary.get(fname, {})

            trend_items.append(TrendFeatureItem(
                feature_index=feat_idx,
                feature_name=fname,
                modality=meta.modality,
                normalized_slope=round(norm_slope, 4),
                raw_slope=round(raw_slope, 4),
                direction=direction,
                latest_sma3=round(float(ma_data["sma_3"]), 4) if ma_data.get("sma_3") is not None else None,
                latest_sma5=round(float(ma_data["sma_5"]), 4) if ma_data.get("sma_5") is not None else None,
                latest_ema3=round(float(ma_data["ema_3"]), 4) if ma_data.get("ema_3") is not None else None,
                latest_ema5=round(float(ma_data["ema_5"]), 4) if ma_data.get("ema_5") is not None else None,
                latest_roc=round(float(roc_data["roc_normalized"]), 4) if roc_data.get("roc_normalized") is not None else None,
                valid_observation_count=int(slope_data.get("K_i", len(temporal_records)))
            ))

        # Sort: absolute normalized slope descending, feature index ascending
        trend_items.sort(key=lambda x: (-abs(x.normalized_slope), x.feature_index))

        return TrendSummary(
            increasing_count=increasing_c,
            decreasing_count=decreasing_c,
            stable_count=stable_c,
            top_trends=trend_items[:limit]
        )

    @staticmethod
    def build_trajectory_summary(
        temporal_records: List[Any],
        message_timestamp_map: Optional[Dict[str, datetime]] = None
    ) -> TrajectorySummary:
        """Constructs ordered longitudinal trajectory points and time-series arrays."""
        points: List[LongitudinalTrajectoryPoint] = []
        timestamps_str: List[str] = []
        mean_abs_series: List[float] = []
        rms_series: List[float] = []
        euc_series: List[float] = []

        for r in temporal_records:
            msg_id = getattr(r, "message_id", "")
            eval_time = (
                message_timestamp_map.get(msg_id)
                if message_timestamp_map and msg_id in message_timestamp_map
                else getattr(r, "evaluated_at", get_utc_now())
            )

            top_dev_name = None
            top_devs = getattr(r, "top_deviations_summary", [])
            if top_devs and len(top_devs) > 0:
                top_dev_name = top_devs[0].get("name") if isinstance(top_devs[0], dict) else getattr(top_devs[0], "name", None)

            pt = LongitudinalTrajectoryPoint(
                message_id=msg_id,
                timestamp=eval_time,
                mean_absolute_z=round(float(r.mean_absolute_z), 4),
                rms_z=round(float(r.rms_z), 4),
                standardized_euclidean_distance=round(float(r.standardized_euclidean_distance), 4),
                valid_dimension_count=int(getattr(r, "valid_dimension_count", 0)),
                top_deviation_feature=top_dev_name
            )
            points.append(pt)
            timestamps_str.append(eval_time.isoformat())
            mean_abs_series.append(pt.mean_absolute_z)
            rms_series.append(pt.rms_z)
            euc_series.append(pt.standardized_euclidean_distance)

        return TrajectorySummary(
            total_points=len(points),
            points=points,
            timestamps=timestamps_str,
            mean_absolute_z_series=mean_abs_series,
            rms_z_series=rms_series,
            standardized_euclidean_distance_series=euc_series
        )

    @staticmethod
    def generate_limitations(
        observation_count: int,
        baseline_sample_count: int,
        is_baseline_established: bool,
        modality_coverage: ModalityCoverageSummary
    ) -> List[str]:
        """Generates clear, explicit research limitations based on data completeness."""
        limits: List[str] = [
            "This analysis reflects descriptive statistical variations relative to an individual's own observed baseline, not normative clinical benchmarks."
        ]

        if not is_baseline_established or baseline_sample_count < 3:
            limits.append(
                f"Personal baseline is still calibrating (N={baseline_sample_count}/3 prior observations). Initial deviation measurements have reduced statistical power."
            )
        elif baseline_sample_count < 10:
            limits.append(
                f"Baseline sample size is moderate (N={baseline_sample_count}). Continued observation will refine empirical mean and variance estimates."
            )

        if modality_coverage.acoustic.percentage < 50.0:
            limits.append(
                f"Acoustic feature coverage is limited ({modality_coverage.acoustic.percentage}% of observations). Voice-derived dynamics are partially unmeasured."
            )

        if modality_coverage.nlp.percentage < 50.0:
            limits.append(
                f"NLP linguistic coverage is limited ({modality_coverage.nlp.percentage}% of observations). Text-derived linguistic dynamics are partially unmeasured."
            )

        limits.append(
            "Missing modalities on individual turns are appropriately masked out and excluded from aggregate deviation scoring without zero-imputation bias."
        )

        return limits

    @classmethod
    def assemble_report(
        cls,
        report_id: str,
        user_id: str,
        room_id: Optional[str],
        period_start: Optional[datetime],
        period_end: Optional[datetime],
        baseline_record: Optional[Any],
        temporal_records: List[Any],
        message_timestamp_map: Optional[Dict[str, datetime]] = None
    ) -> AnalyticalReport:
        """
        Orchestrates full report assembly from Milestone 9 records.
        Enforces deterministic calculation, modality masking, and point-in-time reproducibility.
        """
        if not temporal_records:
            raise InsufficientReportDataError(
                f"No temporal trend observations found for user '{user_id}' in requested period."
            )

        # 1. Baseline Summary
        if baseline_record:
            sample_count = int(baseline_record.sample_count)
            is_established = bool(baseline_record.is_baseline_established)
            valid_dims = int(baseline_record.valid_dimensions_count)
            status_str = "established" if is_established else "insufficient_history"
        else:
            sample_count = 0
            is_established = False
            valid_dims = 140
            status_str = "not_established"

        baseline_summary = BaselineSummary(
            status=status_str,
            sample_count=sample_count,
            window_type="all_time",
            feature_vector_version=cls.FEATURE_VECTOR_VERSION,
            is_baseline_established=is_established,
            valid_dimensions_count=valid_dims
        )

        # 2. Modality Coverage
        modality_coverage = cls.aggregate_modality_coverage(temporal_records)

        # 3. Deviations Aggregation
        deviation_summary = cls.aggregate_deviations(temporal_records)

        # 4. Feature Shifts
        feature_shifts = cls.aggregate_feature_shifts(temporal_records, limit=10)

        # 5. Temporal Trends
        trend_summary = cls.aggregate_temporal_trends(temporal_records, limit=10)

        # 6. Trajectory Summary
        trajectory_summary = cls.build_trajectory_summary(temporal_records, message_timestamp_map)

        # 7. Limitations
        limitations = cls.generate_limitations(
            observation_count=len(temporal_records),
            baseline_sample_count=sample_count,
            is_baseline_established=is_established,
            modality_coverage=modality_coverage
        )

        # 8. Metadata
        metadata = ReportMetadata(
            report_id=report_id,
            report_type="longitudinal",
            feature_vector_version=cls.FEATURE_VECTOR_VERSION,
            user_id=user_id,
            room_id=room_id,
            observation_count=len(temporal_records),
            baseline_sample_count=sample_count,
            baseline_established=is_established,
            generated_at=get_utc_now()
        )

        return AnalyticalReport(
            metadata=metadata,
            period=ReportPeriod(period_start=period_start, period_end=period_end),
            baseline=baseline_summary,
            modality_coverage=modality_coverage,
            deviation_summary=deviation_summary,
            feature_shifts=feature_shifts,
            trend_summary=trend_summary,
            trajectory_summary=trajectory_summary,
            limitations=limitations,
            disclaimer=MANDATORY_RESEARCH_DISCLAIMER
        )
