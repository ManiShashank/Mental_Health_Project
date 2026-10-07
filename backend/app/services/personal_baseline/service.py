"""
Milestone 9: Personal Baseline & Temporal Trend Coordinator Service.
Enforces strict data leakage prevention, point-in-time snapshotting,
modality availability masking, and idempotent repository persistence.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import logging
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.personal_baseline.models import (
    PersonalBaselineResult,
    TemporalTrendEvaluationResult,
    CANONICAL_FEATURE_REGISTRY
)
from app.services.personal_baseline.exceptions import (
    PersonalBaselineException,
    InsufficientHistoryError,
    UnsupportedFeatureVectorError,
    DataLeakageError
)
from app.services.personal_baseline.engine import PersonalBaselineEngine
from app.db.repository import ChatRepository

logger = logging.getLogger("personal_baseline_service")


class PersonalBaselineService:
    """
    Coordinator service for idiographic personal baseline construction,
    message-level deviation evaluation, and longitudinal trajectory analysis.
    """
    _instance: Optional["PersonalBaselineService"] = None

    MIN_BASELINE_OBSERVATIONS: int = 3
    FEATURE_VECTOR_VERSION: str = "v1.0-140d"

    @classmethod
    def get_instance(cls) -> "PersonalBaselineService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def extract_modality_mask(cls, record: Any) -> List[float]:
        """Extracts 140-d binary validity mask from a MultimodalFeatureRecord."""
        mask = [0.0] * 140
        # Communication [000..009] always valid
        for i in range(10):
            mask[i] = 1.0

        # Acoustic [010..068] valid if has_acoustic is True
        if getattr(record, "has_acoustic", False):
            for i in range(10, 69):
                mask[i] = 1.0

        # NLP [069..134] valid if has_nlp is True
        if getattr(record, "has_nlp", False):
            for i in range(69, 135):
                mask[i] = 1.0

        # Mask flags [135..139] are metadata, set to 0.0 for behavioral scoring
        return mask

    async def compute_or_update_user_baseline(
        self,
        user_id: str,
        db: AsyncSession,
        window_type: str = "all_time",
        before_timestamp: Optional[datetime] = None
    ) -> PersonalBaselineResult:
        """
        Recomputes and idempotently persists a participant's longitudinal personal baseline.
        """
        # Fetch completed historical multimodal records
        records = await ChatRepository.get_user_multimodal_history(
            session=db,
            user_id=user_id,
            before_timestamp=before_timestamp,
            window_type=window_type,
            feature_version=self.FEATURE_VECTOR_VERSION
        )

        sample_count = len(records)
        is_established = (sample_count >= self.MIN_BASELINE_OBSERVATIONS)

        if sample_count == 0:
            zeros = [0.0] * 140
            result = PersonalBaselineResult(
                user_id=user_id,
                window_type=window_type,
                feature_vector_version=self.FEATURE_VECTOR_VERSION,
                sample_count=0,
                is_baseline_established=False,
                valid_dimensions_count=0,
                per_feature_sample_counts=[0] * 140,
                raw_mean_vector=zeros,
                raw_std_vector=zeros,
                normalized_mean_vector=zeros,
                normalized_std_vector=zeros,
                first_observed_at=None,
                last_observed_at=None,
                computed_at=datetime.now(timezone.utc)
            )
        else:
            raw_vecs = [r.raw_fused_vector for r in records]
            norm_vecs = [r.normalized_fused_vector for r in records]
            masks = [self.extract_modality_mask(r) for r in records]

            raw_mean, raw_std, norm_mean, norm_std, counts = PersonalBaselineEngine.compute_baseline_statistics(
                raw_vectors=raw_vecs,
                normalized_vectors=norm_vecs,
                masks=masks
            )

            first_obs = records[0].created_at if records else None
            last_obs = records[-1].created_at if records else None

            result = PersonalBaselineResult(
                user_id=user_id,
                window_type=window_type,
                feature_vector_version=self.FEATURE_VECTOR_VERSION,
                sample_count=sample_count,
                is_baseline_established=is_established,
                valid_dimensions_count=int(sum(1 for c in counts[:135] if c >= 2)),
                per_feature_sample_counts=counts,
                raw_mean_vector=raw_mean,
                raw_std_vector=raw_std,
                normalized_mean_vector=norm_mean,
                normalized_std_vector=norm_std,
                first_observed_at=first_obs,
                last_observed_at=last_obs,
                computed_at=datetime.now(timezone.utc)
            )

        # Persist baseline snapshot idempotently
        await ChatRepository.save_or_update_personal_baseline(
            session=db,
            user_id=user_id,
            window_type=window_type,
            feature_vector_version=self.FEATURE_VECTOR_VERSION,
            sample_count=result.sample_count,
            is_baseline_established=result.is_baseline_established,
            valid_dimensions_count=result.valid_dimensions_count,
            raw_mean_vector=result.raw_mean_vector,
            raw_std_vector=result.raw_std_vector,
            normalized_mean_vector=result.normalized_mean_vector,
            normalized_std_vector=result.normalized_std_vector,
            first_observed_at=result.first_observed_at,
            last_observed_at=result.last_observed_at
        )

        return result

    async def evaluate_message_temporal_deviation(
        self,
        message_id: str,
        db: AsyncSession,
        window_type: str = "all_time"
    ) -> TemporalTrendEvaluationResult:
        """
        Evaluates current message deviation and temporal trajectory against strictly preceding baseline.
        Enforces strict leakage prevention: message_id is NEVER included in its reference baseline.
        """
        current_record = await ChatRepository.get_multimodal_features_by_message_id(db, message_id=message_id)
        if not current_record:
            raise PersonalBaselineException(f"Multimodal feature record for message '{message_id}' not found.")

        if current_record.feature_vector_version != self.FEATURE_VECTOR_VERSION:
            raise UnsupportedFeatureVectorError(
                f"Feature version '{current_record.feature_vector_version}' incompatible with expected '{self.FEATURE_VECTOR_VERSION}'"
            )

        user_id = current_record.sender_id
        room_id = current_record.room_id

        msg = await ChatRepository.get_message_by_id(db, message_id=message_id)
        current_time = msg.timestamp if msg else current_record.created_at

        # 1. Fetch strictly preceding historical records (t < current_time)
        prior_records = await ChatRepository.get_user_multimodal_history(
            session=db,
            user_id=user_id,
            before_timestamp=current_time,
            exclude_message_id=message_id,
            window_type=window_type,
            feature_version=self.FEATURE_VECTOR_VERSION
        )

        prior_count = len(prior_records)
        is_established = (prior_count >= self.MIN_BASELINE_OBSERVATIONS)

        current_mask = self.extract_modality_mask(current_record)
        modality_cov = {
            "has_communication": bool(current_record.has_communication),
            "has_acoustic": bool(current_record.has_acoustic),
            "has_nlp": bool(current_record.has_nlp)
        }

        # 2. If insufficient history, return explicit non-established status
        if not is_established:
            zeros = [0.0] * 140
            falses = [False] * 140

            result = TemporalTrendEvaluationResult(
                message_id=message_id,
                room_id=room_id,
                user_id=user_id,
                status="insufficient_history",
                feature_vector_version=self.FEATURE_VECTOR_VERSION,
                sample_count_at_eval=prior_count,
                is_baseline_established=False,
                baseline_snapshot_mean=zeros,
                baseline_snapshot_std=zeros,
                zscore_validity_mask=falses,
                valid_dimension_count=0,
                unavailable_dimension_count=140,
                modality_coverage=modality_cov,
                raw_zscore_vector=zeros,
                normalized_zscore_vector=zeros,
                clipped_zscore_vector=zeros,
                mean_absolute_z=0.0,
                rms_z=0.0,
                standardized_euclidean_distance=0.0,
                top_deviations=[],
                trend_slopes_summary={},
                moving_averages_summary={},
                rate_of_change_summary={},
                evaluation_error="Insufficient prior observations to establish personal baseline (min required: 3).",
                evaluated_at=datetime.now(timezone.utc)
            )
        else:
            # 3. Compute baseline snapshot from strictly prior observations
            prior_raw = [r.raw_fused_vector for r in prior_records]
            prior_norm = [r.normalized_fused_vector for r in prior_records]
            prior_masks = [self.extract_modality_mask(r) for r in prior_records]

            raw_mean, raw_std, norm_mean, norm_std, counts = PersonalBaselineEngine.compute_baseline_statistics(
                raw_vectors=prior_raw,
                normalized_vectors=prior_norm,
                masks=prior_masks
            )

            # Compute regularized z-scores on normalized space (scale-invariant primary space)
            raw_z, clipped_z, val_mask, mean_abs, rms, std_euc = PersonalBaselineEngine.compute_zscores(
                current_vector=current_record.normalized_fused_vector,
                current_mask=current_mask,
                baseline_mean=norm_mean,
                baseline_std=norm_std,
                per_feature_counts=counts
            )

            # Extract top deviations
            top_devs = PersonalBaselineEngine.extract_top_deviations(
                raw_zscores=raw_z,
                clipped_zscores=clipped_z,
                validity_mask=val_mask,
                current_raw_vector=current_record.raw_fused_vector,
                baseline_raw_mean=raw_mean,
                baseline_raw_std=raw_std,
                limit=5
            )

            trajectory_records = prior_records + [current_record]
            timestamps = [
                (r.message.timestamp if getattr(r, "message", None) else r.created_at).timestamp()
                for r in trajectory_records
            ]
            traj_raw = [r.raw_fused_vector for r in trajectory_records]
            traj_norm = [r.normalized_fused_vector for r in trajectory_records]
            traj_masks = [self.extract_modality_mask(r) for r in trajectory_records]

            sma_sum, slopes_sum, roc_sum = PersonalBaselineEngine.compute_temporal_dynamics(
                timestamps=timestamps,
                raw_series=traj_raw,
                norm_series=traj_norm,
                mask_series=traj_masks
            )

            val_dim_count = int(sum(1 for v in val_mask if v is True))
            unavail_count = 140 - val_dim_count

            result = TemporalTrendEvaluationResult(
                message_id=message_id,
                room_id=room_id,
                user_id=user_id,
                status="completed",
                feature_vector_version=self.FEATURE_VECTOR_VERSION,
                sample_count_at_eval=prior_count,
                is_baseline_established=True,
                baseline_snapshot_mean=norm_mean,
                baseline_snapshot_std=norm_std,
                zscore_validity_mask=val_mask,
                valid_dimension_count=val_dim_count,
                unavailable_dimension_count=unavail_count,
                modality_coverage=modality_cov,
                raw_zscore_vector=raw_z,
                normalized_zscore_vector=raw_z,
                clipped_zscore_vector=clipped_z,
                mean_absolute_z=mean_abs,
                rms_z=rms,
                standardized_euclidean_distance=std_euc,
                top_deviations=top_devs,
                trend_slopes_summary=slopes_sum,
                moving_averages_summary=sma_sum,
                rate_of_change_summary=roc_sum,
                evaluation_error=None,
                evaluated_at=datetime.now(timezone.utc)
            )

        # 5. Persist evaluation record idempotently
        await ChatRepository.save_or_update_temporal_trend_record(
            session=db,
            message_id=message_id,
            room_id=room_id,
            user_id=user_id,
            status=result.status,
            feature_vector_version=result.feature_vector_version,
            sample_count_at_eval=result.sample_count_at_eval,
            is_baseline_established=result.is_baseline_established,
            baseline_snapshot_mean=result.baseline_snapshot_mean,
            baseline_snapshot_std=result.baseline_snapshot_std,
            zscore_validity_mask=result.zscore_validity_mask,
            valid_dimension_count=result.valid_dimension_count,
            unavailable_dimension_count=result.unavailable_dimension_count,
            modality_coverage=result.modality_coverage,
            raw_zscore_vector=result.raw_zscore_vector,
            normalized_zscore_vector=result.normalized_zscore_vector,
            clipped_zscore_vector=result.clipped_zscore_vector,
            mean_absolute_z=result.mean_absolute_z,
            rms_z=result.rms_z,
            standardized_euclidean_distance=result.standardized_euclidean_distance,
            top_deviations_summary=[item.model_dump() for item in result.top_deviations],
            trend_slopes_summary=result.trend_slopes_summary,
            moving_averages_summary=result.moving_averages_summary,
            rate_of_change_summary=result.rate_of_change_summary,
            evaluation_error=result.evaluation_error
        )

        return result
