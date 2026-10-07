"""
Milestone 10: Analytical Report Service Coordinator
Orchestrates report generation, database persistence, period filtering, and historical reproducibility.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import uuid
import logging
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repository import ChatRepository
from app.services.analytical_report.models import (
    AnalyticalReport,
    ReportPeriod,
    MANDATORY_RESEARCH_DISCLAIMER
)
from app.services.analytical_report.engine import AnalyticalReportEngine
from app.services.analytical_report.exceptions import (
    AnalyticalReportException,
    InsufficientReportDataError,
    UnsupportedFeatureVersionError,
    InvalidReportPeriodError,
    ReportNotFoundError,
    UserAccessDeniedError,
    ReportGenerationError
)

logger = logging.getLogger(__name__)


class AnalyticalReportService:
    """
    Service coordinator for Milestone 10 Analytical Reports.
    Enforces authorization, period boundaries, point-in-time snapshot persistence, and determinism.
    """

    FEATURE_VECTOR_VERSION = "v1.0-140d"
    _instance: Optional["AnalyticalReportService"] = None

    @classmethod
    def get_instance(cls) -> "AnalyticalReportService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def generate_user_report(
        self,
        user_id: str,
        db: AsyncSession,
        room_id: Optional[str] = None,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
        report_type: str = "longitudinal",
        requesting_user_id: Optional[str] = None
    ) -> AnalyticalReport:
        """
        Generates and persists a deterministic, self-contained analytical report for a participant.
        """
        # 1. Authorization check
        if requesting_user_id and requesting_user_id != user_id:
            raise UserAccessDeniedError(f"User '{requesting_user_id}' is not authorized to generate reports for user '{user_id}'.")

        # 2. Validate period
        if period_start and period_end and period_start > period_end:
            raise InvalidReportPeriodError(f"period_start ({period_start}) cannot be after period_end ({period_end}).")

        # 3. Verify user existence
        user = await ChatRepository.get_user_by_id(db, user_id=user_id)
        if not user:
            raise AnalyticalReportException(f"User '{user_id}' not found.")

        # 4. Fetch baseline record (M9)
        baseline_record = await ChatRepository.get_personal_baseline_by_user(
            session=db,
            user_id=user_id,
            window_type="all_time",
            feature_vector_version=self.FEATURE_VECTOR_VERSION
        )

        # 5. Fetch temporal trend records (M9) for user in period
        temporal_records = await ChatRepository.get_user_temporal_records_for_period(
            session=db,
            user_id=user_id,
            room_id=room_id,
            period_start=period_start,
            period_end=period_end,
            feature_version=self.FEATURE_VECTOR_VERSION
        )

        if not temporal_records:
            raise InsufficientReportDataError(
                f"No temporal trend observations available for user '{user_id}' in the specified period."
            )

        # 6. Map message timestamps
        msg_time_map: Dict[str, datetime] = {}
        for r in temporal_records:
            if hasattr(r, "message") and r.message:
                msg_time_map[r.message_id] = r.message.timestamp

        # 7. Assemble Report using pure math engine
        report_id = str(uuid.uuid4())
        try:
            report = AnalyticalReportEngine.assemble_report(
                report_id=report_id,
                user_id=user_id,
                room_id=room_id,
                period_start=period_start,
                period_end=period_end,
                baseline_record=baseline_record,
                temporal_records=temporal_records,
                message_timestamp_map=msg_time_map
            )
        except Exception as exc:
            logger.error(f"Engine failed to assemble report for user '{user_id}': {exc}", exc_info=True)
            raise ReportGenerationError(f"Failed to assemble analytical report: {str(exc)}") from exc

        # 8. Persist report record for historical reproducibility
        report_dict = report.model_dump(mode="json")
        await ChatRepository.save_analytical_report(
            session=db,
            report_id=report_id,
            user_id=user_id,
            room_id=room_id,
            report_type=report_type,
            feature_vector_version=self.FEATURE_VECTOR_VERSION,
            status="completed",
            period_start=period_start,
            period_end=period_end,
            observation_count=report.metadata.observation_count,
            baseline_sample_count=report.metadata.baseline_sample_count,
            baseline_established=report.metadata.baseline_established,
            modality_coverage=report.modality_coverage.model_dump(mode="json"),
            deviation_summary=report.deviation_summary.model_dump(mode="json"),
            trend_summary=report.trend_summary.model_dump(mode="json"),
            feature_shift_summary=[item.model_dump(mode="json") for item in report.feature_shifts],
            trajectory_summary=report.trajectory_summary.model_dump(mode="json"),
            report_payload=report_dict
        )

        logger.info(f"Generated analytical report '{report_id}' for user '{user_id}' with {len(temporal_records)} observations.")
        return report

    async def get_report_by_id(
        self,
        report_id: str,
        db: AsyncSession,
        requesting_user_id: Optional[str] = None
    ) -> AnalyticalReport:
        """
        Retrieves an exact point-in-time persisted analytical report by ID.
        """
        record = await ChatRepository.get_analytical_report_by_id(db, report_id=report_id)
        if not record:
            raise ReportNotFoundError(f"Analytical report '{report_id}' not found.")

        if requesting_user_id and requesting_user_id != record.user_id:
            raise UserAccessDeniedError(f"User '{requesting_user_id}' is not authorized to access report '{report_id}'.")

        # Return the exact persisted snapshot payload
        return AnalyticalReport.model_validate(record.report_payload)

    async def get_latest_user_report(
        self,
        user_id: str,
        db: AsyncSession,
        room_id: Optional[str] = None,
        requesting_user_id: Optional[str] = None
    ) -> Optional[AnalyticalReport]:
        """
        Retrieves the latest persisted report for a user, or generates a new one if none exists.
        """
        if requesting_user_id and requesting_user_id != user_id:
            raise UserAccessDeniedError(f"User '{requesting_user_id}' is not authorized to access reports for '{user_id}'.")

        record = await ChatRepository.get_latest_analytical_report_for_user(db, user_id=user_id, room_id=room_id)
        if record:
            return AnalyticalReport.model_validate(record.report_payload)

        # Generate on the fly if user has observations
        try:
            return await self.generate_user_report(
                user_id=user_id,
                db=db,
                room_id=room_id,
                requesting_user_id=requesting_user_id
            )
        except InsufficientReportDataError:
            return None


analytical_report_service = AnalyticalReportService.get_instance()
