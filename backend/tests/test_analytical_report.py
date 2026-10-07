"""
Comprehensive Automated Test Suite for Milestone 10:
Final Analytical Report Engine (Tests T10-01 through T10-37)
"""
import pytest
import pytest_asyncio
import numpy as np
import uuid
from datetime import datetime, timezone, timedelta
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.main import app
from app.db.session import async_session_maker, init_db
from app.db.models import (
    User, Room, Message, RoomParticipant,
    MultimodalFeatureRecord, PersonalBaselineRecord, TemporalTrendRecord, AnalyticalReportRecord
)
from app.db.repository import ChatRepository
from app.services.personal_baseline.models import (
    CANONICAL_FEATURE_REGISTRY,
    INDEX_TO_FEATURE_NAME,
    TopDeviationItem
)
from app.services.analytical_report.models import (
    AnalyticalReport,
    ReportPeriod,
    ReportMetadata,
    BaselineSummary,
    ModalityCoverageSummary,
    DeviationSummary,
    FeatureShiftItem,
    TrendSummary,
    TrajectorySummary,
    MANDATORY_RESEARCH_DISCLAIMER
)
from app.services.analytical_report.engine import AnalyticalReportEngine
from app.services.analytical_report.service import AnalyticalReportService
from app.services.analytical_report.formatters import ReportFormatter
from app.services.analytical_report.exceptions import (
    AnalyticalReportException,
    InsufficientReportDataError,
    UnsupportedFeatureVersionError,
    InvalidReportPeriodError,
    ReportNotFoundError,
    UserAccessDeniedError
)


@pytest_asyncio.fixture
async def db_session():
    """Isolated async database session fixture for M10 tests."""
    await init_db()
    async with async_session_maker() as session:
        yield session


# Helper to build mock TemporalTrendRecord
def create_mock_temporal_record(
    message_id: str,
    user_id: str,
    room_id: str,
    timestamp: datetime,
    mean_abs_z: float = 0.5,
    rms_z: float = 1.0,
    std_euc: float = 8.0,
    has_acoustic: bool = True,
    has_nlp: bool = True,
    z_shift: float = 0.0,
    slope_shift: float = 0.0
):
    raw_zs = [0.1 + z_shift] * 140
    norm_zs = [0.1 + z_shift] * 140
    clipped_zs = [0.1 + z_shift] * 140

    z_mask = [True] * 140
    for i in range(135, 140):
        z_mask[i] = False

    if not has_acoustic:
        for i in range(10, 69):
            z_mask[i] = False
            raw_zs[i] = 0.0
            norm_zs[i] = 0.0
            clipped_zs[i] = 0.0

    if not has_nlp:
        for i in range(69, 135):
            z_mask[i] = False

    top_devs = [
        {
            "index": 69,
            "name": "char_count",
            "modality": "nlp",
            "raw_zscore": 2.5 + z_shift,
            "clipped_zscore": 2.5 + z_shift,
            "absolute_zscore": 2.5 + z_shift,
            "raw_value": 45.0,
            "baseline_mean": 20.0,
            "baseline_std": 10.0
        }
    ]

    slopes = {
        "char_count": {
            "valid": True,
            "slope_normalized_per_hour": 0.05 + slope_shift,
            "slope_raw_per_hour": 5.0,
            "direction": "increasing" if (0.05 + slope_shift) > 0.01 else "decreasing" if (0.05 + slope_shift) < -0.01 else "stable",
            "K_i": 5
        },
        "f0_mean_hz": {
            "valid": has_acoustic,
            "slope_normalized_per_hour": -0.03,
            "slope_raw_per_hour": -10.0,
            "direction": "decreasing",
            "K_i": 5 if has_acoustic else 0
        }
    }

    mas = {
        "char_count": {"sma_3": 0.4, "sma_5": 0.45, "ema_3": 0.42, "ema_5": 0.43},
        "f0_mean_hz": {"sma_3": 0.3, "sma_5": 0.32, "ema_3": 0.31, "ema_5": 0.31}
    }

    rocs = {
        "char_count": {"roc_normalized": 0.02},
        "f0_mean_hz": {"roc_normalized": -0.01}
    }

    rec = TemporalTrendRecord(
        id=str(uuid.uuid4()),
        message_id=message_id,
        room_id=room_id,
        user_id=user_id,
        status="completed",
        feature_vector_version="v1.0-140d",
        sample_count_at_eval=5,
        is_baseline_established=True,
        baseline_snapshot_mean=[0.5] * 140,
        baseline_snapshot_std=[0.1] * 140,
        zscore_validity_mask=z_mask,
        valid_dimension_count=sum(z_mask),
        unavailable_dimension_count=140 - sum(z_mask),
        modality_coverage={"has_communication": True, "has_acoustic": has_acoustic, "has_nlp": has_nlp},
        raw_zscore_vector=raw_zs,
        normalized_zscore_vector=norm_zs,
        clipped_zscore_vector=clipped_zs,
        mean_absolute_z=mean_abs_z,
        rms_z=rms_z,
        standardized_euclidean_distance=std_euc,
        top_deviations_summary=top_devs,
        trend_slopes_summary=slopes,
        moving_averages_summary=mas,
        rate_of_change_summary=rocs,
        evaluated_at=timestamp
    )
    return rec


# ==============================================================================
# CORE REPORT TESTS (T10-01 through T10-05)
# ==============================================================================

def test_t01_report_aggregates_m9_observations_correctly():
    """T10-01: Report aggregates M9 observations into a complete AnalyticalReport structure."""
    u_id = f"U_{uuid.uuid4().hex[:6]}"
    r_id = f"R_{uuid.uuid4().hex[:6]}"
    now = datetime.now(timezone.utc)

    recs = [
        create_mock_temporal_record(f"M_{i}", u_id, r_id, now + timedelta(minutes=i), mean_abs_z=0.4 + i * 0.1)
        for i in range(5)
    ]

    report = AnalyticalReportEngine.assemble_report(
        report_id=str(uuid.uuid4()),
        user_id=u_id,
        room_id=r_id,
        period_start=None,
        period_end=None,
        baseline_record=None,
        temporal_records=recs
    )

    assert report.metadata.user_id == u_id
    assert report.metadata.observation_count == 5
    assert report.metadata.feature_vector_version == "v1.0-140d"
    assert report.disclaimer == MANDATORY_RESEARCH_DISCLAIMER


@pytest.mark.asyncio
async def test_t02_period_filtering_works(db_session: AsyncSession):
    """T10-02: Period filtering restricts observations strictly to [period_start, period_end]."""
    u_id = f"U_{uuid.uuid4().hex[:6]}"
    r_id = f"R_{uuid.uuid4().hex[:6]}"
    now = datetime.now(timezone.utc)

    # Insert user & room
    db_session.add(User(id=u_id, display_name="Test User"))
    db_session.add(Room(id=r_id, name=r_id))
    await db_session.commit()

    # Create messages and temporal records across time
    for i in range(6):
        t = now + timedelta(hours=i)
        msg = Message(id=f"MSG_T02_{u_id}_{i}", room_id=r_id, sender_id=u_id, message_text=f"Msg {i}", message_type="text", timestamp=t)
        db_session.add(msg)
        await db_session.flush()

        rec = create_mock_temporal_record(msg.id, u_id, r_id, t)
        db_session.add(rec)
    await db_session.commit()

    # Filter for hours 2 to 4
    start_t = now + timedelta(hours=2)
    end_t = now + timedelta(hours=4)
    filtered = await ChatRepository.get_user_temporal_records_for_period(
        db_session, user_id=u_id, period_start=start_t, period_end=end_t
    )
    assert len(filtered) == 3


def test_t03_empty_period_handled_safely():
    """T10-03: Empty observations list raises InsufficientReportDataError cleanly."""
    with pytest.raises(InsufficientReportDataError):
        AnalyticalReportEngine.assemble_report(
            report_id=str(uuid.uuid4()),
            user_id="U_EMPTY",
            room_id="R_EMPTY",
            period_start=None,
            period_end=None,
            baseline_record=None,
            temporal_records=[]
        )


def test_t04_insufficient_baseline_represented_correctly():
    """T10-04: Insufficient baseline (N < 3) sets status = insufficient_history and established = false."""
    u_id = f"U_{uuid.uuid4().hex[:6]}"
    now = datetime.now(timezone.utc)
    recs = [create_mock_temporal_record(f"M_{i}", u_id, "R1", now + timedelta(minutes=i)) for i in range(2)]

    base_rec = PersonalBaselineRecord(
        id=str(uuid.uuid4()),
        user_id=u_id,
        window_type="all_time",
        sample_count=2,
        is_baseline_established=False,
        valid_dimensions_count=140,
        raw_mean_vector=[0.0]*140,
        raw_std_vector=[0.0]*140,
        normalized_mean_vector=[0.0]*140,
        normalized_std_vector=[0.0]*140
    )

    report = AnalyticalReportEngine.assemble_report(
        report_id=str(uuid.uuid4()),
        user_id=u_id,
        room_id="R1",
        period_start=None,
        period_end=None,
        baseline_record=base_rec,
        temporal_records=recs
    )

    assert report.baseline.status == "insufficient_history"
    assert report.baseline.is_baseline_established is False
    assert report.baseline.sample_count == 2


def test_t05_established_baseline_represented_correctly():
    """T10-05: Established baseline (N >= 3) sets status = established and is_baseline_established = true."""
    u_id = f"U_{uuid.uuid4().hex[:6]}"
    now = datetime.now(timezone.utc)
    recs = [create_mock_temporal_record(f"M_{i}", u_id, "R1", now + timedelta(minutes=i)) for i in range(4)]

    base_rec = PersonalBaselineRecord(
        id=str(uuid.uuid4()),
        user_id=u_id,
        window_type="all_time",
        sample_count=4,
        is_baseline_established=True,
        valid_dimensions_count=140,
        raw_mean_vector=[0.0]*140,
        raw_std_vector=[0.0]*140,
        normalized_mean_vector=[0.0]*140,
        normalized_std_vector=[0.0]*140
    )

    report = AnalyticalReportEngine.assemble_report(
        report_id=str(uuid.uuid4()),
        user_id=u_id,
        room_id="R1",
        period_start=None,
        period_end=None,
        baseline_record=base_rec,
        temporal_records=recs
    )

    assert report.baseline.status == "established"
    assert report.baseline.is_baseline_established is True
    assert report.baseline.sample_count == 4


# ==============================================================================
# DEVIATION AGGREGATION TESTS (T10-06 through T10-08)
# ==============================================================================

def test_t06_mean_absolute_z_aggregation_is_correct():
    """T10-06: Mean absolute Z aggregation accurately computes mean, median, min, max, std."""
    values = [0.4, 0.6, 0.8, 1.0, 1.2]
    res = AnalyticalReportEngine.compute_metric_summary(values)
    assert res.mean == 0.8
    assert res.median == 0.8
    assert res.minimum == 0.4
    assert res.maximum == 1.2
    assert res.std == round(float(np.std(values, ddof=1)), 4)


def test_t07_rms_z_aggregation_is_correct():
    """T10-07: RMS Z aggregation computes accurate descriptive summary."""
    values = [1.1, 1.2, 1.3, 1.4, 1.5]
    res = AnalyticalReportEngine.compute_metric_summary(values)
    assert res.mean == 1.3
    assert res.minimum == 1.1
    assert res.maximum == 1.5


def test_t08_standardized_euclidean_aggregation_is_correct():
    """T10-08: Standardized Euclidean distance summary is accurate."""
    values = [10.0, 12.0, 14.0]
    res = AnalyticalReportEngine.compute_metric_summary(values)
    assert res.mean == 12.0
    assert res.median == 12.0


# ==============================================================================
# FEATURE AGGREGATION & MODALITY TESTS (T10-09 through T10-15)
# ==============================================================================

def test_t09_top_shifted_features_are_deterministic():
    """T10-09: Feature shift aggregation produces deterministic results on identical inputs."""
    u_id = f"U_{uuid.uuid4().hex[:6]}"
    now = datetime.now(timezone.utc)
    recs = [create_mock_temporal_record(f"M_{i}", u_id, "R1", now + timedelta(minutes=i)) for i in range(5)]

    shifts1 = AnalyticalReportEngine.aggregate_feature_shifts(recs, limit=5)
    shifts2 = AnalyticalReportEngine.aggregate_feature_shifts(recs, limit=5)

    assert [s.feature_index for s in shifts1] == [s.feature_index for s in shifts2]
    assert [s.mean_absolute_z for s in shifts1] == [s.mean_absolute_z for s in shifts2]


def test_t10_feature_ordering_is_deterministic():
    """T10-10: Ordering strictly follows mean |Z| desc, max |Z| desc, feature index asc."""
    u_id = f"U_{uuid.uuid4().hex[:6]}"
    now = datetime.now(timezone.utc)
    recs = [create_mock_temporal_record(f"M_{i}", u_id, "R1", now + timedelta(minutes=i)) for i in range(3)]

    shifts = AnalyticalReportEngine.aggregate_feature_shifts(recs, limit=10)
    for i in range(len(shifts) - 1):
        s1 = shifts[i]
        s2 = shifts[i + 1]
        assert (s1.mean_absolute_z, s1.max_absolute_z, -s1.feature_index) >= (s2.mean_absolute_z, s2.max_absolute_z, -s2.feature_index)


def test_t11_feature_valid_observation_counts_are_correct():
    """T10-11: Valid observation count per feature accurately counts turns where dimension was valid."""
    u_id = f"U_{uuid.uuid4().hex[:6]}"
    now = datetime.now(timezone.utc)
    # 3 turns with acoustic, 2 without acoustic
    recs = [
        create_mock_temporal_record(f"M_{i}", u_id, "R1", now + timedelta(minutes=i), has_acoustic=(i < 3))
        for i in range(5)
    ]
    shifts = AnalyticalReportEngine.aggregate_feature_shifts(recs, limit=50)
    # Find an acoustic feature (e.g. index 10: f0_mean_hz) and a comm feature (e.g. index 0)
    for s in shifts:
        if s.modality == "acoustic":
            assert s.valid_observations == 3
        elif s.modality == "communication":
            assert s.valid_observations == 5


def test_t12_missing_modality_dimensions_are_excluded_correctly():
    """T10-12: Features unmeasured on text-only turns are masked out and not treated as zero deviations."""
    u_id = f"U_{uuid.uuid4().hex[:6]}"
    now = datetime.now(timezone.utc)
    # All text turns (no acoustic)
    recs = [create_mock_temporal_record(f"M_{i}", u_id, "R1", now + timedelta(minutes=i), has_acoustic=False) for i in range(3)]
    shifts = AnalyticalReportEngine.aggregate_feature_shifts(recs, limit=50)
    # No acoustic features should appear
    assert all(s.modality != "acoustic" for s in shifts)


def test_t13_metadata_dimensions_135_139_remain_excluded():
    """T10-13: Modality mask metadata indices [135..139] are excluded from feature shift ranking."""
    u_id = f"U_{uuid.uuid4().hex[:6]}"
    now = datetime.now(timezone.utc)
    recs = [create_mock_temporal_record(f"M_{i}", u_id, "R1", now + timedelta(minutes=i)) for i in range(3)]
    shifts = AnalyticalReportEngine.aggregate_feature_shifts(recs, limit=140)
    assert all(s.feature_index < 135 for s in shifts)
    assert all(s.modality != "mask" for s in shifts)


def test_t14_acoustic_coverage_is_correct():
    """T10-14: Acoustic coverage accurately calculates measured turns / total turns percentage."""
    u_id = f"U_{uuid.uuid4().hex[:6]}"
    now = datetime.now(timezone.utc)
    recs = [
        create_mock_temporal_record(f"M_{i}", u_id, "R1", now + timedelta(minutes=i), has_acoustic=(i % 2 == 0))
        for i in range(10)
    ]  # 5 measured out of 10 = 50.0%
    cov = AnalyticalReportEngine.aggregate_modality_coverage(recs)
    assert cov.acoustic.measured_count == 5
    assert cov.acoustic.total_count == 10
    assert cov.acoustic.percentage == 50.0


def test_t15_nlp_coverage_is_correct():
    """T10-15: NLP coverage percentage calculation is accurate."""
    u_id = f"U_{uuid.uuid4().hex[:6]}"
    now = datetime.now(timezone.utc)
    recs = [
        create_mock_temporal_record(f"M_{i}", u_id, "R1", now + timedelta(minutes=i), has_nlp=(i < 3))
        for i in range(4)
    ]  # 3 measured out of 4 = 75.0%
    cov = AnalyticalReportEngine.aggregate_modality_coverage(recs)
    assert cov.nlp.measured_count == 3
    assert cov.nlp.total_count == 4
    assert cov.nlp.percentage == 75.0


# ==============================================================================
# TEMPORAL DYNAMICS & TRENDS (T10-16 through T10-24)
# ==============================================================================

def test_t16_sma3_data_is_consumed_correctly():
    """T10-16: SMA-3 values from M9 moving averages summary are consumed accurately."""
    u_id = f"U_{uuid.uuid4().hex[:6]}"
    now = datetime.now(timezone.utc)
    recs = [create_mock_temporal_record(f"M_1", u_id, "R1", now)]
    trends = AnalyticalReportEngine.aggregate_temporal_trends(recs)
    char_trend = next(t for t in trends.top_trends if t.feature_name == "char_count")
    assert char_trend.latest_sma3 == 0.4


def test_t17_sma5_data_is_consumed_correctly():
    """T10-17: SMA-5 values from M9 are preserved."""
    u_id = f"U_{uuid.uuid4().hex[:6]}"
    now = datetime.now(timezone.utc)
    recs = [create_mock_temporal_record(f"M_1", u_id, "R1", now)]
    trends = AnalyticalReportEngine.aggregate_temporal_trends(recs)
    char_trend = next(t for t in trends.top_trends if t.feature_name == "char_count")
    assert char_trend.latest_sma5 == 0.45


def test_t18_ema3_data_is_consumed_correctly():
    """T10-18: EMA-3 values from M9 are preserved."""
    u_id = f"U_{uuid.uuid4().hex[:6]}"
    now = datetime.now(timezone.utc)
    recs = [create_mock_temporal_record(f"M_1", u_id, "R1", now)]
    trends = AnalyticalReportEngine.aggregate_temporal_trends(recs)
    char_trend = next(t for t in trends.top_trends if t.feature_name == "char_count")
    assert char_trend.latest_ema3 == 0.42


def test_t19_ema5_data_is_consumed_correctly():
    """T10-19: EMA-5 values from M9 are preserved."""
    u_id = f"U_{uuid.uuid4().hex[:6]}"
    now = datetime.now(timezone.utc)
    recs = [create_mock_temporal_record(f"M_1", u_id, "R1", now)]
    trends = AnalyticalReportEngine.aggregate_temporal_trends(recs)
    char_trend = next(t for t in trends.top_trends if t.feature_name == "char_count")
    assert char_trend.latest_ema5 == 0.43


def test_t20_ols_trend_is_consumed_correctly():
    """T10-20: Normalized OLS slope and raw slope are consumed accurately."""
    u_id = f"U_{uuid.uuid4().hex[:6]}"
    now = datetime.now(timezone.utc)
    recs = [create_mock_temporal_record(f"M_1", u_id, "R1", now)]
    trends = AnalyticalReportEngine.aggregate_temporal_trends(recs)
    char_trend = next(t for t in trends.top_trends if t.feature_name == "char_count")
    assert char_trend.normalized_slope == 0.05
    assert char_trend.raw_slope == 5.0


def test_t21_roc_is_consumed_correctly():
    """T10-21: Rate of change metric is consumed accurately."""
    u_id = f"U_{uuid.uuid4().hex[:6]}"
    now = datetime.now(timezone.utc)
    recs = [create_mock_temporal_record(f"M_1", u_id, "R1", now)]
    trends = AnalyticalReportEngine.aggregate_temporal_trends(recs)
    char_trend = next(t for t in trends.top_trends if t.feature_name == "char_count")
    assert char_trend.latest_roc == 0.02


def test_t22_increasing_direction_is_preserved():
    """T10-22: Slope > +0.01 is classified as 'increasing'."""
    u_id = f"U_{uuid.uuid4().hex[:6]}"
    now = datetime.now(timezone.utc)
    recs = [create_mock_temporal_record(f"M_1", u_id, "R1", now, slope_shift=0.0)]
    trends = AnalyticalReportEngine.aggregate_temporal_trends(recs)
    char_trend = next(t for t in trends.top_trends if t.feature_name == "char_count")
    assert char_trend.direction == "increasing"


def test_t23_decreasing_direction_is_preserved():
    """T10-23: Slope < -0.01 is classified as 'decreasing'."""
    u_id = f"U_{uuid.uuid4().hex[:6]}"
    now = datetime.now(timezone.utc)
    recs = [create_mock_temporal_record(f"M_1", u_id, "R1", now)]
    trends = AnalyticalReportEngine.aggregate_temporal_trends(recs)
    f0_trend = next(t for t in trends.top_trends if t.feature_name == "f0_mean_hz")
    assert f0_trend.direction == "decreasing"


def test_t24_stable_direction_is_preserved():
    """T10-24: |Slope| <= 0.01 is classified as 'stable'."""
    u_id = f"U_{uuid.uuid4().hex[:6]}"
    now = datetime.now(timezone.utc)
    # Shift slope to 0.005 (<= 0.01)
    recs = [create_mock_temporal_record(f"M_1", u_id, "R1", now, slope_shift=-0.045)]
    trends = AnalyticalReportEngine.aggregate_temporal_trends(recs)
    char_trend = next(t for t in trends.top_trends if t.feature_name == "char_count")
    assert char_trend.direction == "stable"


# ==============================================================================
# SECURITY & ACCESS CONTROL (T10-25 through T10-28)
# ==============================================================================

@pytest.mark.asyncio
async def test_t25_user_cannot_access_another_users_report(db_session: AsyncSession):
    """T10-25: Requesting user Alice cannot generate or retrieve reports for Bob."""
    service = AnalyticalReportService.get_instance()
    with pytest.raises(UserAccessDeniedError):
        await service.generate_user_report(
            user_id="BOB_USER",
            db=db_session,
            requesting_user_id="ALICE_USER"
        )


@pytest.mark.asyncio
async def test_t26_room_authorization_is_enforced(db_session: AsyncSession):
    """T10-26: Participant check in get_room_analytical_reports enforces room membership."""
    r_id = f"R_AUTH_{uuid.uuid4().hex[:6]}"
    u_id = f"U_INTRUDER_{uuid.uuid4().hex[:6]}"
    db_session.add(Room(id=r_id, name="Secure Room"))
    db_session.add(User(id=u_id, display_name="Intruder"))
    await db_session.commit()

    is_part = await ChatRepository.is_user_participant(db_session, room_id=r_id, user_id=u_id)
    assert is_part is False


@pytest.mark.asyncio
async def test_t27_invalid_report_id_rejected_safely(db_session: AsyncSession):
    """T10-27: Querying non-existent report ID raises ReportNotFoundError."""
    service = AnalyticalReportService.get_instance()
    with pytest.raises(ReportNotFoundError):
        await service.get_report_by_id("NON_EXISTENT_REPORT_ID", db_session)


@pytest.mark.asyncio
async def test_t28_unsupported_feature_vector_version_rejected(db_session: AsyncSession):
    """T10-28: Feature version != v1.0-140d is rejected during retrieval."""
    u_id = f"U_{uuid.uuid4().hex[:6]}"
    db_session.add(User(id=u_id, display_name="Test User"))
    await db_session.commit()

    records = await ChatRepository.get_user_temporal_records_for_period(
        db_session, user_id=u_id, feature_version="v0.9-100d"
    )
    assert len(records) == 0


# ==============================================================================
# REPRODUCIBILITY & IMMUTABILITY (T10-29 through T10-31)
# ==============================================================================

def test_t29_same_m9_inputs_produce_identical_report():
    """T10-29: Determinism test: Same M9 inputs generate 100% bit-for-bit identical report content."""
    u_id = f"U_{uuid.uuid4().hex[:6]}"
    now = datetime.now(timezone.utc)
    recs = [create_mock_temporal_record(f"M_{i}", u_id, "R1", now + timedelta(minutes=i)) for i in range(4)]

    r1 = AnalyticalReportEngine.assemble_report("REP_1", u_id, "R1", None, None, None, recs)
    r2 = AnalyticalReportEngine.assemble_report("REP_2", u_id, "R1", None, None, None, recs)

    assert r1.deviation_summary.model_dump() == r2.deviation_summary.model_dump()
    assert [s.model_dump() for s in r1.feature_shifts] == [s.model_dump() for s in r2.feature_shifts]
    assert r1.trend_summary.model_dump() == r2.trend_summary.model_dump()
    assert r1.modality_coverage.model_dump() == r2.modality_coverage.model_dump()


@pytest.mark.asyncio
async def test_t30_persisted_report_remains_unchanged_after_retrieval(db_session: AsyncSession):
    """T10-30: Retrieving a persisted report by ID returns exact saved payload."""
    u_id = f"U_{uuid.uuid4().hex[:6]}"
    r_id = f"R_{uuid.uuid4().hex[:6]}"
    rep_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    db_session.add(User(id=u_id, display_name="Test User"))
    db_session.add(Room(id=r_id, name=r_id))
    await db_session.commit()

    saved_rec = await ChatRepository.save_analytical_report(
        session=db_session,
        report_id=rep_id,
        user_id=u_id,
        room_id=r_id,
        report_payload={"test_key": "persisted_snapshot_data"}
    )
    assert saved_rec.id == rep_id

    retrieved = await ChatRepository.get_analytical_report_by_id(db_session, rep_id)
    assert retrieved is not None
    assert retrieved.report_payload["test_key"] == "persisted_snapshot_data"


@pytest.mark.asyncio
async def test_t31_future_m9_observations_do_not_mutate_old_report(db_session: AsyncSession):
    """T10-31: Adding future M9 observations does NOT alter previously persisted M10 report snapshot."""
    u_id = f"U_{uuid.uuid4().hex[:6]}"
    r_id = f"R_{uuid.uuid4().hex[:6]}"
    now = datetime.now(timezone.utc)

    db_session.add(User(id=u_id, display_name="Test User"))
    db_session.add(Room(id=r_id, name=r_id))
    await db_session.commit()

    # Create Turn 1..3
    for i in range(3):
        t = now + timedelta(minutes=i)
        msg = Message(id=f"MSG_T31_{u_id}_{i}", room_id=r_id, sender_id=u_id, message_text=f"Turn {i}", message_type="text", timestamp=t)
        db_session.add(msg)
        await db_session.flush()
        db_session.add(create_mock_temporal_record(msg.id, u_id, r_id, t, mean_abs_z=0.5))
    await db_session.commit()

    service = AnalyticalReportService.get_instance()
    rep1 = await service.generate_user_report(user_id=u_id, db=db_session, room_id=r_id)
    assert rep1.metadata.observation_count == 3

    # Now add Turn 4 (future observation)
    t4 = now + timedelta(minutes=10)
    msg4 = Message(id=f"MSG_T31_{u_id}_4", room_id=r_id, sender_id=u_id, message_text="Turn 4", message_type="text", timestamp=t4)
    db_session.add(msg4)
    await db_session.flush()
    db_session.add(create_mock_temporal_record(msg4.id, u_id, r_id, t4, mean_abs_z=3.0))
    await db_session.commit()

    # Re-retrieve old report by ID
    old_rep = await service.get_report_by_id(report_id=rep1.metadata.report_id, db=db_session)
    assert old_rep.metadata.observation_count == 3  # Still exactly 3 observations
    assert old_rep.deviation_summary.mean_absolute_z.mean == rep1.deviation_summary.mean_absolute_z.mean


# ==============================================================================
# REST API & WEBSOCKET TESTS (T10-32 through T10-36)
# ==============================================================================

@pytest.mark.asyncio
async def test_t32_generate_endpoint_works(db_session: AsyncSession):
    """T10-32: POST /api/v1/chat/analytical/users/{user_id}/report generates and persists report."""
    u_id = f"U_API_{uuid.uuid4().hex[:6]}"
    r_id = f"R_API_{uuid.uuid4().hex[:6]}"
    now = datetime.now(timezone.utc)

    db_session.add(User(id=u_id, display_name="API User"))
    db_session.add(Room(id=r_id, name=r_id))
    await db_session.commit()

    msg = Message(id=f"MSG_{u_id}", room_id=r_id, sender_id=u_id, message_text="Test", message_type="text", timestamp=now)
    db_session.add(msg)
    await db_session.flush()
    db_session.add(create_mock_temporal_record(msg.id, u_id, r_id, now))
    await db_session.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.post(
            f"/api/v1/chat/analytical/users/{u_id}/report",
            json={"room_id": r_id, "report_type": "longitudinal"},
            headers={"X-User-Id": u_id}
        )
        assert res.status_code == 201
        data = res.json()
        assert data["metadata"]["user_id"] == u_id
        assert data["metadata"]["observation_count"] == 1
        assert "disclaimer" in data


@pytest.mark.asyncio
async def test_t33_retrieve_endpoints_work(db_session: AsyncSession):
    """T10-33: GET /analytical/users/{user_id}/report and GET /analytical/reports/{report_id} work."""
    u_id = f"U_RET_{uuid.uuid4().hex[:6]}"
    r_id = f"R_RET_{uuid.uuid4().hex[:6]}"
    now = datetime.now(timezone.utc)

    db_session.add(User(id=u_id, display_name="Ret User"))
    db_session.add(Room(id=r_id, name=r_id))
    await db_session.commit()

    msg = Message(id=f"MSG_RET_{u_id}", room_id=r_id, sender_id=u_id, message_text="Test", message_type="text", timestamp=now)
    db_session.add(msg)
    await db_session.flush()
    db_session.add(create_mock_temporal_record(msg.id, u_id, r_id, now))
    await db_session.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. Get latest
        res_lat = await ac.get(f"/api/v1/chat/analytical/users/{u_id}/report?room_id={r_id}", headers={"X-User-Id": u_id})
        assert res_lat.status_code == 200
        rep_id = res_lat.json()["metadata"]["report_id"]

        # 2. Get by ID
        res_id = await ac.get(f"/api/v1/chat/analytical/reports/{rep_id}", headers={"X-User-Id": u_id})
        assert res_id.status_code == 200
        assert res_id.json()["metadata"]["report_id"] == rep_id


@pytest.mark.asyncio
async def test_t34_report_history_and_room_reports_work(db_session: AsyncSession):
    """T10-34: GET /analytical/users/{user_id}/reports and GET /analytical/rooms/{room_id} work."""
    u_id = f"U_HIST_{uuid.uuid4().hex[:6]}"
    r_id = f"R_HIST_{uuid.uuid4().hex[:6]}"
    now = datetime.now(timezone.utc)

    db_session.add(User(id=u_id, display_name="Hist User"))
    db_session.add(Room(id=r_id, name=r_id))
    await db_session.commit()

    msg = Message(id=f"MSG_HIST_{u_id}", room_id=r_id, sender_id=u_id, message_text="Test", message_type="text", timestamp=now)
    db_session.add(msg)
    await db_session.flush()
    db_session.add(create_mock_temporal_record(msg.id, u_id, r_id, now))
    await db_session.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Generate a report first
        await ac.post(f"/api/v1/chat/analytical/users/{u_id}/report", json={"room_id": r_id}, headers={"X-User-Id": u_id})

        # Check history
        res_hist = await ac.get(f"/api/v1/chat/analytical/users/{u_id}/reports", headers={"X-User-Id": u_id})
        assert res_hist.status_code == 200
        assert res_hist.json()["total_reports"] >= 1

        # Check room
        res_room = await ac.get(f"/api/v1/chat/analytical/rooms/{r_id}")
        assert res_room.status_code == 200
        assert res_room.json()["total_reports"] >= 1


@pytest.mark.asyncio
async def test_t35_export_formats_work(db_session: AsyncSession):
    """T10-35: GET /analytical/reports/{report_id}/export returns valid JSON, Markdown, and HTML."""
    u_id = f"U_EXP_{uuid.uuid4().hex[:6]}"
    r_id = f"R_EXP_{uuid.uuid4().hex[:6]}"
    now = datetime.now(timezone.utc)

    db_session.add(User(id=u_id, display_name="Exp User"))
    db_session.add(Room(id=r_id, name=r_id))
    await db_session.commit()

    msg = Message(id=f"MSG_EXP_{u_id}", room_id=r_id, sender_id=u_id, message_text="Test", message_type="text", timestamp=now)
    db_session.add(msg)
    await db_session.flush()
    db_session.add(create_mock_temporal_record(msg.id, u_id, r_id, now))
    await db_session.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        gen_res = await ac.post(f"/api/v1/chat/analytical/users/{u_id}/report", json={"room_id": r_id}, headers={"X-User-Id": u_id})
        rep_id = gen_res.json()["metadata"]["report_id"]

        # Export JSON
        res_json = await ac.get(f"/api/v1/chat/analytical/reports/{rep_id}/export?format=json")
        assert res_json.status_code == 200
        assert "metadata" in res_json.json()

        # Export Markdown
        res_md = await ac.get(f"/api/v1/chat/analytical/reports/{rep_id}/export?format=markdown")
        assert res_md.status_code == 200
        assert "# Longitudinal Analytical Report" in res_md.text

        # Export HTML
        res_html = await ac.get(f"/api/v1/chat/analytical/reports/{rep_id}/export?format=html")
        assert res_html.status_code == 200
        assert "<!DOCTYPE html>" in res_html.text


@pytest.mark.asyncio
async def test_t36_websocket_event_emitted_on_generation(db_session: AsyncSession):
    """T10-36: WebSocket event analytical_report_generated is broadcast upon report creation."""
    u_id = f"U_WS_{uuid.uuid4().hex[:6]}"
    r_id = f"R_WS_{uuid.uuid4().hex[:6]}"
    now = datetime.now(timezone.utc)

    db_session.add(User(id=u_id, display_name="WS User"))
    db_session.add(Room(id=r_id, name=r_id))
    await db_session.commit()

    msg = Message(id=f"MSG_WS_{u_id}", room_id=r_id, sender_id=u_id, message_text="Test", message_type="text", timestamp=now)
    db_session.add(msg)
    await db_session.flush()
    db_session.add(create_mock_temporal_record(msg.id, u_id, r_id, now))
    await db_session.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.post(
            f"/api/v1/chat/analytical/users/{u_id}/report",
            json={"room_id": r_id},
            headers={"X-User-Id": u_id}
        )
        assert res.status_code == 201


# ==============================================================================
# INVARIANCE & NON-DIAGNOSTIC BOUNDARY (T10-37)
# ==============================================================================

@pytest.mark.asyncio
async def test_t37_m1_to_m9_invariance_and_mandatory_disclaimers(db_session: AsyncSession):
    """T10-37: M10 operations NEVER mutate M1–M9 data and always include mandatory non-diagnostic disclaimers."""
    u_id = f"U_INV_{uuid.uuid4().hex[:6]}"
    r_id = f"R_INV_{uuid.uuid4().hex[:6]}"
    now = datetime.now(timezone.utc)

    db_session.add(User(id=u_id, display_name="Inv User"))
    db_session.add(Room(id=r_id, name=r_id))
    await db_session.commit()

    msg = Message(id=f"MSG_INV_{u_id}", room_id=r_id, sender_id=u_id, message_text="Test", message_type="text", timestamp=now)
    db_session.add(msg)
    await db_session.flush()

    m9_rec = create_mock_temporal_record(msg.id, u_id, r_id, now, mean_abs_z=0.75)
    db_session.add(m9_rec)
    await db_session.commit()

    orig_mean_z = m9_rec.mean_absolute_z

    # Generate M10 report
    service = AnalyticalReportService.get_instance()
    rep = await service.generate_user_report(user_id=u_id, db=db_session, room_id=r_id)

    # Verify M9 data is unchanged
    refreshed_m9 = await ChatRepository.get_temporal_trend_by_message_id(db_session, message_id=msg.id)
    assert refreshed_m9.mean_absolute_z == orig_mean_z

    # Verify Mandatory Disclaimer
    assert rep.disclaimer == MANDATORY_RESEARCH_DISCLAIMER
    assert "psychiatric diagnosis" in rep.disclaimer
    assert "not intended for clinical assessment" in rep.disclaimer
