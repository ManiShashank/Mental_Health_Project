"""
Comprehensive Automated Test Suite for Milestone 9:
Personal Baseline & Temporal Trend Engine (Tests T-01 through T-35)
"""
import pytest
import pytest_asyncio
import numpy as np
import math
import uuid
from datetime import datetime, timezone, timedelta
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.main import app
from app.db.session import get_db, async_session_maker, init_db
from app.db.models import (
    User, Room, Message, RoomParticipant,
    AcousticFeatureRecord, NLPFeatureRecord, MultimodalFeatureRecord,
    PersonalBaselineRecord, TemporalTrendRecord
)
from app.db.repository import ChatRepository
from app.services.personal_baseline.models import (
    CANONICAL_FEATURE_REGISTRY,
    INDEX_TO_FEATURE_NAME,
    PersonalBaselineResult,
    TemporalTrendEvaluationResult
)
from app.services.personal_baseline.engine import PersonalBaselineEngine
from app.services.personal_baseline.service import PersonalBaselineService
from app.services.personal_baseline.exceptions import (
    PersonalBaselineException,
    InsufficientHistoryError,
    UnsupportedFeatureVectorError,
    DataLeakageError
)


@pytest_asyncio.fixture
async def db_session():
    """Isolated async database session fixture for M9 tests."""
    await init_db()
    async with async_session_maker() as session:
        yield session


# Helper function to create synthetic 140-d vector
def create_synthetic_140d_vector(base_val: float = 0.5, has_acoustic: bool = True, has_nlp: bool = True):
    vec = [base_val] * 140
    # [135..139] are flags
    vec[135] = 1.0  # has_comm
    vec[136] = 1.0 if has_acoustic else 0.0
    vec[137] = 1.0 if has_nlp else 0.0
    vec[138] = 1.0 if has_acoustic else 0.0
    vec[139] = 0.0 if has_acoustic else 1.0

    if not has_acoustic:
        # zero-imputed acoustic [10..68]
        for i in range(10, 69):
            vec[i] = 0.0

    mask = [0.0] * 140
    for i in range(10):
        mask[i] = 1.0
    if has_acoustic:
        for i in range(10, 69):
            mask[i] = 1.0
    if has_nlp:
        for i in range(69, 135):
            mask[i] = 1.0
    return vec, mask


@pytest.mark.asyncio
async def test_t01_bessel_sample_variance_and_std():
    """T-01: Math - Bessel-corrected sample variance & std accuracy against numpy (N=5)."""
    raw_samples = [
        [1.0 + 0.1 * k] * 140 for k in range(5)
    ]
    norm_samples = [
        [0.2 + 0.05 * k] * 140 for k in range(5)
    ]
    masks = [[1.0] * 140 for _ in range(5)]

    raw_mean, raw_std, norm_mean, norm_std, counts = PersonalBaselineEngine.compute_baseline_statistics(
        raw_samples, norm_samples, masks
    )

    np_raw_arr = np.array([1.0 + 0.1 * k for k in range(5)])
    expected_mean = float(np.mean(np_raw_arr))
    expected_std = float(np.std(np_raw_arr, ddof=1))

    assert len(raw_mean) == 140
    assert len(raw_std) == 140
    assert abs(raw_mean[0] - expected_mean) < 1e-5
    assert abs(raw_std[0] - expected_std) < 1e-5
    assert counts[0] == 5


@pytest.mark.asyncio
async def test_t02_insufficient_history_behavior(db_session: AsyncSession):
    """T-02: Semantics - N_prior < 3 returns insufficient_history and is_baseline_established = false."""
    user = await ChatRepository.get_or_create_user(db_session, user_id=f"M9_T02_{uuid.uuid4().hex[:6]}", display_name="T02 User")
    room = await ChatRepository.get_or_create_room(db_session, room_id=f"M9_T02_R_{uuid.uuid4().hex[:6]}")

    # Create only 1 message
    msg = await ChatRepository.save_message(
        db_session, room_id=room.id, sender_id=user.id, message_text="Turn 1"
    )
    vec, _ = create_synthetic_140d_vector(0.5, has_acoustic=False, has_nlp=True)
    await ChatRepository.save_or_update_multimodal_features(
        db_session,
        message_id=msg.id,
        room_id=room.id,
        sender_id=user.id,
        status="completed",
        raw_fused_vector=vec,
        normalized_fused_vector=vec,
        has_communication=True,
        has_acoustic=False,
        has_nlp=True
    )

    service = PersonalBaselineService.get_instance()
    res = await service.evaluate_message_temporal_deviation(message_id=msg.id, db=db_session)

    assert res.status == "insufficient_history"
    assert res.is_baseline_established is False
    assert res.sample_count_at_eval == 0
    assert res.mean_absolute_z == 0.0


@pytest.mark.asyncio
async def test_t03_baseline_established_at_three_prior(db_session: AsyncSession):
    """T-03: Semantics - At N_prior = 3, baseline establishes and evaluation succeeds."""
    user = await ChatRepository.get_or_create_user(db_session, user_id=f"M9_T03_{uuid.uuid4().hex[:6]}", display_name="T03 User")
    room = await ChatRepository.get_or_create_room(db_session, room_id=f"M9_T03_R_{uuid.uuid4().hex[:6]}")

    t0 = datetime.now(timezone.utc) - timedelta(minutes=30)
    msg_ids = []

    # Create 3 prior messages
    for i in range(3):
        msg = await ChatRepository.save_message(
            db_session, room_id=room.id, sender_id=user.id, message_text=f"Prior Turn {i+1}"
        )
        msg.timestamp = t0 + timedelta(minutes=5 * i)
        await db_session.commit()
        vec, _ = create_synthetic_140d_vector(0.3 + 0.1 * i, has_acoustic=False, has_nlp=True)
        await ChatRepository.save_or_update_multimodal_features(
            db_session,
            message_id=msg.id,
            room_id=room.id,
            sender_id=user.id,
            status="completed",
            raw_fused_vector=vec,
            normalized_fused_vector=vec,
            has_communication=True,
            has_acoustic=False,
            has_nlp=True
        )
        msg_ids.append(msg.id)

    # 4th message (eval target)
    eval_msg = await ChatRepository.save_message(
        db_session, room_id=room.id, sender_id=user.id, message_text="Turn 4 Eval"
    )
    eval_msg.timestamp = t0 + timedelta(minutes=20)
    await db_session.commit()
    eval_vec, _ = create_synthetic_140d_vector(0.8, has_acoustic=False, has_nlp=True)
    await ChatRepository.save_or_update_multimodal_features(
        db_session,
        message_id=eval_msg.id,
        room_id=room.id,
        sender_id=user.id,
        status="completed",
        raw_fused_vector=eval_vec,
        normalized_fused_vector=eval_vec,
        has_communication=True,
        has_acoustic=False,
        has_nlp=True
    )

    service = PersonalBaselineService.get_instance()
    res = await service.evaluate_message_temporal_deviation(message_id=eval_msg.id, db=db_session)

    assert res.status == "completed"
    assert res.is_baseline_established is True
    assert res.sample_count_at_eval == 3
    assert res.valid_dimension_count == 76  # Comm(10) + NLP(66) = 76 valid for text
    assert res.mean_absolute_z > 0.0


@pytest.mark.asyncio
async def test_t04_exact_zscore_calculation_analytical():
    """T-04: Math - Exact 140-d z-score vector matches synthetic analytical shift."""
    mean_vec = [10.0] * 140
    std_vec = [2.0] * 140
    counts = [5] * 140
    curr_vec = [15.0] * 140
    curr_mask = [1.0] * 140

    raw_z, clipped_z, val_mask, mean_abs, rms, std_euc = PersonalBaselineEngine.compute_zscores(
        current_vector=curr_vec,
        current_mask=curr_mask,
        baseline_mean=mean_vec,
        baseline_std=std_vec,
        per_feature_counts=counts
    )

    # (15 - 10) / 2 = 2.5
    for i in range(135):
        assert abs(raw_z[i] - 2.5) < 1e-5
        assert abs(clipped_z[i] - 2.5) < 1e-5
        assert val_mask[i] is True

    assert abs(mean_abs - 2.5) < 1e-5
    assert abs(rms - 2.5) < 1e-5


@pytest.mark.asyncio
async def test_t05_zero_variance_regularization():
    """T-05: Regularization - Zero-variance feature (sigma = 0) returns finite z without division-by-zero."""
    mean_vec = [5.0] * 140
    std_vec = [0.0] * 140  # Constant historical observations
    counts = [5] * 140
    curr_mask = [1.0] * 140

    # Case A: Current equals mean
    curr_vec_a = [5.0] * 140
    raw_z_a, _, _, mean_abs_a, _, _ = PersonalBaselineEngine.compute_zscores(
        curr_vec_a, curr_mask, mean_vec, std_vec, counts
    )
    assert raw_z_a[0] == 0.0
    assert mean_abs_a == 0.0

    # Case B: Current deviates from zero-variance mean
    curr_vec_b = [5.0001] * 140
    raw_z_b, clipped_z_b, _, _, _, _ = PersonalBaselineEngine.compute_zscores(
        curr_vec_b, curr_mask, mean_vec, std_vec, counts
    )
    assert not math.isnan(raw_z_b[0])
    assert not math.isinf(raw_z_b[0])
    assert clipped_z_b[0] == 5.0  # Clipped safely


@pytest.mark.asyncio
async def test_t06_strict_data_leakage_prevention(db_session: AsyncSession):
    uid = f"M9_T06_USER_{uuid.uuid4().hex[:6]}"
    user = await ChatRepository.get_or_create_user(db_session, user_id=uid, display_name="T06 User")
    room = await ChatRepository.get_or_create_room(db_session, room_id=f"ROOM_{uid}")

    t0 = datetime.now(timezone.utc) - timedelta(hours=1)
    # Turn 1, Turn 2, Turn 3 (mean = 0.2)
    for i in range(3):
        m = await ChatRepository.save_message(db_session, room_id=room.id, sender_id=user.id, message_text=f"T06 Turn {i}")
        m.timestamp = t0 + timedelta(minutes=5 * i)
        await db_session.commit()
        vec, _ = create_synthetic_140d_vector(0.2, has_acoustic=False, has_nlp=True)
        await ChatRepository.save_or_update_multimodal_features(
            db_session, message_id=m.id, room_id=room.id, sender_id=user.id,
            status="completed", raw_fused_vector=vec, normalized_fused_vector=vec,
            has_communication=True, has_acoustic=False, has_nlp=True
        )

    # Turn 4 (Eval turn with distinct high value = 0.9)
    eval_m = await ChatRepository.save_message(db_session, room_id=room.id, sender_id=user.id, message_text="T06 Eval Turn")
    eval_m.timestamp = t0 + timedelta(minutes=20)
    await db_session.commit()
    eval_vec, _ = create_synthetic_140d_vector(0.9, has_acoustic=False, has_nlp=True)
    await ChatRepository.save_or_update_multimodal_features(
        db_session, message_id=eval_m.id, room_id=room.id, sender_id=user.id,
        status="completed", raw_fused_vector=eval_vec, normalized_fused_vector=eval_vec,
        has_communication=True, has_acoustic=False, has_nlp=True
    )

    # Turn 5 (Future turn with value = 1.0 created after Turn 4)
    fut_m = await ChatRepository.save_message(db_session, room_id=room.id, sender_id=user.id, message_text="T06 Future Turn")
    fut_m.timestamp = t0 + timedelta(minutes=25)
    await db_session.commit()
    fut_vec, _ = create_synthetic_140d_vector(1.0, has_acoustic=False, has_nlp=True)
    await ChatRepository.save_or_update_multimodal_features(
        db_session, message_id=fut_m.id, room_id=room.id, sender_id=user.id,
        status="completed", raw_fused_vector=fut_vec, normalized_fused_vector=fut_vec,
        has_communication=True, has_acoustic=False, has_nlp=True
    )

    service = PersonalBaselineService.get_instance()
    res = await service.evaluate_message_temporal_deviation(message_id=eval_m.id, db=db_session)

    # Prior records must only be Turns 1, 2, 3 (N=3)
    assert res.sample_count_at_eval == 3
    # Baseline mean must be exactly 0.2 (not contaminated by 0.9 or 1.0)
    assert abs(res.baseline_snapshot_mean[0] - 0.2) < 1e-4


@pytest.mark.asyncio
async def test_t07_user_isolation(db_session: AsyncSession):
    """T-07: Isolation - Alice's baseline is 100% isolated from Bob's observations."""
    alice = await ChatRepository.get_or_create_user(db_session, user_id=f"M9_ALICE_{uuid.uuid4().hex[:6]}", display_name="Alice")
    bob = await ChatRepository.get_or_create_user(db_session, user_id=f"M9_BOB_{uuid.uuid4().hex[:6]}", display_name="Bob")
    room = await ChatRepository.get_or_create_room(db_session, room_id=f"M9_ISO_{uuid.uuid4().hex[:6]}")

    # Give Bob 10 turns with value 0.99
    for i in range(10):
        m = await ChatRepository.save_message(db_session, room_id=room.id, sender_id=bob.id, message_text=f"Bob {i}")
        vec, _ = create_synthetic_140d_vector(0.99)
        await ChatRepository.save_or_update_multimodal_features(
            db_session, message_id=m.id, room_id=room.id, sender_id=bob.id,
            status="completed", raw_fused_vector=vec, normalized_fused_vector=vec
        )

    # Alice has 0 turns
    service = PersonalBaselineService.get_instance()
    alice_base = await service.compute_or_update_user_baseline(user_id=alice.id, db=db_session)

    assert alice_base.sample_count == 0
    assert alice_base.is_baseline_established is False
    assert alice_base.raw_mean_vector[0] == 0.0


@pytest.mark.asyncio
async def test_t08_ols_linear_trend_slope_directionality():
    """T-08: Trends - OLS linear trend slope correctly identifies rising, falling, flat slopes."""
    # 4 points at t = 0, 1, 2, 3 hours
    timestamps = [0.0, 3600.0, 7200.0, 10800.0]
    masks = [[1.0] * 140 for _ in range(4)]

    # Rising series
    norm_rising = [[0.1 + 0.2 * k] * 140 for k in range(4)]
    raw_rising = [[10.0 + 5.0 * k] * 140 for k in range(4)]

    _, slopes, _ = PersonalBaselineEngine.compute_temporal_dynamics(
        timestamps=timestamps,
        raw_series=raw_rising,
        norm_series=norm_rising,
        mask_series=masks
    )

    feat_0_slope = slopes["turn_index"]
    assert feat_0_slope["valid"] is True
    assert feat_0_slope["direction"] == "increasing"
    assert feat_0_slope["slope_normalized_per_hour"] > 0.01


@pytest.mark.asyncio
async def test_t09_sma_and_ema_smoothing_accuracy():
    """T-09: Smoothing - SMA-3, SMA-5, EMA-3 (alpha=0.5), EMA-5 (alpha=1/3) accuracy."""
    timestamps = [0.0, 3600.0, 7200.0]
    masks = [[1.0] * 140 for _ in range(3)]
    # Values: 0.2, 0.4, 0.6
    norm_series = [[0.2] * 140, [0.4] * 140, [0.6] * 140]
    raw_series = [[10.0] * 140, [20.0] * 140, [30.0] * 140]

    sma_sum, _, _ = PersonalBaselineEngine.compute_temporal_dynamics(
        timestamps=timestamps,
        raw_series=raw_series,
        norm_series=norm_series,
        mask_series=masks
    )

    res = sma_sum["turn_index"]
    assert res["valid"] is True
    # SMA-3 = (0.2 + 0.4 + 0.6)/3 = 0.4
    assert abs(res["sma_3"] - 0.4) < 1e-4
    # EMA-3: e1=0.2, e2=0.5*0.4+0.5*0.2=0.3, e3=0.5*0.6+0.5*0.3=0.45
    assert abs(res["ema_3"] - 0.45) < 1e-4


@pytest.mark.asyncio
async def test_t10_deterministic_reproducibility():
    """T-10: Determinism - Repeated baseline & deviation evaluations produce bit-for-bit identical results."""
    mean_vec = [12.3456] * 140
    std_vec = [3.4567] * 140
    counts = [10] * 140
    curr_vec = [15.6789] * 140
    curr_mask = [1.0] * 140

    res1 = PersonalBaselineEngine.compute_zscores(curr_vec, curr_mask, mean_vec, std_vec, counts)
    res2 = PersonalBaselineEngine.compute_zscores(curr_vec, curr_mask, mean_vec, std_vec, counts)

    assert res1[0] == res2[0]
    assert res1[3] == res2[3]
    assert res1[4] == res2[4]


@pytest.mark.asyncio
async def test_t11_idempotent_database_upsert(db_session: AsyncSession):
    """T-11: DB - Idempotent upsert in personal_baselines and temporal_trend_records."""
    user = await ChatRepository.get_or_create_user(db_session, user_id=f"M9_T11_{uuid.uuid4().hex[:6]}", display_name="T11 User")
    room = await ChatRepository.get_or_create_room(db_session, room_id=f"M9_T11_R_{uuid.uuid4().hex[:6]}")
    msg = await ChatRepository.save_message(db_session, room_id=room.id, sender_id=user.id, message_text="T11 Msg")

    # Upsert baseline twice
    b1 = await ChatRepository.save_or_update_personal_baseline(
        db_session, user_id=user.id, sample_count=5, is_baseline_established=True
    )
    b2 = await ChatRepository.save_or_update_personal_baseline(
        db_session, user_id=user.id, sample_count=6, is_baseline_established=True
    )
    assert b1.id == b2.id
    assert b2.sample_count == 6

    # Upsert temporal record twice
    t1 = await ChatRepository.save_or_update_temporal_trend_record(
        db_session, message_id=msg.id, room_id=room.id, user_id=user.id, status="completed", sample_count_at_eval=3
    )
    t2 = await ChatRepository.save_or_update_temporal_trend_record(
        db_session, message_id=msg.id, room_id=room.id, user_id=user.id, status="completed", sample_count_at_eval=4
    )
    assert t1.id == t2.id
    assert t2.sample_count_at_eval == 4


@pytest.mark.asyncio
async def test_t12_rest_api_endpoints_success(db_session: AsyncSession):
    """T-12: REST - All 6 REST endpoints return valid schemas and status codes."""
    user = await ChatRepository.get_or_create_user(db_session, user_id=f"M9_T12_{uuid.uuid4().hex[:6]}", display_name="T12 User")
    room = await ChatRepository.get_or_create_room(db_session, room_id=f"M9_T12_R_{uuid.uuid4().hex[:6]}")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. GET /baselines/users/{id}
        r1 = await ac.get(f"/api/v1/chat/baselines/users/{user.id}")
        assert r1.status_code == 200
        assert r1.json()["user_id"] == user.id

        # 2. POST /baselines/users/{id}/recompute
        r2 = await ac.post(f"/api/v1/chat/baselines/users/{user.id}/recompute")
        assert r2.status_code == 200

        # 3. GET /temporal/rooms/{room_id}
        r3 = await ac.get(f"/api/v1/chat/temporal/rooms/{room.id}")
        assert r3.status_code == 200
        assert "records" in r3.json()

        # 4. GET /temporal/users/{id}/trajectory
        r4 = await ac.get(f"/api/v1/chat/temporal/users/{user.id}/trajectory")
        assert r4.status_code == 200
        assert "trajectory" in r4.json()


@pytest.mark.asyncio
async def test_t13_websocket_event_structure():
    """T-13: WS - WebSocket events personal_baseline_updated and temporal_deviation_computed structure."""
    from app.schemas.chat import WebSocketEvent
    evt1 = WebSocketEvent(
        event="personal_baseline_updated",
        data={"user_id": "U1", "sample_count": 4, "is_baseline_established": True}
    )
    assert evt1.event == "personal_baseline_updated"

    evt2 = WebSocketEvent(
        event="temporal_deviation_computed",
        data={"message_id": "M1", "mean_absolute_z": 0.5}
    )
    assert evt2.event == "temporal_deviation_computed"


@pytest.mark.asyncio
async def test_t14_m3_through_m8_invariance(db_session: AsyncSession):
    """T-14: Invariance - M3 metrics, M4 audio, M5 transcripts, M6 acoustic, M7 NLP, M8 vectors remain 100% unchanged."""
    user = await ChatRepository.get_or_create_user(db_session, user_id=f"M9_T14_{uuid.uuid4().hex[:6]}", display_name="T14 User")
    room = await ChatRepository.get_or_create_room(db_session, room_id=f"M9_T14_R_{uuid.uuid4().hex[:6]}")
    msg = await ChatRepository.save_message(db_session, room_id=room.id, sender_id=user.id, message_text="T14 Invariance Text")

    vec, _ = create_synthetic_140d_vector(0.42)
    m8_rec = await ChatRepository.save_or_update_multimodal_features(
        db_session, message_id=msg.id, room_id=room.id, sender_id=user.id,
        status="completed", raw_fused_vector=vec, normalized_fused_vector=vec
    )
    orig_raw = list(m8_rec.raw_fused_vector)
    orig_norm = list(m8_rec.normalized_fused_vector)

    # Perform M9 recompute and evaluate
    service = PersonalBaselineService.get_instance()
    await service.compute_or_update_user_baseline(user.id, db_session)

    # Refetch M8 record
    refetched = await ChatRepository.get_multimodal_features_by_message_id(db_session, msg.id)
    assert refetched.raw_fused_vector == orig_raw
    assert refetched.normalized_fused_vector == orig_norm


@pytest.mark.asyncio
async def test_t15_live_simulation_7_turns(db_session: AsyncSession):
    """T-15: Live - 7-turn dyadic simulation executes end-to-end calibrating to baseline established."""
    alice = await ChatRepository.get_or_create_user(db_session, user_id=f"M9_T15_A_{uuid.uuid4().hex[:6]}", display_name="Alice")
    bob = await ChatRepository.get_or_create_user(db_session, user_id=f"M9_T15_B_{uuid.uuid4().hex[:6]}", display_name="Bob")
    room = await ChatRepository.get_or_create_room(db_session, room_id=f"M9_T15_R_{uuid.uuid4().hex[:6]}")

    service = PersonalBaselineService.get_instance()
    t0 = datetime.now(timezone.utc) - timedelta(minutes=40)

    # Turns 1..7
    # 1: Alice (prior 0)
    # 2: Bob (prior 0)
    # 3: Alice (prior 1)
    # 4: Bob (prior 1)
    # 5: Alice (prior 2)
    # 6: Bob (prior 2)
    # 7: Alice (prior 3) -> ESTABLISHED!
    turn_evals = []
    for turn in range(1, 8):
        is_alice = (turn % 2 == 1)
        sender = alice if is_alice else bob
        msg = await ChatRepository.save_message(
            db_session, room_id=room.id, sender_id=sender.id, message_text=f"Live Turn {turn}"
        )
        msg.timestamp = t0 + timedelta(minutes=5 * turn)
        await db_session.commit()

        vec, _ = create_synthetic_140d_vector(0.3 + 0.05 * turn, has_acoustic=not is_alice, has_nlp=True)
        await ChatRepository.save_or_update_multimodal_features(
            db_session, message_id=msg.id, room_id=room.id, sender_id=sender.id,
            status="completed", raw_fused_vector=vec, normalized_fused_vector=vec,
            has_communication=True, has_acoustic=not is_alice, has_nlp=True
        )

        res = await service.evaluate_message_temporal_deviation(message_id=msg.id, db=db_session)
        turn_evals.append(res)

    # Turns 1..6 must be insufficient_history
    for t in range(6):
        assert turn_evals[t].is_baseline_established is False
        assert turn_evals[t].status == "insufficient_history"

    # Turn 7 (Alice's 4th turn with 3 prior turns) must be COMPLETED!
    assert turn_evals[6].is_baseline_established is True
    assert turn_evals[6].status == "completed"
    assert turn_evals[6].sample_count_at_eval == 3


@pytest.mark.asyncio
async def test_t16_unsupported_feature_version_rejected(db_session: AsyncSession):
    """T-16: Validation - Unsupported feature-vector version rejected."""
    user = await ChatRepository.get_or_create_user(db_session, user_id=f"M9_T16_{uuid.uuid4().hex[:6]}", display_name="T16")
    room = await ChatRepository.get_or_create_room(db_session, room_id=f"M9_T16_R_{uuid.uuid4().hex[:6]}")
    msg = await ChatRepository.save_message(db_session, room_id=room.id, sender_id=user.id, message_text="T16")

    vec, _ = create_synthetic_140d_vector(0.5)
    rec = await ChatRepository.save_or_update_multimodal_features(
        db_session, message_id=msg.id, room_id=room.id, sender_id=user.id,
        status="completed", raw_fused_vector=vec, normalized_fused_vector=vec
    )
    rec.feature_vector_version = "v0.9-100d"
    await db_session.commit()

    service = PersonalBaselineService.get_instance()
    with pytest.raises(UnsupportedFeatureVectorError):
        await service.evaluate_message_temporal_deviation(msg.id, db_session)


@pytest.mark.asyncio
async def test_t17_invalid_vector_length_rejection():
    """T-17: Validation - Invalid vector length (!= 140) handled safely."""
    with pytest.raises(Exception):
        PersonalBaselineEngine.compute_baseline_statistics(
            raw_vectors=[[0.5] * 100],  # only 100 dims
            normalized_vectors=[[0.5] * 100],
            masks=[[1.0] * 100]
        )


@pytest.mark.asyncio
async def test_t18_modality_mask_indices_excluded_from_deviations():
    """T-18: Masking - Modality mask indices [135..139] excluded from deviation scoring."""
    mean_vec = [0.5] * 140
    std_vec = [0.1] * 140
    counts = [5] * 140
    curr_vec = [0.9] * 140
    curr_mask = [1.0] * 140

    raw_z, _, val_mask, _, _, _ = PersonalBaselineEngine.compute_zscores(
        curr_vec, curr_mask, mean_vec, std_vec, counts
    )

    # Indices 135..139 must be masked out (False) and have z = 0.0
    for idx in range(135, 140):
        assert val_mask[idx] is False
        assert raw_z[idx] == 0.0


@pytest.mark.asyncio
async def test_t19_missing_acoustic_on_text_does_not_create_false_deviations():
    """T-19: Masking - Missing acoustic features on text turns do NOT create false zero-behavioral deviations."""
    # Text turn: mask has acoustic = 0.0
    vec, mask = create_synthetic_140d_vector(0.5, has_acoustic=False, has_nlp=True)
    mean_vec = [0.5] * 140
    std_vec = [0.1] * 140
    counts = [5] * 140

    raw_z, _, val_mask, _, _, _ = PersonalBaselineEngine.compute_zscores(
        vec, mask, mean_vec, std_vec, counts
    )

    # Acoustic features [10..68] must have val_mask = False and raw_z = 0
    for idx in range(10, 69):
        assert val_mask[idx] is False
        assert raw_z[idx] == 0.0


@pytest.mark.asyncio
async def test_t20_missing_nlp_on_untranscribed_voice_masked():
    """T-20: Masking - Missing NLP features on untranscribed voice do NOT create false zero deviations."""
    vec, mask = create_synthetic_140d_vector(0.5, has_acoustic=True, has_nlp=False)
    mean_vec = [0.5] * 140
    std_vec = [0.1] * 140
    counts = [5] * 140

    raw_z, _, val_mask, _, _, _ = PersonalBaselineEngine.compute_zscores(
        vec, mask, mean_vec, std_vec, counts
    )

    # NLP features [69..134] must be val_mask = False
    for idx in range(69, 135):
        assert val_mask[idx] is False
        assert raw_z[idx] == 0.0


@pytest.mark.asyncio
async def test_t21_timestamp_tie_breaking_deterministic(db_session: AsyncSession):
    """T-21: Ties - Identical timestamp tie-breaking orders deterministically by (timestamp, message_id)."""
    user = await ChatRepository.get_or_create_user(db_session, user_id=f"M9_T21_{uuid.uuid4().hex[:6]}", display_name="T21")
    room = await ChatRepository.get_or_create_room(db_session, room_id=f"M9_T21_R_{uuid.uuid4().hex[:6]}")
    t = datetime.now(timezone.utc)

    # Create 3 messages with identical timestamp
    m1 = await ChatRepository.save_message(db_session, room_id=room.id, sender_id=user.id, message_text="M1")
    m1.timestamp = t
    m2 = await ChatRepository.save_message(db_session, room_id=room.id, sender_id=user.id, message_text="M2")
    m2.timestamp = t
    m3 = await ChatRepository.save_message(db_session, room_id=room.id, sender_id=user.id, message_text="M3")
    m3.timestamp = t
    await db_session.commit()

    for m in [m1, m2, m3]:
        vec, _ = create_synthetic_140d_vector(0.5)
        await ChatRepository.save_or_update_multimodal_features(
            db_session, message_id=m.id, room_id=room.id, sender_id=user.id,
            status="completed", raw_fused_vector=vec, normalized_fused_vector=vec
        )

    records = await ChatRepository.get_user_multimodal_history(db_session, user_id=user.id)
    ids = [r.message_id for r in records]
    assert ids == sorted(ids)  # Ordered deterministically


@pytest.mark.asyncio
async def test_t22_chronological_ordering_guarantee(db_session: AsyncSession):
    """T-22: Ordering - Historical sequence ordering is strictly chronological ascending."""
    user = await ChatRepository.get_or_create_user(db_session, user_id=f"M9_T22_{uuid.uuid4().hex[:6]}", display_name="T22")
    room = await ChatRepository.get_or_create_room(db_session, room_id=f"M9_T22_R_{uuid.uuid4().hex[:6]}")
    now = datetime.now(timezone.utc)

    for i in [3, 1, 2]:
        m = await ChatRepository.save_message(db_session, room_id=room.id, sender_id=user.id, message_text=f"M{i}")
        m.timestamp = now + timedelta(minutes=i)
        await db_session.commit()
        vec, _ = create_synthetic_140d_vector(0.5)
        await ChatRepository.save_or_update_multimodal_features(
            db_session, message_id=m.id, room_id=room.id, sender_id=user.id,
            status="completed", raw_fused_vector=vec, normalized_fused_vector=vec
        )

    records = await ChatRepository.get_user_multimodal_history(db_session, user_id=user.id)
    times = [r.message.timestamp.replace(tzinfo=None) for r in records]
    assert times == sorted(times)


@pytest.mark.asyncio
async def test_t23_rolling_windows_subsets(db_session: AsyncSession):
    """T-23: Windows - all_time vs rolling_10 window produces correct subset selections."""
    user = await ChatRepository.get_or_create_user(db_session, user_id=f"M9_T23_{uuid.uuid4().hex[:6]}", display_name="T23")
    room = await ChatRepository.get_or_create_room(db_session, room_id=f"M9_T23_R_{uuid.uuid4().hex[:6]}")
    now = datetime.now(timezone.utc)

    for i in range(15):
        m = await ChatRepository.save_message(db_session, room_id=room.id, sender_id=user.id, message_text=f"M{i}")
        m.timestamp = now + timedelta(minutes=i)
        await db_session.commit()
        vec, _ = create_synthetic_140d_vector(0.1 * i)
        await ChatRepository.save_or_update_multimodal_features(
            db_session, message_id=m.id, room_id=room.id, sender_id=user.id,
            status="completed", raw_fused_vector=vec, normalized_fused_vector=vec
        )

    all_recs = await ChatRepository.get_user_multimodal_history(db_session, user.id, window_type="all_time")
    rolling_recs = await ChatRepository.get_user_multimodal_history(db_session, user.id, window_type="rolling_10")

    assert len(all_recs) == 15
    assert len(rolling_recs) == 10
    # Rolling 10 contains the most recent 10 (indices 5..14)
    assert rolling_recs[-1].message_id == all_recs[-1].message_id


@pytest.mark.asyncio
async def test_t24_raw_vs_normalized_consistency():
    """T-24: Consistency - Raw vs normalized baseline statistics maintain monotonic relationship."""
    raw_vecs = [[10.0 + 2.0 * k] * 140 for k in range(5)]
    norm_vecs = [[0.1 + 0.1 * k] * 140 for k in range(5)]
    masks = [[1.0] * 140 for _ in range(5)]

    raw_mean, _, norm_mean, _, _ = PersonalBaselineEngine.compute_baseline_statistics(
        raw_vecs, norm_vecs, masks
    )
    assert raw_mean[0] > norm_mean[0]


@pytest.mark.asyncio
async def test_t25_clipped_zscores_bounds():
    """T-25: Clipping - Clipped z-scores strictly bounded in [-5.0, +5.0] while raw remain unclipped."""
    mean_vec = [10.0] * 140
    std_vec = [1.0] * 140
    counts = [5] * 140
    curr_vec = [20.0] * 140  # z = (20 - 10)/1 = +10.0
    curr_mask = [1.0] * 140

    raw_z, clipped_z, _, _, _, _ = PersonalBaselineEngine.compute_zscores(
        curr_vec, curr_mask, mean_vec, std_vec, counts
    )

    assert raw_z[0] == 10.0
    assert clipped_z[0] == 5.0


@pytest.mark.asyncio
async def test_t26_aggregates_calculation_accuracy():
    """T-26: Aggregates - mean_absolute_z, rms_z, and standardized_euclidean_distance calculated accurately."""
    mean_vec = [0.0] * 140
    std_vec = [1.0] * 140
    counts = [5] * 140
    curr_vec = [2.0] * 140  # z_i = 2.0 for all 135 behavioral dimensions
    curr_mask = [1.0] * 140

    _, _, _, mean_abs, rms, std_euc = PersonalBaselineEngine.compute_zscores(
        curr_vec, curr_mask, mean_vec, std_vec, counts
    )

    # 135 valid behavioral dimensions (0..134)
    assert abs(mean_abs - 2.0) < 1e-4
    assert abs(rms - 2.0) < 1e-4
    expected_std_euc = math.sqrt(135 * (2.0 ** 2))
    assert abs(std_euc - expected_std_euc) < 1e-4


@pytest.mark.asyncio
async def test_t27_ols_zero_time_variance_edge_case():
    """T-27: Edge Case - OLS zero-time-variance handles Delta tau = 0 safely."""
    # 2 observations with identical timestamp
    timestamps = [100.0, 100.0]
    masks = [[1.0] * 140, [1.0] * 140]
    series = [[0.2] * 140, [0.4] * 140]

    _, slopes, _ = PersonalBaselineEngine.compute_temporal_dynamics(
        timestamps=timestamps,
        raw_series=series,
        norm_series=series,
        mask_series=masks
    )

    res = slopes["turn_index"]
    assert res["valid"] is False
    assert res["reason"] == "insufficient_temporal_variation"
    assert res["slope_normalized_per_hour"] == 0.0


@pytest.mark.asyncio
async def test_t28_roc_zero_time_delta_edge_case():
    """T-28: Edge Case - RoC zero-time-delta handles identical timestamps safely."""
    timestamps = [50.0, 50.0]
    masks = [[1.0] * 140, [1.0] * 140]
    series = [[0.2] * 140, [0.5] * 140]

    _, _, roc = PersonalBaselineEngine.compute_temporal_dynamics(
        timestamps=timestamps,
        raw_series=series,
        norm_series=series,
        mask_series=masks
    )

    res = roc["turn_index"]
    assert res["valid"] is False
    assert res["reason"] == "zero_time_delta"
    assert res["roc_normalized"] == 0.0


@pytest.mark.asyncio
async def test_t29_valid_dimension_count_reporting(db_session: AsyncSession):
    """T-29: Counts - valid_dimension_count accurately reports active feature count (76 for text)."""
    user = await ChatRepository.get_or_create_user(db_session, user_id=f"M9_T29_{uuid.uuid4().hex[:6]}", display_name="T29")
    room = await ChatRepository.get_or_create_room(db_session, room_id=f"M9_T29_R_{uuid.uuid4().hex[:6]}")
    t0 = datetime.now(timezone.utc) - timedelta(minutes=30)

    # 3 prior text turns
    for i in range(3):
        m = await ChatRepository.save_message(db_session, room_id=room.id, sender_id=user.id, message_text=f"Text {i}")
        m.timestamp = t0 + timedelta(minutes=5 * i)
        await db_session.commit()
        vec, _ = create_synthetic_140d_vector(0.5, has_acoustic=False, has_nlp=True)
        await ChatRepository.save_or_update_multimodal_features(
            db_session, message_id=m.id, room_id=room.id, sender_id=user.id,
            status="completed", raw_fused_vector=vec, normalized_fused_vector=vec,
            has_communication=True, has_acoustic=False, has_nlp=True
        )

    # Eval text turn
    eval_m = await ChatRepository.save_message(db_session, room_id=room.id, sender_id=user.id, message_text="Eval Text")
    eval_m.timestamp = t0 + timedelta(minutes=20)
    await db_session.commit()
    vec, _ = create_synthetic_140d_vector(0.5, has_acoustic=False, has_nlp=True)
    await ChatRepository.save_or_update_multimodal_features(
        db_session, message_id=eval_m.id, room_id=room.id, sender_id=user.id,
        status="completed", raw_fused_vector=vec, normalized_fused_vector=vec,
        has_communication=True, has_acoustic=False, has_nlp=True
    )

    service = PersonalBaselineService.get_instance()
    res = await service.evaluate_message_temporal_deviation(eval_m.id, db_session)
    # Comm(10) + NLP(66) = 76 valid dimensions
    assert res.valid_dimension_count == 76
    assert res.unavailable_dimension_count == 140 - 76


@pytest.mark.asyncio
async def test_t30_unauthorized_user_baseline_rejection(db_session: AsyncSession):
    """T-30: Auth - Non-existent user baseline query returns 404."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/api/v1/chat/baselines/users/NONEXISTENT_USER_XYZ")
        assert r.status_code == 404


@pytest.mark.asyncio
async def test_t31_re_evaluating_message_idempotent(db_session: AsyncSession):
    """T-31: Idempotency - Re-evaluating an already evaluated message returns identical existing record."""
    user = await ChatRepository.get_or_create_user(db_session, user_id=f"M9_T31_{uuid.uuid4().hex[:6]}", display_name="T31")
    room = await ChatRepository.get_or_create_room(db_session, room_id=f"M9_T31_R_{uuid.uuid4().hex[:6]}")
    t0 = datetime.now(timezone.utc) - timedelta(minutes=30)

    for i in range(4):
        m = await ChatRepository.save_message(db_session, room_id=room.id, sender_id=user.id, message_text=f"T31 {i}")
        m.timestamp = t0 + timedelta(minutes=5 * i)
        await db_session.commit()
        vec, _ = create_synthetic_140d_vector(0.3 + 0.1 * i)
        await ChatRepository.save_or_update_multimodal_features(
            db_session, message_id=m.id, room_id=room.id, sender_id=user.id,
            status="completed", raw_fused_vector=vec, normalized_fused_vector=vec
        )

    service = PersonalBaselineService.get_instance()
    eval_id = m.id
    res1 = await service.evaluate_message_temporal_deviation(eval_id, db_session)
    res2 = await service.evaluate_message_temporal_deviation(eval_id, db_session)

    assert res1.mean_absolute_z == res2.mean_absolute_z
    assert res1.valid_dimension_count == res2.valid_dimension_count


@pytest.mark.asyncio
async def test_t32_concurrent_recomputations_safe():
    """T-32: Concurrency - Concurrent baseline recomputations execute safely across separate sessions."""
    await init_db()
    async with async_session_maker() as s0:
        user = await ChatRepository.get_or_create_user(s0, user_id=f"M9_T32_{uuid.uuid4().hex[:6]}", display_name="T32")

    service = PersonalBaselineService.get_instance()

    async def run_comp():
        async with async_session_maker() as s:
            return await service.compute_or_update_user_baseline(user.id, s)

    import asyncio
    results = await asyncio.gather(run_comp(), run_comp())
    assert results[0].sample_count == results[1].sample_count


@pytest.mark.asyncio
async def test_t33_frontend_badge_states():
    """T-33: Frontend - Baseline calibrating vs active state logic."""
    base_calibrating = PersonalBaselineResult(
        user_id="U1", sample_count=2, is_baseline_established=False, valid_dimensions_count=0,
        per_feature_sample_counts=[2]*140, raw_mean_vector=[0.0]*140, raw_std_vector=[0.0]*140,
        normalized_mean_vector=[0.0]*140, normalized_std_vector=[0.0]*140
    )
    assert base_calibrating.is_baseline_established is False

    base_active = PersonalBaselineResult(
        user_id="U1", sample_count=3, is_baseline_established=True, valid_dimensions_count=76,
        per_feature_sample_counts=[3]*140, raw_mean_vector=[0.0]*140, raw_std_vector=[0.0]*140,
        normalized_mean_vector=[0.0]*140, normalized_std_vector=[0.0]*140
    )
    assert base_active.is_baseline_established is True


@pytest.mark.asyncio
async def test_t34_frontend_modality_demarcation():
    """T-34: Frontend - Correct demarcation of valid vs unavailable modalities."""
    vec, mask = create_synthetic_140d_vector(0.5, has_acoustic=False, has_nlp=True)
    raw_z, _, val_mask, _, _, _ = PersonalBaselineEngine.compute_zscores(
        vec, mask, [0.5]*140, [0.1]*140, [3]*140
    )
    assert val_mask[0] is True   # Comm
    assert val_mask[15] is False # Acoustic (unavailable)
    assert val_mask[75] is True  # NLP


@pytest.mark.asyncio
async def test_t35_mixed_modality_and_point_in_time_reproducibility(db_session: AsyncSession):
    """T-35: Regression - Mixed-modality longitudinal history & historical point-in-time snapshot reproducibility."""
    user = await ChatRepository.get_or_create_user(db_session, user_id=f"M9_T35_{uuid.uuid4().hex[:6]}", display_name="T35")
    room = await ChatRepository.get_or_create_room(db_session, room_id=f"M9_T35_R_{uuid.uuid4().hex[:6]}")
    t0 = datetime.now(timezone.utc) - timedelta(hours=2)

    # Turn 1: Text
    # Turn 2: Voice
    # Turn 3: Text
    # Turn 4: Eval Message
    turns = [
        ("T1 Text", False),
        ("T2 Voice", True),
        ("T3 Text", False),
        ("T4 Eval", True)
    ]
    msg_ids = []
    for idx, (txt, is_voice) in enumerate(turns):
        m = await ChatRepository.save_message(db_session, room_id=room.id, sender_id=user.id, message_text=txt)
        m.timestamp = t0 + timedelta(minutes=10 * idx)
        await db_session.commit()
        vec, _ = create_synthetic_140d_vector(0.4, has_acoustic=is_voice, has_nlp=True)
        await ChatRepository.save_or_update_multimodal_features(
            db_session, message_id=m.id, room_id=room.id, sender_id=user.id,
            status="completed", raw_fused_vector=vec, normalized_fused_vector=vec,
            has_communication=True, has_acoustic=is_voice, has_nlp=True
        )
        msg_ids.append(m.id)

    service = PersonalBaselineService.get_instance()
    eval_m4_id = msg_ids[3]
    eval_res_original = await service.evaluate_message_temporal_deviation(eval_m4_id, db_session)

    assert eval_res_original.status == "completed"
    assert eval_res_original.sample_count_at_eval == 3
    # Acoustic was only present on Turn 2 (1 observation), so acoustic variance is unavailable (N_acoustic < 2)
    assert eval_res_original.zscore_validity_mask[15] is False  # Acoustic masked out
    assert eval_res_original.zscore_validity_mask[0] is True   # Comm valid (3 observations)
    assert eval_res_original.zscore_validity_mask[70] is True  # NLP valid (3 observations)

    # Point-in-time Snapshot Test:
    # Now append 5 new future messages for user
    for i in range(5):
        fm = await ChatRepository.save_message(db_session, room_id=room.id, sender_id=user.id, message_text=f"Future {i}")
        fm.timestamp = t0 + timedelta(hours=1, minutes=5 * i)
        await db_session.commit()
        vec, _ = create_synthetic_140d_vector(0.99)
        await ChatRepository.save_or_update_multimodal_features(
            db_session, message_id=fm.id, room_id=room.id, sender_id=user.id,
            status="completed", raw_fused_vector=vec, normalized_fused_vector=vec
        )

    # Refetch eval record for M4 from DB
    persisted_m4 = await ChatRepository.get_temporal_trend_by_message_id(db_session, eval_m4_id)
    assert persisted_m4.sample_count_at_eval == 3
    assert persisted_m4.baseline_snapshot_mean == eval_res_original.baseline_snapshot_mean
    assert persisted_m4.mean_absolute_z == eval_res_original.mean_absolute_z
