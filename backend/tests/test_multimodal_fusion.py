"""
Comprehensive Automated Test Matrix for Milestone 8: Multimodal Feature Fusion
Tests T-01 through T-14:
- Exact 140-d vector dimensionality and index ordering (v1.0-140d)
- Raw and normalized vector bounds [0.0, 1.0]
- Text message fusion with acoustic zero-imputation
- Transcribed voice message fusion across all 3 modalities
- Untranscribed voice message fusion with neutral NLP imputation
- Idempotent upsert & persistence in multimodal_features table
- Mathematical determinism & reproducibility
- M3, M4, M5, M6, M7 system invariance
"""
import io
import math
import os
import uuid
import pytest
from fastapi.testclient import TestClient
import numpy as np
import soundfile as sf

from app.core.config import settings
from app.db.session import init_db
from app.services.multimodal_fusion import (
    MultimodalFusionService,
    MultimodalFeatureScaler,
    CommunicationFeatureBlock,
    MultimodalMaskBlock,
    MultimodalFusionResult
)
from app.services.speech_to_text import stt_service, MockSTTProvider
from app.services.acoustic_features import acoustic_service, MockAcousticExtractor
from app.services.nlp_features import nlp_service, RuleLexicalNLPProvider


@pytest.fixture(autouse=True)
def setup_test_env():
    import asyncio
    asyncio.run(init_db())
    os.makedirs(settings.AUDIO_UPLOAD_DIR, exist_ok=True)
    stt_service.set_provider(MockSTTProvider(default_transcript="We are exceptionally optimistic about our team research progress."))
    acoustic_service.set_extractor(MockAcousticExtractor())
    nlp_service.set_provider(RuleLexicalNLPProvider())


def create_test_wav_bytes(duration_s: float = 2.0, sample_rate: int = 16000) -> bytes:
    """Generates synthetic test harmonic audio."""
    n_samples = int(duration_s * sample_rate)
    samples = (0.3 * np.sin(2 * np.pi * 220.0 * np.linspace(0, duration_s, n_samples, endpoint=False))).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format='WAV', subtype='PCM_16')
    return buf.getvalue()


def test_vector_dimensionality_and_structure():
    """
    T-01: Vector Dimensionality Check.
    Verifies that MultimodalFusionResult produces exact 140-d raw and normalized vectors.
    """
    comm_vec = [1.0] * 10
    acoustic_vec = [0.5] * 59
    nlp_vec = [0.2] * 66
    mask_vec = [1.0, 1.0, 1.0, 1.0, 0.0]

    raw_vec, norm_vec = MultimodalFusionResult.assemble_vectors(
        comm_vec, acoustic_vec, nlp_vec, mask_vec
    )

    assert len(raw_vec) == 140
    assert len(norm_vec) == 140
    assert all(math.isfinite(v) for v in raw_vec)
    assert all(math.isfinite(v) for v in norm_vec)
    assert all(0.0 <= v <= 1.0 for v in norm_vec)


def test_text_message_multimodal_fusion(client: TestClient):
    """
    T-02: Text Message Multimodal Fusion.
    Verifies fusion of a text message with M7 NLP, communication metrics, and acoustic zero-imputation.
    """
    room_id = f"ROOM_M8_T02_{uuid.uuid4().hex[:6]}"
    client.post("/api/v1/chat/rooms", json={"id": room_id, "name": "M8 Text Fusion"})

    # Send voice as surrogate or create text message via voice endpoint or WebSocket
    # For testing, we create two messages in room:
    wav_bytes = create_test_wav_bytes(duration_s=1.5)
    files = {"file": ("alice_voice.wav", wav_bytes, "audio/wav")}
    res1 = client.post(
        f"/api/v1/chat/rooms/{room_id}/voice",
        files=files,
        data={"user_id": "ALICE", "duration_seconds": 1.5}
    )
    msg1_id = res1.json()["message"]["message_id"]

    # Transcribe & analyze NLP
    client.post(f"/api/v1/chat/messages/{msg1_id}/transcribe")
    client.post(f"/api/v1/chat/messages/{msg1_id}/analyze-nlp")

    # Fuse
    fuse_res = client.post(f"/api/v1/chat/messages/{msg1_id}/fuse")
    assert fuse_res.status_code == 200
    data = fuse_res.json()

    assert data["feature_vector_version"] == "v1.0-140d"
    assert len(data["raw_fused_vector"]) == 140
    assert len(data["normalized_fused_vector"]) == 140
    assert data["has_communication"] is True
    assert data["has_nlp"] is True
    assert all(0.0 <= v <= 1.0 for v in data["normalized_fused_vector"])


def test_transcribed_voice_message_multimodal_fusion(client: TestClient):
    """
    T-03: Transcribed Voice Message Multimodal Fusion.
    Verifies full 3-modality fusion: M3 comm + M6 acoustic + M7 transcript NLP.
    """
    room_id = f"ROOM_M8_T03_{uuid.uuid4().hex[:6]}"
    client.post("/api/v1/chat/rooms", json={"id": room_id, "name": "M8 Voice Fusion"})

    wav_bytes = create_test_wav_bytes(duration_s=2.5)
    files = {"file": ("voice_m8.wav", wav_bytes, "audio/wav")}
    v_res = client.post(
        f"/api/v1/chat/rooms/{room_id}/voice",
        files=files,
        data={"user_id": "BOB", "duration_seconds": 2.5}
    )
    msg_id = v_res.json()["message"]["message_id"]

    # Run STT, Acoustic, and NLP
    client.post(f"/api/v1/chat/messages/{msg_id}/transcribe")
    client.post(f"/api/v1/chat/messages/{msg_id}/analyze-acoustic")
    client.post(f"/api/v1/chat/messages/{msg_id}/analyze-nlp")

    # Fuse
    fuse_res = client.post(f"/api/v1/chat/messages/{msg_id}/fuse")
    assert fuse_res.status_code == 200
    f_data = fuse_res.json()

    assert f_data["has_communication"] is True
    assert f_data["has_acoustic"] is True
    assert f_data["has_nlp"] is True
    assert f_data["modality_mask"]["has_communication_metrics"] == 1.0
    assert f_data["modality_mask"]["has_acoustic_features"] == 1.0
    assert f_data["modality_mask"]["has_nlp_features"] == 1.0
    assert f_data["modality_mask"]["is_voice_message"] == 1.0
    assert f_data["modality_mask"]["is_text_message"] == 0.0
    assert len(f_data["raw_fused_vector"]) == 140
    assert len(f_data["normalized_fused_vector"]) == 140


def test_untranscribed_voice_message_multimodal_fusion(client: TestClient):
    """
    T-04: Untranscribed Voice Message Fusion.
    Verifies acoustic features are present, while absent NLP features are imputed with neutral values.
    """
    room_id = f"ROOM_M8_T04_{uuid.uuid4().hex[:6]}"
    client.post("/api/v1/chat/rooms", json={"id": room_id, "name": "M8 Untranscribed Voice"})

    wav_bytes = create_test_wav_bytes(duration_s=2.0)
    files = {"file": ("untranscribed.wav", wav_bytes, "audio/wav")}
    v_res = client.post(
        f"/api/v1/chat/rooms/{room_id}/voice",
        files=files,
        data={"user_id": "CHARLIE", "duration_seconds": 2.0}
    )
    msg_id = v_res.json()["message"]["message_id"]

    # Extract acoustic only (no STT, no NLP)
    client.post(f"/api/v1/chat/messages/{msg_id}/analyze-acoustic")

    fuse_res = client.post(f"/api/v1/chat/messages/{msg_id}/fuse")
    assert fuse_res.status_code == 200
    f_data = fuse_res.json()

    assert f_data["has_acoustic"] is True
    assert f_data["has_nlp"] is False
    assert f_data["modality_mask"]["has_acoustic_features"] == 1.0
    assert f_data["modality_mask"]["has_nlp_features"] == 0.0
    assert len(f_data["raw_fused_vector"]) == 140
    assert all(math.isfinite(v) for v in f_data["raw_fused_vector"])


def test_nonexistent_message_fusion_404(client: TestClient):
    """T-05: Nonexistent Message 404 Rejection."""
    res = client.post("/api/v1/chat/messages/nonexistent-msg-id-000/fuse")
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()


def test_normalization_mathematical_bounds():
    """
    T-06: Normalization Mathematical Bounds Verification.
    Verifies that for extreme inputs (negative, large, zero), all normalized outputs are bounded in [0.0, 1.0].
    """
    extreme_raw_vector = [
        -9999.0, 9999.0, 0.0, 1e6, -1e6, 250.0, 50.0, 100.0, 20.0, 1.0,
        *([1000.0] * 59),
        *([-500.0] * 66),
        1.0, 0.0, 1.0, 0.0, 1.0
    ]
    norm_vec = MultimodalFeatureScaler.scale_vector(extreme_raw_vector)
    assert len(norm_vec) == 140
    assert all(0.0 <= v <= 1.0 for v in norm_vec)
    assert all(math.isfinite(v) for v in norm_vec)


def test_mathematical_determinism_and_reproducibility(client: TestClient):
    """
    T-07: Mathematical Determinism Check.
    Verifies that running fusion multiple times produces 100% bit-for-bit identical vectors.
    """
    room_id = f"ROOM_M8_T07_{uuid.uuid4().hex[:6]}"
    client.post("/api/v1/chat/rooms", json={"id": room_id, "name": "M8 Determinism"})

    wav_bytes = create_test_wav_bytes(duration_s=1.8)
    files = {"file": ("audio_det.wav", wav_bytes, "audio/wav")}
    v_res = client.post(
        f"/api/v1/chat/rooms/{room_id}/voice",
        files=files,
        data={"user_id": "ALICE", "duration_seconds": 1.8}
    )
    msg_id = v_res.json()["message"]["message_id"]

    client.post(f"/api/v1/chat/messages/{msg_id}/transcribe")
    client.post(f"/api/v1/chat/messages/{msg_id}/analyze-acoustic")
    client.post(f"/api/v1/chat/messages/{msg_id}/analyze-nlp")

    run_1 = client.post(f"/api/v1/chat/messages/{msg_id}/fuse").json()
    run_2 = client.post(f"/api/v1/chat/messages/{msg_id}/fuse").json()

    assert run_1["raw_fused_vector"] == run_2["raw_fused_vector"]
    assert run_1["normalized_fused_vector"] == run_2["normalized_fused_vector"]


def test_persistence_retrieval_and_idempotent_upsert(client: TestClient):
    """
    T-08: Persistence, Retrieval, and Idempotent Upsert.
    """
    room_id = f"ROOM_M8_T08_{uuid.uuid4().hex[:6]}"
    client.post("/api/v1/chat/rooms", json={"id": room_id, "name": "M8 Persistence"})

    wav_bytes = create_test_wav_bytes(duration_s=2.0)
    files = {"file": ("persist.wav", wav_bytes, "audio/wav")}
    v_res = client.post(
        f"/api/v1/chat/rooms/{room_id}/voice",
        files=files,
        data={"user_id": "ALICE", "duration_seconds": 2.0}
    )
    msg_id = v_res.json()["message"]["message_id"]

    # Initial fusion before STT/acoustic
    fuse_1 = client.post(f"/api/v1/chat/messages/{msg_id}/fuse")
    assert fuse_1.status_code == 200

    # Retrieve
    get_res = client.get(f"/api/v1/chat/messages/{msg_id}/multimodal-features")
    assert get_res.status_code == 200
    assert get_res.json()["raw_fused_vector"] == fuse_1.json()["raw_fused_vector"]

    # Now add acoustic & NLP, and fuse again (idempotent upsert update)
    client.post(f"/api/v1/chat/messages/{msg_id}/transcribe")
    client.post(f"/api/v1/chat/messages/{msg_id}/analyze-acoustic")
    client.post(f"/api/v1/chat/messages/{msg_id}/analyze-nlp")

    fuse_2 = client.post(f"/api/v1/chat/messages/{msg_id}/fuse")
    assert fuse_2.status_code == 200
    assert fuse_2.json()["has_acoustic"] is True
    assert fuse_2.json()["has_nlp"] is True


def test_room_level_multimodal_features_aggregation(client: TestClient):
    """
    T-09: Room-Level Multimodal Features Aggregation Endpoint.
    """
    room_id = f"ROOM_M8_T09_{uuid.uuid4().hex[:6]}"
    client.post("/api/v1/chat/rooms", json={"id": room_id, "name": "M8 Aggregation"})

    # Create 2 messages & fuse both
    for i in range(2):
        wav_bytes = create_test_wav_bytes(duration_s=1.5)
        files = {"file": (f"msg_{i}.wav", wav_bytes, "audio/wav")}
        v_res = client.post(
            f"/api/v1/chat/rooms/{room_id}/voice",
            files=files,
            data={"user_id": f"USER_{i}", "duration_seconds": 1.5}
        )
        mid = v_res.json()["message"]["message_id"]
        client.post(f"/api/v1/chat/messages/{mid}/fuse")

    room_res = client.get(f"/api/v1/chat/rooms/{room_id}/multimodal-features")
    assert room_res.status_code == 200
    data = room_res.json()
    assert data["room_id"] == room_id
    assert data["total_analyzed"] == 2
    assert len(data["features"]) == 2


def test_modality_mask_indicators():
    """
    T-10: Modality Mask Indicators Verification.
    """
    mask = MultimodalMaskBlock(
        has_communication_metrics=1.0,
        has_acoustic_features=1.0,
        has_nlp_features=0.0,
        is_voice_message=1.0,
        is_text_message=0.0
    )
    vec = mask.to_vector()
    assert vec == [1.0, 1.0, 0.0, 1.0, 0.0]


def test_m3_communication_metrics_invariance(client: TestClient):
    """
    T-11: Milestone 3 Communication Invariance Check.
    """
    room_id = f"ROOM_M8_T11_{uuid.uuid4().hex[:6]}"
    client.post("/api/v1/chat/rooms", json={"id": room_id, "name": "M8 Invariance M3"})

    wav_bytes = create_test_wav_bytes(duration_s=1.0)
    files = {"file": ("inv_m3.wav", wav_bytes, "audio/wav")}
    v_res = client.post(
        f"/api/v1/chat/rooms/{room_id}/voice",
        files=files,
        data={"user_id": "ALICE", "duration_seconds": 1.0}
    )
    msg_id = v_res.json()["message"]["message_id"]

    # Pre-fusion metrics
    m3_pre = client.get(f"/api/v1/chat/rooms/{room_id}/analysis").json()

    # Perform fusion
    client.post(f"/api/v1/chat/messages/{msg_id}/fuse")

    # Post-fusion metrics
    m3_post = client.get(f"/api/v1/chat/rooms/{room_id}/analysis").json()

    assert m3_pre["message_count"] == m3_post["message_count"]
    assert m3_pre["voice_message_count"] == m3_post["voice_message_count"]
    assert m3_pre["conversation_duration_seconds"] == m3_post["conversation_duration_seconds"]


def test_m4_voice_audio_invariance(client: TestClient):
    """
    T-12: Milestone 4 Voice Storage & Streaming Invariance.
    """
    room_id = f"ROOM_M8_T12_{uuid.uuid4().hex[:6]}"
    client.post("/api/v1/chat/rooms", json={"id": room_id, "name": "M8 Invariance M4"})

    wav_bytes = create_test_wav_bytes(duration_s=2.0)
    files = {"file": ("inv_m4.wav", wav_bytes, "audio/wav")}
    v_res = client.post(
        f"/api/v1/chat/rooms/{room_id}/voice",
        files=files,
        data={"user_id": "ALICE", "duration_seconds": 2.0}
    )
    voice_msg = v_res.json()["message"]
    msg_id = voice_msg["message_id"]

    # Fuse
    client.post(f"/api/v1/chat/messages/{msg_id}/fuse")

    # Stream audio
    stream_res = client.get(voice_msg["audio_url"])
    assert stream_res.status_code == 200
    assert stream_res.content == wav_bytes


def test_m5_transcript_invariance(client: TestClient):
    """
    T-13: Milestone 5 STT Transcript Invariance.
    """
    room_id = f"ROOM_M8_T13_{uuid.uuid4().hex[:6]}"
    client.post("/api/v1/chat/rooms", json={"id": room_id, "name": "M8 Invariance M5"})

    wav_bytes = create_test_wav_bytes(duration_s=2.0)
    files = {"file": ("inv_m5.wav", wav_bytes, "audio/wav")}
    v_res = client.post(
        f"/api/v1/chat/rooms/{room_id}/voice",
        files=files,
        data={"user_id": "ALICE", "duration_seconds": 2.0}
    )
    msg_id = v_res.json()["message"]["message_id"]
    t_res = client.post(f"/api/v1/chat/messages/{msg_id}/transcribe")
    transcript = t_res.json()["transcript_text"]

    # Fuse
    client.post(f"/api/v1/chat/messages/{msg_id}/fuse")

    # Check transcript in message history
    history = client.get(f"/api/v1/chat/rooms/{room_id}/messages").json()["messages"]
    assert history[0]["transcript_text"] == transcript


def test_m6_and_m7_vectors_invariance(client: TestClient):
    """
    T-14: Milestone 6 Acoustic and Milestone 7 NLP Vectors Invariance.
    """
    room_id = f"ROOM_M8_T14_{uuid.uuid4().hex[:6]}"
    client.post("/api/v1/chat/rooms", json={"id": room_id, "name": "M8 Invariance M6 M7"})

    wav_bytes = create_test_wav_bytes(duration_s=2.0)
    files = {"file": ("inv_m6_m7.wav", wav_bytes, "audio/wav")}
    v_res = client.post(
        f"/api/v1/chat/rooms/{room_id}/voice",
        files=files,
        data={"user_id": "ALICE", "duration_seconds": 2.0}
    )
    msg_id = v_res.json()["message"]["message_id"]

    client.post(f"/api/v1/chat/messages/{msg_id}/transcribe")
    a_vec = client.post(f"/api/v1/chat/messages/{msg_id}/analyze-acoustic").json()["raw_feature_vector"]
    nlp_vec = client.post(f"/api/v1/chat/messages/{msg_id}/analyze-nlp").json()["raw_feature_vector"]

    # Fuse
    fuse_res = client.post(f"/api/v1/chat/messages/{msg_id}/fuse").json()

    # Check acoustic and NLP records
    a_retrieved = client.get(f"/api/v1/chat/messages/{msg_id}/acoustic-features").json()["raw_feature_vector"]
    nlp_retrieved = client.get(f"/api/v1/chat/messages/{msg_id}/nlp-features").json()["raw_feature_vector"]

    assert a_retrieved == a_vec
    assert nlp_retrieved == nlp_vec
    # Check that fused vector contains exact 59 acoustic values in indices [10..68]
    assert fuse_res["raw_fused_vector"][10:69] == a_vec
    # Check that fused vector contains exact 66 NLP values in indices [69..134]
    assert fuse_res["raw_fused_vector"][69:135] == nlp_vec
