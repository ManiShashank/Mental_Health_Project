import io
import math
import os
import uuid
from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient
import numpy as np
import soundfile as sf

from app.core.config import settings
from app.db.session import init_db, async_session_maker
from app.db.repository import ChatRepository
from app.services.acoustic_features import (
    acoustic_service,
    MockAcousticExtractor,
    LibrosaAcousticExtractor,
    AcousticFeaturesResult,
    AcousticExtractionError
)
from app.services.speech_to_text import stt_service, MockSTTProvider


@pytest.fixture(autouse=True)
def setup_test_env():
    import asyncio
    asyncio.run(init_db())
    os.makedirs(settings.AUDIO_UPLOAD_DIR, exist_ok=True)
    stt_service.set_provider(MockSTTProvider(default_transcript="Test voice message transcript."))
    acoustic_service.set_extractor(LibrosaAcousticExtractor())


def create_synthetic_wav_bytes(
    duration_s: float = 2.0,
    sample_rate: int = 16000,
    frequency_hz: float = 220.0,
    silence: bool = False
) -> bytes:
    """Generates an in-memory mono WAV audio payload for testing."""
    n_samples = int(duration_s * sample_rate)
    if silence:
        samples = np.zeros(n_samples, dtype=np.float32)
    else:
        t = np.linspace(0, duration_s, n_samples, endpoint=False)
        # Fundamental tone + harmonic + small modulation
        samples = 0.5 * np.sin(2 * np.pi * frequency_hz * t) + 0.2 * np.sin(2 * np.pi * frequency_hz * 2 * t)
        # Add a 0.3s pause in the middle if duration >= 1.5s
        if duration_s >= 1.5:
            pause_start = int(0.6 * sample_rate)
            pause_end = int(0.95 * sample_rate)
            samples[pause_start:pause_end] = 0.0
        samples = samples.astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format='WAV', subtype='PCM_16')
    return buf.getvalue()


def test_extractor_switching_and_service_initialization():
    """1. Test service extractor switching between Librosa and Mock."""
    mock_ext = MockAcousticExtractor()
    acoustic_service.set_extractor(mock_ext)
    assert acoustic_service.extractor.extractor_name == "mock-acoustic-extractor-v1"

    librosa_ext = LibrosaAcousticExtractor()
    acoustic_service.set_extractor(librosa_ext)
    assert acoustic_service.extractor.extractor_name == "librosa-v1.0.0"


def test_normal_speech_acoustic_extraction(client: TestClient):
    """
    2. Test normal speech audio extraction with Librosa:
       - All 6 feature families populated
       - Exactly 59 finite floats in raw_feature_vector
       - Pitch, energy, pause, speech rate, MFCCs, and spectral values valid.
    """
    room_id = f"M6_NORM_{uuid.uuid4().hex[:8]}"
    client.post("/api/v1/chat/rooms", json={"id": room_id, "name": "M6 Normal Speech Room"})

    # Upload synthetic voice message
    audio_bytes = create_synthetic_wav_bytes(duration_s=2.5, frequency_hz=200.0)
    files = {"file": ("normal_speech.wav", io.BytesIO(audio_bytes), "audio/wav")}
    upload_res = client.post(
        f"/api/v1/chat/rooms/{room_id}/voice",
        files=files,
        data={"user_id": "PID-ALICE", "duration_seconds": 2.5}
    )
    assert upload_res.status_code == 201
    msg_id = upload_res.json()["message"]["message_id"]

    # Trigger acoustic analysis
    res = client.post(f"/api/v1/chat/messages/{msg_id}/analyze-acoustic")
    assert res.status_code == 200
    data = res.json()

    assert data["message_id"] == msg_id
    assert data["status"] == "completed"
    assert data["has_voiced_frames"] is True
    assert data["feature_vector_version"] == "v1.0-59d"

    # Verify Pitch Features
    pitch = data["pitch_features"]
    assert pitch["f0_mean_hz"] is not None
    assert 150.0 <= pitch["f0_mean_hz"] <= 250.0
    assert pitch["voicing_fraction"] > 0.0

    # Verify Energy Features
    energy = data["energy_features"]
    assert energy["rms_mean"] > 0.0
    assert energy["rms_max"] >= energy["rms_mean"]

    # Verify Pause Features (we added a 0.35s pause in synthetic audio)
    pauses = data["pause_features"]
    assert pauses["pause_count"] >= 1
    assert pauses["total_pause_duration_seconds"] > 0.0

    # Verify MFCCs (13 coefficients)
    mfcc = data["mfcc_features"]
    assert len(mfcc["mfcc_means"]) == 13
    assert len(mfcc["mfcc_stds"]) == 13
    assert mfcc["mfcc_dimension"] == 26

    # Verify Spectral Descriptors
    spectral = data["spectral_features"]
    assert spectral["spectral_centroid_mean"] > 0.0
    assert spectral["spectral_rolloff_mean"] > 0.0

    # Verify Exact 59-Dimensional Vector
    raw_vec = data["raw_feature_vector"]
    assert len(raw_vec) == 59
    assert all(math.isfinite(val) for val in raw_vec)


def test_pure_silence_handling_and_null_pitch(client: TestClient):
    """
    3. Test pure silence audio:
       - No divide-by-zero crashes
       - Pitch/jitter/shimmer are None (null)
       - Voicing fraction is 0.0
       - Raw vector contains 59 finite floats.
    """
    room_id = f"M6_SILENCE_{uuid.uuid4().hex[:8]}"
    client.post("/api/v1/chat/rooms", json={"id": room_id})

    # Upload pure silence WAV
    audio_bytes = create_synthetic_wav_bytes(duration_s=2.0, silence=True)
    files = {"file": ("silence.wav", io.BytesIO(audio_bytes), "audio/wav")}
    upload_res = client.post(
        f"/api/v1/chat/rooms/{room_id}/voice",
        files=files,
        data={"user_id": "PID-BOB", "duration_seconds": 2.0}
    )
    msg_id = upload_res.json()["message"]["message_id"]

    res = client.post(f"/api/v1/chat/messages/{msg_id}/analyze-acoustic")
    assert res.status_code == 200
    data = res.json()

    assert data["has_voiced_frames"] is False
    assert data["pitch_features"]["f0_mean_hz"] is None
    assert data["pitch_features"]["voicing_fraction"] == 0.0
    assert data["pitch_features"]["f0_jitter_local"] is None
    assert data["energy_features"]["rms_mean"] == 0.0

    # Raw vector remains exactly 59 finite numbers
    assert len(data["raw_feature_vector"]) == 59
    assert all(math.isfinite(val) for val in data["raw_feature_vector"])


def test_very_short_audio_handling(client: TestClient):
    """
    4. Test very short audio (< 150ms) extraction without crashing.
    """
    room_id = f"M6_SHORT_{uuid.uuid4().hex[:8]}"
    client.post("/api/v1/chat/rooms", json={"id": room_id})

    # 100ms audio
    audio_bytes = create_synthetic_wav_bytes(duration_s=0.1, frequency_hz=300.0)
    files = {"file": ("short.wav", io.BytesIO(audio_bytes), "audio/wav")}
    upload_res = client.post(
        f"/api/v1/chat/rooms/{room_id}/voice",
        files=files,
        data={"user_id": "PID-ALICE", "duration_seconds": 0.1}
    )
    msg_id = upload_res.json()["message"]["message_id"]

    res = client.post(f"/api/v1/chat/messages/{msg_id}/analyze-acoustic")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "completed"
    assert len(data["raw_feature_vector"]) == 59


def test_text_message_rejection_for_acoustic_analysis(client: TestClient):
    """
    5. Test that text messages are rejected with 400 Bad Request.
    """
    room_id = f"M6_TEXT_REJ_{uuid.uuid4().hex[:8]}"
    client.post("/api/v1/chat/rooms", json={"id": room_id})

    with client.websocket_connect(f"/api/v1/chat/ws/{room_id}/PID-ALICE?display_name=Alice") as ws:
        _ = ws.receive_json()
        _ = ws.receive_json()
        ws.send_text('{"event": "message", "data": {"message_text": "I am a text message"}}')
        echo = ws.receive_json()
        text_msg_id = echo["data"]["message_id"]

    res = client.post(f"/api/v1/chat/messages/{text_msg_id}/analyze-acoustic")
    assert res.status_code == 400
    assert "only voice messages" in res.json()["detail"].lower()


def test_nonexistent_message_404(client: TestClient):
    """
    6. Non-existent message returns 404.
    """
    fake_id = str(uuid.uuid4())
    res = client.post(f"/api/v1/chat/messages/{fake_id}/analyze-acoustic")
    assert res.status_code == 404


def test_missing_audio_file_404(client: TestClient):
    """
    7. Missing audio file on disk returns 404.
    """
    room_id = f"M6_NO_AUDIO_{uuid.uuid4().hex[:8]}"
    client.post("/api/v1/chat/rooms", json={"id": room_id})

    import asyncio
    async def create_orphan():
        async with async_session_maker() as db:
            await ChatRepository.get_or_create_user(db, "PID-TEST", "Tester")
            return await ChatRepository.save_message(
                session=db,
                room_id=room_id,
                sender_id="PID-TEST",
                message_text="[Voice]",
                message_type="voice",
                audio_url="/api/v1/chat/audio/NON_EXISTENT_FILE.webm"
            )

    orphan = asyncio.run(create_orphan())
    res = client.post(f"/api/v1/chat/messages/{orphan.id}/analyze-acoustic")
    assert res.status_code == 404
    assert "does not exist" in res.json()["detail"].lower()


def test_persistence_retrieval_and_idempotent_upsert(client: TestClient):
    """
    8. Test persistence, idempotent repeated extraction, and GET endpoints.
    """
    room_id = f"M6_PERSIST_{uuid.uuid4().hex[:8]}"
    client.post("/api/v1/chat/rooms", json={"id": room_id})

    audio_bytes = create_synthetic_wav_bytes(duration_s=1.8, frequency_hz=180.0)
    files = {"file": ("v1.wav", io.BytesIO(audio_bytes), "audio/wav")}
    upload_res = client.post(
        f"/api/v1/chat/rooms/{room_id}/voice",
        files=files,
        data={"user_id": "PID-ALICE", "duration_seconds": 1.8}
    )
    msg_id = upload_res.json()["message"]["message_id"]

    # First extraction
    res1 = client.post(f"/api/v1/chat/messages/{msg_id}/analyze-acoustic")
    assert res1.status_code == 200

    # Second extraction (idempotent upsert)
    res2 = client.post(f"/api/v1/chat/messages/{msg_id}/analyze-acoustic")
    assert res2.status_code == 200
    assert res2.json()["message_id"] == msg_id

    # GET /messages/{msg_id}/acoustic-features
    get_res = client.get(f"/api/v1/chat/messages/{msg_id}/acoustic-features")
    assert get_res.status_code == 200
    assert get_res.json()["message_id"] == msg_id
    assert len(get_res.json()["raw_feature_vector"]) == 59

    # GET /rooms/{room_id}/acoustic-features
    room_res = client.get(f"/api/v1/chat/rooms/{room_id}/acoustic-features")
    assert room_res.status_code == 200
    assert room_res.json()["total_analyzed"] == 1
    assert len(room_res.json()["features"]) == 1


def test_m3_m4_m5_invariance_after_acoustic_analysis(client: TestClient):
    """
    9. Strict invariance verification:
       - M3 communication metrics unchanged
       - M4 voice message audio streaming unchanged
       - M5 speech-to-text transcript unchanged
    """
    room_id = f"M6_INVARIANCE_{uuid.uuid4().hex[:8]}"
    client.post("/api/v1/chat/rooms", json={"id": room_id})

    # Upload 1 voice message
    audio_bytes = create_synthetic_wav_bytes(duration_s=3.0, frequency_hz=210.0)
    files = {"file": ("invariance_voice.wav", io.BytesIO(audio_bytes), "audio/wav")}
    upload_res = client.post(
        f"/api/v1/chat/rooms/{room_id}/voice",
        files=files,
        data={"user_id": "PID-ALICE", "duration_seconds": 3.0}
    )
    v_msg = upload_res.json()["message"]
    v_msg_id = v_msg["message_id"]

    # Transcribe with M5
    t_res = client.post(f"/api/v1/chat/messages/{v_msg_id}/transcribe")
    assert t_res.status_code == 200
    expected_transcript = t_res.json()["transcript_text"]

    # Send 2 text messages
    with client.websocket_connect(f"/api/v1/chat/ws/{room_id}/PID-BOB?display_name=Bob") as ws:
        _ = ws.receive_json()
        _ = ws.receive_json()
        ws.send_text('{"event": "message", "data": {"message_text": "Text A"}}')
        _ = ws.receive_json()
        ws.send_text('{"event": "message", "data": {"message_text": "Text B"}}')
        _ = ws.receive_json()

    # Capture M3 metrics BEFORE acoustic analysis
    m3_before = client.get(f"/api/v1/chat/rooms/{room_id}/analysis").json()

    # Perform M6 Acoustic Extraction
    acoustic_res = client.post(f"/api/v1/chat/messages/{v_msg_id}/analyze-acoustic")
    assert acoustic_res.status_code == 200

    # 1. Verify M3 metrics AFTER acoustic analysis are identical
    m3_after = client.get(f"/api/v1/chat/rooms/{room_id}/analysis").json()
    assert m3_after["message_count"] == m3_before["message_count"] == 3
    assert m3_after["text_message_count"] == m3_before["text_message_count"] == 2
    assert m3_after["voice_message_count"] == m3_before["voice_message_count"] == 1
    assert m3_after["messages_by_participant"] == m3_before["messages_by_participant"]
    assert m3_after["average_message_length"] == m3_before["average_message_length"]

    # 2. Verify M4 Voice Playback remains identical
    audio_filename = os.path.basename(v_msg["audio_url"])
    stream_res = client.get(f"/api/v1/chat/audio/{audio_filename}")
    assert stream_res.status_code == 200
    assert stream_res.content == audio_bytes

    # 3. Verify M5 Transcript remains intact
    hist_res = client.get(f"/api/v1/chat/rooms/{room_id}/messages")
    retrieved_v_msg = [m for m in hist_res.json()["messages"] if m["message_id"] == v_msg_id][0]
    assert retrieved_v_msg["transcript_text"] == expected_transcript
    assert retrieved_v_msg["transcription_status"] == "completed"
