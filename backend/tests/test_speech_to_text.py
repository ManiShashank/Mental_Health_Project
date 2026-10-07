import io
import os
import uuid
from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.db.session import init_db, async_session_maker
from app.db.repository import ChatRepository
from app.services.speech_to_text import (
    stt_service,
    MockSTTProvider,
    WhisperSTTProvider,
    TranscriptionResult,
    SpeechToTextError
)


@pytest.fixture(autouse=True)
def setup_test_env():
    import asyncio
    asyncio.run(init_db())
    os.makedirs(settings.AUDIO_UPLOAD_DIR, exist_ok=True)
    # Ensure testing uses deterministic mock provider by default
    stt_service.set_provider(MockSTTProvider(default_transcript="Hello Bob, this is a test recording."))


def test_stt_service_initialization_and_provider_switching():
    """1. Test STT service initialization and dynamic provider switching."""
    mock_p = MockSTTProvider(default_transcript="Custom testing transcript")
    stt_service.set_provider(mock_p)
    assert stt_service.provider.provider_name == "mock-stt"

    whisper_p = WhisperSTTProvider(model_name="tiny")
    stt_service.set_provider(whisper_p)
    assert stt_service.provider.provider_name == "whisper-tiny"

    # Reset back to mock
    stt_service.set_provider(mock_p)


def test_valid_voice_message_transcription_and_persistence(client: TestClient):
    """
    2. Valid audio transcription
    3. Transcript persistence
    4. Retrieval of transcript through API
    9. Existing M4 voice message remains playable
    """
    room_id = f"STT_ROOM_{uuid.uuid4().hex[:8]}"
    client.post("/api/v1/chat/rooms", json={"id": room_id, "name": "STT Testing Room"})

    # Step 1: Upload a voice message
    audio_bytes = b"RIFF....WAVEfmt ....data...." + b"\x00" * 512
    files = {"file": ("test_voice.webm", io.BytesIO(audio_bytes), "audio/webm")}
    upload_res = client.post(
        f"/api/v1/chat/rooms/{room_id}/voice",
        files=files,
        data={"user_id": "PID-ALICE", "duration_seconds": 3.8}
    )
    assert upload_res.status_code == 201
    msg_data = upload_res.json()["message"]
    msg_id = msg_data["message_id"]

    # Step 2: Trigger transcription via POST /api/v1/chat/messages/{msg_id}/transcribe
    transcribe_res = client.post(f"/api/v1/chat/messages/{msg_id}/transcribe")
    assert transcribe_res.status_code == 200
    t_data = transcribe_res.json()

    assert t_data["message_id"] == msg_id
    assert t_data["room_id"] == room_id
    assert t_data["transcription_status"] == "completed"
    assert t_data["transcript_text"] == "Hello Bob, this is a test recording."
    assert t_data["transcription_model"] == "mock-stt-v1"
    assert t_data["transcribed_at"] is not None

    # Step 3: Verify message history retrieval contains transcript & audio remains playable
    hist_res = client.get(f"/api/v1/chat/rooms/{room_id}/messages")
    assert hist_res.status_code == 200
    hist_msgs = hist_res.json()["messages"]
    assert len(hist_msgs) == 1
    persisted_msg = hist_msgs[0]

    assert persisted_msg["message_type"] == "voice"
    assert persisted_msg["transcript_text"] == "Hello Bob, this is a test recording."
    assert persisted_msg["transcription_status"] == "completed"
    assert persisted_msg["audio_url"] is not None

    # Step 4: Verify audio file playback endpoint remains functional
    audio_filename = os.path.basename(persisted_msg["audio_url"])
    audio_stream_res = client.get(f"/api/v1/chat/audio/{audio_filename}")
    assert audio_stream_res.status_code == 200
    assert audio_stream_res.content == audio_bytes


def test_text_message_transcription_rejection(client: TestClient):
    """
    5. Text message cannot be transcribed (400 Bad Request).
    """
    room_id = f"STT_TEXT_ERR_{uuid.uuid4().hex[:8]}"
    client.post("/api/v1/chat/rooms", json={"id": room_id})

    # Send text message via WebSocket or direct DB seed
    with client.websocket_connect(f"/api/v1/chat/ws/{room_id}/PID-BOB?display_name=Bob") as ws:
        _ = ws.receive_json()  # history
        _ = ws.receive_json()  # user_joined
        ws.send_text('{"event": "message", "data": {"message_text": "Hello, this is pure text"}}')
        echo = ws.receive_json()
        text_msg_id = echo["data"]["message_id"]

    # Attempt to transcribe text message
    res = client.post(f"/api/v1/chat/messages/{text_msg_id}/transcribe")
    assert res.status_code == 400
    assert "only voice messages" in res.json()["detail"].lower()


def test_nonexistent_message_transcription_404(client: TestClient):
    """
    6. Missing message returns 404 error.
    """
    fake_msg_id = str(uuid.uuid4())
    res = client.post(f"/api/v1/chat/messages/{fake_msg_id}/transcribe")
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()


def test_missing_audio_file_transcription_error(client: TestClient):
    """
    7. Missing audio file on disk returns 404 error.
    """
    room_id = f"STT_NO_FILE_{uuid.uuid4().hex[:8]}"
    client.post("/api/v1/chat/rooms", json={"id": room_id})

    # Create voice message record in DB pointing to non-existent audio file
    import asyncio
    async def create_orphan_message():
        async with async_session_maker() as db:
            await ChatRepository.get_or_create_user(db, "PID-TEST", "Tester")
            return await ChatRepository.save_message(
                session=db,
                room_id=room_id,
                sender_id="PID-TEST",
                message_text="[Voice Message]",
                message_type="voice",
                audio_url="/api/v1/chat/audio/DELETED_OR_MISSING_AUDIO.webm"
            )

    orphan_msg = asyncio.run(create_orphan_message())

    res = client.post(f"/api/v1/chat/messages/{orphan_msg.id}/transcribe")
    assert res.status_code == 404
    assert "does not exist" in res.json()["detail"].lower()


def test_failed_transcription_handling_and_status_update(client: TestClient):
    """
    8. Failed transcription safely updates database status to 'failed'.
    """
    room_id = f"STT_FAIL_ROOM_{uuid.uuid4().hex[:8]}"
    client.post("/api/v1/chat/rooms", json={"id": room_id})

    # Upload voice file
    files = {"file": ("err.webm", io.BytesIO(b"AUDIO_FAIL_TEST"), "audio/webm")}
    upload_res = client.post(
        f"/api/v1/chat/rooms/{room_id}/voice",
        files=files,
        data={"user_id": "PID-FAIL", "duration_seconds": 1.0}
    )
    msg_id = upload_res.json()["message"]["message_id"]

    # Configure provider to raise an exception
    class FailingSTTProvider(MockSTTProvider):
        async def transcribe(self, audio_path: str, language=None):
            raise SpeechToTextError("Inference engine CUDA out-of-memory or model crashed.")

    stt_service.set_provider(FailingSTTProvider())

    # Trigger transcription -> expect 500
    res = client.post(f"/api/v1/chat/messages/{msg_id}/transcribe")
    assert res.status_code == 500
    assert "transcription failed" in res.json()["detail"].lower()

    # Verify DB status updated to 'failed'
    hist_res = client.get(f"/api/v1/chat/rooms/{room_id}/messages")
    msg = hist_res.json()["messages"][0]
    assert msg["transcription_status"] == "failed"
    assert msg["transcription_error"] is not None
    assert "crashed" in msg["transcription_error"].lower()


def test_m3_metrics_invariance_after_transcription(client: TestClient):
    """
    10. Existing M3 metrics remain unchanged after voice transcription.
    """
    room_id = f"STT_M3_INVARIANCE_{uuid.uuid4().hex[:8]}"
    client.post("/api/v1/chat/rooms", json={"id": room_id})

    # Upload 1 voice message
    files = {"file": ("v1.webm", io.BytesIO(b"AUDIO_DATA_FOR_M3"), "audio/webm")}
    upload_res = client.post(
        f"/api/v1/chat/rooms/{room_id}/voice",
        files=files,
        data={"user_id": "PID-ALICE", "duration_seconds": 5.0}
    )
    v_msg_id = upload_res.json()["message"]["message_id"]

    # Send 2 text messages
    with client.websocket_connect(f"/api/v1/chat/ws/{room_id}/PID-BOB?display_name=Bob") as ws:
        _ = ws.receive_json()
        _ = ws.receive_json()
        ws.send_text('{"event": "message", "data": {"message_text": "Text 1"}}')
        _ = ws.receive_json()
        ws.send_text('{"event": "message", "data": {"message_text": "Text 2"}}')
        _ = ws.receive_json()

    # Transcribe the voice message
    stt_service.set_provider(MockSTTProvider(default_transcript="M3 invariance transcript"))
    t_res = client.post(f"/api/v1/chat/messages/{v_msg_id}/transcribe")
    assert t_res.status_code == 200

    # Query M3 analysis
    m3_res = client.get(f"/api/v1/chat/rooms/{room_id}/analysis")
    assert m3_res.status_code == 200
    m3_data = m3_res.json()

    # Assert invariant: 2 text + 1 voice = 3 total messages
    assert m3_data["message_count"] == 3
    assert m3_data["text_message_count"] == 2
    assert m3_data["voice_message_count"] == 1
    assert m3_data["messages_by_participant"]["PID-ALICE"] == 1
    assert m3_data["messages_by_participant"]["PID-BOB"] == 2
