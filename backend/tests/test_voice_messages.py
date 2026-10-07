import io
import os
import uuid
from datetime import datetime, timezone, timedelta
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.db.session import init_db, async_session_maker
from app.db.repository import ChatRepository
from app.analysis.communication.metrics import CommunicationMetricsEngine, MessageRecord


@pytest.fixture(autouse=True)
def ensure_db():
    import asyncio
    asyncio.run(init_db())
    os.makedirs(settings.AUDIO_UPLOAD_DIR, exist_ok=True)


def test_voice_upload_validation_and_persistence(client: TestClient):
    """
    1. Test voice message schema / validation
    2. Valid voice message creation and file persistence
    3. Retrieval via GET /api/v1/chat/audio/{filename}
    4. Database persistence and message history
    """
    room_id = f"VOICE_ROOM_{uuid.uuid4().hex[:8]}"
    user_id = "PID-ALICE"

    # Step 1: Create room
    res_room = client.post("/api/v1/chat/rooms", json={"id": room_id, "name": "Voice Testing Room"})
    assert res_room.status_code == 201

    # Step 2: Upload valid audio file
    fake_audio_content = b"RIFF....WAVEfmt ....data...."  # Simulated audio binary
    files = {
        "file": ("sample_recording.webm", io.BytesIO(fake_audio_content), "audio/webm")
    }
    data = {
        "user_id": user_id,
        "duration_seconds": 4.5,
    }

    res_upload = client.post(f"/api/v1/chat/rooms/{room_id}/voice", files=files, data=data)
    assert res_upload.status_code == 201
    upload_json = res_upload.json()

    assert "message" in upload_json
    msg = upload_json["message"]
    assert msg["message_type"] == "voice"
    assert msg["audio_duration_seconds"] == 4.5
    assert msg["audio_format"] == "audio/webm"
    assert msg["audio_size_bytes"] == len(fake_audio_content)
    assert msg["audio_url"] is not None
    assert msg["audio_url"].startswith("/api/v1/chat/audio/")

    # Step 3: Verify audio file retrieval
    audio_filename = os.path.basename(msg["audio_url"])
    res_audio = client.get(f"/api/v1/chat/audio/{audio_filename}")
    assert res_audio.status_code == 200
    assert res_audio.content == fake_audio_content
    assert "audio/webm" in res_audio.headers.get("content-type", "")

    # Step 4: Verify message in room history
    res_hist = client.get(f"/api/v1/chat/rooms/{room_id}/messages")
    assert res_hist.status_code == 200
    hist_json = res_hist.json()
    assert hist_json["total_messages"] == 1
    stored_msg = hist_json["messages"][0]
    assert stored_msg["message_type"] == "voice"
    assert stored_msg["audio_url"] == msg["audio_url"]
    assert stored_msg["audio_duration_seconds"] == 4.5


def test_invalid_audio_upload_rejection(client: TestClient):
    """
    Test safe validation:
    - 0-byte audio file rejection
    - Unsupported audio format rejection
    """
    room_id = f"VOICE_ERR_ROOM_{uuid.uuid4().hex[:8]}"
    client.post("/api/v1/chat/rooms", json={"id": room_id})

    # 1. Zero-byte audio file
    empty_file = {
        "file": ("empty.webm", io.BytesIO(b""), "audio/webm")
    }
    res_empty = client.post(f"/api/v1/chat/rooms/{room_id}/voice", files=empty_file, data={"user_id": "PID-TEST"})
    assert res_empty.status_code == 400
    assert "empty" in res_empty.json()["detail"].lower()

    # 2. Unsupported format (e.g. application/pdf or text/plain)
    invalid_format_file = {
        "file": ("document.pdf", io.BytesIO(b"%PDF-1.4..."), "application/pdf")
    }
    res_bad_format = client.post(f"/api/v1/chat/rooms/{room_id}/voice", files=invalid_format_file, data={"user_id": "PID-TEST"})
    assert res_bad_format.status_code == 400
    assert "unsupported audio format" in res_bad_format.json()["detail"].lower()


def test_nonexistent_audio_file_404(client: TestClient):
    """
    Test audio streaming 404 for nonexistent file
    """
    res = client.get("/api/v1/chat/audio/NON_EXISTENT_AUDIO_FILE_12345.webm")
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()


def test_voice_and_text_m3_metrics_integration(client: TestClient):
    """
    Test M3 analytical integration:
    - 3 text messages + 1 voice message
    - Verify M3 reports:
      total_messages = 4
      text_message_count = 3
      voice_message_count = 1
    """
    room_id = f"M3_VOICE_ROOM_{uuid.uuid4().hex[:8]}"
    client.post("/api/v1/chat/rooms", json={"id": room_id})

    # Seed 3 text messages
    client.post(f"/api/v1/chat/rooms/{room_id}/voice", files={
        "file": ("voice_1.webm", io.BytesIO(b"AUDIO_BYTES_SAMPLE_123"), "audio/webm")
    }, data={"user_id": "PID-ALICE", "duration_seconds": 3.2})

    # Connect WebSocket and send 3 text messages
    with client.websocket_connect(f"/api/v1/chat/ws/{room_id}/PID-BOB?display_name=Bob") as ws:
        _ = ws.receive_json()  # history
        _ = ws.receive_json()  # user_joined

        # Text 1
        ws.send_text('{"event": "message", "data": {"message_text": "Text Message 1"}}')
        _ = ws.receive_json()

        # Text 2
        ws.send_text('{"event": "message", "data": {"message_text": "Text Message 2"}}')
        _ = ws.receive_json()

        # Text 3
        ws.send_text('{"event": "message", "data": {"message_text": "Text Message 3"}}')
        _ = ws.receive_json()

    # Query M3 Analysis Endpoint
    res_analysis = client.get(f"/api/v1/chat/rooms/{room_id}/analysis")
    assert res_analysis.status_code == 200
    m3_data = res_analysis.json()

    assert m3_data["message_count"] == 4
    assert m3_data["text_message_count"] == 3
    assert m3_data["voice_message_count"] == 1
    assert m3_data["conversation_id"] == room_id
    assert m3_data["messages_by_participant"]["PID-ALICE"] == 1
    assert m3_data["messages_by_participant"]["PID-BOB"] == 3
