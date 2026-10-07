from datetime import datetime, timezone, timedelta
import asyncio
import pytest
from fastapi.testclient import TestClient
from app.db.session import init_db, async_session_maker
from app.db.repository import ChatRepository
from app.analysis.communication.metrics import CommunicationMetricsEngine, MessageRecord


@pytest.fixture(autouse=True)
def setup_db():
    """Ensure database schema initialized."""
    asyncio.run(init_db())


def test_unit_metrics_engine_empty_conversation():
    """
    10. Test empty conversation handling without division by zero.
    """
    res = CommunicationMetricsEngine.calculate_metrics(
        room_id="EMPTY_ROOM",
        messages=[],
        participant_names={"PID-ALICE": "Alice", "PID-BOB": "Bob"}
    )
    assert res["message_count"] == 0
    assert res["text_message_count"] == 0
    assert res["voice_message_count"] == 0
    assert res["average_message_length"] == 0.0
    assert res["average_message_length_words"] == 0.0
    assert res["conversation_duration_seconds"] == 0.0
    assert res["average_response_time_seconds"] is None
    assert res["min_response_time_seconds"] is None
    assert res["max_response_time_seconds"] is None
    assert res["number_of_communication_gaps"] == 0
    assert res["message_ratio"]["PID-ALICE"] == 0.0
    assert res["message_ratio"]["PID-BOB"] == 0.0


def test_unit_metrics_engine_deterministic_calculations():
    """
    Test deterministic calculations for:
    3. Message counting
    4. Participant message counts
    5. Message ratio
    6. Message length (avg, min, max)
    7. Conversation duration
    8. Response-time calculation (avg, min, max)
    9. Message-type counting
    """
    base_time = datetime(2026, 8, 30, 12, 0, 0, tzinfo=timezone.utc)
    
    # 4 messages:
    # m1: Alice (10 chars, 2 words) at t=0s
    # m2: Bob (20 chars, 4 words) at t=5s (response time = 5s)
    # m3: Bob (15 chars, 3 words) at t=10s (no speaker switch -> response time not recorded here)
    # m4: Alice (30 chars, 5 words) at t=80s (response time = 70s -> gap >= 60s!)
    msgs = [
        MessageRecord(
            message_id="m1",
            sender_id="PID-A",
            sender_name="Alice",
            receiver_id="PID-B",
            message_text="Hello Bob!",  # 10 chars, 2 words
            timestamp=base_time,
            message_type="text"
        ),
        MessageRecord(
            message_id="m2",
            sender_id="PID-B",
            sender_name="Bob",
            receiver_id="PID-A",
            message_text="Hi Alice, how are you?",  # 22 chars, 5 words
            timestamp=base_time + timedelta(seconds=5),
            message_type="text"
        ),
        MessageRecord(
            message_id="m3",
            sender_id="PID-B",
            sender_name="Bob",
            receiver_id="PID-A",
            message_text="Good to connect.",  # 16 chars, 3 words
            timestamp=base_time + timedelta(seconds=10),
            message_type="text"
        ),
        MessageRecord(
            message_id="m4",
            sender_id="PID-A",
            sender_name="Alice",
            receiver_id="PID-B",
            message_text="Likewise Bob, let us start our review.",  # 38 chars, 7 words
            timestamp=base_time + timedelta(seconds=80),
            message_type="text"
        )
    ]

    metrics = CommunicationMetricsEngine.calculate_metrics(
        room_id="ROOM_CALC",
        messages=msgs,
        gap_threshold_seconds=60.0
    )

    # Message counts
    assert metrics["message_count"] == 4
    assert metrics["text_message_count"] == 4
    assert metrics["voice_message_count"] == 0
    assert metrics["messages_by_participant"]["PID-A"] == 2
    assert metrics["messages_by_participant"]["PID-B"] == 2
    assert metrics["message_ratio"]["PID-A"] == 0.5
    assert metrics["message_ratio"]["PID-B"] == 0.5

    # Message lengths (chars: 10, 22, 16, 38 -> sum=86, avg=21.5)
    assert metrics["min_message_length"] == 10
    assert metrics["max_message_length"] == 38
    assert metrics["average_message_length"] == 21.5

    # Message lengths (words: 2, 5, 3, 7 -> sum=17, avg=4.25)
    assert metrics["average_message_length_words"] == 4.25

    # Duration: 80s - 0s = 80s
    assert metrics["conversation_duration_seconds"] == 80.0

    # Turn latencies:
    # m1 (A) -> m2 (B): latency = 5.0s
    # m2 (B) -> m3 (B): same sender, no switch
    # m3 (B) -> m4 (A): latency = 70.0s
    # Total transitions = 2, avg = (5 + 70) / 2 = 37.5s
    assert metrics["min_response_time_seconds"] == 5.0
    assert metrics["max_response_time_seconds"] == 70.0
    assert metrics["average_response_time_seconds"] == 37.5

    # Gaps: 70.0s >= 60.0s -> 1 gap
    assert metrics["number_of_communication_gaps"] == 1


def test_api_conversation_lifecycle_and_analysis(client: TestClient):
    """
    1. Conversation creation
    2. Conversation retrieval
    11. Invalid room handling
    12. API response structure verification
    """
    import uuid
    room_id = f"ROOM_ANALYTICS_{uuid.uuid4().hex[:8]}"

    # Step 1: Create Room
    res_create = client.post("/api/v1/chat/rooms", json={"id": room_id, "name": "Analytics Testing Room"})
    assert res_create.status_code == 201

    # Step 2: Seed messages in database via repository
    async def seed_messages():
        async with async_session_maker() as db:
            await ChatRepository.get_or_create_user(db, "USER_1", "Alice")
            await ChatRepository.get_or_create_user(db, "USER_2", "Bob")
            await ChatRepository.add_or_update_participant(db, room_id, "USER_1")
            await ChatRepository.add_or_update_participant(db, room_id, "USER_2")
            
            t0 = datetime(2026, 8, 30, 15, 0, 0, tzinfo=timezone.utc)
            await ChatRepository.save_message(db, room_id, "USER_1", "Hello there!", timestamp=t0)
            await ChatRepository.save_message(db, room_id, "USER_2", "General Kenobi!", timestamp=t0 + timedelta(seconds=3))

    asyncio.run(seed_messages())

    # Step 3: GET /api/v1/chat/conversations/{room_id}
    res_conv = client.get(f"/api/v1/chat/conversations/{room_id}")
    assert res_conv.status_code == 200
    conv_data = res_conv.json()
    assert conv_data["room_id"] == room_id
    assert conv_data["status"] == "active"
    assert conv_data["participant_count"] == 2
    assert conv_data["message_count"] == 2

    # Step 4: GET /api/v1/chat/rooms/{room_id}/analysis
    res_anal = client.get(f"/api/v1/chat/rooms/{room_id}/analysis")
    assert res_anal.status_code == 200
    anal_data = res_anal.json()

    assert anal_data["conversation_id"] == room_id
    assert anal_data["message_count"] == 2
    assert anal_data["messages_by_participant"]["USER_1"] == 1
    assert anal_data["messages_by_participant"]["USER_2"] == 1
    assert anal_data["message_ratio"]["USER_1"] == 0.5
    assert anal_data["message_ratio"]["USER_2"] == 0.5
    assert anal_data["average_response_time_seconds"] == 3.0
    assert anal_data["conversation_duration_seconds"] == 3.0
    assert anal_data["number_of_communication_gaps"] == 0
    assert len(anal_data["participant_breakdown"]) == 2

    # Step 5: End conversation session
    res_end = client.post(f"/api/v1/chat/conversations/{room_id}/end")
    assert res_end.status_code == 200
    end_data = res_end.json()
    assert end_data["status"] == "completed"
    assert end_data["ended_at"] is not None

    # Step 6: Test invalid room handling (404)
    res_404 = client.get("/api/v1/chat/rooms/NON_EXISTENT_ROOM_XYZ/analysis")
    assert res_404.status_code == 404
    assert "not found" in res_404.json()["detail"].lower()
