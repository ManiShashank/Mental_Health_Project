import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.db.session import init_db
import asyncio


@pytest.fixture(autouse=True)
def setup_db():
    """Ensure database tables are initialized before tests."""
    asyncio.run(init_db())


def test_rest_create_and_list_rooms(client: TestClient):
    """
    Test REST endpoint for creating and listing rooms.
    """
    # Create room
    res_create = client.post(
        "/api/v1/chat/rooms",
        json={"id": "ROOM_TEST_01", "name": "Research Room Alpha"}
    )
    assert res_create.status_code == 201
    data = res_create.json()
    assert data["id"] == "ROOM_TEST_01"
    assert data["name"] == "Research Room Alpha"
    assert data["is_active"] is True

    # List rooms
    res_list = client.get("/api/v1/chat/rooms")
    assert res_list.status_code == 200
    rooms = res_list.json()
    assert any(r["id"] == "ROOM_TEST_01" for r in rooms)


def test_websocket_single_user_connection(client: TestClient):
    """
    1. Test single user WebSocket connection, history reception, and user_joined broadcast.
    """
    room_id = "ROOM_WS_01"
    user_id = "USER_ALICE"
    display_name = "Alice"

    with client.websocket_connect(f"/api/v1/chat/ws/{room_id}/{user_id}?display_name={display_name}") as ws:
        # First event: history
        ev1 = ws.receive_json()
        assert ev1["event"] == "history"
        assert ev1["data"]["room_id"] == room_id
        assert isinstance(ev1["data"]["messages"], list)

        # Second event: user_joined broadcast
        ev2 = ws.receive_json()
        assert ev2["event"] == "user_joined"
        assert ev2["data"]["user_id"] == user_id
        assert ev2["data"]["display_name"] == display_name


def test_two_users_communication_and_persistence(client: TestClient):
    """
    2. Two users joining the same room.
    3. Alice sends message -> Bob receives it immediately.
    4. Bob replies -> Alice receives it immediately.
    5. Database message persistence.
    """
    room_id = "ROOM_DYAD_01"
    
    # Alice connects
    with client.websocket_connect(f"/api/v1/chat/ws/{room_id}/ALICE_ID?display_name=Alice") as ws_alice:
        # Alice history + user_joined
        _ = ws_alice.receive_json()  # history
        _ = ws_alice.receive_json()  # user_joined Alice

        # Bob connects
        with client.websocket_connect(f"/api/v1/chat/ws/{room_id}/BOB_ID?display_name=Bob") as ws_bob:
            # Bob receives history + user_joined Bob
            _ = ws_bob.receive_json()  # history
            ev_bob_joined_for_bob = ws_bob.receive_json()  # user_joined
            assert ev_bob_joined_for_bob["event"] == "user_joined"

            # Alice receives notification that Bob joined
            ev_bob_joined_for_alice = ws_alice.receive_json()
            assert ev_bob_joined_for_alice["event"] == "user_joined"
            assert ev_bob_joined_for_alice["data"]["user_id"] == "BOB_ID"

            # 1. Alice sends "Hello Bob"
            alice_msg_payload = {
                "event": "message",
                "data": {
                    "message_text": "Hello Bob",
                    "receiver_id": "BOB_ID"
                }
            }
            ws_alice.send_json(alice_msg_payload)

            # Both Alice and Bob receive the broadcasted message
            alice_recv = ws_alice.receive_json()
            bob_recv = ws_bob.receive_json()

            assert alice_recv["event"] == "message"
            assert alice_recv["data"]["message_text"] == "Hello Bob"
            assert alice_recv["data"]["sender_id"] == "ALICE_ID"
            assert "message_id" in alice_recv["data"]
            assert "timestamp" in alice_recv["data"]

            assert bob_recv["event"] == "message"
            assert bob_recv["data"]["message_text"] == "Hello Bob"
            assert bob_recv["data"]["sender_name"] == "Alice"

            # 2. Bob replies "Hello Alice"
            bob_msg_payload = {
                "event": "message",
                "data": {
                    "message_text": "Hello Alice",
                    "receiver_id": "ALICE_ID"
                }
            }
            ws_bob.send_json(bob_msg_payload)

            alice_recv2 = ws_alice.receive_json()
            bob_recv2 = ws_bob.receive_json()

            assert alice_recv2["data"]["message_text"] == "Hello Alice"
            assert alice_recv2["data"]["sender_id"] == "BOB_ID"
            assert bob_recv2["data"]["message_text"] == "Hello Alice"

    # 3. Verify Database Message Persistence via REST API
    res_hist = client.get(f"/api/v1/chat/rooms/{room_id}/messages")
    assert res_hist.status_code == 200
    hist_data = res_hist.json()
    assert hist_data["total_messages"] >= 2
    messages = hist_data["messages"]
    assert any(m["message_text"] == "Hello Bob" and m["sender_id"] == "ALICE_ID" for m in messages)
    assert any(m["message_text"] == "Hello Alice" and m["sender_id"] == "BOB_ID" for m in messages)


def test_empty_message_rejection(client: TestClient):
    """
    6. Test empty message rejection and server-side validation error.
    """
    room_id = "ROOM_VALIDATE_01"
    with client.websocket_connect(f"/api/v1/chat/ws/{room_id}/USER_VAL?display_name=Val") as ws:
        _ = ws.receive_json()  # history
        _ = ws.receive_json()  # user_joined

        # Send empty message
        ws.send_json({
            "event": "message",
            "data": {"message_text": "   "}
        })

        err_event = ws.receive_json()
        assert err_event["event"] == "error"
        assert "Validation Error" in err_event["data"]["detail"]


def test_invalid_json_payload_handling(client: TestClient):
    """
    7. Test malformed JSON text handling without crashing socket.
    """
    room_id = "ROOM_INVALID_01"
    with client.websocket_connect(f"/api/v1/chat/ws/{room_id}/USER_INV?display_name=Inv") as ws:
        _ = ws.receive_json()  # history
        _ = ws.receive_json()  # user_joined

        # Send invalid non-JSON string
        ws.send_text("this-is-not-json-format")

        err_event = ws.receive_json()
        assert err_event["event"] == "error"
        assert "Invalid JSON" in err_event["data"]["detail"]


def test_disconnect_and_user_left_broadcast(client: TestClient):
    """
    8. Test disconnect handling and user_left event broadcast.
    """
    room_id = "ROOM_DISCONNECT_01"
    
    with client.websocket_connect(f"/api/v1/chat/ws/{room_id}/USER_STAY?display_name=Stay") as ws_stay:
        _ = ws_stay.receive_json()  # history
        _ = ws_stay.receive_json()  # user_joined Stay

        with client.websocket_connect(f"/api/v1/chat/ws/{room_id}/USER_LEAVE?display_name=Leaver") as ws_leave:
            _ = ws_leave.receive_json()  # history
            _ = ws_leave.receive_json()  # user_joined Leaver
            _ = ws_stay.receive_json()   # user_joined Leaver for Stay

        # ws_leave is now closed. ws_stay should receive user_left event
        ev_left = ws_stay.receive_json()
        assert ev_left["event"] == "user_left"
        assert ev_left["data"]["user_id"] == "USER_LEAVE"
        assert ev_left["data"]["display_name"] == "Leaver"
