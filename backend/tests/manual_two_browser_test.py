import asyncio
import json
import websockets
import urllib.request


async def test_two_clients_live():
    base_ws_url = "ws://127.0.0.1:8000/api/v1/chat/ws/TEST001"
    room_id = "TEST001"

    print("=" * 60)
    print("STARTING LIVE TWO-CLIENT INTEGRATION TEST (ALICE & BOB)")
    print("=" * 60)

    # 1. Connect Alice (Browser A)
    alice_url = f"{base_ws_url}/PID-ALICE?display_name=Alice"
    bob_url = f"{base_ws_url}/PID-BOB?display_name=Bob"

    async with websockets.connect(alice_url) as ws_alice:
        print("[Browser A / Alice] Connected to Room TEST001")
        
        # Read Alice initial events (history + user_joined)
        ev1 = json.loads(await ws_alice.recv())
        print(f"[Browser A / Alice] Received initial event: {ev1['event']}")
        ev2 = json.loads(await ws_alice.recv())
        print(f"[Browser A / Alice] Received self-join event: {ev2['event']}")

        # 2. Connect Bob (Browser B)
        async with websockets.connect(bob_url) as ws_bob:
            print("[Browser B / Bob] Connected to Room TEST001")
            
            # Read Bob initial events
            bob_ev1 = json.loads(await ws_bob.recv())
            print(f"[Browser B / Bob] Received initial event: {bob_ev1['event']}")
            bob_ev2 = json.loads(await ws_bob.recv())
            print(f"[Browser B / Bob] Received self-join event: {bob_ev2['event']}")

            # Alice receives notification that Bob joined
            alice_saw_bob = json.loads(await ws_alice.recv())
            print(f"[Browser A / Alice] Received peer join: {alice_saw_bob['data']['display_name']} joined")
            assert alice_saw_bob["data"]["user_id"] == "PID-BOB"

            # 3. Alice sends: "Hello Bob"
            alice_msg = {
                "event": "message",
                "data": {
                    "message_text": "Hello Bob",
                    "receiver_id": "PID-BOB"
                }
            }
            print("[Browser A / Alice] Sending message: 'Hello Bob'")
            await ws_alice.send(json.dumps(alice_msg))

            # Alice and Bob both receive the broadcast
            alice_echo = json.loads(await ws_alice.recv())
            bob_recv = json.loads(await ws_bob.recv())

            print(f"[Browser B / Bob] Received message from {bob_recv['data']['sender_name']}: '{bob_recv['data']['message_text']}'")
            print(f"[Browser B / Bob] Message UUID: {bob_recv['data']['message_id']}, Timestamp (UTC): {bob_recv['data']['timestamp']}")
            assert bob_recv["data"]["message_text"] == "Hello Bob"
            assert bob_recv["data"]["sender_id"] == "PID-ALICE"

            # 4. Bob replies: "Hello Alice"
            bob_reply = {
                "event": "message",
                "data": {
                    "message_text": "Hello Alice",
                    "receiver_id": "PID-ALICE"
                }
            }
            print("[Browser B / Bob] Sending reply: 'Hello Alice'")
            await ws_bob.send(json.dumps(bob_reply))

            alice_recv_reply = json.loads(await ws_alice.recv())
            bob_echo_reply = json.loads(await ws_bob.recv())

            print(f"[Browser A / Alice] Received reply from {alice_recv_reply['data']['sender_name']}: '{alice_recv_reply['data']['message_text']}'")
            assert alice_recv_reply["data"]["message_text"] == "Hello Alice"
            assert alice_recv_reply["data"]["sender_id"] == "PID-BOB"

            # 5. Test empty message validation
            print("[Browser A / Alice] Testing empty message submission...")
            await ws_alice.send(json.dumps({"event": "message", "data": {"message_text": "   "}}))
            alice_err = json.loads(await ws_alice.recv())
            print(f"[Browser A / Alice] Received expected error event: {alice_err['data']['detail']}")
            assert alice_err["event"] == "error"

    # 6. Verify Database Persistence via REST API
    print("\nVerifying Database Persistence via REST API GET /api/v1/chat/rooms/TEST001/messages...")
    url = "http://127.0.0.1:8000/api/v1/chat/rooms/TEST001/messages"
    req = urllib.request.urlopen(url)
    data = json.loads(req.read().decode("utf-8"))
    
    print(f"Total Messages Persisted in DB for Room TEST001: {data['total_messages']}")
    for m in data["messages"]:
        print(f" - [{m['timestamp']}] {m['sender_name']} ({m['sender_id']}): {m['message_text']} (ID: {m['message_id']})")

    assert any(m["message_text"] == "Hello Bob" for m in data["messages"])
    assert any(m["message_text"] == "Hello Alice" for m in data["messages"])

    print("\n" + "=" * 60)
    print("ALL TWO-CLIENT LIVE WEBSOCKET & PERSISTENCE CHECKS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_two_clients_live())
