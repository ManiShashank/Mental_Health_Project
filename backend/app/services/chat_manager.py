from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
import json
import logging
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("chat_manager")


class ConnectionManager:
    """
    Manages active WebSocket connections organized by room.
    Room mapping: room_id -> { user_id: (WebSocket, display_name) }
    """

    def __init__(self):
        self._rooms: Dict[str, Dict[str, Tuple[WebSocket, str]]] = {}

    def get_room_participants(self, room_id: str) -> List[Dict[str, str]]:
        if room_id not in self._rooms:
            return []
        return [
            {"user_id": uid, "display_name": dname}
            for uid, (_, dname) in self._rooms[room_id].items()
        ]

    async def connect(
        self,
        websocket: WebSocket,
        room_id: str,
        user_id: str,
        display_name: str
    ) -> None:
        await websocket.accept()
        if room_id not in self._rooms:
            self._rooms[room_id] = {}
        self._rooms[room_id][user_id] = (websocket, display_name)
        logger.info(f"User {user_id} ({display_name}) connected to room {room_id}")

    def disconnect(self, room_id: str, user_id: str) -> Optional[str]:
        display_name = None
        if room_id in self._rooms and user_id in self._rooms[room_id]:
            _, display_name = self._rooms[room_id].pop(user_id)
            if not self._rooms[room_id]:
                del self._rooms[room_id]
            logger.info(f"User {user_id} disconnected from room {room_id}")
        return display_name

    async def send_personal(self, websocket: WebSocket, event_data: Dict[str, Any]) -> None:
        try:
            # Ensure UTC timestamp in payload
            if "timestamp" not in event_data:
                event_data["timestamp"] = datetime.now(timezone.utc).isoformat()
            elif isinstance(event_data["timestamp"], datetime):
                event_data["timestamp"] = event_data["timestamp"].isoformat()
            await websocket.send_json(event_data)
        except Exception as exc:
            logger.warning(f"Error sending personal message: {exc}")

    async def broadcast_to_room(
        self,
        room_id: str,
        event_data: Dict[str, Any],
        exclude_user_id: Optional[str] = None
    ) -> None:
        if room_id not in self._rooms:
            return

        if "timestamp" not in event_data:
            event_data["timestamp"] = datetime.now(timezone.utc).isoformat()
        elif isinstance(event_data["timestamp"], datetime):
            event_data["timestamp"] = event_data["timestamp"].isoformat()

        stale_user_ids = []
        for user_id, (ws, _) in self._rooms[room_id].items():
            if exclude_user_id and user_id == exclude_user_id:
                continue
            try:
                await ws.send_json(event_data)
            except Exception as exc:
                logger.warning(f"Failed to send to user {user_id} in room {room_id}: {exc}")
                stale_user_ids.append(user_id)

        # Clean up stale connections
        for user_id in stale_user_ids:
            self.disconnect(room_id, user_id)


manager = ConnectionManager()
