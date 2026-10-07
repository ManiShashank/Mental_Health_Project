from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
from dataclasses import dataclass


@dataclass
class MessageRecord:
    """
    Standardized in-memory message object for metric extraction.
    """
    message_id: str
    sender_id: str
    sender_name: str
    receiver_id: Optional[str]
    message_text: str
    timestamp: datetime
    message_type: str = "text"


class CommunicationMetricsEngine:
    """
    Deterministic communication statistics and turn-taking dynamics calculation engine.
    Conforms to Milestone 3 requirements with zero clinical or mental-health claims.
    """

    DEFAULT_GAP_THRESHOLD_SECONDS: float = 60.0

    @classmethod
    def calculate_metrics(
        cls,
        room_id: str,
        messages: List[MessageRecord],
        participant_names: Optional[Dict[str, str]] = None,
        gap_threshold_seconds: float = DEFAULT_GAP_THRESHOLD_SECONDS,
        room_status: str = "active",
        started_at: Optional[datetime] = None,
        ended_at: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Computes descriptive communication metrics from stored message records.
        """
        names_map = dict(participant_names or {})

        # Handle empty conversation edge-case
        if not messages:
            return {
                "conversation_id": room_id,
                "room_id": room_id,
                "status": room_status,
                "started_at": started_at,
                "ended_at": ended_at,
                "participants": list(names_map.keys()),
                "participant_names": names_map,
                "message_count": 0,
                "text_message_count": 0,
                "voice_message_count": 0,
                "messages_by_participant": {uid: 0 for uid in names_map},
                "message_ratio": {uid: 0.0 for uid in names_map},
                "average_message_length": 0.0,
                "average_message_length_words": 0.0,
                "min_message_length": 0,
                "max_message_length": 0,
                "conversation_duration_seconds": 0.0,
                "average_response_time_seconds": None,
                "min_response_time_seconds": None,
                "max_response_time_seconds": None,
                "number_of_communication_gaps": 0,
                "gap_threshold_seconds": gap_threshold_seconds,
                "participant_breakdown": []
            }

        # Sort messages chronologically by timestamp
        sorted_msgs = sorted(messages, key=lambda m: m.timestamp)
        total_messages = len(sorted_msgs)

        # Populate participant names from message records if not already provided
        for m in sorted_msgs:
            if m.sender_id not in names_map and m.sender_name:
                names_map[m.sender_id] = m.sender_name

        participants = sorted(list(names_map.keys()))

        # 1. Message Type Counts
        text_msgs = [m for m in sorted_msgs if m.message_type == "text"]
        voice_msgs = [m for m in sorted_msgs if m.message_type == "voice"]
        text_count = len(text_msgs)
        voice_count = len(voice_msgs)

        # 2. Messages by Participant & Ratio
        messages_by_participant: Dict[str, int] = {p: 0 for p in participants}
        for m in sorted_msgs:
            messages_by_participant[m.sender_id] = messages_by_participant.get(m.sender_id, 0) + 1

        message_ratio: Dict[str, float] = {
            p: round(count / total_messages, 4) if total_messages > 0 else 0.0
            for p, count in messages_by_participant.items()
        }

        # 3. Message Lengths (Character and Word Counts)
        char_lengths = [len(m.message_text) for m in text_msgs] if text_msgs else [0]
        word_counts = [len(m.message_text.split()) for m in text_msgs] if text_msgs else [0]

        avg_char_length = round(sum(char_lengths) / text_count, 2) if text_count > 0 else 0.0
        avg_word_length = round(sum(word_counts) / text_count, 2) if text_count > 0 else 0.0
        min_char_length = min(char_lengths) if text_msgs else 0
        max_char_length = max(char_lengths) if text_msgs else 0

        # 4. Conversation Duration (Seconds)
        first_time = sorted_msgs[0].timestamp
        last_time = sorted_msgs[-1].timestamp
        duration_seconds = max(0.0, (last_time - first_time).total_seconds())

        # 5. Turn-Taking Response Times & Communication Gaps
        response_times: List[float] = []
        communication_gaps = 0

        for i in range(len(sorted_msgs) - 1):
            curr_msg = sorted_msgs[i]
            next_msg = sorted_msgs[i + 1]

            # Only calculate response latency on turn switch between different participants
            if curr_msg.sender_id != next_msg.sender_id:
                latency = (next_msg.timestamp - curr_msg.timestamp).total_seconds()
                # Protect against potential clock skew
                latency = max(0.0, latency)
                response_times.append(latency)

                if latency >= gap_threshold_seconds:
                    communication_gaps += 1

        if response_times:
            avg_response_time = round(sum(response_times) / len(response_times), 3)
            min_response_time = round(min(response_times), 3)
            max_response_time = round(max(response_times), 3)
        else:
            avg_response_time = None
            min_response_time = None
            max_response_time = None

        # 6. Granular Per-Participant Breakdown
        participant_breakdown = []
        for p_id in participants:
            p_msgs = [m for m in text_msgs if m.sender_id == p_id]
            p_count = messages_by_participant.get(p_id, 0)
            p_chars = [len(m.message_text) for m in p_msgs]
            p_words = [len(m.message_text.split()) for m in p_msgs]
            total_words = sum(p_words)

            participant_breakdown.append({
                "user_id": p_id,
                "display_name": names_map.get(p_id, p_id),
                "message_count": p_count,
                "message_percentage": message_ratio.get(p_id, 0.0),
                "average_message_length_chars": round(sum(p_chars) / len(p_msgs), 2) if p_msgs else 0.0,
                "average_message_length_words": round(total_words / len(p_msgs), 2) if p_msgs else 0.0,
                "total_words": total_words
            })

        return {
            "conversation_id": room_id,
            "room_id": room_id,
            "status": room_status,
            "started_at": started_at or first_time,
            "ended_at": ended_at,
            "participants": participants,
            "participant_names": names_map,
            "message_count": total_messages,
            "text_message_count": text_count,
            "voice_message_count": voice_count,
            "messages_by_participant": messages_by_participant,
            "message_ratio": message_ratio,
            "average_message_length": avg_char_length,
            "average_message_length_words": avg_word_length,
            "min_message_length": min_char_length,
            "max_message_length": max_char_length,
            "conversation_duration_seconds": round(duration_seconds, 2),
            "average_response_time_seconds": avg_response_time,
            "min_response_time_seconds": min_response_time,
            "max_response_time_seconds": max_response_time,
            "number_of_communication_gaps": communication_gaps,
            "gap_threshold_seconds": gap_threshold_seconds,
            "participant_breakdown": participant_breakdown
        }
