from datetime import datetime, timezone
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, ConfigDict


class ParticipantStats(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    user_id: str
    display_name: str
    message_count: int
    message_percentage: float = Field(..., description="Proportion of total messages (0.0 - 1.0)")
    average_message_length_chars: float
    average_message_length_words: float
    total_words: int


class CommunicationMetricsResponse(BaseModel):
    """
    Standardized communication analysis schema conforming to Milestone 3 requirements.
    All calculations are traceable to underlying stored messages without clinical claims.
    """
    model_config = ConfigDict(from_attributes=True)
    conversation_id: str = Field(..., description="Conversation / Room unique identifier")
    room_id: str = Field(..., description="Room identifier")
    status: str = Field(..., description="Conversation state: active | completed | idle")
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    participants: List[str] = Field(..., description="List of participant IDs")
    participant_names: Dict[str, str] = Field(default_factory=dict, description="Mapping of participant ID to display name")
    
    # Message Counts
    message_count: int = Field(..., description="Total message count (text + voice)")
    text_message_count: int = Field(..., description="Count of text messages")
    voice_message_count: int = Field(..., description="Count of voice messages")
    messages_by_participant: Dict[str, int] = Field(..., description="Message count mapped by participant ID")
    message_ratio: Dict[str, float] = Field(..., description="Participation ratio per user (sums to 1.0)")

    # Message Length Metrics
    average_message_length: float = Field(..., description="Mean message character count")
    average_message_length_words: float = Field(..., description="Mean message word count")
    min_message_length: int = Field(..., description="Minimum message character length")
    max_message_length: int = Field(..., description="Maximum message character length")

    # Temporal & Turn-Taking Dynamics
    conversation_duration_seconds: float = Field(..., description="Total elapsed seconds from first to last message")
    average_response_time_seconds: Optional[float] = Field(None, description="Mean turn-taking latency between alternating speakers")
    min_response_time_seconds: Optional[float] = Field(None, description="Minimum turn-taking latency in seconds")
    max_response_time_seconds: Optional[float] = Field(None, description="Maximum turn-taking latency in seconds")
    number_of_communication_gaps: int = Field(..., description="Number of turn intervals exceeding gap threshold (>= 60s)")
    gap_threshold_seconds: float = Field(default=60.0, description="Inactivity latency threshold defining a communication gap")

    # Detailed Breakdown
    participant_breakdown: List[ParticipantStats] = Field(default_factory=list, description="Per-participant granular statistics")


class ConversationDetailsResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    conversation_id: str
    room_id: str
    name: Optional[str] = None
    status: str
    is_active: bool
    created_at: datetime
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    participant_count: int
    message_count: int
    participants: List[Dict[str, Any]]
