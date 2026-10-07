from enum import Enum
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field


class TranscriptionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TranscriptionResult(BaseModel):
    """
    Standardized result contract returned by any STT provider.
    """
    text: str = Field(..., description="Transcribed textual representation of speech")
    language: str = Field(default="en", description="Detected or configured BCP-47 language code")
    duration_seconds: Optional[float] = Field(None, description="Audio duration processed in seconds")
    model_name: str = Field(..., description="Name/identifier of STT model used")
    confidence: Optional[float] = Field(None, description="Mean transcription confidence score (0.0 - 1.0)")
    created_at: datetime = Field(default_factory=datetime.now)
