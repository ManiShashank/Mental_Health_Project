from abc import ABC, abstractmethod
from typing import Optional
from app.services.speech_to_text.models import TranscriptionResult


class BaseSTTProvider(ABC):
    """
    Abstract Base Class for Speech-to-Text Providers.
    Ensures modularity and swappability between local Whisper, cloud APIs, or test mocks.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name identifying this provider implementation."""
        pass

    @abstractmethod
    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Transcribes the given audio file and returns a standardized TranscriptionResult.
        """
        pass
