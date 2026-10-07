import os
from typing import Optional
from app.services.speech_to_text.providers.base import BaseSTTProvider
from app.services.speech_to_text.models import TranscriptionResult
from app.services.speech_to_text.exceptions import AudioFileNotFoundError


class MockSTTProvider(BaseSTTProvider):
    """
    Deterministic Mock STT Provider for automated unit testing, CI pipelines,
    and development environments without heavy deep-learning model downloads.
    """

    def __init__(self, default_transcript: str = "This is a verified test transcription."):
        self.default_transcript = default_transcript
        self._provider_name = "mock-stt"

    @property
    def provider_name(self) -> str:
        return self._provider_name

    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None
    ) -> TranscriptionResult:
        if not os.path.exists(audio_path):
            raise AudioFileNotFoundError(f"Audio file '{audio_path}' does not exist.")

        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            return TranscriptionResult(
                text="",
                language=language or "en",
                duration_seconds=0.0,
                model_name="mock-stt-v1",
                confidence=0.0
            )

        return TranscriptionResult(
            text=self.default_transcript,
            language=language or "en",
            duration_seconds=3.5,
            model_name="mock-stt-v1",
            confidence=0.98
        )
