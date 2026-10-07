import os
import logging
from typing import Optional
from app.core.config import settings
from app.services.speech_to_text.models import TranscriptionResult, TranscriptionStatus
from app.services.speech_to_text.exceptions import (
    SpeechToTextError,
    AudioFileNotFoundError,
    InvalidAudioFormatError,
    TranscriptionFailedError
)
from app.services.speech_to_text.providers.base import BaseSTTProvider
from app.services.speech_to_text.providers.mock_provider import MockSTTProvider
from app.services.speech_to_text.providers.whisper_provider import WhisperSTTProvider

logger = logging.getLogger("stt_service")


class SpeechToTextService:
    """
    Modular Speech-to-Text orchestrator managing providers and transcription workflows.
    """

    def __init__(self, provider: Optional[BaseSTTProvider] = None):
        if provider:
            self._provider = provider
        else:
            # Instantiate provider based on application configuration
            provider_type = getattr(settings, "STT_PROVIDER", "whisper").lower()
            model_name = getattr(settings, "STT_MODEL_NAME", "base")

            if provider_type == "mock":
                self._provider = MockSTTProvider()
            else:
                self._provider = WhisperSTTProvider(model_name=model_name)

        logger.info(f"Initialized SpeechToTextService with provider: {self._provider.provider_name}")

    @property
    def provider(self) -> BaseSTTProvider:
        return self._provider

    def set_provider(self, provider: BaseSTTProvider) -> None:
        """Allows dynamic switching of provider (useful for testing or fallback)."""
        self._provider = provider
        logger.info(f"Switched SpeechToTextService provider to: {self._provider.provider_name}")

    async def transcribe_audio_file(
        self,
        audio_path: str,
        language: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Validates audio file and executes transcription through the active provider.
        """
        if not os.path.exists(audio_path):
            raise AudioFileNotFoundError(f"Audio file '{audio_path}' does not exist on disk.")

        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            raise InvalidAudioFormatError(f"Audio file '{audio_path}' is empty (0 bytes).")

        try:
            result = await self._provider.transcribe(audio_path, language=language)
            logger.info(
                f"Transcribed '{os.path.basename(audio_path)}' via {result.model_name} "
                f"({len(result.text)} chars, lang={result.language})"
            )
            return result
        except SpeechToTextError:
            raise
        except Exception as exc:
            logger.error(f"Unexpected error in STT service: {exc}", exc_info=True)
            raise TranscriptionFailedError(f"Transcription execution failed: {str(exc)}")


# Global singleton instance
stt_service = SpeechToTextService()
