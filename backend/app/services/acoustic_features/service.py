import os
import logging
from typing import Optional

from app.core.config import settings
from app.services.acoustic_features.models import (
    AcousticFeaturesResult,
    AcousticAnalysisStatus
)
from app.services.acoustic_features.exceptions import (
    AcousticExtractionError,
    AudioFileNotFoundError,
    InvalidAudioFormatError,
    AudioSignalProcessingError
)
from app.services.acoustic_features.extractors.base import BaseAcousticExtractor
from app.services.acoustic_features.extractors.mock_extractor import MockAcousticExtractor
from app.services.acoustic_features.extractors.librosa_extractor import LibrosaAcousticExtractor

logger = logging.getLogger("acoustic_service")


class AcousticFeaturesService:
    """
    Service layer orchestrator for acoustic feature extraction.
    Manages extractor providers, configuration routing, and validation.
    """

    def __init__(self, extractor: Optional[BaseAcousticExtractor] = None):
        if extractor:
            self._extractor = extractor
        else:
            provider_type = getattr(settings, "ACOUSTIC_EXTRACTOR_PROVIDER", "librosa").lower()
            if provider_type == "mock":
                self._extractor = MockAcousticExtractor()
            else:
                self._extractor = LibrosaAcousticExtractor(
                    sample_rate=getattr(settings, "ACOUSTIC_SAMPLE_RATE", 16000),
                    n_mfcc=getattr(settings, "ACOUSTIC_N_MFCC", 13),
                    fmin=getattr(settings, "ACOUSTIC_FMIN", 65.0),
                    fmax=getattr(settings, "ACOUSTIC_FMAX", 500.0),
                    silence_threshold_db=getattr(settings, "ACOUSTIC_SILENCE_THRESHOLD_DB", -30.0),
                    min_pause_duration=getattr(settings, "ACOUSTIC_MIN_PAUSE_DURATION", 0.15)
                )

        logger.info(f"Initialized AcousticFeaturesService with extractor: {self._extractor.extractor_name}")

    @property
    def extractor(self) -> BaseAcousticExtractor:
        return self._extractor

    def set_extractor(self, extractor: BaseAcousticExtractor) -> None:
        """Dynamically switch active extractor (useful for isolated unit testing)."""
        self._extractor = extractor
        logger.info(f"Switched AcousticFeaturesService extractor to: {self._extractor.extractor_name}")

    async def extract_features(
        self,
        audio_path: str,
        sample_rate: Optional[int] = None
    ) -> AcousticFeaturesResult:
        """
        Validates audio file and extracts all 6 acoustic feature families.
        """
        if not os.path.exists(audio_path):
            raise AudioFileNotFoundError(f"Audio file '{audio_path}' does not exist on disk.")

        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            raise InvalidAudioFormatError(f"Audio file '{audio_path}' is empty (0 bytes).")

        try:
            result = await self._extractor.extract(audio_path, sample_rate=sample_rate)
            logger.info(
                f"Extracted acoustic features from '{os.path.basename(audio_path)}' "
                f"via {result.engine} (duration={result.duration_seconds:.2f}s, "
                f"vector_dim={len(result.raw_feature_vector)}, voiced={result.has_voiced_frames})"
            )
            return result
        except AcousticExtractionError:
            raise
        except Exception as exc:
            logger.error(f"Unexpected acoustic extraction error on '{audio_path}': {exc}", exc_info=True)
            raise AudioSignalProcessingError(f"Acoustic extraction failed: {str(exc)}")


# Global singleton instance
acoustic_service = AcousticFeaturesService()
