from abc import ABC, abstractmethod
from typing import Optional
from app.services.acoustic_features.models import AcousticFeaturesResult


class BaseAcousticExtractor(ABC):
    """
    Abstract Base Class for all acoustic feature extraction engines.
    """

    @property
    @abstractmethod
    def extractor_name(self) -> str:
        """Returns the identifier name of this extractor engine."""
        pass

    @abstractmethod
    async def extract(
        self,
        audio_path: str,
        sample_rate: Optional[int] = 16000
    ) -> AcousticFeaturesResult:
        """
        Asynchronously extracts the complete standardized acoustic feature set from an audio file.
        """
        pass
