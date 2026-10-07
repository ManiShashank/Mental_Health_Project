from abc import ABC, abstractmethod
from app.services.nlp_features.models import NLPFeaturesResult


class BaseNLPProvider(ABC):
    """
    Abstract interface for NLP & Linguistic Feature Extraction providers.
    All concrete providers must return an NLPFeaturesResult with the exact
    66-dimensional raw feature vector (v1.0-66d).
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Unique provider identifier."""
        pass

    @abstractmethod
    async def extract(
        self,
        text: str,
        source_type: str = "text_message"
    ) -> NLPFeaturesResult:
        """
        Extracts all 6 linguistic feature families from input text.
        """
        pass
