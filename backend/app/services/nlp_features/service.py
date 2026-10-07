import logging
from typing import Optional

from app.core.config import settings
from app.services.nlp_features.models import NLPFeaturesResult
from app.services.nlp_features.exceptions import (
    NLPException,
    EmptyTextError,
    NLPProcessingError
)
from app.services.nlp_features.providers.base import BaseNLPProvider
from app.services.nlp_features.providers.rule_lexical_provider import RuleLexicalNLPProvider
from app.services.nlp_features.providers.transformer_provider import TransformerNLPProvider
from app.services.nlp_features.providers.mock_provider import MockNLPProvider

logger = logging.getLogger("nlp_service")


class NLPFeaturesService:
    """
    Service layer coordinator for NLP & Linguistic Feature Extraction.
    Routes text payloads to active provider, handles validation, and manages exceptions.
    """

    def __init__(self, provider: Optional[BaseNLPProvider] = None):
        if provider:
            self._provider = provider
        else:
            provider_type = getattr(settings, "NLP_PROVIDER", "rule_lexical").lower()
            if provider_type == "mock":
                self._provider = MockNLPProvider()
            elif provider_type == "transformer":
                self._provider = TransformerNLPProvider(
                    embedding_model=getattr(settings, "NLP_EMBEDDING_MODEL", "distilbert-base-uncased"),
                    emotion_model=getattr(settings, "NLP_EMOTION_MODEL", "bhadresh-psav/roberta-base-go_emotions")
                )
            else:
                self._provider = RuleLexicalNLPProvider()

        logger.info(f"Initialized NLPFeaturesService with provider: {self._provider.provider_name}")

    @property
    def provider(self) -> BaseNLPProvider:
        return self._provider

    def set_provider(self, provider: BaseNLPProvider) -> None:
        """Dynamically switch active NLP provider (useful for unit testing)."""
        self._provider = provider
        logger.info(f"Switched NLPFeaturesService provider to: {self._provider.provider_name}")

    async def extract_features(
        self,
        text: str,
        source_type: str = "text_message"
    ) -> NLPFeaturesResult:
        """
        Validates text input and coordinates linguistic, sentiment, affective, and topic extraction.
        """
        if not text or not text.strip():
            raise EmptyTextError("Cannot perform NLP feature extraction on empty or whitespace-only text.")

        try:
            result = await self._provider.extract(text, source_type=source_type)
            logger.info(
                f"Extracted NLP features via {result.engine} "
                f"(source={source_type}, words={result.lexical.word_count}, "
                f"vector_dim={len(result.raw_feature_vector)}, sentiment_compound={result.sentiment.vader_compound:.2f})"
            )
            return result
        except NLPException:
            raise
        except Exception as exc:
            logger.error(f"Unexpected NLP extraction error: {exc}", exc_info=True)
            raise NLPProcessingError(f"NLP extraction failed: {str(exc)}")


# Global singleton instance
nlp_service = NLPFeaturesService()
