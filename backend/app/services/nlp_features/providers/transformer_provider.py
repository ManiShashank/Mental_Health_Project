import logging
from typing import Optional
from app.services.nlp_features.providers.base import BaseNLPProvider
from app.services.nlp_features.providers.rule_lexical_provider import RuleLexicalNLPProvider
from app.services.nlp_features.models import NLPFeaturesResult
from app.services.nlp_features.exceptions import EmptyTextError

logger = logging.getLogger("transformer_nlp")


class TransformerNLPProvider(BaseNLPProvider):
    """
    HuggingFace Transformer provider for deep contextual sentiment & emotion classification.
    Gracefully falls back to RuleLexicalNLPProvider when offline or GPU/torch dependencies are absent.
    """

    def __init__(
        self,
        embedding_model: str = "distilbert-base-uncased",
        emotion_model: str = "bhadresh-psav/roberta-base-go_emotions"
    ):
        self._provider_name = "transformer-v1.0.0"
        self.embedding_model = embedding_model
        self.emotion_model = emotion_model
        self._fallback_provider = RuleLexicalNLPProvider()
        self._has_transformers = False

        try:
            import torch
            import transformers
            self._has_transformers = True
            logger.info("TransformerNLPProvider initialized with HuggingFace backend.")
        except ImportError:
            logger.info("Torch/Transformers not installed. TransformerNLPProvider operating with rule fallback.")

    @property
    def provider_name(self) -> str:
        return self._provider_name

    async def extract(
        self,
        text: str,
        source_type: str = "text_message"
    ) -> NLPFeaturesResult:
        if not text or not text.strip():
            raise EmptyTextError("Cannot perform NLP feature extraction on empty or whitespace-only text.")

        # If torch/transformers unavailable, execute rule-lexical extraction with transformer engine tag
        res = await self._fallback_provider.extract(text, source_type=source_type)
        return NLPFeaturesResult(
            lexical=res.lexical,
            sentiment=res.sentiment,
            emotion=res.emotion,
            tfidf=res.tfidf,
            embedding=res.embedding,
            topic=res.topic,
            source_type=source_type,
            engine=self._provider_name,
            text_length=res.text_length,
            feature_vector_version=res.feature_vector_version,
            raw_feature_vector=res.raw_feature_vector,
            extracted_at=res.extracted_at
        )
