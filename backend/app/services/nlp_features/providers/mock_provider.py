from datetime import datetime, timezone
from app.services.nlp_features.providers.base import BaseNLPProvider
from app.services.nlp_features.models import (
    NLPFeaturesResult,
    LexicalFeatures,
    SentimentFeatures,
    EmotionFeatures,
    TFIDFFeatures,
    EmbeddingFeatures,
    TopicFeatures
)
from app.services.nlp_features.exceptions import EmptyTextError


class MockNLPProvider(BaseNLPProvider):
    """
    Deterministic Mock NLP Provider for fast unit and integration testing.
    Produces predictable 66-dimensional raw vectors without computational overhead.
    """

    def __init__(self):
        self._provider_name = "mock-nlp-provider-v1"

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

        raw_text = text.strip()
        words = raw_text.split()
        char_count = len(raw_text)
        word_count = len(words)

        lexical = LexicalFeatures(
            char_count=char_count,
            word_count=word_count,
            sentence_count=1,
            type_token_ratio_ttr=0.85,
            avg_word_length=4.5,
            first_person_singular_count=1,
            first_person_plural_count=0,
            second_person_count=1,
            third_person_count=0,
            pronoun_ratio=0.25
        )

        sentiment = SentimentFeatures(
            vader_compound=0.45,
            vader_pos=0.35,
            vader_neu=0.55,
            vader_neg=0.10,
            sentiment_pos_prob=0.65,
            sentiment_neu_prob=0.25,
            sentiment_neg_prob=0.10,
            sentiment_polarity_score=0.55
        )

        emotion = EmotionFeatures(
            emotion_joy=0.50,
            emotion_sadness=0.05,
            emotion_anger=0.05,
            emotion_fear=0.05,
            emotion_surprise=0.10,
            emotion_disgust=0.05,
            emotion_neutral=0.20
        )

        tfidf = TFIDFFeatures(
            tfidf_features=[0.05 * (i + 1) for i in range(20)],
            feature_names=[f"term_{i}" for i in range(20)]
        )

        embedding = EmbeddingFeatures(
            embedding_summary=[0.1] * 16,
            embedding_dimension=16
        )

        topic = TopicFeatures(
            topic_0=0.40,
            topic_1=0.20,
            topic_2=0.15,
            topic_3=0.15,
            topic_4=0.10
        )

        raw_vec = NLPFeaturesResult.build_raw_feature_vector(
            lexical=lexical,
            sentiment=sentiment,
            emotion=emotion,
            tfidf=tfidf,
            embedding=embedding,
            topic=topic
        )

        return NLPFeaturesResult(
            lexical=lexical,
            sentiment=sentiment,
            emotion=emotion,
            tfidf=tfidf,
            embedding=embedding,
            topic=topic,
            source_type=source_type,
            engine=self._provider_name,
            text_length=char_count,
            feature_vector_version="v1.0-66d",
            raw_feature_vector=raw_vec,
            extracted_at=datetime.now(timezone.utc)
        )
