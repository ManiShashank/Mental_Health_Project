import math
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class LexicalFeatures(BaseModel):
    """10 Lexical, syntactic, and pronoun distribution features (Indices 00..09)."""
    char_count: int = Field(..., description="Total character count")
    word_count: int = Field(..., description="Total token/word count")
    sentence_count: int = Field(..., description="Total sentence count")
    type_token_ratio_ttr: float = Field(..., description="Lexical diversity: unique tokens / total words")
    avg_word_length: float = Field(..., description="Average word length in characters")
    first_person_singular_count: int = Field(..., description="Count of 1st person singular pronouns (I, me, my, mine, myself)")
    first_person_plural_count: int = Field(..., description="Count of 1st person plural pronouns (we, us, our, ours, ourselves)")
    second_person_count: int = Field(..., description="Count of 2nd person pronouns (you, your, yours, yourself)")
    third_person_count: int = Field(..., description="Count of 3rd person pronouns (he, she, it, they, him, her, them, etc.)")
    pronoun_ratio: float = Field(..., description="Ratio of total pronouns to total words")


class SentimentFeatures(BaseModel):
    """8 Sentiment polarity metrics (Indices 10..17)."""
    vader_compound: float = Field(..., description="VADER composite normalized valence score in [-1.0, 1.0]")
    vader_pos: float = Field(..., description="VADER positive lexicon proportion in [0.0, 1.0]")
    vader_neu: float = Field(..., description="VADER neutral lexicon proportion in [0.0, 1.0]")
    vader_neg: float = Field(..., description="VADER negative lexicon proportion in [0.0, 1.0]")
    sentiment_pos_prob: float = Field(..., description="Contextual positive probability in [0.0, 1.0]")
    sentiment_neu_prob: float = Field(..., description="Contextual neutral probability in [0.0, 1.0]")
    sentiment_neg_prob: float = Field(..., description="Contextual negative probability in [0.0, 1.0]")
    sentiment_polarity_score: float = Field(..., description="Polarity score: pos_prob - neg_prob in [-1.0, 1.0]")


class EmotionFeatures(BaseModel):
    """7 Affective emotion distribution probabilities summing to 1.0 (Indices 18..24)."""
    emotion_joy: float = Field(..., description="Joy probability in [0.0, 1.0]")
    emotion_sadness: float = Field(..., description="Sadness probability in [0.0, 1.0]")
    emotion_anger: float = Field(..., description="Anger probability in [0.0, 1.0]")
    emotion_fear: float = Field(..., description="Fear / apprehension probability in [0.0, 1.0]")
    emotion_surprise: float = Field(..., description="Surprise / novelty probability in [0.0, 1.0]")
    emotion_disgust: float = Field(..., description="Disgust / aversion probability in [0.0, 1.0]")
    emotion_neutral: float = Field(..., description="Neutral / objective probability in [0.0, 1.0]")


class TFIDFFeatures(BaseModel):
    """20 Canonical Lexical TF-IDF Projections (Indices 25..44)."""
    tfidf_features: List[float] = Field(..., description="Exact 20 canonical TF-IDF n-gram weights")
    feature_names: List[str] = Field(default_factory=list, description="Associated n-gram vocabulary tokens")

    @field_validator("tfidf_features")
    @classmethod
    def validate_dim(cls, v: List[float]) -> List[float]:
        if len(v) != 20:
            raise ValueError(f"TFIDF features must have length 20, got {len(v)}")
        return v


class EmbeddingFeatures(BaseModel):
    """16 Dense Semantic Embedding Summary Dimensions (Indices 45..60)."""
    embedding_summary: List[float] = Field(..., description="Exact 16 sentence embedding summary dimensions")
    embedding_dimension: int = Field(default=16, description="Dimension of embedding summary")

    @field_validator("embedding_summary")
    @classmethod
    def validate_dim(cls, v: List[float]) -> List[float]:
        if len(v) != 16:
            raise ValueError(f"Embedding summary must have length 16, got {len(v)}")
        return v


class TopicFeatures(BaseModel):
    """5 Latent Dirichlet Allocation (LDA) Topic Probabilities (Indices 61..65)."""
    topic_0: float = Field(..., description="Topic 0: Task / Academic / Project discussion")
    topic_1: float = Field(..., description="Topic 1: Interpersonal / Social / Greetings")
    topic_2: float = Field(..., description="Topic 2: Cognitive / Problem solving / Reflection")
    topic_3: float = Field(..., description="Topic 3: Affective / Emotional expressions")
    topic_4: float = Field(..., description="Topic 4: Procedural / Logistics / Time management")


class NLPFeaturesResult(BaseModel):
    """
    Standardized NLP Feature Extraction Domain Container (Milestone 7).
    Encapsulates all 6 feature families and constructs the exact 66-dimensional raw vector (v1.0-66d).
    """
    lexical: LexicalFeatures
    sentiment: SentimentFeatures
    emotion: EmotionFeatures
    tfidf: TFIDFFeatures
    embedding: EmbeddingFeatures
    topic: TopicFeatures

    source_type: str = Field(..., description="'text_message' | 'voice_transcript'")
    engine: str = Field(default="rule-lexical-v1.0.0", description="Extraction engine identifier")
    text_length: int = Field(..., description="Length of analyzed text in characters")
    feature_vector_version: str = Field(default="v1.0-66d", description="Schema version of raw vector")
    raw_feature_vector: List[float] = Field(..., description="Deterministic 66-dimensional numeric feature vector")
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @staticmethod
    def build_raw_feature_vector(
        lexical: LexicalFeatures,
        sentiment: SentimentFeatures,
        emotion: EmotionFeatures,
        tfidf: TFIDFFeatures,
        embedding: EmbeddingFeatures,
        topic: TopicFeatures
    ) -> List[float]:
        """
        Builds the immutable, deterministic 66-dimensional raw feature vector (v1.0-66d).
        Indices:
        [00..09]: Lexical & Syntactic (10 features)
        [10..17]: Sentiment Polarity Dynamics (8 features)
        [18..24]: Affective Emotion Probabilities (7 features)
        [25..44]: TF-IDF Projections (20 features)
        [45..60]: Semantic Embedding Dimensions (16 features)
        [61..65]: Thematic LDA Topic Proportions (5 features)
        Total: Exactly 66 elements.
        """
        vec: List[float] = [
            # [00..09] Lexical & Syntactic
            float(lexical.char_count),
            float(lexical.word_count),
            float(lexical.sentence_count),
            float(lexical.type_token_ratio_ttr),
            float(lexical.avg_word_length),
            float(lexical.first_person_singular_count),
            float(lexical.first_person_plural_count),
            float(lexical.second_person_count),
            float(lexical.third_person_count),
            float(lexical.pronoun_ratio),

            # [10..17] Sentiment Polarity
            float(sentiment.vader_compound),
            float(sentiment.vader_pos),
            float(sentiment.vader_neu),
            float(sentiment.vader_neg),
            float(sentiment.sentiment_pos_prob),
            float(sentiment.sentiment_neu_prob),
            float(sentiment.sentiment_neg_prob),
            float(sentiment.sentiment_polarity_score),

            # [18..24] Affective Emotion Probabilities
            float(emotion.emotion_joy),
            float(emotion.emotion_sadness),
            float(emotion.emotion_anger),
            float(emotion.emotion_fear),
            float(emotion.emotion_surprise),
            float(emotion.emotion_disgust),
            float(emotion.emotion_neutral),

            # [25..44] Lexical TF-IDF Projections (20 values)
            *[float(val) for val in tfidf.tfidf_features[:20]],

            # [45..60] Semantic Embedding Dimensions (16 values)
            *[float(val) for val in embedding.embedding_summary[:16]],

            # [61..65] Thematic LDA Topics (5 values)
            float(topic.topic_0),
            float(topic.topic_1),
            float(topic.topic_2),
            float(topic.topic_3),
            float(topic.topic_4),
        ]

        # Sanitization: Ensure all elements are finite numbers
        sanitized_vec: List[float] = []
        for v in vec:
            if math.isnan(v) or math.isinf(v):
                sanitized_vec.append(0.0)
            else:
                sanitized_vec.append(round(float(v), 6))

        if len(sanitized_vec) != 66:
            raise ValueError(f"Built raw feature vector must have exactly 66 elements, got {len(sanitized_vec)}")

        return sanitized_vec
