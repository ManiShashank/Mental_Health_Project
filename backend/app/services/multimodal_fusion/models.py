"""
Multimodal Feature Fusion Domain Models, Feature Vectors, and Scaling Engine
"""
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timezone
import math
from pydantic import BaseModel, Field


def get_utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CommunicationFeatureBlock(BaseModel):
    """10-dimensional M3 communication and turn dynamics feature block."""
    turn_index: int = Field(..., description="Chronological turn position in conversation (0-indexed)")
    sender_message_ratio: float = Field(..., description="Sender message contribution ratio in session [0.0, 1.0]")
    response_latency_seconds: float = Field(..., description="Response delay in seconds since partner message (0.0 if first/same)")
    is_communication_gap: float = Field(..., description="1.0 if response latency >= 60.0s gap threshold, else 0.0")
    conversation_duration_so_far_seconds: float = Field(..., description="Elapsed conversation time in seconds up to this turn")
    message_char_length: int = Field(..., description="Character count of current message")
    message_word_length: int = Field(..., description="Word count of current message")
    avg_char_length_session: float = Field(..., description="Average message character length across session")
    avg_word_length_session: float = Field(..., description="Average message word count across session")
    turn_switch_flag: float = Field(..., description="1.0 if current turn switched speaker, else 0.0")

    def to_vector(self) -> List[float]:
        """Returns 10-dimensional vector for indices [000..009]."""
        return [
            float(self.turn_index),
            float(self.sender_message_ratio),
            float(self.response_latency_seconds),
            float(self.is_communication_gap),
            float(self.conversation_duration_so_far_seconds),
            float(self.message_char_length),
            float(self.message_word_length),
            float(self.avg_char_length_session),
            float(self.avg_word_length_session),
            float(self.turn_switch_flag),
        ]


class MultimodalMaskBlock(BaseModel):
    """5-dimensional modality availability and alignment mask flags."""
    has_communication_metrics: float = Field(default=1.0, description="1.0 if communication features present, else 0.0")
    has_acoustic_features: float = Field(default=0.0, description="1.0 if acoustic features present, else 0.0")
    has_nlp_features: float = Field(default=0.0, description="1.0 if NLP features present, else 0.0")
    is_voice_message: float = Field(default=0.0, description="1.0 if voice message, else 0.0")
    is_text_message: float = Field(default=0.0, description="1.0 if text message, else 0.0")

    def to_vector(self) -> List[float]:
        """Returns 5-dimensional mask vector for indices [135..139]."""
        return [
            float(self.has_communication_metrics),
            float(self.has_acoustic_features),
            float(self.has_nlp_features),
            float(self.is_voice_message),
            float(self.is_text_message),
        ]


class MultimodalFeatureScaler:
    """
    Deterministic Canonical Domain Scaling Engine for Multimodal Fusion (v1.0-140d).
    Maps raw feature vectors into strictly bounded [0.0, 1.0] normalized space
    using pre-established empirical reference priors without dataset leakage.
    """

    # Reference bounds [x_min_ref, x_max_ref] for each of the 140 dimensions:
    CANONICAL_REFERENCE_BOUNDS: List[Tuple[float, float]] = [
        # [000..009] Communication Dynamics (10 dims)
        (0.0, 100.0),    # 00: turn_index
        (0.0, 1.0),      # 01: sender_message_ratio
        (0.0, 300.0),    # 02: response_latency_seconds
        (0.0, 1.0),      # 03: is_communication_gap
        (0.0, 3600.0),   # 04: conversation_duration_so_far
        (0.0, 500.0),    # 05: char_length
        (0.0, 100.0),    # 06: word_length
        (0.0, 500.0),    # 07: avg_char_length_session
        (0.0, 100.0),    # 08: avg_word_length_session
        (0.0, 1.0),      # 09: turn_switch_flag

        # [010..017] Acoustic Pitch (8 dims)
        (50.0, 500.0),   # 10: f0_mean_hz
        (0.0, 150.0),    # 11: f0_std_hz
        (50.0, 500.0),   # 12: f0_min_hz
        (50.0, 500.0),   # 13: f0_max_hz
        (50.0, 500.0),   # 14: f0_median_hz
        (0.0, 450.0),    # 15: f0_range_hz
        (0.0, 1.0),      # 16: voicing_fraction
        (0.0, 0.10),     # 17: f0_jitter_local

        # [018..023] Acoustic Energy & Shimmer (6 dims)
        (0.0, 0.50),     # 18: rms_mean
        (0.0, 0.25),     # 19: rms_std
        (0.0, 0.50),     # 20: rms_min
        (0.0, 0.50),     # 21: rms_max
        (0.0, 0.50),     # 22: rms_energy_range
        (0.0, 0.20),     # 23: shimmer_local

        # [024..027] Acoustic Speech Rate (4 dims)
        (0.0, 50.0),     # 24: syllable_count
        (0.0, 60.0),     # 25: speech_duration_seconds
        (0.0, 10.0),     # 26: speaking_rate
        (0.0, 10.0),     # 27: articulation_rate

        # [028..032] Acoustic Pause Dynamics (5 dims)
        (0.0, 20.0),     # 28: pause_count
        (0.0, 20.0),     # 29: total_pause_duration
        (0.0, 5.0),      # 30: mean_pause_duration
        (0.0, 10.0),     # 31: max_pause_duration
        (0.0, 2.0),      # 32: pause_to_speech_ratio

        # [033..058] Acoustic MFCC Means (13 dims) & Stds (13 dims)
        *((-50.0, 50.0) for _ in range(13)),  # 33..45: MFCC 1..13 Means
        *((0.0, 30.0) for _ in range(13)),    # 46..58: MFCC 1..13 Stds

        # [059..068] Acoustic Spectral Descriptors (10 dims: 5 means + 5 stds)
        (0.0, 8000.0),   # 59: spectral_centroid_mean
        (0.0, 4000.0),   # 60: spectral_centroid_std
        (0.0, 8000.0),   # 61: spectral_bandwidth_mean
        (0.0, 4000.0),   # 62: spectral_bandwidth_std
        (0.0, 8000.0),   # 63: spectral_rolloff_mean
        (0.0, 4000.0),   # 64: spectral_rolloff_std
        (0.0, 1.0),      # 65: spectral_flatness_mean
        (0.0, 0.50),     # 66: spectral_flatness_std
        (0.0, 0.50),     # 67: zcr_mean
        (0.0, 0.25),     # 68: zcr_std

        # [069..078] NLP Lexical & Diversity (10 dims)
        (0.0, 500.0),    # 69: char_count
        (0.0, 100.0),    # 70: word_count
        (0.0, 20.0),     # 71: sentence_count
        (0.0, 1.0),      # 72: type_token_ratio_ttr
        (0.0, 15.0),     # 73: avg_word_length
        (0.0, 20.0),     # 74: first_person_singular_count
        (0.0, 20.0),     # 75: first_person_plural_count
        (0.0, 20.0),     # 76: second_person_count
        (0.0, 20.0),     # 77: third_person_count
        (0.0, 1.0),      # 78: pronoun_ratio

        # [079..086] NLP Sentiment Polarity (8 dims)
        (-1.0, 1.0),     # 79: vader_compound
        (0.0, 1.0),      # 80: vader_pos
        (0.0, 1.0),      # 81: vader_neu
        (0.0, 1.0),      # 82: vader_neg
        (0.0, 1.0),      # 83: sentiment_pos_prob
        (0.0, 1.0),      # 84: sentiment_neu_prob
        (0.0, 1.0),      # 85: sentiment_neg_prob
        (-1.0, 1.0),     # 86: sentiment_polarity_score

        # [087..093] NLP Affective Emotion Probabilities (7 dims)
        (0.0, 1.0),      # 87: emotion_joy
        (0.0, 1.0),      # 88: emotion_sadness
        (0.0, 1.0),      # 89: emotion_anger
        (0.0, 1.0),      # 90: emotion_fear
        (0.0, 1.0),      # 91: emotion_surprise
        (0.0, 1.0),      # 92: emotion_disgust
        (0.0, 1.0),      # 93: emotion_neutral

        # [094..113] NLP TF-IDF Projections (20 dims)
        *((0.0, 1.0) for _ in range(20)),

        # [114..129] NLP Semantic Embedding Summary (16 dims)
        *((-1.0, 1.0) for _ in range(16)),

        # [130..134] NLP Thematic LDA Topics (5 dims)
        *((0.0, 1.0) for _ in range(5)),

        # [135..139] Modality Mask Indicators (5 dims)
        (0.0, 1.0),      # 135: has_communication_metrics
        (0.0, 1.0),      # 136: has_acoustic_features
        (0.0, 1.0),      # 137: has_nlp_features
        (0.0, 1.0),      # 138: is_voice_message
        (0.0, 1.0),      # 139: is_text_message
    ]

    @classmethod
    def scale_feature(cls, idx: int, val: float) -> float:
        """Scales a single feature index using its canonical domain reference bounds."""
        if math.isnan(val) or math.isinf(val):
            return 0.0
        min_ref, max_ref = cls.CANONICAL_REFERENCE_BOUNDS[idx]
        if max_ref <= min_ref:
            return 0.0
        normalized = (val - min_ref) / (max_ref - min_ref)
        clamped = max(0.0, min(1.0, normalized))
        return round(float(clamped), 6)

    @classmethod
    def scale_vector(cls, raw_vec: List[float]) -> List[float]:
        """Transforms 140-d raw vector to 140-d normalized vector strictly bounded in [0.0, 1.0]."""
        if len(raw_vec) != 140:
            raise ValueError(f"Raw vector must have exactly 140 dimensions, got {len(raw_vec)}")
        return [cls.scale_feature(i, raw_vec[i]) for i in range(140)]


class MultimodalFusionResult(BaseModel):
    """
    Standardized Domain Transfer Object for Multimodal Feature Fusion.
    Contains both the 140-d raw vector and the 140-d normalized vector.
    """
    message_id: str
    room_id: str
    sender_id: str
    fusion_strategy: str = Field(default="early_concatenation_v1", description="Fusion algorithm strategy identifier")
    feature_vector_version: str = Field(default="v1.0-140d", description="Schema version of multimodal feature vector")

    has_communication: bool = True
    has_acoustic: bool = False
    has_nlp: bool = False

    communication_features: CommunicationFeatureBlock
    acoustic_features_summary: Optional[Dict[str, Any]] = None
    nlp_features_summary: Optional[Dict[str, Any]] = None
    modality_mask: MultimodalMaskBlock

    raw_fused_vector: List[float] = Field(..., description="Exactly 140-dimensional raw feature vector")
    normalized_fused_vector: List[float] = Field(..., description="Exactly 140-dimensional normalized feature vector in [0.0, 1.0]")
    fused_at: datetime = Field(default_factory=get_utc_now)

    @classmethod
    def assemble_vectors(
        cls,
        comm_vec: List[float],
        acoustic_vec: List[float],
        nlp_vec: List[float],
        mask_vec: List[float]
    ) -> Tuple[List[float], List[float]]:
        """
        Combines 10 comm + 59 acoustic + 66 nlp + 5 mask features into raw (140-d)
        and scaled normalized (140-d) vectors.
        """
        if len(comm_vec) != 10:
            raise ValueError(f"Communication vector must be 10-d, got {len(comm_vec)}")
        if len(acoustic_vec) != 59:
            raise ValueError(f"Acoustic vector must be 59-d, got {len(acoustic_vec)}")
        if len(nlp_vec) != 66:
            raise ValueError(f"NLP vector must be 66-d, got {len(nlp_vec)}")
        if len(mask_vec) != 5:
            raise ValueError(f"Mask vector must be 5-d, got {len(mask_vec)}")

        raw_vec: List[float] = []
        for val in comm_vec + acoustic_vec + nlp_vec + mask_vec:
            if math.isnan(val) or math.isinf(val):
                raw_vec.append(0.0)
            else:
                raw_vec.append(round(float(val), 6))

        normalized_vec = MultimodalFeatureScaler.scale_vector(raw_vec)
        return raw_vec, normalized_vec
