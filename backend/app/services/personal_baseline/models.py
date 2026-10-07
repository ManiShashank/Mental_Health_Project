"""
Milestone 9: Domain Models and Authoritative 140-d Feature Registry
"""
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timezone
import math
from pydantic import BaseModel, Field


def get_utc_now() -> datetime:
    return datetime.now(timezone.utc)


class FeatureMetadata(BaseModel):
    """Metadata specification for each dimension in v1.0-140d."""
    index: int
    name: str
    modality: str  # "communication" | "acoustic" | "nlp" | "mask"
    is_behavioral: bool = True
    description: str


def build_canonical_feature_registry() -> List[FeatureMetadata]:
    """Builds authoritative list of 140 feature definitions matching v1.0-140d."""
    features: List[FeatureMetadata] = []

    # [000..009] Communication Dynamics (10 dims)
    comm_names = [
        ("turn_index", "Turn index in conversation"),
        ("sender_message_ratio", "Participant message ratio in conversation"),
        ("response_latency_seconds", "Response latency since partner turn"),
        ("is_communication_gap", "Flag if latency >= 60.0s gap threshold"),
        ("conversation_duration_so_far_seconds", "Elapsed conversation time in seconds"),
        ("message_char_length", "Message character length"),
        ("message_word_length", "Message word length"),
        ("avg_char_length_session", "Session average character length"),
        ("avg_word_length_session", "Session average word length"),
        ("turn_switch_flag", "Turn switch indicator flag"),
    ]
    for idx, (name, desc) in enumerate(comm_names):
        features.append(FeatureMetadata(index=idx, name=name, modality="communication", is_behavioral=True, description=desc))

    # [010..068] Acoustic Descriptors (59 dims)
    acoustic_names = [
        # Pitch (8)
        ("f0_mean_hz", "Pitch F0 mean in Hz"),
        ("f0_std_hz", "Pitch F0 standard deviation in Hz"),
        ("f0_min_hz", "Pitch F0 minimum in Hz"),
        ("f0_max_hz", "Pitch F0 maximum in Hz"),
        ("f0_median_hz", "Pitch F0 median in Hz"),
        ("f0_range_hz", "Pitch F0 range in Hz"),
        ("voicing_fraction", "Voicing fraction ratio"),
        ("f0_jitter_local", "Pitch jitter local relative"),
        # Energy & Shimmer (6)
        ("rms_mean", "RMS energy mean"),
        ("rms_std", "RMS energy standard deviation"),
        ("rms_min", "RMS energy minimum"),
        ("rms_max", "RMS energy maximum"),
        ("rms_energy_range", "RMS energy dynamic range"),
        ("shimmer_local", "Shimmer local relative"),
        # Speech rate (4)
        ("syllable_count_estimated", "Estimated syllable count"),
        ("speech_duration_seconds", "Speech duration in seconds"),
        ("speaking_rate_estimated", "Estimated speaking rate syllables per second"),
        ("articulation_rate_estimated", "Estimated articulation rate syllables per speech second"),
        # Pause dynamics (5)
        ("pause_count", "Detected pause count"),
        ("total_pause_duration_seconds", "Total pause duration in seconds"),
        ("mean_pause_duration_seconds", "Mean pause duration in seconds"),
        ("max_pause_duration_seconds", "Max pause duration in seconds"),
        ("pause_to_speech_ratio", "Ratio of pause duration to speech duration"),
        # MFCC 1..13 Means (13)
        *[ (f"mfcc_{k}_mean", f"MFCC coefficient {k} mean") for k in range(1, 14) ],
        # MFCC 1..13 Stds (13)
        *[ (f"mfcc_{k}_std", f"MFCC coefficient {k} standard deviation") for k in range(1, 14) ],
        # Spectral Descriptors (10)
        ("spectral_centroid_mean", "Spectral centroid mean in Hz"),
        ("spectral_centroid_std", "Spectral centroid standard deviation in Hz"),
        ("spectral_bandwidth_mean", "Spectral bandwidth mean in Hz"),
        ("spectral_bandwidth_std", "Spectral bandwidth standard deviation in Hz"),
        ("spectral_rolloff_mean", "Spectral rolloff mean in Hz"),
        ("spectral_rolloff_std", "Spectral rolloff standard deviation in Hz"),
        ("spectral_flatness_mean", "Spectral flatness mean"),
        ("spectral_flatness_std", "Spectral flatness standard deviation"),
        ("zero_crossing_rate_mean", "Zero crossing rate mean"),
        ("zero_crossing_rate_std", "Zero crossing rate standard deviation"),
    ]
    for idx, (name, desc) in enumerate(acoustic_names, start=10):
        features.append(FeatureMetadata(index=idx, name=name, modality="acoustic", is_behavioral=True, description=desc))

    # [069..134] NLP & Linguistic Descriptors (66 dims)
    nlp_names = [
        # Lexical (10)
        ("char_count", "Character count"),
        ("word_count", "Word count"),
        ("sentence_count", "Sentence count"),
        ("type_token_ratio_ttr", "Type-token lexical diversity ratio"),
        ("avg_word_length", "Average word length in characters"),
        ("first_person_singular_count", "1st person singular pronoun count"),
        ("first_person_plural_count", "1st person plural pronoun count"),
        ("second_person_count", "2nd person pronoun count"),
        ("third_person_count", "3rd person pronoun count"),
        ("pronoun_ratio", "Total pronoun ratio to words"),
        # Sentiment (8)
        ("vader_compound", "VADER compound sentiment score"),
        ("vader_positive", "VADER positive proportion"),
        ("vader_neutral", "VADER neutral proportion"),
        ("vader_negative", "VADER negative proportion"),
        ("sentiment_positive_prob", "Contextual positive probability"),
        ("sentiment_neutral_prob", "Contextual neutral probability"),
        ("sentiment_negative_prob", "Contextual negative probability"),
        ("sentiment_polarity_score", "Contextual polarity score"),
        # Affective Emotions (7)
        ("emotion_joy", "Affective joy probability"),
        ("emotion_sadness", "Affective sadness probability"),
        ("emotion_anger", "Affective anger probability"),
        ("emotion_fear", "Affective fear probability"),
        ("emotion_surprise", "Affective surprise probability"),
        ("emotion_disgust", "Affective disgust probability"),
        ("emotion_neutral", "Affective neutral probability"),
        # TF-IDF (20)
        *[ (f"tfidf_projection_{k}", f"TF-IDF projection dimension {k}") for k in range(20) ],
        # Embedding (16)
        *[ (f"semantic_embedding_dim_{k}", f"Semantic embedding axis {k}") for k in range(16) ],
        # LDA Topics (5)
        *[ (f"lda_topic_{k}", f"Thematic LDA topic {k} proportion") for k in range(5) ],
    ]
    for idx, (name, desc) in enumerate(nlp_names, start=69):
        features.append(FeatureMetadata(index=idx, name=name, modality="nlp", is_behavioral=True, description=desc))

    # [135..139] Modality Mask Flags (5 dims)
    mask_names = [
        ("has_communication_metrics", "Communication metrics availability flag"),
        ("has_acoustic_features", "Acoustic features availability flag"),
        ("has_nlp_features", "NLP features availability flag"),
        ("is_voice_message", "Voice message indicator flag"),
        ("is_text_message", "Text message indicator flag"),
    ]
    for idx, (name, desc) in enumerate(mask_names, start=135):
        features.append(FeatureMetadata(index=idx, name=name, modality="mask", is_behavioral=False, description=desc))

    return features


CANONICAL_FEATURE_REGISTRY = build_canonical_feature_registry()
FEATURE_NAME_TO_INDEX = {f.name: f.index for f in CANONICAL_FEATURE_REGISTRY}
INDEX_TO_FEATURE_NAME = {f.index: f.name for f in CANONICAL_FEATURE_REGISTRY}


class TopDeviationItem(BaseModel):
    """Structured record of a significantly deviating dimension."""
    index: int
    name: str
    modality: str
    raw_zscore: float
    clipped_zscore: float
    absolute_zscore: float
    raw_value: float
    baseline_mean: float
    baseline_std: float


class PersonalBaselineResult(BaseModel):
    """Domain model holding user longitudinal baseline vectors and state."""
    user_id: str
    window_type: str = "all_time"
    feature_vector_version: str = "v1.0-140d"
    sample_count: int
    is_baseline_established: bool  # True if sample_count >= 3
    valid_dimensions_count: int
    per_feature_sample_counts: List[int]

    raw_mean_vector: List[float]
    raw_std_vector: List[float]
    normalized_mean_vector: List[float]
    normalized_std_vector: List[float]

    first_observed_at: Optional[datetime] = None
    last_observed_at: Optional[datetime] = None
    computed_at: datetime = Field(default_factory=get_utc_now)


class TemporalTrendEvaluationResult(BaseModel):
    """Domain model holding message deviation and temporal trajectory evaluation."""
    message_id: str
    room_id: str
    user_id: str
    status: str = "completed"  # completed | insufficient_history | failed
    feature_vector_version: str = "v1.0-140d"
    sample_count_at_eval: int
    is_baseline_established: bool

    # Point-in-time snapshot of baseline used (Guarantees historical reproducibility)
    baseline_snapshot_mean: List[float]
    baseline_snapshot_std: List[float]

    # Validity Masks
    zscore_validity_mask: List[bool]
    valid_dimension_count: int
    unavailable_dimension_count: int
    modality_coverage: Dict[str, bool]

    # Z-Score Vectors
    raw_zscore_vector: List[float]
    normalized_zscore_vector: List[float]
    clipped_zscore_vector: List[float]

    # Aggregate Metrics
    mean_absolute_z: float
    rms_z: float
    standardized_euclidean_distance: float

    # Summaries
    top_deviations: List[TopDeviationItem]
    trend_slopes_summary: Dict[str, Any]
    moving_averages_summary: Dict[str, Any]
    rate_of_change_summary: Dict[str, Any]

    evaluation_error: Optional[str] = None
    evaluated_at: datetime = Field(default_factory=get_utc_now)
