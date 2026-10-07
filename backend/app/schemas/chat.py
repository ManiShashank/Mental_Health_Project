from datetime import datetime, timezone
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field, field_validator, ConfigDict


def get_utc_now() -> datetime:
    return datetime.now(timezone.utc)


class UserBase(BaseModel):
    id: str = Field(..., description="Unique User/Participant ID", min_length=1, max_length=64)
    display_name: str = Field(..., description="User display name", min_length=1, max_length=128)


class UserCreate(UserBase):
    pass


class UserResponse(UserBase):
    model_config = ConfigDict(from_attributes=True)
    created_at: datetime


class RoomBase(BaseModel):
    id: str = Field(..., description="Room identifier / code", min_length=1, max_length=64)
    name: Optional[str] = Field(None, description="Human readable room name", max_length=128)


class RoomCreate(RoomBase):
    pass


class RoomResponse(RoomBase):
    model_config = ConfigDict(from_attributes=True)
    is_active: bool
    created_at: datetime


class MessageCreate(BaseModel):
    """
    Schema for incoming message payload.
    """
    message_text: str = Field(..., description="Message text content", min_length=1, max_length=4000)
    receiver_id: Optional[str] = Field(None, description="Optional target user ID")
    message_type: str = Field(default="text", description="Message type: text | voice | system")
    audio_url: Optional[str] = Field(None, description="URL or reference to stored audio")
    audio_duration_seconds: Optional[float] = Field(None, description="Voice recording duration in seconds")
    audio_format: Optional[str] = Field(None, description="Audio MIME format, e.g. audio/webm")
    audio_size_bytes: Optional[int] = Field(None, description="Audio file size in bytes")

    @field_validator("message_text")
    @classmethod
    def validate_message_text_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Message text cannot be empty or whitespace only.")
        return stripped


class MessageResponse(BaseModel):
    """
    Standardized message contract conforming to project specification.
    """
    model_config = ConfigDict(from_attributes=True)
    message_id: str = Field(..., description="Unique message UUID")
    conversation_id: str = Field(..., description="Conversation or room ID")
    room_id: str = Field(..., description="Room identifier")
    sender_id: str = Field(..., description="Sender user ID")
    sender_name: Optional[str] = Field(None, description="Sender display name")
    receiver_id: Optional[str] = Field(None, description="Target recipient ID (null for broadcast)")
    message_text: str = Field(..., description="Text content or voice label")
    timestamp: datetime = Field(default_factory=get_utc_now, description="UTC timestamp")
    message_type: str = Field(default="text", description="Type: text | voice | system")
    
    # Milestone 4: Voice Attributes
    audio_url: Optional[str] = Field(None, description="Static or streaming URL for recorded audio")
    audio_duration_seconds: Optional[float] = Field(None, description="Duration in seconds")
    audio_format: Optional[str] = Field(None, description="Audio container / codec MIME type")
    audio_size_bytes: Optional[int] = Field(None, description="Audio file byte size")

    # Milestone 5: Speech-to-Text Attributes
    transcript_text: Optional[str] = Field(None, description="Speech-to-text transcript")
    transcription_status: Optional[str] = Field(None, description="Status: pending | processing | completed | failed")
    transcription_model: Optional[str] = Field(None, description="STT model used for transcription")
    transcription_language: Optional[str] = Field(None, description="Language code of transcription")
    transcription_error: Optional[str] = Field(None, description="Error message if transcription failed")
    transcribed_at: Optional[datetime] = Field(None, description="UTC timestamp when transcription was generated")

    # Milestone 6: Acoustic Analysis Flag
    has_acoustic_features: bool = Field(default=False, description="Whether acoustic features have been extracted for this voice message")

    # Milestone 7: NLP Analysis Flag
    has_nlp_features: bool = Field(default=False, description="Whether NLP & linguistic features have been extracted for this message")

    # Milestone 8: Multimodal Feature Fusion Flag
    has_multimodal_features: bool = Field(default=False, description="Whether multimodal feature fusion has been executed for this message")

    # Milestone 9: Temporal Trends & Personal Deviation Flag
    has_temporal_trends: bool = Field(default=False, description="Whether personal deviation and temporal trends have been evaluated for this message")


class VoiceUploadResponse(BaseModel):
    message: MessageResponse
    detail: str = Field(default="Voice message uploaded and broadcast successfully.")


class TranscriptionResponse(BaseModel):
    message_id: str = Field(..., description="Message UUID")
    room_id: str = Field(..., description="Room identifier")
    transcript_text: Optional[str] = Field(None, description="Transcribed text")
    transcription_status: str = Field(..., description="Status: pending | processing | completed | failed")
    transcription_model: Optional[str] = Field(None, description="STT engine used")
    transcription_language: Optional[str] = Field(None, description="Language detected or used")
    transcription_error: Optional[str] = Field(None, description="Error details if failed")
    transcribed_at: Optional[datetime] = Field(None, description="UTC timestamp of completion")
    detail: str = Field(default="Transcription processed successfully.")


class AcousticFeatureResponse(BaseModel):
    """
    Standardized response schema for extracted acoustic descriptors (Milestone 6).
    """
    model_config = ConfigDict(from_attributes=True)

    message_id: str = Field(..., description="Message UUID")
    room_id: str = Field(..., description="Room identifier")
    sender_id: str = Field(..., description="Sender User ID")
    status: str = Field(..., description="Analysis status: pending | processing | completed | failed")
    engine: str = Field(..., description="Acoustic extraction engine name and version")
    duration_seconds: float = Field(..., description="Audio duration in seconds")
    sample_rate: int = Field(..., description="Sampling rate in Hz")
    channels: int = Field(default=1, description="Audio channel count")
    
    pitch_features: Dict[str, Any] = Field(..., description="Pitch (F0) statistics block")
    energy_features: Dict[str, Any] = Field(..., description="RMS energy & shimmer statistics block")
    speech_rate_features: Dict[str, Any] = Field(..., description="Estimated speech rate & syllable rhythm block")
    pause_features: Dict[str, Any] = Field(..., description="Pause detection & silence duration block")
    mfcc_features: Dict[str, Any] = Field(..., description="13-coefficient MFCC representation block")
    spectral_features: Dict[str, Any] = Field(..., description="Spectral shape descriptors block")

    feature_vector_version: str = Field(default="v1.0-59d", description="Standardized feature vector schema version")
    raw_feature_vector: List[float] = Field(..., description="Exact 59-dimensional ordered numeric feature vector")
    has_voiced_frames: bool = Field(default=True, description="Voiced speech frame detection flag")
    extraction_error: Optional[str] = Field(None, description="Error detail if extraction failed")
    extracted_at: datetime = Field(..., description="UTC timestamp of extraction")
    detail: str = Field(default="Acoustic features extracted successfully.")


class RoomAcousticFeaturesResponse(BaseModel):
    room_id: str = Field(..., description="Room identifier")
    total_analyzed: int = Field(..., description="Count of analyzed voice messages")
    features: List[AcousticFeatureResponse] = Field(default_factory=list, description="Acoustic feature records for the room")


class NLPFeatureResponse(BaseModel):
    """
    Standardized response schema for extracted NLP & linguistic descriptors (Milestone 7).
    """
    model_config = ConfigDict(from_attributes=True)

    message_id: str = Field(..., description="Message UUID")
    room_id: str = Field(..., description="Room identifier")
    sender_id: str = Field(..., description="Sender User ID")
    source_type: str = Field(..., description="'text_message' | 'voice_transcript'")
    status: str = Field(..., description="Analysis status: pending | processing | completed | failed")
    engine: str = Field(..., description="NLP extraction engine identifier")
    text_content: str = Field(..., description="Raw text or speech transcript analyzed")

    lexical_features: Dict[str, Any] = Field(..., description="Lexical, syntactic, and pronoun distribution block (10 metrics)")
    sentiment_features: Dict[str, Any] = Field(..., description="VADER and contextual sentiment polarity block (8 metrics)")
    emotion_features: Dict[str, Any] = Field(..., description="7-class affective emotion probability distribution block")
    tfidf_features: Dict[str, Any] = Field(..., description="20 canonical TF-IDF projections block")
    embedding_features: Dict[str, Any] = Field(..., description="16 semantic embedding summary dimensions block")
    topic_features: Dict[str, Any] = Field(..., description="5 thematic LDA topic proportions block")

    feature_vector_version: str = Field(default="v1.0-66d", description="Standardized feature vector schema version")
    raw_feature_vector: List[float] = Field(..., description="Exact 66-dimensional ordered numeric feature vector")
    extraction_error: Optional[str] = Field(None, description="Error detail if extraction failed")
    extracted_at: datetime = Field(..., description="UTC timestamp of extraction")
    detail: str = Field(default="NLP and linguistic features extracted successfully.")


class RoomNLPFeaturesResponse(BaseModel):
    room_id: str = Field(..., description="Room identifier")
    total_analyzed: int = Field(..., description="Count of analyzed messages")
    features: List[NLPFeatureResponse] = Field(default_factory=list, description="NLP feature records for the room")


class MultimodalFeatureResponse(BaseModel):
    """
    Standardized response schema for early multimodal feature fusion (Milestone 8).
    """
    model_config = ConfigDict(from_attributes=True)

    message_id: str = Field(..., description="Message UUID")
    room_id: str = Field(..., description="Room identifier")
    sender_id: str = Field(..., description="Sender User ID")
    status: str = Field(..., description="Fusion status: completed | failed")
    fusion_strategy: str = Field(default="early_concatenation_v1", description="Fusion algorithm strategy identifier")
    feature_vector_version: str = Field(default="v1.0-140d", description="Standardized feature vector schema version")

    has_communication: bool = Field(default=True, description="Whether communication dynamics features are present")
    has_acoustic: bool = Field(default=False, description="Whether acoustic features are present")
    has_nlp: bool = Field(default=False, description="Whether NLP features are present")

    communication_features: Dict[str, Any] = Field(..., description="10-dimensional M3 communication dynamics block")
    acoustic_features_summary: Optional[Dict[str, Any]] = Field(None, description="Summary of M6 acoustic descriptors")
    nlp_features_summary: Optional[Dict[str, Any]] = Field(None, description="Summary of M7 NLP descriptors")
    modality_mask: Dict[str, Any] = Field(..., description="5-dimensional modality availability mask flags")

    raw_fused_vector: List[float] = Field(..., description="Exact 140-dimensional raw feature vector")
    normalized_fused_vector: List[float] = Field(..., description="Exact 140-dimensional normalized feature vector strictly in [0.0, 1.0]")
    fusion_error: Optional[str] = Field(None, description="Error detail if fusion failed")
    fused_at: datetime = Field(..., description="UTC timestamp of fusion")
    detail: str = Field(default="Multimodal features fused successfully.")


class RoomMultimodalFeaturesResponse(BaseModel):
    room_id: str = Field(..., description="Room identifier")
    total_analyzed: int = Field(..., description="Count of fused messages")
    features: List[MultimodalFeatureResponse] = Field(default_factory=list, description="Multimodal feature records for the room")


# ---------------------------------------------------------
# Milestone 9: Personal Baseline & Temporal Trend Schemas
# ---------------------------------------------------------

class PersonalBaselineResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    user_id: str = Field(..., description="User identifier")
    window_type: str = Field(default="all_time", description="Historical window type")
    feature_vector_version: str = Field(default="v1.0-140d", description="Schema version of feature vector")
    sample_count: int = Field(..., description="Number of historical observations in baseline")
    is_baseline_established: bool = Field(..., description="True if sample_count >= 3")
    valid_dimensions_count: int = Field(..., description="Count of dimensions with >= 2 observations")
    
    raw_mean_vector: List[float] = Field(..., description="140-dimensional raw mean vector")
    raw_std_vector: List[float] = Field(..., description="140-dimensional raw sample standard deviation vector")
    normalized_mean_vector: List[float] = Field(..., description="140-dimensional normalized mean vector")
    normalized_std_vector: List[float] = Field(..., description="140-dimensional normalized standard deviation vector")

    first_observed_at: Optional[datetime] = Field(None, description="UTC timestamp of first observation")
    last_observed_at: Optional[datetime] = Field(None, description="UTC timestamp of last observation")
    computed_at: datetime = Field(..., description="UTC timestamp of baseline computation")
    detail: str = Field(default="Personal baseline retrieved successfully.")


class TemporalTrendResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    message_id: str = Field(..., description="Message UUID")
    room_id: str = Field(..., description="Room identifier")
    user_id: str = Field(..., description="User identifier")
    status: str = Field(..., description="Status: completed | insufficient_history | failed")
    feature_vector_version: str = Field(default="v1.0-140d", description="Feature vector schema version")
    sample_count_at_eval: int = Field(..., description="Count of strictly preceding observations used")
    is_baseline_established: bool = Field(..., description="True if sample_count_at_eval >= 3")

    baseline_snapshot_mean: List[float] = Field(..., description="140-d normalized baseline mean snapshot used at evaluation")
    baseline_snapshot_std: List[float] = Field(..., description="140-d normalized baseline std snapshot used at evaluation")

    zscore_validity_mask: List[bool] = Field(..., description="140 boolean flags: True if feature is evaluable")
    valid_dimension_count: int = Field(..., description="Count of evaluable behavioral dimensions")
    unavailable_dimension_count: int = Field(..., description="Count of unavailable dimensions")
    modality_coverage: Dict[str, bool] = Field(..., description="Modality presence indicators")

    raw_zscore_vector: List[float] = Field(..., description="Exact 140-dimensional raw z-score vector")
    normalized_zscore_vector: List[float] = Field(..., description="Exact 140-dimensional normalized z-score vector")
    clipped_zscore_vector: List[float] = Field(..., description="140-dimensional z-score vector clipped to [-5.0, +5.0]")

    mean_absolute_z: float = Field(..., description="Mean absolute z-score over valid dimensions")
    rms_z: float = Field(..., description="Root mean square z-score over valid dimensions")
    standardized_euclidean_distance: float = Field(..., description="Standardized Euclidean distance over valid dimensions")

    top_deviations: List[Dict[str, Any]] = Field(default_factory=list, description="Top deviating dimensions sorted by |z| desc")
    trend_slopes_summary: Dict[str, Any] = Field(default_factory=dict, description="OLS linear trend slopes and directions")
    moving_averages_summary: Dict[str, Any] = Field(default_factory=dict, description="SMA-3, SMA-5, EMA-3, EMA-5 summaries")
    rate_of_change_summary: Dict[str, Any] = Field(default_factory=dict, description="Rate of change metrics")

    evaluation_error: Optional[str] = Field(None, description="Error detail if evaluation failed")
    evaluated_at: datetime = Field(..., description="UTC timestamp of evaluation")
    detail: str = Field(default="Temporal trend evaluation completed successfully.")


class RoomTemporalTrendsResponse(BaseModel):
    room_id: str = Field(..., description="Room identifier")
    total_analyzed: int = Field(..., description="Count of evaluated records")
    records: List[TemporalTrendResponse] = Field(default_factory=list, description="Temporal trend records for room")


class UserTrajectoryResponse(BaseModel):
    user_id: str = Field(..., description="User identifier")
    room_id: Optional[str] = Field(None, description="Room filter if applied")
    total_points: int = Field(..., description="Count of trajectory points")
    trajectory: List[TemporalTrendResponse] = Field(default_factory=list, description="Longitudinal trajectory series")


class WebSocketEvent(BaseModel):
    """
    Structured envelope for real-time WebSocket events.
    """
    event: str = Field(..., description="Event name: join | message | leave | user_joined | user_left | history | error | ping | pong | transcription_completed | acoustic_analysis_completed | nlp_analysis_completed | multimodal_fusion_completed | personal_baseline_updated | temporal_deviation_computed | analytical_report_generated")
    data: Any = Field(default_factory=dict, description="Event payload dictionary")
    timestamp: datetime = Field(default_factory=get_utc_now, description="UTC event timestamp")


class ConversationHistoryResponse(BaseModel):
    room_id: str
    messages: List[MessageResponse]
    total_messages: int


# --------------------------------------------------
# Milestone 10: Analytical Report Engine Schemas
# --------------------------------------------------

class AnalyticalReportRequest(BaseModel):
    room_id: Optional[str] = Field(None, description="Optional room ID filter")
    start_time: Optional[datetime] = Field(None, description="Optional start datetime filter")
    end_time: Optional[datetime] = Field(None, description="Optional end datetime filter")
    report_type: str = Field(default="longitudinal", description="Report type: longitudinal")


class ReportMetadataSchema(BaseModel):
    report_id: str
    report_type: str = "longitudinal"
    feature_vector_version: str = "v1.0-140d"
    user_id: str
    room_id: Optional[str] = None
    observation_count: int
    baseline_sample_count: int
    baseline_established: bool
    generated_at: datetime


class BaselineSummarySchema(BaseModel):
    status: str
    sample_count: int
    window_type: str = "all_time"
    feature_vector_version: str = "v1.0-140d"
    is_baseline_established: bool
    valid_dimensions_count: int


class ModalityCoverageItemSchema(BaseModel):
    modality: str
    measured_count: int
    total_count: int
    percentage: float


class ModalityCoverageSummarySchema(BaseModel):
    communication: ModalityCoverageItemSchema
    acoustic: ModalityCoverageItemSchema
    nlp: ModalityCoverageItemSchema


class DescriptiveMetricSummarySchema(BaseModel):
    mean: float
    median: float
    minimum: float
    maximum: float
    std: float


class DeviationSummarySchema(BaseModel):
    mean_absolute_z: DescriptiveMetricSummarySchema
    rms_z: DescriptiveMetricSummarySchema
    standardized_euclidean_distance: DescriptiveMetricSummarySchema
    statistical_variation_category: str
    category_description: str


class FeatureShiftItemSchema(BaseModel):
    feature_index: int
    feature_name: str
    modality: str
    valid_observations: int
    mean_absolute_z: float
    max_absolute_z: float
    latest_signed_z: float
    latest_absolute_z: float
    baseline_mean: float
    baseline_std: float


class TrendFeatureItemSchema(BaseModel):
    feature_index: int
    feature_name: str
    modality: str
    normalized_slope: float
    raw_slope: float
    direction: str
    latest_sma3: Optional[float] = None
    latest_sma5: Optional[float] = None
    latest_ema3: Optional[float] = None
    latest_ema5: Optional[float] = None
    latest_roc: Optional[float] = None
    valid_observation_count: int


class TrendSummarySchema(BaseModel):
    increasing_count: int
    decreasing_count: int
    stable_count: int
    top_trends: List[TrendFeatureItemSchema]


class LongitudinalTrajectoryPointSchema(BaseModel):
    message_id: str
    timestamp: datetime
    mean_absolute_z: float
    rms_z: float
    standardized_euclidean_distance: float
    valid_dimension_count: int
    top_deviation_feature: Optional[str] = None


class TrajectorySummarySchema(BaseModel):
    total_points: int
    points: List[LongitudinalTrajectoryPointSchema]
    timestamps: List[str]
    mean_absolute_z_series: List[float]
    rms_z_series: List[float]
    standardized_euclidean_distance_series: List[float]


class ReportPeriodSchema(BaseModel):
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None


class AnalyticalReportResponse(BaseModel):
    metadata: ReportMetadataSchema
    period: ReportPeriodSchema
    baseline: BaselineSummarySchema
    modality_coverage: ModalityCoverageSummarySchema
    deviation_summary: DeviationSummarySchema
    feature_shifts: List[FeatureShiftItemSchema]
    trend_summary: TrendSummarySchema
    trajectory_summary: TrajectorySummarySchema
    limitations: List[str]
    disclaimer: str
    detail: str = "Analytical report generated successfully."


class AnalyticalReportHistoryResponse(BaseModel):
    user_id: str
    total_reports: int
    reports: List[AnalyticalReportResponse]


class RoomAnalyticalReportsResponse(BaseModel):
    room_id: str
    total_reports: int
    reports: List[AnalyticalReportResponse]



