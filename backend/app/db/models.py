from datetime import datetime, timezone
import uuid
from sqlalchemy import Column, String, Text, DateTime, Boolean, ForeignKey, Index, Float, Integer, JSON
from sqlalchemy.orm import relationship
from app.db.session import Base


def get_utc_now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(String(64), primary_key=True, index=True)  # User ID / Anonymous PID
    display_name = Column(String(128), nullable=False)
    created_at = Column(DateTime(timezone=True), default=get_utc_now, nullable=False)

    messages = relationship("Message", back_populates="sender", foreign_keys="Message.sender_id")
    participations = relationship("RoomParticipant", back_populates="user", cascade="all, delete-orphan")


class Room(Base):
    __tablename__ = "rooms"

    id = Column(String(64), primary_key=True, index=True)  # Room code, e.g., TEST001
    name = Column(String(128), nullable=True)
    status = Column(String(32), default="active", nullable=False)  # "active" | "completed" | "idle"
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=get_utc_now, nullable=False)
    started_at = Column(DateTime(timezone=True), default=get_utc_now, nullable=True)
    ended_at = Column(DateTime(timezone=True), nullable=True)

    messages = relationship("Message", back_populates="room", cascade="all, delete-orphan", order_by="Message.timestamp")
    participants = relationship("RoomParticipant", back_populates="room", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    room_id = Column(String(64), ForeignKey("rooms.id", ondelete="CASCADE"), nullable=False, index=True)
    sender_id = Column(String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    receiver_id = Column(String(64), nullable=True)  # Nullable for room broadcasts
    message_text = Column(Text, nullable=False)
    message_type = Column(String(32), default="text", nullable=False)  # "text" | "voice" | "system"
    
    # Milestone 4: Voice Data Fields
    audio_url = Column(String(256), nullable=True)
    audio_duration_seconds = Column(Float, nullable=True)
    audio_format = Column(String(32), nullable=True)
    audio_size_bytes = Column(Integer, nullable=True)
    
    # Milestone 5: Speech-to-Text Transcription Fields
    transcript_text = Column(Text, nullable=True)
    transcription_status = Column(String(32), nullable=True)  # pending | processing | completed | failed
    transcription_model = Column(String(64), nullable=True)
    transcription_language = Column(String(16), nullable=True)
    transcription_error = Column(Text, nullable=True)
    transcribed_at = Column(DateTime(timezone=True), nullable=True)
    
    timestamp = Column(DateTime(timezone=True), default=get_utc_now, nullable=False, index=True)

    sender = relationship("User", back_populates="messages", foreign_keys=[sender_id])
    room = relationship("Room", back_populates="messages")
    acoustic_features = relationship("AcousticFeatureRecord", back_populates="message", uselist=False, cascade="all, delete-orphan")
    nlp_features = relationship("NLPFeatureRecord", back_populates="message", uselist=False, cascade="all, delete-orphan")
    multimodal_features = relationship("MultimodalFeatureRecord", back_populates="message", uselist=False, cascade="all, delete-orphan")
    temporal_trend_record = relationship("TemporalTrendRecord", back_populates="message", uselist=False, cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_messages_room_timestamp", "room_id", "timestamp"),
    )


class AcousticFeatureRecord(Base):
    """
    Milestone 6: Relational record of acoustic signal measurements extracted from a voice message.
    Preserves exact 59-dimensional numeric feature vector (v1.0-59d) and typed metric blocks.
    """
    __tablename__ = "acoustic_features"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    message_id = Column(String(64), ForeignKey("messages.id", ondelete="CASCADE"), unique=True, nullable=False, index=True)
    room_id = Column(String(64), ForeignKey("rooms.id", ondelete="CASCADE"), nullable=False, index=True)
    sender_id = Column(String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    status = Column(String(32), default="completed", nullable=False)  # pending | processing | completed | failed
    engine = Column(String(64), default="librosa-v1.0.0", nullable=False)
    duration_seconds = Column(Float, nullable=False)
    sample_rate = Column(Integer, default=16000, nullable=False)
    channels = Column(Integer, default=1, nullable=False)

    # Typed Structured Feature Dictionaries
    pitch_features = Column(JSON, nullable=False)
    energy_features = Column(JSON, nullable=False)
    speech_rate_features = Column(JSON, nullable=False)
    pause_features = Column(JSON, nullable=False)
    mfcc_features = Column(JSON, nullable=False)
    spectral_features = Column(JSON, nullable=False)

    # Standardized 59-Dimensional Feature Vector for M8 Fusion
    feature_vector_version = Column(String(32), default="v1.0-59d", nullable=False)
    raw_feature_vector = Column(JSON, nullable=False)  # List[float] of length exactly 59

    has_voiced_frames = Column(Boolean, default=True, nullable=False)
    extraction_error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=get_utc_now, nullable=False)
    extracted_at = Column(DateTime(timezone=True), default=get_utc_now, nullable=False)

    message = relationship("Message", back_populates="acoustic_features")

    __table_args__ = (
        Index("ix_acoustic_room_msg", "room_id", "message_id"),
    )


class NLPFeatureRecord(Base):
    """
    Milestone 7: Relational record of NLP and linguistic descriptors extracted
    from a text message or a voice message transcript.
    Preserves exact 66-dimensional numeric feature vector (v1.0-66d) and typed metric blocks.
    """
    __tablename__ = "nlp_features"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    message_id = Column(String(64), ForeignKey("messages.id", ondelete="CASCADE"), unique=True, nullable=False, index=True)
    room_id = Column(String(64), ForeignKey("rooms.id", ondelete="CASCADE"), nullable=False, index=True)
    sender_id = Column(String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    source_type = Column(String(32), default="text_message", nullable=False)  # "text_message" | "voice_transcript"
    status = Column(String(32), default="completed", nullable=False)          # "pending" | "processing" | "completed" | "failed"
    engine = Column(String(64), default="rule-lexical-v1.0.0", nullable=False)
    text_content = Column(Text, nullable=False)

    # Typed Structured Feature Blocks
    lexical_features = Column(JSON, nullable=False)
    sentiment_features = Column(JSON, nullable=False)
    emotion_features = Column(JSON, nullable=False)
    tfidf_features = Column(JSON, nullable=False)
    embedding_features = Column(JSON, nullable=False)
    topic_features = Column(JSON, nullable=False)

    # Standardized 66-Dimensional Feature Vector for M8 Fusion
    feature_vector_version = Column(String(32), default="v1.0-66d", nullable=False)
    raw_feature_vector = Column(JSON, nullable=False)  # List[float] of length exactly 66

    extraction_error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=get_utc_now, nullable=False)
    extracted_at = Column(DateTime(timezone=True), default=get_utc_now, nullable=False)

    message = relationship("Message", back_populates="nlp_features")

    __table_args__ = (
        Index("ix_nlp_room_msg", "room_id", "message_id"),
    )


class MultimodalFeatureRecord(Base):
    """
    Milestone 8: Relational record of early multimodal feature fusion.
    Preserves exact 140-dimensional raw and normalized feature vectors (v1.0-140d),
    modality availability masks, and structured summary blocks.
    """
    __tablename__ = "multimodal_features"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    message_id = Column(String(64), ForeignKey("messages.id", ondelete="CASCADE"), unique=True, nullable=False, index=True)
    room_id = Column(String(64), ForeignKey("rooms.id", ondelete="CASCADE"), nullable=False, index=True)
    sender_id = Column(String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    status = Column(String(32), default="completed", nullable=False)  # completed | failed
    fusion_strategy = Column(String(64), default="early_concatenation_v1", nullable=False)
    feature_vector_version = Column(String(32), default="v1.0-140d", nullable=False)

    has_communication = Column(Boolean, default=True, nullable=False)
    has_acoustic = Column(Boolean, default=False, nullable=False)
    has_nlp = Column(Boolean, default=False, nullable=False)

    communication_features = Column(JSON, nullable=False)
    acoustic_features_summary = Column(JSON, nullable=True)
    nlp_features_summary = Column(JSON, nullable=True)
    modality_mask = Column(JSON, nullable=False)

    raw_fused_vector = Column(JSON, nullable=False)         # List[float] of length exactly 140
    normalized_fused_vector = Column(JSON, nullable=False)  # List[float] of length exactly 140, bounded in [0.0, 1.0]

    fusion_error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=get_utc_now, nullable=False)
    fused_at = Column(DateTime(timezone=True), default=get_utc_now, nullable=False)

    message = relationship("Message", back_populates="multimodal_features")

    __table_args__ = (
        Index("ix_multimodal_room_msg", "room_id", "message_id"),
    )


class RoomParticipant(Base):
    __tablename__ = "room_participants"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    room_id = Column(String(64), ForeignKey("rooms.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    joined_at = Column(DateTime(timezone=True), default=get_utc_now, nullable=False)
    last_seen_at = Column(DateTime(timezone=True), default=get_utc_now, nullable=False)

    room = relationship("Room", back_populates="participants")
    user = relationship("User", back_populates="participations")

    __table_args__ = (
        Index("ix_participant_room_user", "room_id", "user_id", unique=True),
    )


class PersonalBaselineRecord(Base):
    """
    Milestone 9: Relational record of a participant's longitudinal personal baseline.
    Preserves running empirical sample mean and Bessel-corrected sample standard deviation
    vectors across both raw and canonical normalized v1.0-140d feature spaces.
    """
    __tablename__ = "personal_baselines"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    user_id = Column(String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    window_type = Column(String(32), default="all_time", nullable=False)  # "all_time" | "rolling_10" | "rolling_30"
    feature_vector_version = Column(String(32), default="v1.0-140d", nullable=False)

    sample_count = Column(Integer, default=0, nullable=False)
    is_baseline_established = Column(Boolean, default=False, nullable=False)  # sample_count >= 3
    valid_dimensions_count = Column(Integer, default=140, nullable=False)

    # Statistical Vectors (Length: 140)
    raw_mean_vector = Column(JSON, nullable=False)
    raw_std_vector = Column(JSON, nullable=False)
    normalized_mean_vector = Column(JSON, nullable=False)
    normalized_std_vector = Column(JSON, nullable=False)

    first_observed_at = Column(DateTime(timezone=True), nullable=True)
    last_observed_at = Column(DateTime(timezone=True), nullable=True)
    computed_at = Column(DateTime(timezone=True), default=get_utc_now, nullable=False)

    user = relationship("User", backref="personal_baselines")

    __table_args__ = (
        Index("ix_user_baseline_window", "user_id", "window_type", "feature_vector_version", unique=True),
    )


class TemporalTrendRecord(Base):
    """
    Milestone 9: Relational record of message-level within-subject deviation and temporal trajectory dynamics.
    Stores point-in-time baseline snapshot for historical reproducibility, modality masking diagnostics,
    raw and clipped z-scores, and longitudinal moving averages/trend slopes.
    """
    __tablename__ = "temporal_trend_records"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    message_id = Column(String(64), ForeignKey("messages.id", ondelete="CASCADE"), unique=True, nullable=False, index=True)
    room_id = Column(String(64), ForeignKey("rooms.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    status = Column(String(32), default="completed", nullable=False)  # completed | insufficient_history | failed
    feature_vector_version = Column(String(32), default="v1.0-140d", nullable=False)
    sample_count_at_eval = Column(Integer, default=0, nullable=False)
    is_baseline_established = Column(Boolean, default=False, nullable=False)

    # Point-in-Time Baseline Snapshot (Guarantees Historical Reproducibility)
    baseline_snapshot_mean = Column(JSON, nullable=False)  # Exact 140-d mean used
    baseline_snapshot_std = Column(JSON, nullable=False)   # Exact 140-d std used

    # Modality and Masking Diagnostics
    zscore_validity_mask = Column(JSON, nullable=False)    # 140 booleans: True if dimension is validly scored
    valid_dimension_count = Column(Integer, nullable=False)
    unavailable_dimension_count = Column(Integer, nullable=False)
    modality_coverage = Column(JSON, nullable=False)

    # 140-d Z-Score Vectors
    raw_zscore_vector = Column(JSON, nullable=False)
    normalized_zscore_vector = Column(JSON, nullable=False)
    clipped_zscore_vector = Column(JSON, nullable=False)  # Bounded [-5.0, +5.0]

    # Aggregate Statistics (Over Valid Dimensions Only)
    mean_absolute_z = Column(Float, nullable=False)
    rms_z = Column(Float, nullable=False)
    standardized_euclidean_distance = Column(Float, nullable=False)

    # Top Deviations & Trend Summaries
    top_deviations_summary = Column(JSON, nullable=False)  # Top deviations sorted by |z| desc, index asc
    trend_slopes_summary = Column(JSON, nullable=False)
    moving_averages_summary = Column(JSON, nullable=False)
    rate_of_change_summary = Column(JSON, nullable=False)

    evaluation_error = Column(Text, nullable=True)
    evaluated_at = Column(DateTime(timezone=True), default=get_utc_now, nullable=False)

    message = relationship("Message", back_populates="temporal_trend_record")

    __table_args__ = (
        Index("ix_temporal_room_user", "room_id", "user_id"),
        Index("ix_temporal_eval_time", "user_id", "evaluated_at"),
    )


class AnalyticalReportRecord(Base):
    """
    Milestone 10: Relational record of longitudinal analytical research reports.
    Preserves complete point-in-time report payload, modality completeness metrics,
    within-subject deviations, feature shifts, and trend dynamics for immutable reproducibility.
    """
    __tablename__ = "analytical_report_records"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    user_id = Column(String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    room_id = Column(String(64), ForeignKey("rooms.id", ondelete="CASCADE"), nullable=True, index=True)

    report_type = Column(String(32), default="longitudinal", nullable=False)
    feature_vector_version = Column(String(32), default="v1.0-140d", nullable=False)
    status = Column(String(32), default="completed", nullable=False)  # completed | failed

    period_start = Column(DateTime(timezone=True), nullable=True)
    period_end = Column(DateTime(timezone=True), nullable=True)

    observation_count = Column(Integer, default=0, nullable=False)
    baseline_sample_count = Column(Integer, default=0, nullable=False)
    baseline_established = Column(Boolean, default=False, nullable=False)

    modality_coverage = Column(JSON, nullable=False)
    deviation_summary = Column(JSON, nullable=False)
    trend_summary = Column(JSON, nullable=False)
    feature_shift_summary = Column(JSON, nullable=False)
    trajectory_summary = Column(JSON, nullable=False)
    report_payload = Column(JSON, nullable=False)

    generated_at = Column(DateTime(timezone=True), default=get_utc_now, nullable=False)

    user = relationship("User", backref="analytical_reports")
    room = relationship("Room", backref="analytical_reports")

    __table_args__ = (
        Index("ix_report_user_gen", "user_id", "generated_at"),
        Index("ix_report_room_gen", "room_id", "generated_at"),
    )


