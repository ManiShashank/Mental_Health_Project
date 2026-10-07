from typing import List, Optional
from datetime import datetime, timezone
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from sqlalchemy.orm import selectinload
from app.db.models import (
    User,
    Room,
    Message,
    RoomParticipant,
    AcousticFeatureRecord,
    NLPFeatureRecord,
    MultimodalFeatureRecord,
    PersonalBaselineRecord,
    TemporalTrendRecord,
    AnalyticalReportRecord
)
from app.services.acoustic_features.models import AcousticFeaturesResult
from app.services.nlp_features.models import NLPFeaturesResult


class ChatRepository:
    """
    Repository layer isolating SQL operations from business and WebSocket logic.
    """

    @staticmethod
    async def get_or_create_user(
        session: AsyncSession,
        user_id: str,
        display_name: str
    ) -> User:
        result = await session.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            user = User(id=user_id, display_name=display_name)
            session.add(user)
            await session.commit()
            await session.refresh(user)
        elif user.display_name != display_name:
            user.display_name = display_name
            await session.commit()
            await session.refresh(user)
        return user

    @staticmethod
    async def get_user_by_id(session: AsyncSession, user_id: str) -> Optional[User]:
        result = await session.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def get_or_create_room(
        session: AsyncSession,
        room_id: str,
        name: Optional[str] = None
    ) -> Room:
        result = await session.execute(select(Room).where(Room.id == room_id))
        room = result.scalar_one_or_none()
        if not room:
            room = Room(id=room_id, name=name or f"Room {room_id}", is_active=True)
            session.add(room)
            await session.commit()
            await session.refresh(room)
        return room

    @staticmethod
    async def get_active_rooms(session: AsyncSession) -> List[Room]:
        result = await session.execute(
            select(Room).where(Room.is_active == True).order_by(desc(Room.created_at))
        )
        return list(result.scalars().all())

    @staticmethod
    async def add_or_update_participant(
        session: AsyncSession,
        room_id: str,
        user_id: str
    ) -> RoomParticipant:
        result = await session.execute(
            select(RoomParticipant).where(
                RoomParticipant.room_id == room_id,
                RoomParticipant.user_id == user_id
            )
        )
        participant = result.scalar_one_or_none()
        now = datetime.now(timezone.utc)
        if not participant:
            participant = RoomParticipant(
                id=str(uuid.uuid4()),
                room_id=room_id,
                user_id=user_id,
                joined_at=now,
                last_seen_at=now
            )
            session.add(participant)
        else:
            participant.last_seen_at = now
        await session.commit()
        await session.refresh(participant)
        return participant

    @staticmethod
    async def save_message(
        session: AsyncSession,
        room_id: str,
        sender_id: str,
        message_text: str,
        receiver_id: Optional[str] = None,
        message_type: str = "text",
        timestamp: Optional[datetime] = None,
        message_id: Optional[str] = None,
        audio_url: Optional[str] = None,
        audio_duration_seconds: Optional[float] = None,
        audio_format: Optional[str] = None,
        audio_size_bytes: Optional[int] = None
    ) -> Message:
        msg = Message(
            id=message_id or str(uuid.uuid4()),
            room_id=room_id,
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_text=message_text,
            message_type=message_type,
            audio_url=audio_url,
            audio_duration_seconds=audio_duration_seconds,
            audio_format=audio_format,
            audio_size_bytes=audio_size_bytes,
            timestamp=timestamp or datetime.now(timezone.utc)
        )
        session.add(msg)
        await session.commit()
        await session.refresh(msg)
        return msg

    @staticmethod
    async def get_room_by_id(session: AsyncSession, room_id: str) -> Optional[Room]:
        result = await session.execute(
            select(Room)
            .options(selectinload(Room.participants).selectinload(RoomParticipant.user))
            .where(Room.id == room_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_room_messages(
        session: AsyncSession,
        room_id: str,
        limit: int = 500,
        offset: int = 0
    ) -> List[Message]:
        result = await session.execute(
            select(Message)
            .options(
                selectinload(Message.sender),
                selectinload(Message.acoustic_features),
                selectinload(Message.nlp_features),
                selectinload(Message.multimodal_features),
                selectinload(Message.temporal_trend_record)
            )
            .where(Message.room_id == room_id)
            .order_by(Message.timestamp.asc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    @staticmethod
    async def end_room_conversation(session: AsyncSession, room_id: str) -> Optional[Room]:
        result = await session.execute(select(Room).where(Room.id == room_id))
        room = result.scalar_one_or_none()
        if room:
            room.status = "completed"
            room.ended_at = datetime.now(timezone.utc)
            await session.commit()
            await session.refresh(room)
        return room

    @staticmethod
    async def get_message_by_id(session: AsyncSession, message_id: str) -> Optional[Message]:
        result = await session.execute(
            select(Message)
            .options(selectinload(Message.sender))
            .where(Message.id == message_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def update_message_transcript(
        session: AsyncSession,
        message_id: str,
        transcript_text: Optional[str] = None,
        transcription_status: str = "completed",
        transcription_model: Optional[str] = None,
        transcription_language: Optional[str] = None,
        transcription_error: Optional[str] = None,
        transcribed_at: Optional[datetime] = None
    ) -> Optional[Message]:
        result = await session.execute(
            select(Message)
            .options(selectinload(Message.sender))
            .where(Message.id == message_id)
        )
        msg = result.scalar_one_or_none()
        if msg:
            msg.transcript_text = transcript_text
            msg.transcription_status = transcription_status
            msg.transcription_model = transcription_model
            msg.transcription_language = transcription_language
            msg.transcription_error = transcription_error
            msg.transcribed_at = transcribed_at or datetime.now(timezone.utc)
            await session.commit()
            await session.refresh(msg)
        return msg

    @staticmethod
    async def save_or_update_acoustic_features(
        session: AsyncSession,
        message_id: str,
        room_id: str,
        sender_id: str,
        features: Optional[AcousticFeaturesResult] = None,
        status: str = "completed",
        extraction_error: Optional[str] = None
    ) -> AcousticFeatureRecord:
        result = await session.execute(
            select(AcousticFeatureRecord).where(AcousticFeatureRecord.message_id == message_id)
        )
        record = result.scalar_one_or_none()
        utc_now = datetime.now(timezone.utc)

        if not record:
            record = AcousticFeatureRecord(
                id=str(uuid.uuid4()),
                message_id=message_id,
                room_id=room_id,
                sender_id=sender_id,
                status=status,
                engine=features.engine if features else "librosa-v1.0.0",
                duration_seconds=features.duration_seconds if features else 0.0,
                sample_rate=features.sample_rate if features else 16000,
                channels=features.channels if features else 1,
                pitch_features=features.pitch.model_dump() if features else {},
                energy_features=features.energy.model_dump() if features else {},
                speech_rate_features=features.speech_rate.model_dump() if features else {},
                pause_features=features.pauses.model_dump() if features else {},
                mfcc_features=features.mfcc.model_dump() if features else {},
                spectral_features=features.spectral.model_dump() if features else {},
                feature_vector_version=features.feature_vector_version if features else "v1.0-59d",
                raw_feature_vector=features.raw_feature_vector if features else [0.0] * 59,
                has_voiced_frames=features.has_voiced_frames if features else False,
                extraction_error=extraction_error,
                created_at=utc_now,
                extracted_at=utc_now
            )
            session.add(record)
        else:
            record.status = status
            if features:
                record.engine = features.engine
                record.duration_seconds = features.duration_seconds
                record.sample_rate = features.sample_rate
                record.channels = features.channels
                record.pitch_features = features.pitch.model_dump()
                record.energy_features = features.energy.model_dump()
                record.speech_rate_features = features.speech_rate.model_dump()
                record.pause_features = features.pauses.model_dump()
                record.mfcc_features = features.mfcc.model_dump()
                record.spectral_features = features.spectral.model_dump()
                record.feature_vector_version = features.feature_vector_version
                record.raw_feature_vector = features.raw_feature_vector
                record.has_voiced_frames = features.has_voiced_frames
            record.extraction_error = extraction_error
            record.extracted_at = utc_now

        await session.commit()
        await session.refresh(record)
        return record

    @staticmethod
    async def get_acoustic_features_by_message_id(
        session: AsyncSession,
        message_id: str
    ) -> Optional[AcousticFeatureRecord]:
        result = await session.execute(
            select(AcousticFeatureRecord).where(AcousticFeatureRecord.message_id == message_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_room_acoustic_features(
        session: AsyncSession,
        room_id: str
    ) -> List[AcousticFeatureRecord]:
        result = await session.execute(
            select(AcousticFeatureRecord)
            .where(AcousticFeatureRecord.room_id == room_id)
            .order_by(AcousticFeatureRecord.created_at.asc())
        )
        return list(result.scalars().all())

    @staticmethod
    async def save_or_update_nlp_features(
        session: AsyncSession,
        message_id: str,
        room_id: str,
        sender_id: str,
        source_type: str,
        text_content: str,
        features: Optional[NLPFeaturesResult] = None,
        status: str = "completed",
        extraction_error: Optional[str] = None
    ) -> NLPFeatureRecord:
        result = await session.execute(
            select(NLPFeatureRecord).where(NLPFeatureRecord.message_id == message_id)
        )
        record = result.scalar_one_or_none()
        utc_now = datetime.now(timezone.utc)

        if not record:
            record = NLPFeatureRecord(
                id=str(uuid.uuid4()),
                message_id=message_id,
                room_id=room_id,
                sender_id=sender_id,
                source_type=source_type,
                status=status,
                engine=features.engine if features else "rule-lexical-v1.0.0",
                text_content=text_content,
                lexical_features=features.lexical.model_dump() if features else {},
                sentiment_features=features.sentiment.model_dump() if features else {},
                emotion_features=features.emotion.model_dump() if features else {},
                tfidf_features=features.tfidf.model_dump() if features else {},
                embedding_features=features.embedding.model_dump() if features else {},
                topic_features=features.topic.model_dump() if features else {},
                feature_vector_version=features.feature_vector_version if features else "v1.0-66d",
                raw_feature_vector=features.raw_feature_vector if features else [0.0] * 66,
                extraction_error=extraction_error,
                created_at=utc_now,
                extracted_at=utc_now
            )
            session.add(record)
        else:
            record.status = status
            record.source_type = source_type
            record.text_content = text_content
            if features:
                record.engine = features.engine
                record.lexical_features = features.lexical.model_dump()
                record.sentiment_features = features.sentiment.model_dump()
                record.emotion_features = features.emotion.model_dump()
                record.tfidf_features = features.tfidf.model_dump()
                record.embedding_features = features.embedding.model_dump()
                record.topic_features = features.topic.model_dump()
                record.feature_vector_version = features.feature_vector_version
                record.raw_feature_vector = features.raw_feature_vector
            record.extraction_error = extraction_error
            record.extracted_at = utc_now

        await session.commit()
        await session.refresh(record)
        return record

    @staticmethod
    async def get_nlp_features_by_message_id(
        session: AsyncSession,
        message_id: str
    ) -> Optional[NLPFeatureRecord]:
        result = await session.execute(
            select(NLPFeatureRecord).where(NLPFeatureRecord.message_id == message_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_room_nlp_features(
        session: AsyncSession,
        room_id: str
    ) -> List[NLPFeatureRecord]:
        result = await session.execute(
            select(NLPFeatureRecord)
            .where(NLPFeatureRecord.room_id == room_id)
            .order_by(NLPFeatureRecord.created_at.asc())
        )
        return list(result.scalars().all())

    @staticmethod
    async def save_or_update_multimodal_features(
        session: AsyncSession,
        message_id: str,
        room_id: str,
        sender_id: str,
        fusion_strategy: str = "early_concatenation_v1",
        feature_vector_version: str = "v1.0-140d",
        has_communication: bool = True,
        has_acoustic: bool = False,
        has_nlp: bool = False,
        communication_features: Optional[dict] = None,
        acoustic_features_summary: Optional[dict] = None,
        nlp_features_summary: Optional[dict] = None,
        modality_mask: Optional[dict] = None,
        raw_fused_vector: Optional[list] = None,
        normalized_fused_vector: Optional[list] = None,
        status: str = "completed",
        fusion_error: Optional[str] = None
    ) -> MultimodalFeatureRecord:
        result = await session.execute(
            select(MultimodalFeatureRecord).where(MultimodalFeatureRecord.message_id == message_id)
        )
        record = result.scalar_one_or_none()
        utc_now = datetime.now(timezone.utc)

        if not record:
            record = MultimodalFeatureRecord(
                id=str(uuid.uuid4()),
                message_id=message_id,
                room_id=room_id,
                sender_id=sender_id,
                status=status,
                fusion_strategy=fusion_strategy,
                feature_vector_version=feature_vector_version,
                has_communication=has_communication,
                has_acoustic=has_acoustic,
                has_nlp=has_nlp,
                communication_features=communication_features or {},
                acoustic_features_summary=acoustic_features_summary,
                nlp_features_summary=nlp_features_summary,
                modality_mask=modality_mask or {},
                raw_fused_vector=raw_fused_vector or [0.0] * 140,
                normalized_fused_vector=normalized_fused_vector or [0.0] * 140,
                fusion_error=fusion_error,
                created_at=utc_now,
                fused_at=utc_now
            )
            session.add(record)
        else:
            record.status = status
            record.fusion_strategy = fusion_strategy
            record.feature_vector_version = feature_vector_version
            record.has_communication = has_communication
            record.has_acoustic = has_acoustic
            record.has_nlp = has_nlp
            if communication_features is not None:
                record.communication_features = communication_features
            record.acoustic_features_summary = acoustic_features_summary
            record.nlp_features_summary = nlp_features_summary
            if modality_mask is not None:
                record.modality_mask = modality_mask
            if raw_fused_vector is not None:
                record.raw_fused_vector = raw_fused_vector
            if normalized_fused_vector is not None:
                record.normalized_fused_vector = normalized_fused_vector
            record.fusion_error = fusion_error
            record.fused_at = utc_now

        await session.commit()
        await session.refresh(record)
        return record

    @staticmethod
    async def get_multimodal_features_by_message_id(
        session: AsyncSession,
        message_id: str
    ) -> Optional[MultimodalFeatureRecord]:
        result = await session.execute(
            select(MultimodalFeatureRecord).where(MultimodalFeatureRecord.message_id == message_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_room_multimodal_features(
        session: AsyncSession,
        room_id: str
    ) -> List[MultimodalFeatureRecord]:
        result = await session.execute(
            select(MultimodalFeatureRecord)
            .where(MultimodalFeatureRecord.room_id == room_id)
            .order_by(MultimodalFeatureRecord.created_at.asc())
        )
        return list(result.scalars().all())

    # ---------------------------------------------------------
    # Milestone 9: Personal Baseline & Temporal Trend Engine
    # ---------------------------------------------------------

    @staticmethod
    async def save_or_update_personal_baseline(
        session: AsyncSession,
        user_id: str,
        window_type: str = "all_time",
        feature_vector_version: str = "v1.0-140d",
        sample_count: int = 0,
        is_baseline_established: bool = False,
        valid_dimensions_count: int = 140,
        raw_mean_vector: Optional[list] = None,
        raw_std_vector: Optional[list] = None,
        normalized_mean_vector: Optional[list] = None,
        normalized_std_vector: Optional[list] = None,
        first_observed_at: Optional[datetime] = None,
        last_observed_at: Optional[datetime] = None
    ) -> PersonalBaselineRecord:
        result = await session.execute(
            select(PersonalBaselineRecord).where(
                PersonalBaselineRecord.user_id == user_id,
                PersonalBaselineRecord.window_type == window_type,
                PersonalBaselineRecord.feature_vector_version == feature_vector_version
            )
        )
        record = result.scalar_one_or_none()
        utc_now = datetime.now(timezone.utc)

        if not record:
            record = PersonalBaselineRecord(
                id=str(uuid.uuid4()),
                user_id=user_id,
                window_type=window_type,
                feature_vector_version=feature_vector_version,
                sample_count=sample_count,
                is_baseline_established=is_baseline_established,
                valid_dimensions_count=valid_dimensions_count,
                raw_mean_vector=raw_mean_vector or [0.0] * 140,
                raw_std_vector=raw_std_vector or [0.0] * 140,
                normalized_mean_vector=normalized_mean_vector or [0.0] * 140,
                normalized_std_vector=normalized_std_vector or [0.0] * 140,
                first_observed_at=first_observed_at,
                last_observed_at=last_observed_at,
                computed_at=utc_now
            )
            session.add(record)
        else:
            record.sample_count = sample_count
            record.is_baseline_established = is_baseline_established
            record.valid_dimensions_count = valid_dimensions_count
            if raw_mean_vector is not None:
                record.raw_mean_vector = raw_mean_vector
            if raw_std_vector is not None:
                record.raw_std_vector = raw_std_vector
            if normalized_mean_vector is not None:
                record.normalized_mean_vector = normalized_mean_vector
            if normalized_std_vector is not None:
                record.normalized_std_vector = normalized_std_vector
            if first_observed_at is not None:
                record.first_observed_at = first_observed_at
            if last_observed_at is not None:
                record.last_observed_at = last_observed_at
            record.computed_at = utc_now

        try:
            await session.commit()
            await session.refresh(record)
            return record
        except Exception:
            await session.rollback()
            result = await session.execute(
                select(PersonalBaselineRecord).where(
                    PersonalBaselineRecord.user_id == user_id,
                    PersonalBaselineRecord.window_type == window_type,
                    PersonalBaselineRecord.feature_vector_version == feature_vector_version
                )
            )
            existing = result.scalar_one_or_none()
            if existing:
                existing.sample_count = sample_count
                existing.is_baseline_established = is_baseline_established
                existing.valid_dimensions_count = valid_dimensions_count
                if raw_mean_vector is not None:
                    existing.raw_mean_vector = raw_mean_vector
                if raw_std_vector is not None:
                    existing.raw_std_vector = raw_std_vector
                if normalized_mean_vector is not None:
                    existing.normalized_mean_vector = normalized_mean_vector
                if normalized_std_vector is not None:
                    existing.normalized_std_vector = normalized_std_vector
                if first_observed_at is not None:
                    existing.first_observed_at = first_observed_at
                if last_observed_at is not None:
                    existing.last_observed_at = last_observed_at
                existing.computed_at = utc_now
                await session.commit()
                await session.refresh(existing)
                return existing
            raise

    @staticmethod
    async def get_personal_baseline_by_user(
        session: AsyncSession,
        user_id: str,
        window_type: str = "all_time",
        feature_vector_version: str = "v1.0-140d"
    ) -> Optional[PersonalBaselineRecord]:
        result = await session.execute(
            select(PersonalBaselineRecord).where(
                PersonalBaselineRecord.user_id == user_id,
                PersonalBaselineRecord.window_type == window_type,
                PersonalBaselineRecord.feature_vector_version == feature_vector_version
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def save_or_update_temporal_trend_record(
        session: AsyncSession,
        message_id: str,
        room_id: str,
        user_id: str,
        status: str = "completed",
        feature_vector_version: str = "v1.0-140d",
        sample_count_at_eval: int = 0,
        is_baseline_established: bool = False,
        baseline_snapshot_mean: Optional[list] = None,
        baseline_snapshot_std: Optional[list] = None,
        zscore_validity_mask: Optional[list] = None,
        valid_dimension_count: int = 0,
        unavailable_dimension_count: int = 140,
        modality_coverage: Optional[dict] = None,
        raw_zscore_vector: Optional[list] = None,
        normalized_zscore_vector: Optional[list] = None,
        clipped_zscore_vector: Optional[list] = None,
        mean_absolute_z: float = 0.0,
        rms_z: float = 0.0,
        standardized_euclidean_distance: float = 0.0,
        top_deviations_summary: Optional[list] = None,
        trend_slopes_summary: Optional[dict] = None,
        moving_averages_summary: Optional[dict] = None,
        rate_of_change_summary: Optional[dict] = None,
        evaluation_error: Optional[str] = None
    ) -> TemporalTrendRecord:
        result = await session.execute(
            select(TemporalTrendRecord).where(TemporalTrendRecord.message_id == message_id)
        )
        record = result.scalar_one_or_none()
        utc_now = datetime.now(timezone.utc)

        if not record:
            record = TemporalTrendRecord(
                id=str(uuid.uuid4()),
                message_id=message_id,
                room_id=room_id,
                user_id=user_id,
                status=status,
                feature_vector_version=feature_vector_version,
                sample_count_at_eval=sample_count_at_eval,
                is_baseline_established=is_baseline_established,
                baseline_snapshot_mean=baseline_snapshot_mean or [0.0] * 140,
                baseline_snapshot_std=baseline_snapshot_std or [0.0] * 140,
                zscore_validity_mask=zscore_validity_mask or [False] * 140,
                valid_dimension_count=valid_dimension_count,
                unavailable_dimension_count=unavailable_dimension_count,
                modality_coverage=modality_coverage or {},
                raw_zscore_vector=raw_zscore_vector or [0.0] * 140,
                normalized_zscore_vector=normalized_zscore_vector or [0.0] * 140,
                clipped_zscore_vector=clipped_zscore_vector or [0.0] * 140,
                mean_absolute_z=mean_absolute_z,
                rms_z=rms_z,
                standardized_euclidean_distance=standardized_euclidean_distance,
                top_deviations_summary=top_deviations_summary or [],
                trend_slopes_summary=trend_slopes_summary or {},
                moving_averages_summary=moving_averages_summary or {},
                rate_of_change_summary=rate_of_change_summary or {},
                evaluation_error=evaluation_error,
                evaluated_at=utc_now
            )
            session.add(record)
        else:
            record.status = status
            record.feature_vector_version = feature_vector_version
            record.sample_count_at_eval = sample_count_at_eval
            record.is_baseline_established = is_baseline_established
            if baseline_snapshot_mean is not None:
                record.baseline_snapshot_mean = baseline_snapshot_mean
            if baseline_snapshot_std is not None:
                record.baseline_snapshot_std = baseline_snapshot_std
            if zscore_validity_mask is not None:
                record.zscore_validity_mask = zscore_validity_mask
            record.valid_dimension_count = valid_dimension_count
            record.unavailable_dimension_count = unavailable_dimension_count
            if modality_coverage is not None:
                record.modality_coverage = modality_coverage
            if raw_zscore_vector is not None:
                record.raw_zscore_vector = raw_zscore_vector
            if normalized_zscore_vector is not None:
                record.normalized_zscore_vector = normalized_zscore_vector
            if clipped_zscore_vector is not None:
                record.clipped_zscore_vector = clipped_zscore_vector
            record.mean_absolute_z = mean_absolute_z
            record.rms_z = rms_z
            record.standardized_euclidean_distance = standardized_euclidean_distance
            if top_deviations_summary is not None:
                record.top_deviations_summary = top_deviations_summary
            if trend_slopes_summary is not None:
                record.trend_slopes_summary = trend_slopes_summary
            if moving_averages_summary is not None:
                record.moving_averages_summary = moving_averages_summary
            if rate_of_change_summary is not None:
                record.rate_of_change_summary = rate_of_change_summary
            record.evaluation_error = evaluation_error
            record.evaluated_at = utc_now

        await session.commit()
        await session.refresh(record)
        return record

    @staticmethod
    async def get_temporal_trend_by_message_id(
        session: AsyncSession,
        message_id: str
    ) -> Optional[TemporalTrendRecord]:
        result = await session.execute(
            select(TemporalTrendRecord).where(TemporalTrendRecord.message_id == message_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_room_temporal_trends(
        session: AsyncSession,
        room_id: str
    ) -> List[TemporalTrendRecord]:
        result = await session.execute(
            select(TemporalTrendRecord)
            .where(TemporalTrendRecord.room_id == room_id)
            .order_by(TemporalTrendRecord.evaluated_at.asc())
        )
        return list(result.scalars().all())

    @staticmethod
    async def get_user_multimodal_history(
        session: AsyncSession,
        user_id: str,
        before_timestamp: Optional[datetime] = None,
        exclude_message_id: Optional[str] = None,
        window_type: str = "all_time",
        feature_version: str = "v1.0-140d"
    ) -> List[MultimodalFeatureRecord]:
        """
        Retrieves strictly preceding historical multimodal records for a user.
        Enforces data leakage prevention and deterministic ordering (timestamp ASC, id ASC).
        """
        query = (
            select(MultimodalFeatureRecord)
            .options(selectinload(MultimodalFeatureRecord.message))
            .join(Message, MultimodalFeatureRecord.message_id == Message.id)
            .where(
                MultimodalFeatureRecord.sender_id == user_id,
                MultimodalFeatureRecord.status == "completed",
                MultimodalFeatureRecord.feature_vector_version == feature_version
            )
        )

        if before_timestamp is not None:
            query = query.where(Message.timestamp < before_timestamp)

        if exclude_message_id is not None:
            query = query.where(Message.id != exclude_message_id)

        if window_type == "rolling_10":
            # Fetch the 10 most recent observations then reverse to chronological
            query = query.order_by(Message.timestamp.desc(), Message.id.desc()).limit(10)
            res = await session.execute(query)
            records = list(res.scalars().all())
            records.reverse()
            return records
        elif window_type == "rolling_30":
            query = query.order_by(Message.timestamp.desc(), Message.id.desc()).limit(30)
            res = await session.execute(query)
            records = list(res.scalars().all())
            records.reverse()
            return records
        else:
            # Default "all_time"
            query = query.order_by(Message.timestamp.asc(), Message.id.asc())
            res = await session.execute(query)
            return list(res.scalars().all())

    @staticmethod
    async def get_user_temporal_trajectory(
        session: AsyncSession,
        user_id: str,
        room_id: Optional[str] = None,
        limit: int = 100
    ) -> List[TemporalTrendRecord]:
        query = (
            select(TemporalTrendRecord)
            .where(
                TemporalTrendRecord.user_id == user_id
            )
        )
        if room_id:
            query = query.where(TemporalTrendRecord.room_id == room_id)

        query = query.order_by(TemporalTrendRecord.evaluated_at.asc()).limit(limit)
        res = await session.execute(query)
        return list(res.scalars().all())

    @staticmethod
    async def get_user_temporal_records_for_period(
        session: AsyncSession,
        user_id: str,
        room_id: Optional[str] = None,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
        feature_version: str = "v1.0-140d"
    ) -> List[TemporalTrendRecord]:
        """
        Retrieves chronological TemporalTrendRecord items for a user within an optional period window.
        Orders deterministically by Message.timestamp asc, Message.id asc.
        """
        query = (
            select(TemporalTrendRecord)
            .options(selectinload(TemporalTrendRecord.message))
            .join(Message, TemporalTrendRecord.message_id == Message.id)
            .where(
                TemporalTrendRecord.user_id == user_id,
                TemporalTrendRecord.feature_vector_version == feature_version
            )
        )
        if room_id:
            query = query.where(TemporalTrendRecord.room_id == room_id)
        if period_start is not None:
            query = query.where(Message.timestamp >= period_start)
        if period_end is not None:
            query = query.where(Message.timestamp <= period_end)

        query = query.order_by(Message.timestamp.asc(), Message.id.asc())
        res = await session.execute(query)
        return list(res.scalars().all())

    @staticmethod
    async def save_analytical_report(
        session: AsyncSession,
        report_id: str,
        user_id: str,
        room_id: Optional[str] = None,
        report_type: str = "longitudinal",
        feature_vector_version: str = "v1.0-140d",
        status: str = "completed",
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
        observation_count: int = 0,
        baseline_sample_count: int = 0,
        baseline_established: bool = False,
        modality_coverage: Optional[dict] = None,
        deviation_summary: Optional[dict] = None,
        trend_summary: Optional[dict] = None,
        feature_shift_summary: Optional[list] = None,
        trajectory_summary: Optional[dict] = None,
        report_payload: Optional[dict] = None
    ) -> AnalyticalReportRecord:
        record = AnalyticalReportRecord(
            id=report_id,
            user_id=user_id,
            room_id=room_id,
            report_type=report_type,
            feature_vector_version=feature_vector_version,
            status=status,
            period_start=period_start,
            period_end=period_end,
            observation_count=observation_count,
            baseline_sample_count=baseline_sample_count,
            baseline_established=baseline_established,
            modality_coverage=modality_coverage or {},
            deviation_summary=deviation_summary or {},
            trend_summary=trend_summary or {},
            feature_shift_summary=feature_shift_summary or [],
            trajectory_summary=trajectory_summary or {},
            report_payload=report_payload or {},
            generated_at=datetime.now(timezone.utc)
        )
        session.add(record)
        await session.commit()
        await session.refresh(record)
        return record

    @staticmethod
    async def get_analytical_report_by_id(
        session: AsyncSession,
        report_id: str
    ) -> Optional[AnalyticalReportRecord]:
        result = await session.execute(
            select(AnalyticalReportRecord).where(AnalyticalReportRecord.id == report_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_latest_analytical_report_for_user(
        session: AsyncSession,
        user_id: str,
        room_id: Optional[str] = None
    ) -> Optional[AnalyticalReportRecord]:
        query = select(AnalyticalReportRecord).where(AnalyticalReportRecord.user_id == user_id)
        if room_id:
            query = query.where(AnalyticalReportRecord.room_id == room_id)
        query = query.order_by(desc(AnalyticalReportRecord.generated_at), desc(AnalyticalReportRecord.id)).limit(1)
        result = await session.execute(query)
        return result.scalar_one_or_none()

    @staticmethod
    async def get_analytical_reports_for_user(
        session: AsyncSession,
        user_id: str,
        limit: int = 50
    ) -> List[AnalyticalReportRecord]:
        result = await session.execute(
            select(AnalyticalReportRecord)
            .where(AnalyticalReportRecord.user_id == user_id)
            .order_by(desc(AnalyticalReportRecord.generated_at), desc(AnalyticalReportRecord.id))
            .limit(limit)
        )
        return list(result.scalars().all())

    @staticmethod
    async def get_analytical_reports_for_room(
        session: AsyncSession,
        room_id: str,
        limit: int = 50
    ) -> List[AnalyticalReportRecord]:
        result = await session.execute(
            select(AnalyticalReportRecord)
            .where(AnalyticalReportRecord.room_id == room_id)
            .order_by(desc(AnalyticalReportRecord.generated_at), desc(AnalyticalReportRecord.id))
            .limit(limit)
        )
        return list(result.scalars().all())

    @staticmethod
    async def is_user_participant(
        session: AsyncSession,
        room_id: str,
        user_id: str
    ) -> bool:
        result = await session.execute(
            select(RoomParticipant).where(
                RoomParticipant.room_id == room_id,
                RoomParticipant.user_id == user_id
            )
        )
        return result.scalar_one_or_none() is not None


