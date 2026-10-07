"""
Multimodal Feature Fusion Service Orchestrator
Coordinates M3 communication, M6 acoustic, and M7 NLP modalities into 140-d aligned space.
"""
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
import math
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repository import ChatRepository
from app.db.models import Message, AcousticFeatureRecord, NLPFeatureRecord, MultimodalFeatureRecord
from app.services.multimodal_fusion.models import (
    CommunicationFeatureBlock,
    MultimodalMaskBlock,
    MultimodalFusionResult,
    MultimodalFeatureScaler,
    get_utc_now
)
from app.services.multimodal_fusion.exceptions import (
    MessageNotFoundError,
    FusionProcessingError
)


class MultimodalFusionService:
    """
    Coordinator singleton for Early Multimodal Feature Fusion (v1.0-140d).
    """
    _instance: Optional["MultimodalFusionService"] = None

    @classmethod
    def get_instance(cls) -> "MultimodalFusionService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def get_neutral_acoustic_vector(cls) -> List[float]:
        """Returns 59-dimensional neutral/absent acoustic vector (all zeros)."""
        return [0.0] * 59

    @classmethod
    def get_neutral_nlp_vector(cls) -> List[float]:
        """Returns 66-dimensional neutral default NLP vector."""
        return [
            # [00..09] Lexical (all zeros)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            # [10..17] Sentiment (neutral compound 0.0, neu prob 1.0)
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            # [18..24] Emotion (neutral 1.0, others 0.0)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            # [25..44] TF-IDF (20 zeros)
            *([0.0] * 20),
            # [45..60] Embedding (16 zeros)
            *([0.0] * 16),
            # [61..65] Topic proportions (uniform 0.20 each)
            0.2, 0.2, 0.2, 0.2, 0.2
        ]

    async def fuse_message_features(
        self,
        message_id: str,
        db: AsyncSession
    ) -> MultimodalFusionResult:
        """
        Extracts, aligns, concatenates, and normalizes features across M3, M6, and M7 for a message.
        Persists result idempotently to multimodal_features table.
        """
        # 1. Fetch target message
        msg = await ChatRepository.get_message_by_id(db, message_id)
        if not msg:
            raise MessageNotFoundError(f"Message '{message_id}' not found.")

        # 2. Fetch room messages to compute M3 Communication temporal context
        room_messages = await ChatRepository.get_room_messages(db, msg.room_id)
        sorted_msgs = sorted(room_messages, key=lambda m: m.timestamp)

        # Locate target message index
        target_idx = 0
        for i, m in enumerate(sorted_msgs):
            if m.id == msg.id:
                target_idx = i
                break

        # Compute running communication metrics for this message
        total_so_far = target_idx + 1
        sender_msgs = [m for m in sorted_msgs[:total_so_far] if m.sender_id == msg.sender_id]
        sender_ratio = round(len(sender_msgs) / total_so_far, 4) if total_so_far > 0 else 1.0

        # Turn latency calculation
        response_latency = 0.0
        turn_switch = 0.0
        if target_idx > 0:
            prev_msg = sorted_msgs[target_idx - 1]
            if prev_msg.sender_id != msg.sender_id:
                response_latency = max(0.0, (msg.timestamp - prev_msg.timestamp).total_seconds())
                turn_switch = 1.0

        gap_threshold = 60.0
        is_gap = 1.0 if response_latency >= gap_threshold else 0.0

        # Conversation duration so far
        first_time = sorted_msgs[0].timestamp
        duration_so_far = max(0.0, (msg.timestamp - first_time).total_seconds())

        # Length metrics
        text_content = msg.message_text or ""
        char_len = len(text_content)
        word_len = len(text_content.split()) if text_content else 0

        all_chars = [len(m.message_text or "") for m in sorted_msgs[:total_so_far]]
        all_words = [len((m.message_text or "").split()) for m in sorted_msgs[:total_so_far]]
        avg_char = round(sum(all_chars) / total_so_far, 2) if total_so_far > 0 else 0.0
        avg_word = round(sum(all_words) / total_so_far, 2) if total_so_far > 0 else 0.0

        comm_block = CommunicationFeatureBlock(
            turn_index=target_idx,
            sender_message_ratio=sender_ratio,
            response_latency_seconds=round(response_latency, 3),
            is_communication_gap=is_gap,
            conversation_duration_so_far_seconds=round(duration_so_far, 2),
            message_char_length=char_len,
            message_word_length=word_len,
            avg_char_length_session=avg_char,
            avg_word_length_session=avg_word,
            turn_switch_flag=turn_switch
        )
        comm_vec = comm_block.to_vector()

        # 3. Retrieve M6 Acoustic Features
        acoustic_record = await ChatRepository.get_acoustic_features_by_message_id(db, message_id)
        has_acoustic = False
        acoustic_summary = None
        if acoustic_record and acoustic_record.raw_feature_vector and len(acoustic_record.raw_feature_vector) == 59:
            acoustic_vec = [float(v) for v in acoustic_record.raw_feature_vector]
            has_acoustic = True
            acoustic_summary = {
                "engine": acoustic_record.engine,
                "duration_seconds": acoustic_record.duration_seconds,
                "pitch_mean": acoustic_record.pitch_features.get("f0_mean_hz") if acoustic_record.pitch_features else None,
                "rms_mean": acoustic_record.energy_features.get("rms_mean") if acoustic_record.energy_features else None,
                "pause_count": acoustic_record.pause_features.get("pause_count") if acoustic_record.pause_features else None
            }
        else:
            acoustic_vec = self.get_neutral_acoustic_vector()

        # 4. Retrieve M7 NLP Features
        nlp_record = await ChatRepository.get_nlp_features_by_message_id(db, message_id)
        has_nlp = False
        nlp_summary = None
        if nlp_record and nlp_record.raw_feature_vector and len(nlp_record.raw_feature_vector) == 66:
            nlp_vec = [float(v) for v in nlp_record.raw_feature_vector]
            has_nlp = True
            nlp_summary = {
                "engine": nlp_record.engine,
                "source_type": nlp_record.source_type,
                "vader_compound": nlp_record.sentiment_features.get("vader_compound") if nlp_record.sentiment_features else None,
                "dominant_emotion": max(nlp_record.emotion_features.items(), key=lambda kv: kv[1])[0] if nlp_record.emotion_features else None,
                "word_count": nlp_record.lexical_features.get("word_count") if nlp_record.lexical_features else None
            }
        else:
            nlp_vec = self.get_neutral_nlp_vector()

        # 5. Build Modality Mask
        is_voice = 1.0 if msg.message_type == "voice" else 0.0
        is_text = 1.0 if msg.message_type == "text" else 0.0
        mask_block = MultimodalMaskBlock(
            has_communication_metrics=1.0,
            has_acoustic_features=1.0 if has_acoustic else 0.0,
            has_nlp_features=1.0 if has_nlp else 0.0,
            is_voice_message=is_voice,
            is_text_message=is_text
        )
        mask_vec = mask_block.to_vector()

        # 6. Assemble 140-d Raw and Normalized Vectors
        raw_vec, normalized_vec = MultimodalFusionResult.assemble_vectors(
            comm_vec=comm_vec,
            acoustic_vec=acoustic_vec,
            nlp_vec=nlp_vec,
            mask_vec=mask_vec
        )

        # 7. Persist to Database (Idempotent Upsert)
        db_record = await ChatRepository.save_or_update_multimodal_features(
            session=db,
            message_id=msg.id,
            room_id=msg.room_id,
            sender_id=msg.sender_id,
            fusion_strategy="early_concatenation_v1",
            feature_vector_version="v1.0-140d",
            has_communication=True,
            has_acoustic=has_acoustic,
            has_nlp=has_nlp,
            communication_features=comm_block.model_dump(),
            acoustic_features_summary=acoustic_summary,
            nlp_features_summary=nlp_summary,
            modality_mask=mask_block.model_dump(),
            raw_fused_vector=raw_vec,
            normalized_fused_vector=normalized_vec,
            fusion_error=None
        )

        return MultimodalFusionResult(
            message_id=msg.id,
            room_id=msg.room_id,
            sender_id=msg.sender_id,
            fusion_strategy="early_concatenation_v1",
            feature_vector_version="v1.0-140d",
            has_communication=True,
            has_acoustic=has_acoustic,
            has_nlp=has_nlp,
            communication_features=comm_block,
            acoustic_features_summary=acoustic_summary,
            nlp_features_summary=nlp_summary,
            modality_mask=mask_block,
            raw_fused_vector=raw_vec,
            normalized_fused_vector=normalized_vec,
            fused_at=db_record.fused_at or get_utc_now()
        )
