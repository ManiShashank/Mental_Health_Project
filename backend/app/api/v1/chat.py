import os
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timezone
import json
import logging
import uuid
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query, Header, Response, status, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import get_db, async_session_maker
from app.db.repository import ChatRepository
from app.schemas.chat import (
    RoomCreate,
    RoomResponse,
    MessageCreate,
    MessageResponse,
    VoiceUploadResponse,
    TranscriptionResponse,
    AcousticFeatureResponse,
    RoomAcousticFeaturesResponse,
    NLPFeatureResponse,
    RoomNLPFeaturesResponse,
    MultimodalFeatureResponse,
    RoomMultimodalFeaturesResponse,
    PersonalBaselineResponse,
    TemporalTrendResponse,
    RoomTemporalTrendsResponse,
    UserTrajectoryResponse,
    AnalyticalReportRequest,
    AnalyticalReportResponse,
    AnalyticalReportHistoryResponse,
    RoomAnalyticalReportsResponse,
    ReportMetadataSchema,
    ReportPeriodSchema,
    BaselineSummarySchema,
    ModalityCoverageSummarySchema,
    DeviationSummarySchema,
    FeatureShiftItemSchema,
    TrendSummarySchema,
    TrajectorySummarySchema,
    ConversationHistoryResponse,
    WebSocketEvent
)
from app.schemas.analysis import (
    CommunicationMetricsResponse,
    ConversationDetailsResponse
)
from app.analysis.communication.metrics import (
    CommunicationMetricsEngine,
    MessageRecord
)
from app.services.chat_manager import manager
from app.services.speech_to_text import (
    stt_service,
    SpeechToTextError,
    AudioFileNotFoundError as STTAudioFileNotFoundError,
    NonVoiceMessageError as STTNonVoiceMessageError,
    TranscriptionFailedError
)
from app.services.acoustic_features import (
    acoustic_service,
    AcousticExtractionError,
    AudioFileNotFoundError as AcousticAudioFileNotFoundError,
    InvalidAudioFormatError as AcousticInvalidAudioFormatError,
    NonVoiceMessageError as AcousticNonVoiceMessageError,
    AudioSignalProcessingError
)
from app.services.nlp_features import (
    nlp_service,
    NLPException,
    EmptyTextError,
    MissingTranscriptError,
    InvalidMessageTypeError,
    NLPProcessingError
)
from app.services.multimodal_fusion import (
    MultimodalFusionService,
    MultimodalFusionException,
    MessageNotFoundError as MultimodalMessageNotFoundError,
    FusionProcessingError
)
from app.services.personal_baseline import (
    personal_baseline_service,
    PersonalBaselineException,
    InsufficientHistoryError,
    UnsupportedFeatureVectorError,
    DataLeakageError,
    UserAccessDeniedError
)
from app.services.analytical_report import (
    analytical_report_service,
    AnalyticalReport,
    AnalyticalReportException,
    InsufficientReportDataError,
    UnsupportedFeatureVersionError,
    InvalidReportPeriodError,
    ReportNotFoundError,
    UserAccessDeniedError as ReportUserAccessDeniedError,
    ReportGenerationError,
    ReportFormatter
)

logger = logging.getLogger("chat_api")
router = APIRouter(prefix="/chat", tags=["Communication & WebSockets"])

# Ensure audio storage directory exists
os.makedirs(settings.AUDIO_UPLOAD_DIR, exist_ok=True)


# ---------------------------------------------------------
# REST API Endpoints
# ---------------------------------------------------------

@router.get(
    "/rooms",
    response_model=List[RoomResponse],
    summary="List Active Rooms",
    description="Returns all active chat rooms stored in the system."
)
async def list_rooms(db: AsyncSession = Depends(get_db)):
    rooms = await ChatRepository.get_active_rooms(db)
    return rooms


@router.post(
    "/rooms",
    response_model=RoomResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create or Retrieve Room",
    description="Creates a new conversation room or retrieves an existing room by ID."
)
async def create_or_get_room(
    room_in: RoomCreate,
    db: AsyncSession = Depends(get_db)
):
    room = await ChatRepository.get_or_create_room(db, room_id=room_in.id, name=room_in.name)
    return room


@router.get(
    "/rooms/{room_id}/messages",
    response_model=ConversationHistoryResponse,
    summary="Get Conversation History",
    description="Retrieves timestamped message history for a specific room directly from the database."
)
async def get_room_history(
    room_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    room = await ChatRepository.get_or_create_room(db, room_id=room_id)
    messages = await ChatRepository.get_room_messages(db, room_id=room_id, limit=limit, offset=offset)
    
    formatted_messages = [
        MessageResponse(
            message_id=m.id,
            conversation_id=m.room_id,
            room_id=m.room_id,
            sender_id=m.sender_id,
            sender_name=m.sender.display_name if m.sender else m.sender_id,
            receiver_id=m.receiver_id,
            message_text=m.message_text,
            timestamp=m.timestamp,
            message_type=m.message_type,
            audio_url=m.audio_url,
            audio_duration_seconds=m.audio_duration_seconds,
            audio_format=m.audio_format,
            audio_size_bytes=m.audio_size_bytes,
            transcript_text=m.transcript_text,
            transcription_status=m.transcription_status,
            transcription_model=m.transcription_model,
            transcription_language=m.transcription_language,
            transcription_error=m.transcription_error,
            transcribed_at=m.transcribed_at,
            has_acoustic_features=bool(m.acoustic_features is not None and m.acoustic_features.status == "completed"),
            has_nlp_features=bool(m.nlp_features is not None and m.nlp_features.status == "completed"),
            has_multimodal_features=bool(m.multimodal_features is not None and m.multimodal_features.status == "completed"),
            has_temporal_trends=bool(m.temporal_trend_record is not None and m.temporal_trend_record.status == "completed")
        )
        for m in messages
    ]
    return ConversationHistoryResponse(
        room_id=room_id,
        messages=formatted_messages,
        total_messages=len(formatted_messages)
    )


@router.post(
    "/rooms/{room_id}/voice",
    response_model=VoiceUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and Transmit Voice Message",
    description="Validates and persists audio recording to storage, creates message record, and broadcasts to room."
)
async def upload_voice_message(
    room_id: str,
    file: UploadFile = File(..., description="Audio file payload"),
    user_id: str = Form(..., description="Sender User ID"),
    duration_seconds: Optional[float] = Form(None, description="Audio duration in seconds"),
    receiver_id: Optional[str] = Form(None, description="Optional target receiver ID"),
    db: AsyncSession = Depends(get_db)
):
    # 1. Validation: Ensure room exists
    room = await ChatRepository.get_or_create_room(db, room_id=room_id)
    user = await ChatRepository.get_or_create_user(db, user_id=user_id, display_name=user_id)

    # 2. Validation: File content-type / format
    content_type = file.content_type or "audio/webm"
    if content_type.lower() not in [fmt.lower() for fmt in settings.ALLOWED_AUDIO_FORMATS]:
        # Fallback check on extension if browser sent generic audio/octet-stream
        filename = file.filename or ""
        ext = os.path.splitext(filename)[1].lower()
        valid_exts = [".webm", ".wav", ".ogg", ".mp4", ".m4a", ".aac", ".mp3"]
        if ext not in valid_exts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported audio format '{content_type}'. Allowed formats: {settings.ALLOWED_AUDIO_FORMATS}"
            )

    # 3. Read and validate file size
    audio_bytes = await file.read()
    file_size = len(audio_bytes)
    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded audio file is empty (0 bytes)."
        )
    if file_size > settings.MAX_AUDIO_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file size ({file_size} bytes) exceeds maximum limit of {settings.MAX_AUDIO_SIZE_BYTES} bytes."
        )

    # 4. Save file to storage directory
    msg_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename or "")[1] or ".webm"
    if not ext.startswith("."):
        ext = f".{ext}"
    safe_filename = f"{room_id}_{msg_id}{ext}"
    storage_path = os.path.join(settings.AUDIO_UPLOAD_DIR, safe_filename)

    with open(storage_path, "wb") as f:
        f.write(audio_bytes)

    # 5. Resolve receiver in dyadic room if not specified
    if not receiver_id:
        other_participants = [
            p["user_id"] for p in manager.get_room_participants(room_id)
            if p["user_id"] != user_id
        ]
        if other_participants:
            receiver_id = other_participants[0]

    # 6. Persist to Database
    audio_url = f"/api/v1/chat/audio/{safe_filename}"
    utc_now = datetime.now(timezone.utc)
    saved_msg = await ChatRepository.save_message(
        session=db,
        room_id=room_id,
        sender_id=user_id,
        message_text="[Voice Message]",
        receiver_id=receiver_id,
        message_type="voice",
        timestamp=utc_now,
        message_id=msg_id,
        audio_url=audio_url,
        audio_duration_seconds=duration_seconds,
        audio_format=content_type,
        audio_size_bytes=file_size
    )

    # 7. Construct Message Response
    msg_resp = MessageResponse(
        message_id=saved_msg.id,
        conversation_id=room_id,
        room_id=room_id,
        sender_id=user_id,
        sender_name=user.display_name,
        receiver_id=receiver_id,
        message_text=saved_msg.message_text,
        timestamp=utc_now,
        message_type="voice",
        audio_url=audio_url,
        audio_duration_seconds=duration_seconds,
        audio_format=content_type,
        audio_size_bytes=file_size
    )

    # 8. Broadcast over WebSocket to all active room participants
    broadcast_payload = {
        "event": "message",
        "data": msg_resp.model_dump(mode="json"),
        "timestamp": utc_now.isoformat()
    }
    await manager.broadcast_to_room(room_id, broadcast_payload)

    return VoiceUploadResponse(
        message=msg_resp,
        detail="Voice message uploaded, persisted, and broadcast successfully."
    )


@router.get(
    "/audio/{filename}",
    summary="Stream Stored Audio File",
    description="Securely serves recorded voice audio messages from local storage."
)
async def get_audio_file(filename: str):
    # Path traversal protection
    safe_name = os.path.basename(filename)
    file_path = os.path.join(settings.AUDIO_UPLOAD_DIR, safe_name)
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audio file '{safe_name}' not found."
        )

    # Determine MIME type
    ext = os.path.splitext(safe_name)[1].lower()
    mime_types = {
        ".webm": "audio/webm",
        ".wav": "audio/wav",
        ".ogg": "audio/ogg",
        ".mp4": "audio/mp4",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".mp3": "audio/mpeg",
    }
    media_type = mime_types.get(ext, "application/octet-stream")

    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=safe_name
    )


@router.post(
    "/messages/{message_id}/transcribe",
    response_model=TranscriptionResponse,
    summary="Transcribe Stored Voice Message",
    description="Executes Speech-to-Text inference on a previously uploaded voice message and updates its record in the database."
)
async def transcribe_message(
    message_id: str,
    language: Optional[str] = Query(None, description="Optional target language code (e.g. 'en')"),
    db: AsyncSession = Depends(get_db)
):
    # 1. Fetch message
    msg = await ChatRepository.get_message_by_id(db, message_id=message_id)
    if not msg:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Message '{message_id}' not found."
        )

    # 2. Validate message is a voice message
    if msg.message_type != "voice":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Message '{message_id}' is of type '{msg.message_type}'. Only voice messages can be transcribed."
        )

    # 3. Locate audio file
    if not msg.audio_url:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Voice message '{message_id}' has no audio recording reference."
        )

    audio_filename = os.path.basename(msg.audio_url)
    storage_path = os.path.join(settings.AUDIO_UPLOAD_DIR, audio_filename)
    if not os.path.exists(storage_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audio file '{audio_filename}' for message '{message_id}' does not exist on disk."
        )

    # 4. Set status to processing
    await ChatRepository.update_message_transcript(
        session=db,
        message_id=message_id,
        transcription_status="processing"
    )

    # 5. Run STT service
    try:
        result = await stt_service.transcribe_audio_file(storage_path, language=language)
        utc_now = datetime.now(timezone.utc)

        # 6. Persist transcript
        await ChatRepository.update_message_transcript(
            session=db,
            message_id=message_id,
            transcript_text=result.text,
            transcription_status="completed",
            transcription_model=result.model_name,
            transcription_language=result.language,
            transcription_error=None,
            transcribed_at=utc_now
        )

        # 7. Broadcast WebSocket event to room
        await manager.broadcast_to_room(msg.room_id, {
            "event": "transcription_completed",
            "data": {
                "message_id": message_id,
                "room_id": msg.room_id,
                "transcript_text": result.text,
                "transcription_status": "completed",
                "transcription_model": result.model_name,
                "transcription_language": result.language,
                "transcribed_at": utc_now.isoformat()
            },
            "timestamp": utc_now.isoformat()
        })

        return TranscriptionResponse(
            message_id=message_id,
            room_id=msg.room_id,
            transcript_text=result.text,
            transcription_status="completed",
            transcription_model=result.model_name,
            transcription_language=result.language,
            transcribed_at=utc_now,
            detail="Voice message transcribed successfully."
        )

    except Exception as exc:
        logger.error(f"Transcription failure for message '{message_id}': {exc}")
        await ChatRepository.update_message_transcript(
            session=db,
            message_id=message_id,
            transcription_status="failed",
            transcription_error=str(exc)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(exc)}"
        )


@router.post(
    "/messages/{message_id}/analyze-acoustic",
    response_model=AcousticFeatureResponse,
    summary="Extract Acoustic Features from Voice Message",
    description="Asynchronously computes 6 physical/acoustic feature families (F0 pitch, RMS energy & shimmer, speech rate, pause dynamics, 13 MFCCs, and spectral descriptors) and standardizes them into the exact 59-dimensional v1.0-59d feature vector."
)
async def analyze_message_acoustics(
    message_id: str,
    db: AsyncSession = Depends(get_db)
):
    # 1. Fetch message
    msg = await ChatRepository.get_message_by_id(db, message_id=message_id)
    if not msg:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Message '{message_id}' not found."
        )

    # 2. Validate voice message type
    if msg.message_type != "voice":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Message '{message_id}' is of type '{msg.message_type}'. Only voice messages can undergo acoustic feature extraction."
        )

    # 3. Locate audio file
    if not msg.audio_url:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Voice message '{message_id}' has no associated audio file URL."
        )

    audio_filename = os.path.basename(msg.audio_url)
    storage_path = os.path.join(settings.AUDIO_UPLOAD_DIR, audio_filename)
    if not os.path.exists(storage_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audio file '{audio_filename}' for message '{message_id}' does not exist on disk."
        )

    # 4. Mark status as processing
    await ChatRepository.save_or_update_acoustic_features(
        session=db,
        message_id=message_id,
        room_id=msg.room_id,
        sender_id=msg.sender_id,
        status="processing"
    )

    # 5. Extract acoustic features
    try:
        features_result = await acoustic_service.extract_features(storage_path)

        # 6. Persist completed record
        record = await ChatRepository.save_or_update_acoustic_features(
            session=db,
            message_id=message_id,
            room_id=msg.room_id,
            sender_id=msg.sender_id,
            features=features_result,
            status="completed"
        )

        # 7. Broadcast WebSocket event to room
        await manager.broadcast_to_room(msg.room_id, {
            "event": "acoustic_analysis_completed",
            "data": {
                "message_id": message_id,
                "room_id": msg.room_id,
                "sender_id": msg.sender_id,
                "status": "completed",
                "engine": record.engine,
                "duration_seconds": record.duration_seconds,
                "has_voiced_frames": record.has_voiced_frames,
                "feature_vector_version": record.feature_vector_version,
                "extracted_at": record.extracted_at.isoformat()
            },
            "timestamp": record.extracted_at.isoformat()
        })

        return AcousticFeatureResponse(
            message_id=record.message_id,
            room_id=record.room_id,
            sender_id=record.sender_id,
            status=record.status,
            engine=record.engine,
            duration_seconds=record.duration_seconds,
            sample_rate=record.sample_rate,
            channels=record.channels,
            pitch_features=record.pitch_features,
            energy_features=record.energy_features,
            speech_rate_features=record.speech_rate_features,
            pause_features=record.pause_features,
            mfcc_features=record.mfcc_features,
            spectral_features=record.spectral_features,
            feature_vector_version=record.feature_vector_version,
            raw_feature_vector=record.raw_feature_vector,
            has_voiced_frames=record.has_voiced_frames,
            extraction_error=record.extraction_error,
            extracted_at=record.extracted_at,
            detail="Acoustic features extracted and persisted successfully."
        )

    except (AcousticAudioFileNotFoundError, AcousticInvalidAudioFormatError) as err:
        logger.error(f"Validation error during acoustic extraction for message '{message_id}': {err}")
        await ChatRepository.save_or_update_acoustic_features(
            session=db,
            message_id=message_id,
            room_id=msg.room_id,
            sender_id=msg.sender_id,
            status="failed",
            extraction_error=str(err)
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err)
        )
    except Exception as exc:
        logger.error(f"Acoustic extraction failure for message '{message_id}': {exc}", exc_info=True)
        await ChatRepository.save_or_update_acoustic_features(
            session=db,
            message_id=message_id,
            room_id=msg.room_id,
            sender_id=msg.sender_id,
            status="failed",
            extraction_error=str(exc)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Acoustic extraction failed: {str(exc)}"
        )


@router.get(
    "/messages/{message_id}/acoustic-features",
    response_model=AcousticFeatureResponse,
    summary="Get Acoustic Features for a Message",
    description="Retrieves persisted acoustic features and the exact 59-dimensional vector for a specific voice message."
)
async def get_message_acoustic_features(
    message_id: str,
    db: AsyncSession = Depends(get_db)
):
    record = await ChatRepository.get_acoustic_features_by_message_id(db, message_id=message_id)
    if not record or record.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Acoustic feature analysis for message '{message_id}' not found or not completed."
        )

    return AcousticFeatureResponse(
        message_id=record.message_id,
        room_id=record.room_id,
        sender_id=record.sender_id,
        status=record.status,
        engine=record.engine,
        duration_seconds=record.duration_seconds,
        sample_rate=record.sample_rate,
        channels=record.channels,
        pitch_features=record.pitch_features,
        energy_features=record.energy_features,
        speech_rate_features=record.speech_rate_features,
        pause_features=record.pause_features,
        mfcc_features=record.mfcc_features,
        spectral_features=record.spectral_features,
        feature_vector_version=record.feature_vector_version,
        raw_feature_vector=record.raw_feature_vector,
        has_voiced_frames=record.has_voiced_frames,
        extraction_error=record.extraction_error,
        extracted_at=record.extracted_at,
        detail="Acoustic features retrieved successfully."
    )


@router.get(
    "/rooms/{room_id}/acoustic-features",
    response_model=RoomAcousticFeaturesResponse,
    summary="Get All Acoustic Features for Room",
    description="Retrieves all extracted acoustic feature records for all voice messages in a conversation room."
)
async def get_room_acoustic_features(
    room_id: str,
    db: AsyncSession = Depends(get_db)
):
    records = await ChatRepository.get_room_acoustic_features(db, room_id=room_id)
    completed_records = [
        AcousticFeatureResponse(
            message_id=r.message_id,
            room_id=r.room_id,
            sender_id=r.sender_id,
            status=r.status,
            engine=r.engine,
            duration_seconds=r.duration_seconds,
            sample_rate=r.sample_rate,
            channels=r.channels,
            pitch_features=r.pitch_features,
            energy_features=r.energy_features,
            speech_rate_features=r.speech_rate_features,
            pause_features=r.pause_features,
            mfcc_features=r.mfcc_features,
            spectral_features=r.spectral_features,
            feature_vector_version=r.feature_vector_version,
            raw_feature_vector=r.raw_feature_vector,
            has_voiced_frames=r.has_voiced_frames,
            extraction_error=r.extraction_error,
            extracted_at=r.extracted_at,
            detail="Acoustic features retrieved successfully."
        )
        for r in records
        if r.status == "completed"
    ]
    return RoomAcousticFeaturesResponse(
        room_id=room_id,
        total_analyzed=len(completed_records),
        features=completed_records
    )


@router.post(
    "/messages/{message_id}/analyze-nlp",
    response_model=NLPFeatureResponse,
    summary="Extract NLP & Linguistic Features from Message",
    description="Asynchronously computes 6 linguistic feature families (lexical & syntactic diversity, sentiment polarity dynamics, affective emotion probabilities, 20 TF-IDF projections, 16 semantic embedding summary dimensions, and 5-topic LDA distribution) and standardizes them into the exact 66-dimensional v1.0-66d feature vector."
)
async def analyze_message_nlp(
    message_id: str,
    db: AsyncSession = Depends(get_db)
):
    # 1. Fetch message
    msg = await ChatRepository.get_message_by_id(db, message_id=message_id)
    if not msg:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Message '{message_id}' not found."
        )

    # 2. Determine target text source and validate message state
    if msg.message_type == "text":
        source_type = "text_message"
        target_text = msg.message_text
    elif msg.message_type == "voice":
        source_type = "voice_transcript"
        if not msg.transcript_text or msg.transcription_status != "completed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Voice message '{message_id}' has not been transcribed yet. Run Speech-to-Text transcription (Milestone 5) before NLP analysis."
            )
        target_text = msg.transcript_text
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Message '{message_id}' of type '{msg.message_type}' cannot undergo NLP analysis."
        )

    # 3. Validate non-empty text content
    if not target_text or not target_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot perform NLP analysis on empty or whitespace-only text."
        )

    # 4. Mark status as processing
    await ChatRepository.save_or_update_nlp_features(
        session=db,
        message_id=message_id,
        room_id=msg.room_id,
        sender_id=msg.sender_id,
        source_type=source_type,
        text_content=target_text,
        status="processing"
    )

    # 5. Extract NLP features
    try:
        features_result = await nlp_service.extract_features(target_text, source_type=source_type)

        # 6. Persist completed record
        record = await ChatRepository.save_or_update_nlp_features(
            session=db,
            message_id=message_id,
            room_id=msg.room_id,
            sender_id=msg.sender_id,
            source_type=source_type,
            text_content=target_text,
            features=features_result,
            status="completed"
        )

        # 7. Broadcast WebSocket event to room
        await manager.broadcast_to_room(msg.room_id, {
            "event": "nlp_analysis_completed",
            "data": {
                "message_id": message_id,
                "room_id": msg.room_id,
                "sender_id": msg.sender_id,
                "source_type": record.source_type,
                "status": "completed",
                "engine": record.engine,
                "word_count": record.lexical_features.get("word_count", 0),
                "sentiment_polarity": record.sentiment_features.get("sentiment_polarity_score", 0.0),
                "feature_vector_version": record.feature_vector_version,
                "extracted_at": record.extracted_at.isoformat()
            },
            "timestamp": record.extracted_at.isoformat()
        })

        return NLPFeatureResponse(
            message_id=record.message_id,
            room_id=record.room_id,
            sender_id=record.sender_id,
            source_type=record.source_type,
            status=record.status,
            engine=record.engine,
            text_content=record.text_content,
            lexical_features=record.lexical_features,
            sentiment_features=record.sentiment_features,
            emotion_features=record.emotion_features,
            tfidf_features=record.tfidf_features,
            embedding_features=record.embedding_features,
            topic_features=record.topic_features,
            feature_vector_version=record.feature_vector_version,
            raw_feature_vector=record.raw_feature_vector,
            extraction_error=record.extraction_error,
            extracted_at=record.extracted_at,
            detail="NLP and linguistic features extracted and persisted successfully."
        )

    except EmptyTextError as err:
        logger.error(f"Validation error during NLP extraction for message '{message_id}': {err}")
        await ChatRepository.save_or_update_nlp_features(
            session=db,
            message_id=message_id,
            room_id=msg.room_id,
            sender_id=msg.sender_id,
            source_type=source_type,
            text_content=target_text,
            status="failed",
            extraction_error=str(err)
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err)
        )
    except Exception as exc:
        logger.error(f"NLP extraction failure for message '{message_id}': {exc}", exc_info=True)
        await ChatRepository.save_or_update_nlp_features(
            session=db,
            message_id=message_id,
            room_id=msg.room_id,
            sender_id=msg.sender_id,
            source_type=source_type,
            text_content=target_text,
            status="failed",
            extraction_error=str(exc)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"NLP extraction failed: {str(exc)}"
        )


@router.get(
    "/messages/{message_id}/nlp-features",
    response_model=NLPFeatureResponse,
    summary="Get NLP Features for a Message",
    description="Retrieves persisted NLP features and the exact 66-dimensional vector for a specific message or transcript."
)
async def get_message_nlp_features(
    message_id: str,
    db: AsyncSession = Depends(get_db)
):
    record = await ChatRepository.get_nlp_features_by_message_id(db, message_id=message_id)
    if not record or record.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"NLP feature analysis for message '{message_id}' not found or not completed."
        )

    return NLPFeatureResponse(
        message_id=record.message_id,
        room_id=record.room_id,
        sender_id=record.sender_id,
        source_type=record.source_type,
        status=record.status,
        engine=record.engine,
        text_content=record.text_content,
        lexical_features=record.lexical_features,
        sentiment_features=record.sentiment_features,
        emotion_features=record.emotion_features,
        tfidf_features=record.tfidf_features,
        embedding_features=record.embedding_features,
        topic_features=record.topic_features,
        feature_vector_version=record.feature_vector_version,
        raw_feature_vector=record.raw_feature_vector,
        extraction_error=record.extraction_error,
        extracted_at=record.extracted_at,
        detail="NLP features retrieved successfully."
    )


@router.get(
    "/rooms/{room_id}/nlp-features",
    response_model=RoomNLPFeaturesResponse,
    summary="Get All NLP Features for Room",
    description="Retrieves all extracted NLP feature records for all messages in a conversation room."
)
async def get_room_nlp_features(
    room_id: str,
    db: AsyncSession = Depends(get_db)
):
    records = await ChatRepository.get_room_nlp_features(db, room_id=room_id)
    completed_records = [
        NLPFeatureResponse(
            message_id=r.message_id,
            room_id=r.room_id,
            sender_id=r.sender_id,
            source_type=r.source_type,
            status=r.status,
            engine=r.engine,
            text_content=r.text_content,
            lexical_features=r.lexical_features,
            sentiment_features=r.sentiment_features,
            emotion_features=r.emotion_features,
            tfidf_features=r.tfidf_features,
            embedding_features=r.embedding_features,
            topic_features=r.topic_features,
            feature_vector_version=r.feature_vector_version,
            raw_feature_vector=r.raw_feature_vector,
            extraction_error=r.extraction_error,
            extracted_at=r.extracted_at,
            detail="NLP features retrieved successfully."
        )
        for r in records
        if r.status == "completed"
    ]
    return RoomNLPFeaturesResponse(
        room_id=room_id,
        total_analyzed=len(completed_records),
        features=completed_records
    )


@router.post(
    "/messages/{message_id}/fuse",
    response_model=MultimodalFeatureResponse,
    summary="Fuse Multimodal Features for a Message",
    description="Aligns, concatenates, and normalizes communication (M3), acoustic (M6), and linguistic (M7) features into standardized 140-d vector (v1.0-140d)."
)
async def fuse_message_multimodal_features(
    message_id: str,
    db: AsyncSession = Depends(get_db)
):
    try:
        fusion_svc = MultimodalFusionService.get_instance()
        result = await fusion_svc.fuse_message_features(message_id=message_id, db=db)

        # Broadcast WebSocket event to room
        await manager.broadcast_to_room(result.room_id, {
            "event": "multimodal_fusion_completed",
            "data": {
                "message_id": result.message_id,
                "room_id": result.room_id,
                "sender_id": result.sender_id,
                "fusion_strategy": result.fusion_strategy,
                "feature_vector_version": result.feature_vector_version,
                "raw_feature_vector_length": len(result.raw_fused_vector),
                "has_communication": result.has_communication,
                "has_acoustic": result.has_acoustic,
                "has_nlp": result.has_nlp,
                "fused_at": result.fused_at.isoformat()
            },
            "timestamp": result.fused_at.isoformat()
        })

        return MultimodalFeatureResponse(
            message_id=result.message_id,
            room_id=result.room_id,
            sender_id=result.sender_id,
            status="completed",
            fusion_strategy=result.fusion_strategy,
            feature_vector_version=result.feature_vector_version,
            has_communication=result.has_communication,
            has_acoustic=result.has_acoustic,
            has_nlp=result.has_nlp,
            communication_features=result.communication_features.model_dump(),
            acoustic_features_summary=result.acoustic_features_summary,
            nlp_features_summary=result.nlp_features_summary,
            modality_mask=result.modality_mask.model_dump(),
            raw_fused_vector=result.raw_fused_vector,
            normalized_fused_vector=result.normalized_fused_vector,
            fusion_error=None,
            fused_at=result.fused_at,
            detail="Multimodal features fused successfully."
        )

    except MultimodalMessageNotFoundError as err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(err)
        )
    except Exception as exc:
        logger.error(f"Multimodal fusion error for message '{message_id}': {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Multimodal fusion processing error: {str(exc)}"
        )


@router.get(
    "/messages/{message_id}/multimodal-features",
    response_model=MultimodalFeatureResponse,
    summary="Get Multimodal Features for a Message",
    description="Retrieves persisted multimodal feature vectors (raw and normalized 140-d) and block breakdown for a message."
)
async def get_message_multimodal_features(
    message_id: str,
    db: AsyncSession = Depends(get_db)
):
    record = await ChatRepository.get_multimodal_features_by_message_id(db, message_id=message_id)
    if not record or record.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Multimodal fusion record for message '{message_id}' not found or not completed."
        )

    return MultimodalFeatureResponse(
        message_id=record.message_id,
        room_id=record.room_id,
        sender_id=record.sender_id,
        status=record.status,
        fusion_strategy=record.fusion_strategy,
        feature_vector_version=record.feature_vector_version,
        has_communication=record.has_communication,
        has_acoustic=record.has_acoustic,
        has_nlp=record.has_nlp,
        communication_features=record.communication_features,
        acoustic_features_summary=record.acoustic_features_summary,
        nlp_features_summary=record.nlp_features_summary,
        modality_mask=record.modality_mask,
        raw_fused_vector=record.raw_fused_vector,
        normalized_fused_vector=record.normalized_fused_vector,
        fusion_error=record.fusion_error,
        fused_at=record.fused_at,
        detail="Multimodal features retrieved successfully."
    )


@router.get(
    "/rooms/{room_id}/multimodal-features",
    response_model=RoomMultimodalFeaturesResponse,
    summary="Get All Multimodal Features for Room",
    description="Retrieves all fused multimodal feature records for all messages in a conversation room."
)
async def get_room_multimodal_features(
    room_id: str,
    db: AsyncSession = Depends(get_db)
):
    records = await ChatRepository.get_room_multimodal_features(db, room_id=room_id)
    completed_records = [
        MultimodalFeatureResponse(
            message_id=r.message_id,
            room_id=r.room_id,
            sender_id=r.sender_id,
            status=r.status,
            fusion_strategy=r.fusion_strategy,
            feature_vector_version=r.feature_vector_version,
            has_communication=r.has_communication,
            has_acoustic=r.has_acoustic,
            has_nlp=r.has_nlp,
            communication_features=r.communication_features,
            acoustic_features_summary=r.acoustic_features_summary,
            nlp_features_summary=r.nlp_features_summary,
            modality_mask=r.modality_mask,
            raw_fused_vector=r.raw_fused_vector,
            normalized_fused_vector=r.normalized_fused_vector,
            fusion_error=r.fusion_error,
            fused_at=r.fused_at,
            detail="Multimodal features retrieved successfully."
        )
        for r in records
        if r.status == "completed"
    ]
    return RoomMultimodalFeaturesResponse(
        room_id=room_id,
        total_analyzed=len(completed_records),
        features=completed_records
    )


# ---------------------------------------------------------
# Milestone 9: Personal Baseline & Temporal Trend Endpoints
# ---------------------------------------------------------

@router.post(
    "/baselines/users/{user_id}/recompute",
    response_model=PersonalBaselineResponse,
    summary="Recompute Personal Baseline for User",
    description="Recomputes empirical sample mean and Bessel-corrected sample standard deviation vectors across historical observations."
)
async def recompute_user_baseline(
    user_id: str,
    window_type: str = Query(default="all_time", pattern="^(all_time|rolling_10|rolling_30)$"),
    db: AsyncSession = Depends(get_db)
):
    user = await ChatRepository.get_user_by_id(db, user_id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{user_id}' not found."
        )

    result = await personal_baseline_service.compute_or_update_user_baseline(
        user_id=user_id,
        db=db,
        window_type=window_type
    )

    return PersonalBaselineResponse(
        user_id=result.user_id,
        window_type=result.window_type,
        feature_vector_version=result.feature_vector_version,
        sample_count=result.sample_count,
        is_baseline_established=result.is_baseline_established,
        valid_dimensions_count=result.valid_dimensions_count,
        raw_mean_vector=result.raw_mean_vector,
        raw_std_vector=result.raw_std_vector,
        normalized_mean_vector=result.normalized_mean_vector,
        normalized_std_vector=result.normalized_std_vector,
        first_observed_at=result.first_observed_at,
        last_observed_at=result.last_observed_at,
        computed_at=result.computed_at,
        detail="Personal baseline computed and persisted successfully."
    )


@router.get(
    "/baselines/users/{user_id}",
    response_model=PersonalBaselineResponse,
    summary="Get Personal Baseline for User",
    description="Retrieves persisted personal baseline vectors and sample statistics for a user."
)
async def get_user_baseline(
    user_id: str,
    window_type: str = Query(default="all_time", pattern="^(all_time|rolling_10|rolling_30)$"),
    db: AsyncSession = Depends(get_db)
):
    user = await ChatRepository.get_user_by_id(db, user_id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{user_id}' not found."
        )

    record = await ChatRepository.get_personal_baseline_by_user(
        db,
        user_id=user_id,
        window_type=window_type
    )
    if not record:
        # Recompute on the fly if not cached
        result = await personal_baseline_service.compute_or_update_user_baseline(
            user_id=user_id,
            db=db,
            window_type=window_type
        )
        return PersonalBaselineResponse(
            user_id=result.user_id,
            window_type=result.window_type,
            feature_vector_version=result.feature_vector_version,
            sample_count=result.sample_count,
            is_baseline_established=result.is_baseline_established,
            valid_dimensions_count=result.valid_dimensions_count,
            raw_mean_vector=result.raw_mean_vector,
            raw_std_vector=result.raw_std_vector,
            normalized_mean_vector=result.normalized_mean_vector,
            normalized_std_vector=result.normalized_std_vector,
            first_observed_at=result.first_observed_at,
            last_observed_at=result.last_observed_at,
            computed_at=result.computed_at,
            detail="Personal baseline generated."
        )

    return PersonalBaselineResponse(
        user_id=record.user_id,
        window_type=record.window_type,
        feature_vector_version=record.feature_vector_version,
        sample_count=record.sample_count,
        is_baseline_established=record.is_baseline_established,
        valid_dimensions_count=record.valid_dimensions_count,
        raw_mean_vector=record.raw_mean_vector,
        raw_std_vector=record.raw_std_vector,
        normalized_mean_vector=record.normalized_mean_vector,
        normalized_std_vector=record.normalized_std_vector,
        first_observed_at=record.first_observed_at,
        last_observed_at=record.last_observed_at,
        computed_at=record.computed_at,
        detail="Personal baseline retrieved successfully."
    )


@router.post(
    "/temporal/messages/{message_id}/evaluate",
    response_model=TemporalTrendResponse,
    summary="Evaluate Message Deviation & Temporal Trajectory",
    description="Evaluates within-subject z-score deviation and longitudinal moving averages/trend slopes against strictly preceding baseline."
)
async def evaluate_message_temporal(
    message_id: str,
    window_type: str = Query(default="all_time", pattern="^(all_time|rolling_10|rolling_30)$"),
    db: AsyncSession = Depends(get_db)
):
    msg = await ChatRepository.get_message_by_id(db, message_id=message_id)
    if not msg:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Message '{message_id}' not found."
        )

    try:
        result = await personal_baseline_service.evaluate_message_temporal_deviation(
            message_id=message_id,
            db=db,
            window_type=window_type
        )
    except UnsupportedFeatureVectorError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        )
    except PersonalBaselineException as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc)
        )
    except Exception as exc:
        logger.error(f"Failed to evaluate temporal trends for message '{message_id}': {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Temporal trend evaluation failed: {str(exc)}"
        )

    # Broadcast WebSocket event to room participants
    ws_payload = {
        "event": "temporal_deviation_computed",
        "data": {
            "message_id": result.message_id,
            "room_id": result.room_id,
            "user_id": result.user_id,
            "status": result.status,
            "sample_count_at_eval": result.sample_count_at_eval,
            "is_baseline_established": result.is_baseline_established,
            "valid_dimension_count": result.valid_dimension_count,
            "mean_absolute_z": result.mean_absolute_z,
            "rms_z": result.rms_z,
            "standardized_euclidean_distance": result.standardized_euclidean_distance,
            "top_deviations": [item.model_dump() for item in result.top_deviations[:3]],
            "evaluated_at": result.evaluated_at.isoformat()
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    await manager.broadcast_to_room(result.room_id, ws_payload)

    # Also broadcast personal_baseline_updated event
    ws_baseline_payload = {
        "event": "personal_baseline_updated",
        "data": {
            "user_id": result.user_id,
            "sample_count": result.sample_count_at_eval,
            "is_baseline_established": result.is_baseline_established,
            "valid_dimension_count": result.valid_dimension_count,
            "evaluated_at": result.evaluated_at.isoformat()
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    await manager.broadcast_to_room(result.room_id, ws_baseline_payload)

    return TemporalTrendResponse(
        message_id=result.message_id,
        room_id=result.room_id,
        user_id=result.user_id,
        status=result.status,
        feature_vector_version=result.feature_vector_version,
        sample_count_at_eval=result.sample_count_at_eval,
        is_baseline_established=result.is_baseline_established,
        baseline_snapshot_mean=result.baseline_snapshot_mean,
        baseline_snapshot_std=result.baseline_snapshot_std,
        zscore_validity_mask=result.zscore_validity_mask,
        valid_dimension_count=result.valid_dimension_count,
        unavailable_dimension_count=result.unavailable_dimension_count,
        modality_coverage=result.modality_coverage,
        raw_zscore_vector=result.raw_zscore_vector,
        normalized_zscore_vector=result.normalized_zscore_vector,
        clipped_zscore_vector=result.clipped_zscore_vector,
        mean_absolute_z=result.mean_absolute_z,
        rms_z=result.rms_z,
        standardized_euclidean_distance=result.standardized_euclidean_distance,
        top_deviations=[item.model_dump() for item in result.top_deviations],
        trend_slopes_summary=result.trend_slopes_summary,
        moving_averages_summary=result.moving_averages_summary,
        rate_of_change_summary=result.rate_of_change_summary,
        evaluation_error=result.evaluation_error,
        evaluated_at=result.evaluated_at,
        detail="Temporal trend evaluation completed successfully."
    )


@router.get(
    "/temporal/messages/{message_id}",
    response_model=TemporalTrendResponse,
    summary="Get Message Temporal Trend Record",
    description="Retrieves persisted within-subject z-score deviations, point-in-time snapshot, and trend metrics for a message."
)
async def get_message_temporal(
    message_id: str,
    db: AsyncSession = Depends(get_db)
):
    record = await ChatRepository.get_temporal_trend_by_message_id(db, message_id=message_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Temporal trend record for message '{message_id}' not found."
        )

    return TemporalTrendResponse(
        message_id=record.message_id,
        room_id=record.room_id,
        user_id=record.user_id,
        status=record.status,
        feature_vector_version=record.feature_vector_version,
        sample_count_at_eval=record.sample_count_at_eval,
        is_baseline_established=record.is_baseline_established,
        baseline_snapshot_mean=record.baseline_snapshot_mean,
        baseline_snapshot_std=record.baseline_snapshot_std,
        zscore_validity_mask=record.zscore_validity_mask,
        valid_dimension_count=record.valid_dimension_count,
        unavailable_dimension_count=record.unavailable_dimension_count,
        modality_coverage=record.modality_coverage,
        raw_zscore_vector=record.raw_zscore_vector,
        normalized_zscore_vector=record.normalized_zscore_vector,
        clipped_zscore_vector=record.clipped_zscore_vector,
        mean_absolute_z=record.mean_absolute_z,
        rms_z=record.rms_z,
        standardized_euclidean_distance=record.standardized_euclidean_distance,
        top_deviations=record.top_deviations_summary or [],
        trend_slopes_summary=record.trend_slopes_summary or {},
        moving_averages_summary=record.moving_averages_summary or {},
        rate_of_change_summary=record.rate_of_change_summary or {},
        evaluation_error=record.evaluation_error,
        evaluated_at=record.evaluated_at,
        detail="Temporal trend record retrieved successfully."
    )


@router.get(
    "/temporal/users/{user_id}/trajectory",
    response_model=UserTrajectoryResponse,
    summary="Get User Longitudinal Trajectory",
    description="Returns chronological sequence of temporal deviation points for a user, optionally filtered by room."
)
async def get_user_trajectory(
    user_id: str,
    room_id: Optional[str] = Query(None, description="Optional room ID filter"),
    limit: int = Query(default=100, ge=1, le=500),
    db: AsyncSession = Depends(get_db)
):
    user = await ChatRepository.get_user_by_id(db, user_id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{user_id}' not found."
        )

    records = await ChatRepository.get_user_temporal_trajectory(
        db,
        user_id=user_id,
        room_id=room_id,
        limit=limit
    )

    items = [
        TemporalTrendResponse(
            message_id=r.message_id,
            room_id=r.room_id,
            user_id=r.user_id,
            status=r.status,
            feature_vector_version=r.feature_vector_version,
            sample_count_at_eval=r.sample_count_at_eval,
            is_baseline_established=r.is_baseline_established,
            baseline_snapshot_mean=r.baseline_snapshot_mean,
            baseline_snapshot_std=r.baseline_snapshot_std,
            zscore_validity_mask=r.zscore_validity_mask,
            valid_dimension_count=r.valid_dimension_count,
            unavailable_dimension_count=r.unavailable_dimension_count,
            modality_coverage=r.modality_coverage,
            raw_zscore_vector=r.raw_zscore_vector,
            normalized_zscore_vector=r.normalized_zscore_vector,
            clipped_zscore_vector=r.clipped_zscore_vector,
            mean_absolute_z=r.mean_absolute_z,
            rms_z=r.rms_z,
            standardized_euclidean_distance=r.standardized_euclidean_distance,
            top_deviations=r.top_deviations_summary or [],
            trend_slopes_summary=r.trend_slopes_summary or {},
            moving_averages_summary=r.moving_averages_summary or {},
            rate_of_change_summary=r.rate_of_change_summary or {},
            evaluation_error=r.evaluation_error,
            evaluated_at=r.evaluated_at,
            detail="Trajectory point retrieved."
        )
        for r in records
    ]

    return UserTrajectoryResponse(
        user_id=user_id,
        room_id=room_id,
        total_points=len(items),
        trajectory=items
    )


@router.get(
    "/temporal/rooms/{room_id}",
    response_model=RoomTemporalTrendsResponse,
    summary="Get All Temporal Trends for Room",
    description="Retrieves all evaluated temporal trend records for messages in a specific room."
)
async def get_room_temporal_trends(
    room_id: str,
    db: AsyncSession = Depends(get_db)
):
    room = await ChatRepository.get_room_by_id(db, room_id=room_id)
    if not room:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Room '{room_id}' not found."
        )

    records = await ChatRepository.get_room_temporal_trends(db, room_id=room_id)
    items = [
        TemporalTrendResponse(
            message_id=r.message_id,
            room_id=r.room_id,
            user_id=r.user_id,
            status=r.status,
            feature_vector_version=r.feature_vector_version,
            sample_count_at_eval=r.sample_count_at_eval,
            is_baseline_established=r.is_baseline_established,
            baseline_snapshot_mean=r.baseline_snapshot_mean,
            baseline_snapshot_std=r.baseline_snapshot_std,
            zscore_validity_mask=r.zscore_validity_mask,
            valid_dimension_count=r.valid_dimension_count,
            unavailable_dimension_count=r.unavailable_dimension_count,
            modality_coverage=r.modality_coverage,
            raw_zscore_vector=r.raw_zscore_vector,
            normalized_zscore_vector=r.normalized_zscore_vector,
            clipped_zscore_vector=r.clipped_zscore_vector,
            mean_absolute_z=r.mean_absolute_z,
            rms_z=r.rms_z,
            standardized_euclidean_distance=r.standardized_euclidean_distance,
            top_deviations=r.top_deviations_summary or [],
            trend_slopes_summary=r.trend_slopes_summary or {},
            moving_averages_summary=r.moving_averages_summary or {},
            rate_of_change_summary=r.rate_of_change_summary or {},
            evaluation_error=r.evaluation_error,
            evaluated_at=r.evaluated_at,
            detail="Room temporal record retrieved."
        )
        for r in records
    ]

    return RoomTemporalTrendsResponse(
        room_id=room_id,
        total_analyzed=len(items),
        records=items
    )


# --------------------------------------------------
# Milestone 10: Final Analytical Report Engine Endpoints
# --------------------------------------------------

@router.post(
    "/analytical/users/{user_id}/report",
    response_model=AnalyticalReportResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate & Persist Longitudinal Analytical Report",
    description="Assembles a point-in-time, deterministic, self-contained analytical research report summarizing observed behavioral feature variations relative to personal baseline."
)
async def generate_analytical_report(
    user_id: str,
    request: Optional[AnalyticalReportRequest] = None,
    x_user_id: Optional[str] = Header(None, alias="X-User-Id", description="Requesting user ID for authorization"),
    db: AsyncSession = Depends(get_db)
):
    req_room_id = request.room_id if request else None
    req_start = request.start_time if request else None
    req_end = request.end_time if request else None
    req_type = request.report_type if request else "longitudinal"

    try:
        report = await analytical_report_service.generate_user_report(
            user_id=user_id,
            db=db,
            room_id=req_room_id,
            period_start=req_start,
            period_end=req_end,
            report_type=req_type,
            requesting_user_id=x_user_id
        )
    except (UserAccessDeniedError, ReportUserAccessDeniedError) as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except InvalidReportPeriodError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except InsufficientReportDataError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except UnsupportedFeatureVersionError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except AnalyticalReportException as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        logger.error(f"Failed to generate analytical report for user '{user_id}': {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Report generation failed: {str(exc)}")

    # Broadcast WebSocket event if room_id is present
    if report.metadata.room_id:
        ws_payload = {
            "event": "analytical_report_generated",
            "data": {
                "report_id": report.metadata.report_id,
                "user_id": report.metadata.user_id,
                "room_id": report.metadata.room_id,
                "observation_count": report.metadata.observation_count,
                "baseline_established": report.metadata.baseline_established,
                "generated_at": report.metadata.generated_at.isoformat()
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        await manager.broadcast_to_room(report.metadata.room_id, ws_payload)

    return AnalyticalReportResponse(
        metadata=ReportMetadataSchema(**report.metadata.model_dump()),
        period=ReportPeriodSchema(**report.period.model_dump()),
        baseline=BaselineSummarySchema(**report.baseline.model_dump()),
        modality_coverage=ModalityCoverageSummarySchema(**report.modality_coverage.model_dump()),
        deviation_summary=DeviationSummarySchema(**report.deviation_summary.model_dump()),
        feature_shifts=[FeatureShiftItemSchema(**item.model_dump()) for item in report.feature_shifts],
        trend_summary=TrendSummarySchema(**report.trend_summary.model_dump()),
        trajectory_summary=TrajectorySummarySchema(**report.trajectory_summary.model_dump()),
        limitations=report.limitations,
        disclaimer=report.disclaimer,
        detail="Analytical report generated and persisted successfully."
    )


@router.get(
    "/analytical/users/{user_id}/report",
    response_model=AnalyticalReportResponse,
    summary="Get Latest Analytical Report for User",
    description="Retrieves the most recently generated analytical report for a user, or generates one if observations exist."
)
async def get_latest_user_report(
    user_id: str,
    room_id: Optional[str] = Query(None, description="Optional room filter"),
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    db: AsyncSession = Depends(get_db)
):
    try:
        report = await analytical_report_service.get_latest_user_report(
            user_id=user_id,
            db=db,
            room_id=room_id,
            requesting_user_id=x_user_id
        )
    except (UserAccessDeniedError, ReportUserAccessDeniedError) as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except AnalyticalReportException as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No analytical reports or observations found for user '{user_id}'."
        )

    return AnalyticalReportResponse(
        metadata=ReportMetadataSchema(**report.metadata.model_dump()),
        period=ReportPeriodSchema(**report.period.model_dump()),
        baseline=BaselineSummarySchema(**report.baseline.model_dump()),
        modality_coverage=ModalityCoverageSummarySchema(**report.modality_coverage.model_dump()),
        deviation_summary=DeviationSummarySchema(**report.deviation_summary.model_dump()),
        feature_shifts=[FeatureShiftItemSchema(**item.model_dump()) for item in report.feature_shifts],
        trend_summary=TrendSummarySchema(**report.trend_summary.model_dump()),
        trajectory_summary=TrajectorySummarySchema(**report.trajectory_summary.model_dump()),
        limitations=report.limitations,
        disclaimer=report.disclaimer,
        detail="Latest analytical report retrieved successfully."
    )


@router.get(
    "/analytical/reports/{report_id}",
    response_model=AnalyticalReportResponse,
    summary="Get Persisted Analytical Report by ID",
    description="Retrieves an exact point-in-time persisted analytical report by UUID."
)
async def get_analytical_report_by_id(
    report_id: str,
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    db: AsyncSession = Depends(get_db)
):
    try:
        report = await analytical_report_service.get_report_by_id(
            report_id=report_id,
            db=db,
            requesting_user_id=x_user_id
        )
    except (UserAccessDeniedError, ReportUserAccessDeniedError) as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except ReportNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    return AnalyticalReportResponse(
        metadata=ReportMetadataSchema(**report.metadata.model_dump()),
        period=ReportPeriodSchema(**report.period.model_dump()),
        baseline=BaselineSummarySchema(**report.baseline.model_dump()),
        modality_coverage=ModalityCoverageSummarySchema(**report.modality_coverage.model_dump()),
        deviation_summary=DeviationSummarySchema(**report.deviation_summary.model_dump()),
        feature_shifts=[FeatureShiftItemSchema(**item.model_dump()) for item in report.feature_shifts],
        trend_summary=TrendSummarySchema(**report.trend_summary.model_dump()),
        trajectory_summary=TrajectorySummarySchema(**report.trajectory_summary.model_dump()),
        limitations=report.limitations,
        disclaimer=report.disclaimer,
        detail="Analytical report retrieved successfully."
    )


@router.get(
    "/analytical/users/{user_id}/reports",
    response_model=AnalyticalReportHistoryResponse,
    summary="Get User Analytical Report History",
    description="Returns chronological history of generated analytical reports for a participant."
)
async def get_user_report_history(
    user_id: str,
    limit: int = Query(default=50, ge=1, le=100),
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    db: AsyncSession = Depends(get_db)
):
    if x_user_id and x_user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User '{x_user_id}' is not authorized to view report history for '{user_id}'."
        )

    user = await ChatRepository.get_user_by_id(db, user_id=user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"User '{user_id}' not found.")

    records = await ChatRepository.get_analytical_reports_for_user(db, user_id=user_id, limit=limit)
    report_items = []
    for rec in records:
        rep = AnalyticalReport.model_validate(rec.report_payload)
        report_items.append(AnalyticalReportResponse(
            metadata=ReportMetadataSchema(**rep.metadata.model_dump()),
            period=ReportPeriodSchema(**rep.period.model_dump()),
            baseline=BaselineSummarySchema(**rep.baseline.model_dump()),
            modality_coverage=ModalityCoverageSummarySchema(**rep.modality_coverage.model_dump()),
            deviation_summary=DeviationSummarySchema(**rep.deviation_summary.model_dump()),
            feature_shifts=[FeatureShiftItemSchema(**item.model_dump()) for item in rep.feature_shifts],
            trend_summary=TrendSummarySchema(**rep.trend_summary.model_dump()),
            trajectory_summary=TrajectorySummarySchema(**rep.trajectory_summary.model_dump()),
            limitations=rep.limitations,
            disclaimer=rep.disclaimer,
            detail="Historical report item."
        ))

    return AnalyticalReportHistoryResponse(
        user_id=user_id,
        total_reports=len(report_items),
        reports=report_items
    )


@router.get(
    "/analytical/rooms/{room_id}",
    response_model=RoomAnalyticalReportsResponse,
    summary="Get Room Analytical Reports",
    description="Returns all analytical reports associated with a conversation room."
)
async def get_room_analytical_reports(
    room_id: str,
    limit: int = Query(default=50, ge=1, le=100),
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    db: AsyncSession = Depends(get_db)
):
    room = await ChatRepository.get_room_by_id(db, room_id=room_id)
    if not room:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Room '{room_id}' not found.")

    if x_user_id:
        is_part = await ChatRepository.is_user_participant(db, room_id=room_id, user_id=x_user_id)
        if not is_part:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User '{x_user_id}' is not a participant in room '{room_id}'."
            )

    records = await ChatRepository.get_analytical_reports_for_room(db, room_id=room_id, limit=limit)
    report_items = []
    for rec in records:
        rep = AnalyticalReport.model_validate(rec.report_payload)
        report_items.append(AnalyticalReportResponse(
            metadata=ReportMetadataSchema(**rep.metadata.model_dump()),
            period=ReportPeriodSchema(**rep.period.model_dump()),
            baseline=BaselineSummarySchema(**rep.baseline.model_dump()),
            modality_coverage=ModalityCoverageSummarySchema(**rep.modality_coverage.model_dump()),
            deviation_summary=DeviationSummarySchema(**rep.deviation_summary.model_dump()),
            feature_shifts=[FeatureShiftItemSchema(**item.model_dump()) for item in rep.feature_shifts],
            trend_summary=TrendSummarySchema(**rep.trend_summary.model_dump()),
            trajectory_summary=TrajectorySummarySchema(**rep.trajectory_summary.model_dump()),
            limitations=rep.limitations,
            disclaimer=rep.disclaimer,
            detail="Room report item."
        ))

    return RoomAnalyticalReportsResponse(
        room_id=room_id,
        total_reports=len(report_items),
        reports=report_items
    )


@router.get(
    "/analytical/reports/{report_id}/export",
    summary="Export Analytical Report (JSON, Markdown, HTML)",
    description="Exports a persisted analytical report in the requested presentation format."
)
async def export_analytical_report(
    report_id: str,
    format: str = Query(default="json", pattern="^(json|markdown|html)$"),
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    db: AsyncSession = Depends(get_db)
):
    try:
        report = await analytical_report_service.get_report_by_id(
            report_id=report_id,
            db=db,
            requesting_user_id=x_user_id
        )
    except (UserAccessDeniedError, ReportUserAccessDeniedError) as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except ReportNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    if format == "markdown":
        md_text = ReportFormatter.to_markdown(report)
        return Response(content=md_text, media_type="text/markdown")
    elif format == "html":
        html_text = ReportFormatter.to_html(report)
        return Response(content=html_text, media_type="text/html")
    else:
        return ReportFormatter.to_json_dict(report)



@router.get(
    "/rooms/{room_id}/analysis",
    response_model=CommunicationMetricsResponse,
    summary="Get Communication Metrics & Analysis",
    description="Computes descriptive communication metrics (message count, length, ratio, response latency, gaps) from stored messages."
)
async def get_room_analysis(
    room_id: str,
    gap_threshold: float = Query(default=60.0, ge=1.0, le=3600.0, description="Communication gap latency threshold in seconds"),
    db: AsyncSession = Depends(get_db)
):
    room = await ChatRepository.get_room_by_id(db, room_id=room_id)
    if not room:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Room or conversation '{room_id}' not found."
        )

    messages = await ChatRepository.get_room_messages(db, room_id=room_id, limit=1000)
    
    # Map participant names
    names_map: dict[str, str] = {}
    for p in (room.participants or []):
        if p.user:
            names_map[p.user_id] = p.user.display_name

    # Transform messages to in-memory MessageRecord
    message_records = [
        MessageRecord(
            message_id=m.id,
            sender_id=m.sender_id,
            sender_name=m.sender.display_name if m.sender else names_map.get(m.sender_id, m.sender_id),
            receiver_id=m.receiver_id,
            message_text=m.message_text,
            timestamp=m.timestamp,
            message_type=m.message_type
        )
        for m in messages
    ]

    metrics = CommunicationMetricsEngine.calculate_metrics(
        room_id=room_id,
        messages=message_records,
        participant_names=names_map,
        gap_threshold_seconds=gap_threshold,
        room_status=room.status,
        started_at=room.started_at,
        ended_at=room.ended_at
    )
    return metrics


@router.get(
    "/conversations/{room_id}",
    response_model=ConversationDetailsResponse,
    summary="Get Conversation Lifecycle Details",
    description="Returns metadata, lifecycle status, participants, and message count for a conversation."
)
async def get_conversation_details(
    room_id: str,
    db: AsyncSession = Depends(get_db)
):
    room = await ChatRepository.get_room_by_id(db, room_id=room_id)
    if not room:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{room_id}' not found."
        )

    messages = await ChatRepository.get_room_messages(db, room_id=room_id, limit=1000)
    participants_list = [
        {
            "user_id": p.user_id,
            "display_name": p.user.display_name if p.user else p.user_id,
            "joined_at": p.joined_at,
            "last_seen_at": p.last_seen_at
        }
        for p in room.participants
    ]

    return ConversationDetailsResponse(
        conversation_id=room.id,
        room_id=room.id,
        name=room.name,
        status=room.status,
        is_active=room.is_active,
        created_at=room.created_at,
        started_at=room.started_at,
        ended_at=room.ended_at,
        participant_count=len(participants_list),
        message_count=len(messages),
        participants=participants_list
    )


@router.post(
    "/conversations/{room_id}/end",
    response_model=ConversationDetailsResponse,
    summary="End Conversation Session",
    description="Marks an active conversation as completed and records UTC ended_at timestamp."
)
async def end_conversation(
    room_id: str,
    db: AsyncSession = Depends(get_db)
):
    room = await ChatRepository.get_room_by_id(db, room_id=room_id)
    if not room:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{room_id}' not found."
        )

    updated_room = await ChatRepository.end_room_conversation(db, room_id=room_id)
    messages = await ChatRepository.get_room_messages(db, room_id=room_id, limit=1000)
    
    participants_list = [
        {
            "user_id": p.user_id,
            "display_name": p.user.display_name if p.user else p.user_id,
            "joined_at": p.joined_at,
            "last_seen_at": p.last_seen_at
        }
        for p in room.participants
    ]

    return ConversationDetailsResponse(
        conversation_id=updated_room.id,
        room_id=updated_room.id,
        name=updated_room.name,
        status=updated_room.status,
        is_active=updated_room.is_active,
        created_at=updated_room.created_at,
        started_at=updated_room.started_at,
        ended_at=updated_room.ended_at,
        participant_count=len(participants_list),
        message_count=len(messages),
        participants=participants_list
    )


@router.get(
    "/rooms/{room_id}/participants",
    summary="Get Active Room Participants",
    description="Returns list of currently connected participants in the WebSocket room."
)
async def get_active_participants(room_id: str):
    participants = manager.get_room_participants(room_id)
    return {"room_id": room_id, "participants": participants, "count": len(participants)}


# ---------------------------------------------------------
# WebSocket Real-Time Communication Endpoint
# ---------------------------------------------------------

@router.websocket("/ws/{room_id}/{user_id}")
async def websocket_chat_endpoint(
    websocket: WebSocket,
    room_id: str,
    user_id: str,
    display_name: str = Query(default="Anonymous")
):
    """
    Bidirectional WebSocket endpoint for two-user dyadic real-time messaging.
    Handles connection lifecycle, room joins, validation, DB persistence, and broadcasts.
    """
    # 1. Accept and register connection
    await manager.connect(websocket, room_id=room_id, user_id=user_id, display_name=display_name)

    # 2. Database Sync: Ensure user and room exist, record participant
    async with async_session_maker() as db:
        await ChatRepository.get_or_create_user(db, user_id=user_id, display_name=display_name)
        await ChatRepository.get_or_create_room(db, room_id=room_id)
        await ChatRepository.add_or_update_participant(db, room_id=room_id, user_id=user_id)
        
        # Load previous history for this user
        historical_msgs = await ChatRepository.get_room_messages(db, room_id=room_id, limit=50)
        history_payload = [
            {
                "message_id": m.id,
                "conversation_id": m.room_id,
                "room_id": m.room_id,
                "sender_id": m.sender_id,
                "sender_name": m.sender.display_name if m.sender else m.sender_id,
                "receiver_id": m.receiver_id,
                "message_text": m.message_text,
                "timestamp": m.timestamp.isoformat(),
                "message_type": m.message_type,
                "audio_url": m.audio_url,
                "audio_duration_seconds": m.audio_duration_seconds,
                "audio_format": m.audio_format,
                "audio_size_bytes": m.audio_size_bytes,
                "transcript_text": m.transcript_text,
                "transcription_status": m.transcription_status,
                "transcription_model": m.transcription_model,
                "transcription_language": m.transcription_language,
                "transcription_error": m.transcription_error,
                "transcribed_at": m.transcribed_at.isoformat() if m.transcribed_at else None
            }
            for m in historical_msgs
        ]

    # 3. Send initial history and room state to the newly connected user
    await manager.send_personal(websocket, {
        "event": "history",
        "data": {
            "room_id": room_id,
            "messages": history_payload,
            "participants": manager.get_room_participants(room_id)
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    # 4. Broadcast user_joined event to all participants in the room
    await manager.broadcast_to_room(room_id, {
        "event": "user_joined",
        "data": {
            "user_id": user_id,
            "display_name": display_name,
            "room_id": room_id,
            "participants": manager.get_room_participants(room_id),
            "system_message": f"{display_name} joined the room."
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    # 5. Continuous message receiving loop
    try:
        while True:
            raw_text = await websocket.receive_text()
            try:
                payload = json.loads(raw_text)
            except Exception:
                await manager.send_personal(websocket, {
                    "event": "error",
                    "data": {"detail": "Invalid JSON format payload."},
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                continue

            event_type = payload.get("event", "message")
            data = payload.get("data", payload)

            # Handle PING/PONG heartbeats
            if event_type == "ping":
                await manager.send_personal(websocket, {
                    "event": "pong",
                    "data": {},
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                continue

            # Handle Explicit Leave
            if event_type == "leave":
                break

            # Handle Real-Time Message
            if event_type == "message":
                msg_text = data.get("message_text", "")
                
                # Server-Side Validation: Empty message check
                if not isinstance(msg_text, str) or not msg_text.strip():
                    await manager.send_personal(websocket, {
                        "event": "error",
                        "data": {
                            "detail": "Validation Error: Message text cannot be empty or whitespace only.",
                            "field": "message_text"
                        },
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    continue

                clean_text = msg_text.strip()
                receiver_id = data.get("receiver_id")
                if not receiver_id:
                    # In a dyadic room, automatically resolve the recipient to the other participant if present
                    other_participants = [
                        p["user_id"] for p in manager.get_room_participants(room_id)
                        if p["user_id"] != user_id
                    ]
                    if other_participants:
                        receiver_id = other_participants[0]

                message_type = data.get("message_type", "text")
                audio_url = data.get("audio_url")
                audio_duration_seconds = data.get("audio_duration_seconds")
                audio_format = data.get("audio_format")
                audio_size_bytes = data.get("audio_size_bytes")
                msg_id = str(uuid.uuid4())
                utc_now = datetime.now(timezone.utc)

                # Persist to Database
                async with async_session_maker() as db:
                    saved_msg = await ChatRepository.save_message(
                        session=db,
                        room_id=room_id,
                        sender_id=user_id,
                        message_text=clean_text,
                        receiver_id=receiver_id,
                        message_type=message_type,
                        timestamp=utc_now,
                        message_id=msg_id,
                        audio_url=audio_url,
                        audio_duration_seconds=audio_duration_seconds,
                        audio_format=audio_format,
                        audio_size_bytes=audio_size_bytes
                    )

                # Prepare Standardized Message Payload
                broadcast_msg = {
                    "message_id": msg_id,
                    "conversation_id": room_id,
                    "room_id": room_id,
                    "sender_id": user_id,
                    "sender_name": display_name,
                    "receiver_id": receiver_id,
                    "message_text": clean_text,
                    "timestamp": utc_now.isoformat(),
                    "message_type": message_type,
                    "audio_url": audio_url,
                    "audio_duration_seconds": audio_duration_seconds,
                    "audio_format": audio_format,
                    "audio_size_bytes": audio_size_bytes
                }

                # Broadcast to all room participants
                await manager.broadcast_to_room(room_id, {
                    "event": "message",
                    "data": broadcast_msg,
                    "timestamp": utc_now.isoformat()
                })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for {user_id} in {room_id}")
    except Exception as exc:
        logger.warning(f"Unexpected WebSocket error for {user_id}: {exc}")
    finally:
        # 6. Disconnect cleanup and broadcast user_left event
        left_name = manager.disconnect(room_id, user_id) or display_name
        await manager.broadcast_to_room(room_id, {
            "event": "user_left",
            "data": {
                "user_id": user_id,
                "display_name": left_name,
                "room_id": room_id,
                "participants": manager.get_room_participants(room_id),
                "system_message": f"{left_name} left the room."
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
