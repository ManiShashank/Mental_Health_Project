from datetime import datetime, timezone
from fastapi import APIRouter, status
from app.core.config import settings
from app.schemas.health import HealthResponse, SystemInfoResponse

router = APIRouter(tags=["System & Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="System Health Check",
    description="Returns backend service operational status, current timestamp, and environment metadata."
)
async def get_health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc),
        version=settings.VERSION,
        environment=settings.ENVIRONMENT
    )


@router.get(
    "/info",
    response_model=SystemInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="System & Academic Metadata",
    description="Returns research metadata, supported modalities, baseline statistical approach, and mandatory non-diagnostic disclaimers."
)
async def get_system_info() -> SystemInfoResponse:
    return SystemInfoResponse(
        project_title=settings.PROJECT_NAME,
        system_type=settings.SYSTEM_TYPE,
        disclaimer=settings.CLINICAL_DISCLAIMER,
        version=settings.VERSION,
        modalities_supported=[
            "Modality 1: Text & Language (Lexical TF-IDF, BERT embeddings, VADER/Transformer sentiment, GoEmotions, LDA topics, Linguistic stats)",
            "Modality 2: Voice & Acoustics (Path A: Whisper ASR speech-to-text transcript; Path B: Librosa pitch F0, RMS energy, speech rate, pause duration, MFCCs, spectral flux)",
            "Modality 3: Communication Dynamics (Turn-taking latency, message cadence, participation ratio, initiation tracking)"
        ],
        baseline_paradigm="Personalized Longitudinal Z-Score Deviation (Individual mean mu and standard deviation sigma)"
    )
