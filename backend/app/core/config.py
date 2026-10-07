from typing import List, Union
from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application Settings configured via environment variables and .env file.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    PROJECT_NAME: str = (
        "Multimodal Analysis of Communication and Language Patterns for Personalized Mental Health Risk Screening"
    )
    SYSTEM_TYPE: str = "Analytical Pre-Screening Research Platform"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DATABASE_URL: str = "sqlite+aiosqlite:///./mental_health.db"

    # Non-diagnostic mandatory regulatory disclaimer
    CLINICAL_DISCLAIMER: str = (
        "This platform is an analytical and pre-screening research system. "
        "It does NOT provide clinical diagnoses or replace professional medical/psychiatric evaluation. "
        "System outputs describe observed communication, language, and acoustic patterns and deviations "
        "relative to an individual's personal baseline."
    )

    # Milestone 4: Audio Storage Settings
    AUDIO_UPLOAD_DIR: str = "storage/audio"
    MAX_AUDIO_SIZE_BYTES: int = 15 * 1024 * 1024  # 15 MB
    ALLOWED_AUDIO_FORMATS: List[str] = [
        "audio/webm",
        "audio/wav",
        "audio/x-wav",
        "audio/ogg",
        "audio/mp4",
        "audio/m4a",
        "audio/x-m4a",
        "audio/aac",
        "audio/mpeg",
        "audio/mp3"
    ]

    # Milestone 5: Speech-to-Text (STT) Settings
    STT_PROVIDER: str = "whisper"
    STT_MODEL_NAME: str = "base"
    STT_AUTO_TRANSCRIBE: bool = False

    # Milestone 6: Acoustic Feature Extraction Settings
    ACOUSTIC_EXTRACTOR_PROVIDER: str = "librosa"  # "librosa" | "mock"
    ACOUSTIC_SAMPLE_RATE: int = 16000
    ACOUSTIC_N_MFCC: int = 13
    ACOUSTIC_FMIN: float = 65.0
    ACOUSTIC_FMAX: float = 500.0
    ACOUSTIC_SILENCE_THRESHOLD_DB: float = -30.0
    ACOUSTIC_MIN_PAUSE_DURATION: float = 0.15

    # Milestone 7: NLP & Linguistic Feature Extraction Settings
    NLP_PROVIDER: str = "rule_lexical"  # "rule_lexical" | "transformer" | "mock"
    NLP_EMBEDDING_MODEL: str = "distilbert-base-uncased"
    NLP_EMOTION_MODEL: str = "bhadresh-psav/roberta-base-go_emotions"
    NLP_TFIDF_MAX_FEATURES: int = 20
    NLP_LDA_TOPICS: int = 5

    # CORS settings
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, list):
            return v
        return []


settings = Settings()
