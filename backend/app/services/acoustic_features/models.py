from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


def get_utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AcousticAnalysisStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class PitchFeatures(BaseModel):
    """Pitch (F0) Statistics extracted via pYIN or autocorrelation."""
    model_config = ConfigDict(from_attributes=True)

    f0_mean_hz: Optional[float] = Field(None, description="Mean fundamental frequency across voiced frames (Hz)")
    f0_std_hz: Optional[float] = Field(None, description="Pitch standard deviation (Hz)")
    f0_min_hz: Optional[float] = Field(None, description="Minimum voiced pitch (Hz)")
    f0_max_hz: Optional[float] = Field(None, description="Maximum voiced pitch (Hz)")
    f0_median_hz: Optional[float] = Field(None, description="Median fundamental frequency (Hz)")
    f0_range_hz: Optional[float] = Field(None, description="Pitch dynamic spread (f0_max - f0_min, Hz)")
    voicing_fraction: float = Field(..., description="Ratio of voiced frames to total frames (0.0 to 1.0)")
    f0_jitter_local: Optional[float] = Field(None, description="Local cycle-to-cycle pitch period perturbation ratio")


class EnergyFeatures(BaseModel):
    """RMS Energy Statistics and Amplitude Shimmer."""
    model_config = ConfigDict(from_attributes=True)

    rms_mean: float = Field(..., description="Mean frame RMS energy (linear amplitude)")
    rms_std: float = Field(..., description="RMS energy variability")
    rms_min: float = Field(..., description="Minimum frame RMS energy")
    rms_max: float = Field(..., description="Peak frame RMS energy")
    rms_energy_range: float = Field(..., description="Dynamic energy range (rms_max - rms_min)")
    shimmer_local: Optional[float] = Field(None, description="Local frame-to-frame amplitude perturbation ratio")


class SpeechRateFeatures(BaseModel):
    """Speech Rate and Syllable Rhythm Estimates (Explicitly Labeled as Estimated)."""
    model_config = ConfigDict(from_attributes=True)

    estimated_syllable_count: int = Field(..., description="Estimated vocalic energy peak count")
    speech_duration_seconds: float = Field(..., description="Total active vocalization time excluding pauses (seconds)")
    estimated_speaking_rate_syllables_per_sec: float = Field(..., description="Estimated speaking rate over total audio duration (syllables/s)")
    estimated_articulation_rate_syllables_per_sec: Optional[float] = Field(None, description="Estimated articulation rate over active speech time (syllables/s)")


class PauseFeatures(BaseModel):
    """Pause Detection and Silence Duration Statistics."""
    model_config = ConfigDict(from_attributes=True)

    pause_count: int = Field(..., description="Number of detected silence intervals (>= 150ms)")
    total_pause_duration_seconds: float = Field(..., description="Cumulative duration of all pauses (seconds)")
    mean_pause_duration_seconds: float = Field(..., description="Average pause length (seconds)")
    max_pause_duration_seconds: float = Field(..., description="Duration of longest single pause (seconds)")
    pause_to_speech_ratio: Optional[float] = Field(None, description="Ratio of total pause duration to active speech duration")


class MFCCFeatures(BaseModel):
    """13-Coefficient Mel-Frequency Cepstral Representation."""
    model_config = ConfigDict(from_attributes=True)

    mfcc_means: List[float] = Field(..., description="Means of the 13 MFCC coefficients across time frames")
    mfcc_stds: List[float] = Field(..., description="Standard deviations of the 13 MFCC coefficients across time frames")
    mfcc_dimension: int = Field(default=26, description="Total MFCC descriptor dimension (13 means + 13 stds)")


class SpectralFeatures(BaseModel):
    """Spectral Shape and Timbre Descriptors."""
    model_config = ConfigDict(from_attributes=True)

    spectral_centroid_mean: float = Field(..., description="Mean spectral center of mass (Hz)")
    spectral_centroid_std: float = Field(..., description="Spectral centroid standard deviation (Hz)")
    spectral_bandwidth_mean: float = Field(..., description="Mean spectral spread (Hz)")
    spectral_bandwidth_std: float = Field(..., description="Spectral bandwidth standard deviation (Hz)")
    spectral_rolloff_mean: float = Field(..., description="Mean 85% energy frequency cutoff (Hz)")
    spectral_rolloff_std: float = Field(..., description="Spectral rolloff standard deviation (Hz)")
    spectral_flatness_mean: float = Field(..., description="Mean spectral flatness (0.0 = pure tone, 1.0 = white noise)")
    spectral_flatness_std: float = Field(..., description="Spectral flatness standard deviation")
    zcr_mean: float = Field(..., description="Mean Zero Crossing Rate per frame (0.0 to 1.0)")
    zcr_std: float = Field(..., description="Zero Crossing Rate standard deviation")


class AcousticFeaturesResult(BaseModel):
    """
    Complete container DTO grouping all 6 acoustic feature families and
    the deterministic 59-dimensional raw feature vector (v1.0-59d) for Milestone 8 fusion.
    """
    model_config = ConfigDict(from_attributes=True)

    pitch: PitchFeatures
    energy: EnergyFeatures
    speech_rate: SpeechRateFeatures
    pauses: PauseFeatures
    mfcc: MFCCFeatures
    spectral: SpectralFeatures

    duration_seconds: float = Field(..., description="Total audio signal duration (seconds)")
    sample_rate: int = Field(default=16000, description="Audio sampling rate (Hz)")
    channels: int = Field(default=1, description="Number of audio channels (1 = mono)")
    engine: str = Field(..., description="Acoustic feature extraction engine identifier")
    has_voiced_frames: bool = Field(default=True, description="Whether voiced speech frames were detected")
    feature_vector_version: str = Field(default="v1.0-59d", description="Standardized feature vector schema version")
    raw_feature_vector: List[float] = Field(..., description="Exactly 59-dimensional ordered numeric feature vector")
    extracted_at: datetime = Field(default_factory=get_utc_now, description="UTC timestamp of extraction")

    @classmethod
    def build_raw_feature_vector(
        cls,
        pitch: PitchFeatures,
        energy: EnergyFeatures,
        speech_rate: SpeechRateFeatures,
        pauses: PauseFeatures,
        mfcc: MFCCFeatures,
        spectral: SpectralFeatures
    ) -> List[float]:
        """
        Constructs the exact 59-dimensional deterministic raw feature vector (v1.0-59d)
        for future Milestone 8 Multimodal Feature Fusion.
        Indices 00..58 strictly correspond to the specification in implementation_plan.md.
        """
        vec: List[float] = [
            # [00..07] Pitch Statistics (8 dims)
            float(pitch.f0_mean_hz or 0.0),
            float(pitch.f0_std_hz or 0.0),
            float(pitch.f0_min_hz or 0.0),
            float(pitch.f0_max_hz or 0.0),
            float(pitch.f0_median_hz or 0.0),
            float(pitch.f0_range_hz or 0.0),
            float(pitch.voicing_fraction),
            float(pitch.f0_jitter_local or 0.0),

            # [08..13] RMS Energy & Shimmer Statistics (6 dims)
            float(energy.rms_mean),
            float(energy.rms_std),
            float(energy.rms_min),
            float(energy.rms_max),
            float(energy.rms_energy_range),
            float(energy.shimmer_local or 0.0),

            # [14..17] Speech Rate Estimates (4 dims)
            float(speech_rate.estimated_syllable_count),
            float(speech_rate.speech_duration_seconds),
            float(speech_rate.estimated_speaking_rate_syllables_per_sec),
            float(speech_rate.estimated_articulation_rate_syllables_per_sec or 0.0),

            # [18..22] Pause Dynamics (5 dims)
            float(pauses.pause_count),
            float(pauses.total_pause_duration_seconds),
            float(pauses.mean_pause_duration_seconds),
            float(pauses.max_pause_duration_seconds),
            float(pauses.pause_to_speech_ratio or 0.0),
        ]

        # [23..35] MFCC Means (13 dims: MFCC 1 to 13)
        for val in mfcc.mfcc_means[:13]:
            vec.append(float(val))

        # [36..48] MFCC Stds (13 dims: MFCC 1 to 13)
        for val in mfcc.mfcc_stds[:13]:
            vec.append(float(val))

        # [49..58] Spectral Descriptors (10 dims)
        vec.extend([
            float(spectral.spectral_centroid_mean),
            float(spectral.spectral_centroid_std),
            float(spectral.spectral_bandwidth_mean),
            float(spectral.spectral_bandwidth_std),
            float(spectral.spectral_rolloff_mean),
            float(spectral.spectral_rolloff_std),
            float(spectral.spectral_flatness_mean),
            float(spectral.spectral_flatness_std),
            float(spectral.zcr_mean),
            float(spectral.zcr_std),
        ])

        if len(vec) != 59:
            raise ValueError(f"Acoustic raw feature vector must have exactly 59 elements, got {len(vec)}")

        return vec
