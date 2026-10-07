import os
from datetime import datetime, timezone
from typing import Optional
from app.services.acoustic_features.extractors.base import BaseAcousticExtractor
from app.services.acoustic_features.models import (
    AcousticFeaturesResult,
    PitchFeatures,
    EnergyFeatures,
    SpeechRateFeatures,
    PauseFeatures,
    MFCCFeatures,
    SpectralFeatures
)
from app.services.acoustic_features.exceptions import (
    AudioFileNotFoundError,
    InvalidAudioFormatError
)


class MockAcousticExtractor(BaseAcousticExtractor):
    """
    Deterministic Mock Acoustic Extractor for unit tests, CI pipelines,
    and fast regression testing.
    """

    def __init__(self, is_silent: bool = False):
        self._extractor_name = "mock-acoustic-extractor-v1"
        self.is_silent = is_silent

    @property
    def extractor_name(self) -> str:
        return self._extractor_name

    async def extract(
        self,
        audio_path: str,
        sample_rate: Optional[int] = 16000
    ) -> AcousticFeaturesResult:
        if not os.path.exists(audio_path):
            raise AudioFileNotFoundError(f"Audio file '{audio_path}' does not exist on disk.")

        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            raise InvalidAudioFormatError(f"Audio file '{audio_path}' is empty (0 bytes).")

        sr = sample_rate or 16000
        duration = 4.0

        if self.is_silent:
            # Pure silence representation (unvoiced signal behavior)
            pitch = PitchFeatures(
                f0_mean_hz=None,
                f0_std_hz=None,
                f0_min_hz=None,
                f0_max_hz=None,
                f0_median_hz=None,
                f0_range_hz=None,
                voicing_fraction=0.0,
                f0_jitter_local=None
            )
            energy = EnergyFeatures(
                rms_mean=0.0,
                rms_std=0.0,
                rms_min=0.0,
                rms_max=0.0,
                rms_energy_range=0.0,
                shimmer_local=None
            )
            speech_rate = SpeechRateFeatures(
                estimated_syllable_count=0,
                speech_duration_seconds=0.0,
                estimated_speaking_rate_syllables_per_sec=0.0,
                estimated_articulation_rate_syllables_per_sec=None
            )
            pauses = PauseFeatures(
                pause_count=1,
                total_pause_duration_seconds=duration,
                mean_pause_duration_seconds=duration,
                max_pause_duration_seconds=duration,
                pause_to_speech_ratio=None
            )
            mfcc = MFCCFeatures(
                mfcc_means=[0.0] * 13,
                mfcc_stds=[0.0] * 13,
                mfcc_dimension=26
            )
            spectral = SpectralFeatures(
                spectral_centroid_mean=0.0,
                spectral_centroid_std=0.0,
                spectral_bandwidth_mean=0.0,
                spectral_bandwidth_std=0.0,
                spectral_rolloff_mean=0.0,
                spectral_rolloff_std=0.0,
                spectral_flatness_mean=0.0,
                spectral_flatness_std=0.0,
                zcr_mean=0.0,
                zcr_std=0.0
            )
            has_voiced = False
        else:
            # Deterministic standard speech profile
            pitch = PitchFeatures(
                f0_mean_hz=175.4,
                f0_std_hz=24.8,
                f0_min_hz=112.0,
                f0_max_hz=248.5,
                f0_median_hz=172.1,
                f0_range_hz=136.5,
                voicing_fraction=0.72,
                f0_jitter_local=0.014
            )
            energy = EnergyFeatures(
                rms_mean=0.0482,
                rms_std=0.0195,
                rms_min=0.0012,
                rms_max=0.0984,
                rms_energy_range=0.0972,
                shimmer_local=0.038
            )
            speech_rate = SpeechRateFeatures(
                estimated_syllable_count=14,
                speech_duration_seconds=3.2,
                estimated_speaking_rate_syllables_per_sec=3.5,
                estimated_articulation_rate_syllables_per_sec=4.375
            )
            pauses = PauseFeatures(
                pause_count=2,
                total_pause_duration_seconds=0.8,
                mean_pause_duration_seconds=0.4,
                max_pause_duration_seconds=0.55,
                pause_to_speech_ratio=0.25
            )
            mfcc_means = [-250.4, 98.2, -18.5, 34.2, -8.1, 14.3, -5.2, 8.4, -3.1, 4.2, -2.0, 1.8, -0.9]
            mfcc_stds = [45.2, 22.1, 14.8, 11.2, 9.4, 7.8, 6.5, 5.8, 4.9, 4.2, 3.8, 3.2, 2.9]
            mfcc = MFCCFeatures(
                mfcc_means=mfcc_means,
                mfcc_stds=mfcc_stds,
                mfcc_dimension=26
            )
            spectral = SpectralFeatures(
                spectral_centroid_mean=1840.5,
                spectral_centroid_std=420.2,
                spectral_bandwidth_mean=1980.4,
                spectral_bandwidth_std=310.8,
                spectral_rolloff_mean=3650.0,
                spectral_rolloff_std=820.4,
                spectral_flatness_mean=0.0182,
                spectral_flatness_std=0.0094,
                zcr_mean=0.068,
                zcr_std=0.032
            )
            has_voiced = True

        raw_vec = AcousticFeaturesResult.build_raw_feature_vector(
            pitch=pitch,
            energy=energy,
            speech_rate=speech_rate,
            pauses=pauses,
            mfcc=mfcc,
            spectral=spectral
        )

        return AcousticFeaturesResult(
            pitch=pitch,
            energy=energy,
            speech_rate=speech_rate,
            pauses=pauses,
            mfcc=mfcc,
            spectral=spectral,
            duration_seconds=duration,
            sample_rate=sr,
            channels=1,
            engine=self._extractor_name,
            has_voiced_frames=has_voiced,
            feature_vector_version="v1.0-59d",
            raw_feature_vector=raw_vec,
            extracted_at=datetime.now(timezone.utc)
        )
