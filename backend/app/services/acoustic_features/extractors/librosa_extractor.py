import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Optional, List, Tuple
import numpy as np

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
    InvalidAudioFormatError,
    AudioSignalProcessingError
)

logger = logging.getLogger("acoustic_librosa")


class LibrosaAcousticExtractor(BaseAcousticExtractor):
    """
    Production Acoustic Feature Extractor utilizing Librosa, SoundFile, and SciPy.
    Extracts all 6 acoustic feature families and builds the exact 59-dimensional
    raw feature vector (v1.0-59d) asynchronously via a threadpool executor.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        fmin: float = 65.0,
        fmax: float = 500.0,
        silence_threshold_db: float = -30.0,
        min_pause_duration: float = 0.15
    ):
        self._extractor_name = "librosa-v1.0.0"
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.fmin = fmin
        self.fmax = fmax
        self.silence_threshold_db = silence_threshold_db
        self.min_pause_duration = min_pause_duration

    @property
    def extractor_name(self) -> str:
        return self._extractor_name

    def _sync_extract(self, audio_path: str, target_sr: int) -> AcousticFeaturesResult:
        try:
            import librosa
            import scipy.signal
        except ImportError as e:
            raise AudioSignalProcessingError(f"Missing required scientific dependencies for Librosa extraction: {e}")

        try:
            y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        except Exception as exc:
            logger.error(f"Failed to load audio file '{audio_path}': {exc}")
            raise InvalidAudioFormatError(f"Cannot decode audio file '{os.path.basename(audio_path)}': {str(exc)}")

        if len(y) == 0:
            raise InvalidAudioFormatError(f"Audio file '{os.path.basename(audio_path)}' contains no audio frames (0 samples).")

        duration_seconds = float(len(y) / sr)

        # -------------------------------------------------------------
        # 1. RMS Energy Envelope & Shimmer
        # -------------------------------------------------------------
        frame_length = 2048
        hop_length = 512
        rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        if len(rms_frames) == 0:
            rms_frames = np.array([0.0])

        rms_mean = float(np.mean(rms_frames))
        rms_std = float(np.std(rms_frames))
        rms_min = float(np.min(rms_frames))
        rms_max = float(np.max(rms_frames))
        rms_range = float(rms_max - rms_min)

        # Shimmer (relative frame amplitude perturbation)
        if len(rms_frames) > 1 and rms_mean > 1e-5:
            diff_rms = np.abs(rms_frames[1:] - rms_frames[:-1])
            shimmer_local = float(np.mean(diff_rms) / rms_mean)
        else:
            shimmer_local = None

        energy_feat = EnergyFeatures(
            rms_mean=rms_mean,
            rms_std=rms_std,
            rms_min=rms_min,
            rms_max=rms_max,
            rms_energy_range=rms_range,
            shimmer_local=shimmer_local
        )

        # -------------------------------------------------------------
        # 2. Pitch (F0) Statistics via pYIN
        # -------------------------------------------------------------
        # For very short signals (< 2048 samples), pad signal to allow STFT/pYIN framing
        if len(y) < frame_length:
            y_padded = np.pad(y, (0, frame_length - len(y)), mode='constant')
        else:
            y_padded = y

        try:
            f0_arr, voiced_flag, voiced_probs = librosa.pyin(
                y_padded,
                fmin=self.fmin,
                fmax=self.fmax,
                sr=sr,
                frame_length=frame_length,
                hop_length=hop_length
            )
        except Exception as exc:
            logger.warning(f"pYIN extraction error on '{audio_path}': {exc}. Falling back to unvoiced.")
            f0_arr = np.array([np.nan])

        voiced_f0 = f0_arr[~np.isnan(f0_arr)] if f0_arr is not None else np.array([])
        total_f0_frames = len(f0_arr) if f0_arr is not None and len(f0_arr) > 0 else 1

        if len(voiced_f0) > 0:
            f0_mean = float(np.mean(voiced_f0))
            f0_std = float(np.std(voiced_f0)) if len(voiced_f0) > 1 else 0.0
            f0_min = float(np.min(voiced_f0))
            f0_max = float(np.max(voiced_f0))
            f0_median = float(np.median(voiced_f0))
            f0_range = float(f0_max - f0_min)
            voicing_fraction = float(len(voiced_f0) / total_f0_frames)

            # Local Jitter (cycle-to-cycle period perturbation)
            if len(voiced_f0) > 1:
                periods = 1.0 / voiced_f0
                diff_periods = np.abs(periods[1:] - periods[:-1])
                f0_jitter = float(np.mean(diff_periods) / np.mean(periods))
            else:
                f0_jitter = None

            pitch_feat = PitchFeatures(
                f0_mean_hz=f0_mean,
                f0_std_hz=f0_std,
                f0_min_hz=f0_min,
                f0_max_hz=f0_max,
                f0_median_hz=f0_median,
                f0_range_hz=f0_range,
                voicing_fraction=voicing_fraction,
                f0_jitter_local=f0_jitter
            )
            has_voiced = True
        else:
            # Unvoiced or silent signal: do not fabricate values, return None
            pitch_feat = PitchFeatures(
                f0_mean_hz=None,
                f0_std_hz=None,
                f0_min_hz=None,
                f0_max_hz=None,
                f0_median_hz=None,
                f0_range_hz=None,
                voicing_fraction=0.0,
                f0_jitter_local=None
            )
            has_voiced = False

        # -------------------------------------------------------------
        # 3. Pause Dynamics (-30 dBFS threshold, >= 150ms minimum duration)
        # -------------------------------------------------------------
        ref_peak = rms_max if rms_max > 1e-7 else 1.0
        db_frames = 20.0 * np.log10((rms_frames + 1e-7) / (ref_peak + 1e-7))
        silent_frames = db_frames < self.silence_threshold_db

        frame_duration = hop_length / sr  # 0.032 seconds
        min_consecutive_frames = max(1, int(np.ceil(self.min_pause_duration / frame_duration)))

        # Find contiguous silent segments
        pause_durations: List[float] = []
        current_silent_run = 0

        for is_sil in silent_frames:
            if is_sil:
                current_silent_run += 1
            else:
                if current_silent_run >= min_consecutive_frames:
                    pause_durations.append(current_silent_run * frame_duration)
                current_silent_run = 0

        if current_silent_run >= min_consecutive_frames:
            pause_durations.append(current_silent_run * frame_duration)

        pause_count = len(pause_durations)
        total_pause_duration = float(sum(pause_durations))
        # Cap pause duration at total duration
        total_pause_duration = min(total_pause_duration, duration_seconds)
        mean_pause_duration = float(np.mean(pause_durations)) if pause_count > 0 else 0.0
        max_pause_duration = float(np.max(pause_durations)) if pause_count > 0 else 0.0

        active_speech_duration = max(0.0, duration_seconds - total_pause_duration)
        pause_to_speech_ratio = (
            float(total_pause_duration / active_speech_duration)
            if active_speech_duration > 0.05
            else None
        )

        pause_feat = PauseFeatures(
            pause_count=pause_count,
            total_pause_duration_seconds=total_pause_duration,
            mean_pause_duration_seconds=mean_pause_duration,
            max_pause_duration_seconds=max_pause_duration,
            pause_to_speech_ratio=pause_to_speech_ratio
        )

        # -------------------------------------------------------------
        # 4. Speech Rate & Syllable Nucleus Estimation
        # -------------------------------------------------------------
        # Detect vocalic energy peaks on smoothed intensity envelope
        min_peak_dist = max(1, int(0.12 * sr / hop_length))  # minimum 120ms between syllables
        min_peak_height = max(1e-4, rms_mean + 0.1 * rms_std)

        peaks, _ = scipy.signal.find_peaks(
            rms_frames,
            height=min_peak_height,
            distance=min_peak_dist
        )
        syllable_count = int(len(peaks))
        speaking_rate = float(syllable_count / duration_seconds) if duration_seconds > 0 else 0.0
        articulation_rate = (
            float(syllable_count / active_speech_duration)
            if active_speech_duration > 0.05
            else None
        )

        speech_rate_feat = SpeechRateFeatures(
            estimated_syllable_count=syllable_count,
            speech_duration_seconds=active_speech_duration,
            estimated_speaking_rate_syllables_per_sec=speaking_rate,
            estimated_articulation_rate_syllables_per_sec=articulation_rate
        )

        # -------------------------------------------------------------
        # 5. MFCC Features (13 Coefficients)
        # -------------------------------------------------------------
        try:
            mfcc_mat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, n_mels=40)
            mfcc_means = [float(np.mean(mfcc_mat[i])) for i in range(self.n_mfcc)]
            mfcc_stds = [float(np.std(mfcc_mat[i])) for i in range(self.n_mfcc)]
        except Exception as exc:
            logger.warning(f"MFCC extraction fallback: {exc}")
            mfcc_means = [0.0] * self.n_mfcc
            mfcc_stds = [0.0] * self.n_mfcc

        mfcc_feat = MFCCFeatures(
            mfcc_means=mfcc_means,
            mfcc_stds=mfcc_stds,
            mfcc_dimension=26
        )

        # -------------------------------------------------------------
        # 6. Spectral Descriptors
        # -------------------------------------------------------------
        try:
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
            spec_flat = librosa.feature.spectral_flatness(y=y)[0]
            zcr_arr = librosa.feature.zero_crossing_rate(y)[0]

            spectral_feat = SpectralFeatures(
                spectral_centroid_mean=float(np.mean(spec_cent)),
                spectral_centroid_std=float(np.std(spec_cent)),
                spectral_bandwidth_mean=float(np.mean(spec_bw)),
                spectral_bandwidth_std=float(np.std(spec_bw)),
                spectral_rolloff_mean=float(np.mean(spec_roll)),
                spectral_rolloff_std=float(np.std(spec_roll)),
                spectral_flatness_mean=float(np.mean(spec_flat)),
                spectral_flatness_std=float(np.std(spec_flat)),
                zcr_mean=float(np.mean(zcr_arr)),
                zcr_std=float(np.std(zcr_arr))
            )
        except Exception as exc:
            logger.warning(f"Spectral descriptor fallback: {exc}")
            spectral_feat = SpectralFeatures(
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

        # -------------------------------------------------------------
        # Build Standardized 59-Dimensional Raw Feature Vector
        # -------------------------------------------------------------
        raw_vec = AcousticFeaturesResult.build_raw_feature_vector(
            pitch=pitch_feat,
            energy=energy_feat,
            speech_rate=speech_rate_feat,
            pauses=pause_feat,
            mfcc=mfcc_feat,
            spectral=spectral_feat
        )

        return AcousticFeaturesResult(
            pitch=pitch_feat,
            energy=energy_feat,
            speech_rate=speech_rate_feat,
            pauses=pause_feat,
            mfcc=mfcc_feat,
            spectral=spectral_feat,
            duration_seconds=duration_seconds,
            sample_rate=sr,
            channels=1,
            engine=self._extractor_name,
            has_voiced_frames=has_voiced,
            feature_vector_version="v1.0-59d",
            raw_feature_vector=raw_vec,
            extracted_at=datetime.now(timezone.utc)
        )

    async def extract(
        self,
        audio_path: str,
        sample_rate: Optional[int] = None
    ) -> AcousticFeaturesResult:
        if not os.path.exists(audio_path):
            raise AudioFileNotFoundError(f"Audio file '{audio_path}' does not exist on disk.")

        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            raise InvalidAudioFormatError(f"Audio file '{audio_path}' is empty (0 bytes).")

        target_sr = sample_rate or self.sample_rate

        # Offload CPU/DSP bound calculations to default threadpool executor
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_extract, audio_path, target_sr)
