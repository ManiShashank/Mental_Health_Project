"""
Milestone 9: Pure Mathematical Engine for Personal Baseline, Regularized Z-Scores,
Modality-Mask-Aware Trajectories (SMA, EMA, OLS, RoC).
"""
from typing import List, Dict, Tuple, Optional, Any
import math
import numpy as np

from app.services.personal_baseline.models import (
    CANONICAL_FEATURE_REGISTRY,
    INDEX_TO_FEATURE_NAME,
    TopDeviationItem
)


class PersonalBaselineEngine:
    """
    Stateless, vectorized mathematical engine providing Bessel-corrected sample statistics,
    two-branch zero-variance regularized z-scores, modality-mask-aware moving averages,
    OLS trend regression, and rate-of-change dynamics.
    """
    EPSILON: float = 1e-6
    OLS_MIN_TIME_VAR: float = 1e-8
    ROC_MIN_TIME_DELTA: float = 1e-6  # in hours (~0.0036 seconds)

    @classmethod
    def compute_baseline_statistics(
        cls,
        raw_vectors: List[List[float]],
        normalized_vectors: List[List[float]],
        masks: List[List[float]]
    ) -> Tuple[List[float], List[float], List[float], List[float], List[int]]:
        """
        Computes Bessel-corrected empirical mean and sample standard deviation vectors
        across valid observations for each of the 140 dimensions.
        Returns:
            (raw_mean_vec, raw_std_vec, norm_mean_vec, norm_std_vec, per_feature_counts)
        """
        N = len(raw_vectors)
        if N == 0:
            zeros = [0.0] * 140
            return zeros, zeros, zeros, zeros, [0] * 140

        R = np.array(raw_vectors, dtype=np.float64)        # (N, 140)
        Norm = np.array(normalized_vectors, dtype=np.float64) # (N, 140)
        M = np.array(masks, dtype=np.float64)              # (N, 140)

        # per-feature valid observation counts
        counts = np.sum(M, axis=0).astype(np.int32)        # (140,)

        raw_mean = np.zeros(140, dtype=np.float64)
        raw_std = np.zeros(140, dtype=np.float64)
        norm_mean = np.zeros(140, dtype=np.float64)
        norm_std = np.zeros(140, dtype=np.float64)

        for i in range(140):
            n_i = counts[i]
            if n_i >= 1:
                valid_mask = (M[:, i] == 1.0)
                valid_raw = R[valid_mask, i]
                valid_norm = Norm[valid_mask, i]

                raw_mean[i] = np.mean(valid_raw)
                norm_mean[i] = np.mean(valid_norm)

                if n_i >= 2:
                    # Bessel-corrected sample standard deviation (ddof=1)
                    raw_std[i] = np.std(valid_raw, ddof=1)
                    norm_std[i] = np.std(valid_norm, ddof=1)
                else:
                    raw_std[i] = 0.0
                    norm_std[i] = 0.0

        raw_mean_list = [round(float(v), 6) for v in raw_mean]
        raw_std_list = [round(float(v), 6) for v in raw_std]
        norm_mean_list = [round(float(v), 6) for v in norm_mean]
        norm_std_list = [round(float(v), 6) for v in norm_std]
        counts_list = counts.tolist()

        return raw_mean_list, raw_std_list, norm_mean_list, norm_std_list, counts_list

    @classmethod
    def compute_zscores(
        cls,
        current_vector: List[float],
        current_mask: List[float],
        baseline_mean: List[float],
        baseline_std: List[float],
        per_feature_counts: List[int]
    ) -> Tuple[List[float], List[float], List[bool], float, float, float]:
        """
        Computes regularized z-scores over valid behavioral dimensions.
        Returns:
            (raw_zscores, clipped_zscores, validity_mask, mean_abs_z, rms_z, std_euclidean)
        """
        raw_z: List[float] = [0.0] * 140
        clipped_z: List[float] = [0.0] * 140
        valid_mask: List[bool] = [False] * 140

        valid_z_scores: List[float] = []

        for i in range(140):
            # Exclude mask flags [135..139] from behavioral deviation scoring
            if i >= 135:
                valid_mask[i] = False
                continue

            is_valid_obs = (current_mask[i] == 1.0)
            has_sufficient_history = (per_feature_counts[i] >= 2)

            if is_valid_obs and has_sufficient_history:
                valid_mask[i] = True
                x = float(current_vector[i])
                mu = float(baseline_mean[i])
                sigma = float(baseline_std[i])

                if sigma > cls.EPSILON:
                    z = (x - mu) / sigma
                else:
                    # Historical zero-variance regularization
                    diff = x - mu
                    if abs(diff) <= cls.EPSILON:
                        z = 0.0
                    else:
                        z = diff / cls.EPSILON

                # Check finiteness
                if math.isnan(z) or math.isinf(z):
                    z = 0.0

                z_round = round(float(z), 6)
                z_clip = round(float(np.clip(z_round, -5.0, 5.0)), 6)

                raw_z[i] = z_round
                clipped_z[i] = z_clip
                valid_z_scores.append(z_round)
            else:
                valid_mask[i] = False
                raw_z[i] = 0.0
                clipped_z[i] = 0.0

        if len(valid_z_scores) > 0:
            arr = np.array(valid_z_scores, dtype=np.float64)
            mean_abs = float(np.mean(np.abs(arr)))
            rms = float(np.sqrt(np.mean(arr ** 2)))
            std_euc = float(np.sqrt(np.sum(arr ** 2)))
        else:
            mean_abs = 0.0
            rms = 0.0
            std_euc = 0.0

        return (
            raw_z,
            clipped_z,
            valid_mask,
            round(mean_abs, 6),
            round(rms, 6),
            round(std_euc, 6)
        )

    @classmethod
    def extract_top_deviations(
        cls,
        raw_zscores: List[float],
        clipped_zscores: List[float],
        validity_mask: List[bool],
        current_raw_vector: List[float],
        baseline_raw_mean: List[float],
        baseline_raw_std: List[float],
        limit: int = 5
    ) -> List[TopDeviationItem]:
        """
        Extracts top deviated features sorted by absolute z-score descending, index ascending.
        """
        candidates = []
        for i in range(135):  # Exclude mask flags
            if validity_mask[i]:
                abs_z = abs(raw_zscores[i])
                meta = CANONICAL_FEATURE_REGISTRY[i]
                candidates.append((
                    abs_z,
                    i,
                    TopDeviationItem(
                        index=i,
                        name=meta.name,
                        modality=meta.modality,
                        raw_zscore=raw_zscores[i],
                        clipped_zscore=clipped_zscores[i],
                        absolute_zscore=round(abs_z, 6),
                        raw_value=round(float(current_raw_vector[i]), 6),
                        baseline_mean=round(float(baseline_raw_mean[i]), 6),
                        baseline_std=round(float(baseline_raw_std[i]), 6)
                    )
                ))

        # Sort descending by |z|, ascending by index
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in candidates[:limit]]

    @classmethod
    def compute_temporal_dynamics(
        cls,
        timestamps: List[float],  # epoch seconds
        raw_series: List[List[float]],
        norm_series: List[List[float]],
        mask_series: List[List[float]]
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Computes modality-mask-aware moving averages (SMA-3, SMA-5, EMA-3, EMA-5),
        OLS linear trend slopes (raw & normalized), and rate of change.
        Returns:
            (moving_averages_summary, trend_slopes_summary, rate_of_change_summary)
        """
        K = len(timestamps)
        if K == 0:
            return {}, {}, {}

        t0 = timestamps[0]
        # Convert timestamps to elapsed hours
        tau = [(t - t0) / 3600.0 for t in timestamps]

        R = np.array(raw_series, dtype=np.float64)       # (K, 140)
        Norm = np.array(norm_series, dtype=np.float64)    # (K, 140)
        M = np.array(mask_series, dtype=np.float64)      # (K, 140)

        # Select key indicative behavioral dimensions for trend summaries
        key_indices = [
            0,   # turn_index
            1,   # sender_message_ratio
            2,   # response_latency_seconds
            5,   # message_char_length
            6,   # message_word_length
            10,  # f0_mean_hz
            18,  # rms_mean
            26,  # speaking_rate_estimated
            28,  # pause_count
            69,  # char_count
            70,  # word_count
            72,  # type_token_ratio_ttr
            79,  # vader_compound
            87,  # emotion_joy
            88,  # emotion_sadness
            89,  # emotion_anger
            90,  # emotion_fear
        ]

        sma_summary = {}
        slopes_summary = {}
        roc_summary = {}

        alpha_3 = 2.0 / (3.0 + 1.0)  # 0.5
        alpha_5 = 2.0 / (5.0 + 1.0)  # 1/3

        for idx in key_indices:
            feat_name = INDEX_TO_FEATURE_NAME.get(idx, f"feat_{idx}")
            valid_k = [k for k in range(K) if M[k, idx] == 1.0]
            K_i = len(valid_k)

            if K_i == 0:
                sma_summary[feat_name] = {"valid": False, "K_i": 0}
                slopes_summary[feat_name] = {"valid": False, "K_i": 0}
                roc_summary[feat_name] = {"valid": False, "K_i": 0}
                continue

            valid_norm_vals = [Norm[k, idx] for k in valid_k]
            valid_raw_vals = [R[k, idx] for k in valid_k]
            valid_tau = [tau[k] for k in valid_k]

            # 1. Moving Averages (Normalized)
            sma_3 = float(np.mean(valid_norm_vals[-min(3, K_i):]))
            sma_5 = float(np.mean(valid_norm_vals[-min(5, K_i):]))

            # EMA calculation
            ema_3 = valid_norm_vals[0]
            ema_5 = valid_norm_vals[0]
            for v in valid_norm_vals[1:]:
                ema_3 = alpha_3 * v + (1.0 - alpha_3) * ema_3
                ema_5 = alpha_5 * v + (1.0 - alpha_5) * ema_5

            sma_summary[feat_name] = {
                "valid": True,
                "K_i": K_i,
                "latest_value": round(valid_norm_vals[-1], 6),
                "sma_3": round(sma_3, 6),
                "sma_5": round(sma_5, 6),
                "ema_3": round(float(ema_3), 6),
                "ema_5": round(float(ema_5), 6)
            }

            # 2. OLS Trend Slopes
            if K_i >= 2:
                tau_arr = np.array(valid_tau, dtype=np.float64)
                mean_tau = float(np.mean(tau_arr))
                var_tau = float(np.sum((tau_arr - mean_tau) ** 2))

                if var_tau >= cls.OLS_MIN_TIME_VAR:
                    norm_arr = np.array(valid_norm_vals, dtype=np.float64)
                    raw_arr = np.array(valid_raw_vals, dtype=np.float64)

                    slope_norm = float(np.sum((tau_arr - mean_tau) * (norm_arr - np.mean(norm_arr))) / var_tau)
                    slope_raw = float(np.sum((tau_arr - mean_tau) * (raw_arr - np.mean(raw_arr))) / var_tau)

                    # Direction strictly from normalized slope
                    if slope_norm > 0.01:
                        direction = "increasing"
                    elif slope_norm < -0.01:
                        direction = "decreasing"
                    else:
                        direction = "stable"

                    slopes_summary[feat_name] = {
                        "valid": True,
                        "K_i": K_i,
                        "slope_normalized_per_hour": round(slope_norm, 6),
                        "slope_raw_per_hour": round(slope_raw, 6),
                        "direction": direction
                    }
                else:
                    slopes_summary[feat_name] = {
                        "valid": False,
                        "K_i": K_i,
                        "reason": "insufficient_temporal_variation",
                        "slope_normalized_per_hour": 0.0,
                        "direction": "stable"
                    }
            else:
                slopes_summary[feat_name] = {
                    "valid": False,
                    "K_i": K_i,
                    "reason": "single_observation",
                    "slope_normalized_per_hour": 0.0,
                    "direction": "stable"
                }

            # 3. Rate of Change
            if K_i >= 2:
                d_tau = valid_tau[-1] - valid_tau[-2]
                if d_tau >= cls.ROC_MIN_TIME_DELTA:
                    roc_norm = (valid_norm_vals[-1] - valid_norm_vals[-2]) / d_tau
                    roc_raw = (valid_raw_vals[-1] - valid_raw_vals[-2]) / d_tau
                    roc_summary[feat_name] = {
                        "valid": True,
                        "delta_time_hours": round(d_tau, 6),
                        "roc_normalized": round(float(roc_norm), 6),
                        "roc_raw": round(float(roc_raw), 6)
                    }
                else:
                    roc_summary[feat_name] = {
                        "valid": False,
                        "reason": "zero_time_delta",
                        "roc_normalized": 0.0
                    }
            else:
                roc_summary[feat_name] = {
                    "valid": False,
                    "reason": "insufficient_observations",
                    "roc_normalized": 0.0
                }

        return sma_summary, slopes_summary, roc_summary
