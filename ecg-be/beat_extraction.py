from dataclasses import dataclass
from typing import List, Optional

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_SAMPLING_RATE = 200

# One second per ML beat.
DEFAULT_BEAT_LENGTH = 200

# R peak location inside the 200-sample window.
PRE_R_SAMPLES = 60
POST_R_SAMPLES = 140

MIN_RR_MS = 250
MAX_RR_MS = 2500


# ============================================================
# RESULT
# ============================================================

@dataclass
class ExtractedBeat:
    beat_index: int

    r_peak_sample: int

    start_sample: int
    end_sample: int

    rr_interval_ms: Optional[float]

    heart_rate_bpm: Optional[float]

    quality_score: float

    samples: List[float]


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_beat(samples):
    x = np.asarray(
        samples,
        dtype=np.float64,
    )

    if len(x) == 0:
        return np.array([], dtype=np.float64)

    # Remove local baseline.
    x = x - np.median(x)

    # Robust amplitude normalization.
    scale = np.percentile(
        np.abs(x),
        95,
    )

    if scale < 1e-9:
        return np.zeros_like(x)

    x = x / scale

    # Keep extreme noise from dominating the ML vector.
    x = np.clip(
        x,
        -3.0,
        3.0,
    )

    return x


# ============================================================
# QUALITY OF INDIVIDUAL BEAT
# ============================================================

def beat_quality(samples):
    x = np.asarray(
        samples,
        dtype=np.float64,
    )

    if len(x) == 0:
        return 0.0

    if not np.all(np.isfinite(x)):
        return 0.0

    score = 100.0

    # Excessively flat beat.
    if np.std(x) < 1e-6:
        return 0.0

    # Extremely large jumps.
    differences = np.diff(x)

    if len(differences):
        jump_ratio = np.mean(
            np.abs(differences)
            > (
                np.std(x) * 5
            )
        )

        if jump_ratio > 0.10:
            score -= 35
        elif jump_ratio > 0.05:
            score -= 15

    # Excessive amplitude clipping.
    if np.max(np.abs(x)) > 3:
        score -= 20

    return round(
        max(
            0.0,
            min(100.0, score),
        ),
        2,
    )


# ============================================================
# HEART RATE FROM RR
# ============================================================

def heart_rate_from_rr(rr_ms):
    if rr_ms is None:
        return None

    if rr_ms <= 0:
        return None

    hr = 60000.0 / rr_ms

    if hr < 20 or hr > 240:
        return None

    return round(
        float(hr),
        2,
    )


# ============================================================
# EXTRACT BEATS
# ============================================================

def extract_beats(
    filtered_signal,
    r_peaks,
    sampling_rate=DEFAULT_SAMPLING_RATE,
    quality_threshold=40.0,
):
    x = np.asarray(
        filtered_signal,
        dtype=np.float64,
    )

    peaks = np.asarray(
        r_peaks,
        dtype=int,
    )

    if len(x) == 0 or len(peaks) == 0:
        return []

    pre_r = int(
        PRE_R_SAMPLES
        * sampling_rate
        / DEFAULT_SAMPLING_RATE
    )

    post_r = int(
        POST_R_SAMPLES
        * sampling_rate
        / DEFAULT_SAMPLING_RATE
    )

    beat_length = pre_r + post_r

    beats = []

    beat_index = 0

    for index, r_peak in enumerate(peaks):

        start = r_peak - pre_r
        end = r_peak + post_r

        # Don't create incomplete beats.
        if start < 0:
            continue

        if end > len(x):
            continue

        segment = x[start:end]

        if len(segment) != beat_length:
            continue

        # RR before this beat.
        rr_ms = None

        if index > 0:

            previous_peak = peaks[index - 1]

            rr_samples = (
                r_peak
                - previous_peak
            )

            rr_ms = (
                rr_samples
                / sampling_rate
                * 1000.0
            )

            if (
                rr_ms < MIN_RR_MS
                or rr_ms > MAX_RR_MS
            ):
                rr_ms = None

        quality = beat_quality(
            segment
        )

        if quality < quality_threshold:
            continue

        normalized = normalize_beat(
            segment
        )

        beat_index += 1

        beats.append(
            ExtractedBeat(
                beat_index=beat_index,

                r_peak_sample=int(
                    r_peak
                ),

                start_sample=int(
                    start
                ),

                end_sample=int(
                    end - 1
                ),

                rr_interval_ms=(
                    round(
                        float(rr_ms),
                        2,
                    )
                    if rr_ms is not None
                    else None
                ),

                heart_rate_bpm=(
                    heart_rate_from_rr(
                        rr_ms
                    )
                ),

                quality_score=quality,

                samples=[
                    round(
                        float(value),
                        6,
                    )
                    for value
                    in normalized
                ],
            )
        )

    return beats


# ============================================================
# CONVERT TO ML MATRIX
# ============================================================

def beats_to_matrix(beats):
    if not beats:
        return np.empty(
            (0, DEFAULT_BEAT_LENGTH),
            dtype=np.float32,
        )

    return np.asarray(
        [
            beat.samples
            for beat in beats
        ],
        dtype=np.float32,
    )


# ============================================================
# DATASET EXPORT STRUCTURE
# ============================================================

def build_ml_dataset(
    beats,
    sampling_rate=DEFAULT_SAMPLING_RATE,
):
    matrix = beats_to_matrix(beats)

    return {
        "sampling_rate": sampling_rate,
        "beat_count": int(
            len(beats)
        ),
        "samples_per_beat": (
            int(matrix.shape[1])
            if matrix.ndim == 2
            and matrix.shape[0] > 0
            else DEFAULT_BEAT_LENGTH
        ),
        "shape": list(
            matrix.shape
        ),
        "X": matrix.tolist(),
    }