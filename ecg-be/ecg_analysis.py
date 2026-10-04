"""
ECG signal processing and ML-ready beat segmentation.

Pipeline:

Raw ECG
    ↓
Filtering
    ↓
Signal quality
    ↓
R-peak detection
    ↓
RR / Heart Rate
    ↓
Fixed-length beat segmentation
    ↓
Z-score normalization
    ↓
ML-ready tensor: (N, 200, 1)

Current project sampling rate:
    200 Hz

Beat representation:
    300 ms before R-peak  = 60 samples
    700 ms after R-peak   = 140 samples
    Total                 = 200 samples

R-peak position inside beat:
    sample 60
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    signal = None
    SCIPY_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_FS = 200

LOWCUT_HZ = 0.5
HIGHCUT_HZ = 40.0
NOTCH_HZ = 50.0

BEAT_PRE_MS = 300
BEAT_POST_MS = 700

BEAT_PRE_SAMPLES = int(DEFAULT_FS * BEAT_PRE_MS / 1000)
BEAT_POST_SAMPLES = int(DEFAULT_FS * BEAT_POST_MS / 1000)

SAMPLES_PER_BEAT = BEAT_PRE_SAMPLES + BEAT_POST_SAMPLES

MIN_ANALYSIS_SECONDS = 5
MIN_ANALYSIS_SAMPLES = DEFAULT_FS * MIN_ANALYSIS_SECONDS


# ============================================================
# FILTER STATE
# ============================================================

@dataclass
class FilterState:
    """
    Stateful ECG filter configuration.

    The state object is intentionally lightweight. Filtering is
    performed on each received analysis window.
    """

    sampling_rate: int = DEFAULT_FS


# ============================================================
# BASIC UTILITIES
# ============================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)

        if not np.isfinite(result):
            return default

        return result

    except (TypeError, ValueError):
        return default


def _clean_samples(samples: Sequence[Any]) -> np.ndarray:
    """
    Convert incoming samples to a finite float64 NumPy array.
    """

    if samples is None:
        return np.asarray([], dtype=np.float64)

    try:
        array = np.asarray(samples, dtype=np.float64)
    except (TypeError, ValueError):
        return np.asarray([], dtype=np.float64)

    if array.ndim != 1:
        array = array.reshape(-1)

    finite_mask = np.isfinite(array)

    return array[finite_mask]


# ============================================================
# FILTERING
# ============================================================

def bandpass_filter(
    samples: Sequence[Any],
    fs: float = DEFAULT_FS,
    lowcut: float = LOWCUT_HZ,
    highcut: float = HIGHCUT_HZ,
) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter.

    Passband:
        0.5 Hz - 40 Hz

    If scipy is unavailable or the input is too short,
    the original cleaned signal is returned.
    """

    data = _clean_samples(samples)

    if data.size < 20:
        return data

    if not SCIPY_AVAILABLE:
        return data

    fs = _safe_float(fs, DEFAULT_FS)

    nyquist = fs / 2.0

    low = max(0.01, lowcut / nyquist)
    high = min(0.99, highcut / nyquist)

    if low >= high:
        return data

    try:
        b, a = signal.butter(
            4,
            [low, high],
            btype="bandpass",
        )

        pad_length = 3 * max(len(a), len(b))

        if len(data) <= pad_length:
            return data

        return signal.filtfilt(
            b,
            a,
            data,
        )

    except Exception:
        return data


def notch_filter(
    samples: Sequence[Any],
    fs: float = DEFAULT_FS,
    notch_hz: float = NOTCH_HZ,
) -> np.ndarray:
    """
    Apply a 50 Hz notch filter for mains interference.
    """

    data = _clean_samples(samples)

    if data.size < 20:
        return data

    if not SCIPY_AVAILABLE:
        return data

    fs = _safe_float(fs, DEFAULT_FS)

    if notch_hz >= fs / 2:
        return data

    try:
        quality_factor = 30.0

        b, a = signal.iirnotch(
            notch_hz,
            quality_factor,
            fs,
        )

        pad_length = 3 * max(len(a), len(b))

        if len(data) <= pad_length:
            return data

        return signal.filtfilt(
            b,
            a,
            data,
        )

    except Exception:
        return data


def filter_ecg(
    samples: Sequence[Any],
    fs: float = DEFAULT_FS,
) -> np.ndarray:
    """
    Complete ECG filtering pipeline.

    1. Bandpass 0.5-40 Hz
    2. 50 Hz notch
    """

    data = _clean_samples(samples)

    if data.size == 0:
        return data

    filtered = bandpass_filter(
        data,
        fs=fs,
    )

    filtered = notch_filter(
        filtered,
        fs=fs,
    )

    return filtered


# ============================================================
# SIGNAL QUALITY
# ============================================================

def calculate_signal_quality(
    samples: Sequence[Any],
    fs: float = DEFAULT_FS,
) -> Dict[str, Any]:
    """
    Estimate ECG signal quality.

    This is an engineering quality indicator, not a medical
    diagnostic measurement.
    """

    data = _clean_samples(samples)

    if data.size == 0:
        return {
            "score": 0.0,
            "label": "Waiting",
            "baseline_wander": 0.0,
            "noise_rms": 0.0,
            "clipping_ratio": 0.0,
            "dynamic_range": 0.0,
            "notes": ["No ECG samples available"],
        }

    notes: List[str] = []

    mean_value = float(np.mean(data))

    centered = data - mean_value

    dynamic_range = float(
        np.max(data) - np.min(data)
    )

    rms = float(
        np.sqrt(np.mean(centered ** 2))
    )

    if dynamic_range <= 1e-9:
        return {
            "score": 0.0,
            "label": "Poor",
            "baseline_wander": 0.0,
            "noise_rms": rms,
            "clipping_ratio": 0.0,
            "dynamic_range": dynamic_range,
            "notes": ["Signal has almost no dynamic range"],
        }

    # --------------------------------------------------------
    # Baseline wander estimate
    # --------------------------------------------------------

    baseline_wander = 0.0

    if SCIPY_AVAILABLE and data.size >= 20:

        try:
            fs_value = _safe_float(fs, DEFAULT_FS)

            cutoff = min(
                0.5,
                fs_value * 0.45,
            )

            normalized_cutoff = cutoff / (fs_value / 2.0)

            if 0 < normalized_cutoff < 1:

                b, a = signal.butter(
                    2,
                    normalized_cutoff,
                    btype="low",
                )

                pad_length = 3 * max(
                    len(a),
                    len(b),
                )

                if len(data) > pad_length:

                    baseline = signal.filtfilt(
                        b,
                        a,
                        data,
                    )

                    baseline_wander = float(
                        np.std(baseline)
                    )

        except Exception:
            baseline_wander = 0.0

    # --------------------------------------------------------
    # Clipping estimate
    # --------------------------------------------------------

    min_value = float(np.min(data))
    max_value = float(np.max(data))

    clipping_mask = (
        (data <= min_value)
        | (data >= max_value)
    )

    clipping_ratio = float(
        np.mean(clipping_mask)
    )

    # --------------------------------------------------------
    # Noise estimate
    # --------------------------------------------------------

    noise_rms = rms

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    score = 100.0

    normalized_baseline = (
        baseline_wander / max(dynamic_range, 1e-9)
    )

    normalized_noise = (
        noise_rms / max(dynamic_range, 1e-9)
    )

    score -= min(
        35.0,
        normalized_baseline * 200.0,
    )

    score -= min(
        35.0,
        normalized_noise * 100.0,
    )

    score -= min(
        30.0,
        clipping_ratio * 300.0,
    )

    score = float(
        np.clip(score, 0.0, 100.0)
    )

    if score >= 75:
        label = "Good"

    elif score >= 45:
        label = "Fair"

    else:
        label = "Poor"

    if normalized_baseline > 0.15:
        notes.append("Baseline wander detected")

    if normalized_noise > 0.20:
        notes.append("High noise level")

    if clipping_ratio > 0.05:
        notes.append("Possible ADC clipping")

    if not notes:
        notes.append("Signal quality acceptable")

    return {
        "score": round(score, 2),
        "label": label,
        "baseline_wander": round(
            baseline_wander,
            4,
        ),
        "noise_rms": round(
            noise_rms,
            4,
        ),
        "clipping_ratio": round(
            clipping_ratio,
            6,
        ),
        "dynamic_range": round(
            dynamic_range,
            4,
        ),
        "notes": notes,
    }


# ============================================================
# R-PEAK DETECTION
# ============================================================

def detect_r_peaks(
    samples: Sequence[Any],
    fs: float = DEFAULT_FS,
) -> List[int]:
    """
    Detect candidate R-peaks.

    Uses scipy.signal.find_peaks with ECG-oriented distance
    and prominence constraints.

    Returns sample indices.
    """

    data = _clean_samples(samples)

    if data.size < 3:
        return []

    if not SCIPY_AVAILABLE:
        return []

    fs = _safe_float(fs, DEFAULT_FS)

    filtered = filter_ecg(
        data,
        fs=fs,
    )

    if filtered.size < 3:
        return []

    # R-peaks should generally be separated by at least
    # ~250 ms for this application.
    minimum_distance = max(
        1,
        int(round(fs * 0.25)),
    )

    signal_std = float(
        np.std(filtered)
    )

    signal_range = float(
        np.ptp(filtered)
    )

    if signal_std <= 1e-9:
        return []

    prominence = max(
        signal_std * 0.35,
        signal_range * 0.05,
    )

    try:

        peaks, properties = signal.find_peaks(
            filtered,
            distance=minimum_distance,
            prominence=prominence,
        )

        if len(peaks) == 0:
            return []

        # ----------------------------------------------------
        # Polarity handling
        #
        # AD8232 Lead II is normally positive, but electrode
        # orientation can invert the waveform. If very few
        # positive peaks exist, try the inverted signal.
        # ----------------------------------------------------

        positive_peaks = peaks

        negative_peaks, _ = signal.find_peaks(
            -filtered,
            distance=minimum_distance,
            prominence=prominence,
        )

        if len(negative_peaks) > len(positive_peaks):
            peaks = negative_peaks
        else:
            peaks = positive_peaks

        return [
            int(index)
            for index in peaks
        ]

    except Exception:
        return []


# ============================================================
# RR INTERVALS
# ============================================================

def calculate_rr_intervals(
    r_peaks: Sequence[int],
    fs: float = DEFAULT_FS,
) -> List[float]:
    """
    Calculate RR intervals in milliseconds.
    """

    if not r_peaks:
        return []

    fs = _safe_float(fs, DEFAULT_FS)

    if fs <= 0:
        return []

    intervals: List[float] = []

    for i in range(1, len(r_peaks)):

        try:
            current = int(r_peaks[i])
            previous = int(r_peaks[i - 1])

            difference = current - previous

            if difference <= 0:
                continue

            rr_ms = (
                difference / fs
            ) * 1000.0

            intervals.append(
                float(rr_ms)
            )

        except (TypeError, ValueError):
            continue

    return intervals


def calculate_heart_rate(
    rr_intervals_ms: Sequence[float],
) -> Optional[float]:
    """
    Calculate BPM from the median RR interval.
    """

    valid = [
        float(rr)
        for rr in rr_intervals_ms
        if rr is not None
        and np.isfinite(rr)
        and rr > 0
    ]

    if not valid:
        return None

    median_rr = float(
        np.median(valid)
    )

    if median_rr <= 0:
        return None

    bpm = 60000.0 / median_rr

    return round(
        float(bpm),
        1,
    )


def rr_stability(
    rr_intervals_ms: Sequence[float],
) -> Optional[float]:
    """
    Return a simple RR stability score from 0-100.

    This is not a clinical HRV metric.
    """

    valid = np.asarray(
        [
            float(rr)
            for rr in rr_intervals_ms
            if rr is not None
            and np.isfinite(rr)
            and rr > 0
        ],
        dtype=np.float64,
    )

    if valid.size < 2:
        return None

    mean_rr = float(
        np.mean(valid)
    )

    if mean_rr <= 0:
        return None

    cv = float(
        np.std(valid) / mean_rr
    )

    score = 100.0 - (
        min(cv, 1.0) * 100.0
    )

    return round(
        float(np.clip(score, 0, 100)),
        1,
    )


# ============================================================
# ANALYSIS CONFIDENCE
# ============================================================

def calculate_analysis_confidence(
    sample_count: int,
    quality_score: float,
    r_peak_count: int,
) -> float:
    """
    Estimate confidence that enough clean ECG data exists
    for basic analysis.

    This is an engineering confidence indicator, not a
    diagnostic confidence value.
    """

    if sample_count <= 0:
        return 0.0

    duration_seconds = (
        sample_count / DEFAULT_FS
    )

    duration_score = min(
        duration_seconds / MIN_ANALYSIS_SECONDS,
        1.0,
    )

    peak_score = min(
        r_peak_count / 5.0,
        1.0,
    )

    quality_score = float(
        np.clip(
            _safe_float(
                quality_score,
                0.0,
            ),
            0.0,
            100.0,
        )
    )

    quality_component = (
        quality_score / 100.0
    )

    confidence = (
        duration_score * 0.25
        + peak_score * 0.25
        + quality_component * 0.50
    )

    return round(
        confidence * 100.0,
        1,
    )


# ============================================================
# ECG WINDOW ANALYSIS
# ============================================================

def analyze_window(
    samples: Sequence[Any],
    fs: float = DEFAULT_FS,
) -> Dict[str, Any]:
    """
    Analyze an ECG window.

    Important:
        Signal quality can be reported early.

        Reliable heart-rate / R-peak analysis requires
        approximately 5 seconds of data.
    """

    data = _clean_samples(samples)

    fs = _safe_float(
        fs,
        DEFAULT_FS,
    )

    sample_count = int(
        data.size
    )

    duration_seconds = (
        sample_count / fs
        if fs > 0
        else 0.0
    )

    quality = calculate_signal_quality(
        data,
        fs=fs,
    )

    if sample_count < 3:

        return {
            "heart_rate": None,
            "heart_rate_bpm": None,
            "rr_interval": None,
            "rr_interval_ms": None,
            "rr_intervals": [],
            "r_peaks": [],
            "peak_count": 0,
            "signal_quality": quality,
            "confidence": 0.0,
            "status": "Waiting for ECG data",
            "analysis_ready": False,
            "samples_available": sample_count,
            "required_samples": MIN_ANALYSIS_SAMPLES,
            "duration_seconds": round(
                duration_seconds,
                2,
            ),
            "morphology": {
                "p_duration_ms": None,
                "pr_interval_ms": None,
                "qrs_duration_ms": None,
                "qt_interval_ms": None,
                "qtc_ms": None,
            },
        }

    # --------------------------------------------------------
    # Do not perform aggressive R-peak analysis on tiny
    # windows. The UI can still show signal quality.
    # --------------------------------------------------------

    if sample_count < MIN_ANALYSIS_SAMPLES:

        return {
            "heart_rate": None,
            "heart_rate_bpm": None,
            "rr_interval": None,
            "rr_interval_ms": None,
            "rr_intervals": [],
            "r_peaks": [],
            "peak_count": 0,
            "signal_quality": quality,
            "confidence": calculate_analysis_confidence(
                sample_count,
                quality.get("score", 0),
                0,
            ),
            "status": "Collecting ECG data",
            "analysis_ready": False,
            "samples_available": sample_count,
            "required_samples": MIN_ANALYSIS_SAMPLES,
            "duration_seconds": round(
                duration_seconds,
                2,
            ),
            "morphology": {
                "p_duration_ms": None,
                "pr_interval_ms": None,
                "qrs_duration_ms": None,
                "qt_interval_ms": None,
                "qtc_ms": None,
            },
        }

    # --------------------------------------------------------
    # R-peak detection
    # --------------------------------------------------------

    r_peaks = detect_r_peaks(
        data,
        fs=fs,
    )

    rr_intervals = calculate_rr_intervals(
        r_peaks,
        fs=fs,
    )

    heart_rate = calculate_heart_rate(
        rr_intervals
    )

    rr_interval = (
        float(np.median(rr_intervals))
        if rr_intervals
        else None
    )

    confidence = calculate_analysis_confidence(
        sample_count,
        quality.get("score", 0),
        len(r_peaks),
    )

    if heart_rate is not None:
        status = "Normal monitoring"
    else:
        status = "Analysis unavailable"

    return {
        "heart_rate": heart_rate,
        "heart_rate_bpm": heart_rate,

        "rr_interval": (
            round(rr_interval, 1)
            if rr_interval is not None
            else None
        ),

        "rr_interval_ms": (
            round(rr_interval, 1)
            if rr_interval is not None
            else None
        ),

        "rr_intervals": [
            round(float(rr), 2)
            for rr in rr_intervals
        ],

        "r_peaks": [
            int(index)
            for index in r_peaks
        ],

        "peak_count": len(r_peaks),

        "signal_quality": quality,

        "confidence": confidence,

        "status": status,

        "analysis_ready": True,

        "samples_available": sample_count,

        "required_samples": MIN_ANALYSIS_SAMPLES,

        "duration_seconds": round(
            duration_seconds,
            2,
        ),

        # Morphology is intentionally not estimated here.
        # These fields remain available for future validated
        # morphology processing.
        "morphology": {
            "p_duration_ms": None,
            "pr_interval_ms": None,
            "qrs_duration_ms": None,
            "qt_interval_ms": None,
            "qtc_ms": None,
        },
    }


# ============================================================
# ML-READY BEAT SEGMENTATION
# ============================================================

def segment_ecg_beats(
    samples: Sequence[Any],
    r_peaks: Sequence[int],
    fs: float = DEFAULT_FS,
    normalize: bool = True,
) -> List[Dict[str, Any]]:
    """
    Extract fixed-length heartbeat segments around R-peaks.

    Current representation:

        300 ms before R-peak
        700 ms after R-peak

    At 200 Hz:

        60 samples before
        140 samples after

        = 200 samples / beat

    The R-peak is aligned at index 60.

    Invalid/incomplete beats are rejected.
    """

    data = _clean_samples(samples)

    if data.size == 0:
        return []

    fs = _safe_float(
        fs,
        DEFAULT_FS,
    )

    if fs <= 0:
        return []

    pre_samples = int(
        round(
            fs
            * BEAT_PRE_MS
            / 1000.0
        )
    )

    post_samples = int(
        round(
            fs
            * BEAT_POST_MS
            / 1000.0
        )
    )

    expected_length = (
        pre_samples
        + post_samples
    )

    beats: List[Dict[str, Any]] = []

    for peak_position, peak in enumerate(r_peaks):

        try:
            peak = int(peak)
        except (TypeError, ValueError):
            continue

        start_index = (
            peak
            - pre_samples
        )

        end_index = (
            peak
            + post_samples
        )

        # ----------------------------------------------------
        # Reject incomplete boundary beats
        # ----------------------------------------------------

        if start_index < 0:
            continue

        if end_index > len(data):
            continue

        beat = data[
            start_index:end_index
        ].copy()

        # ----------------------------------------------------
        # Fixed-size validation
        # ----------------------------------------------------

        if len(beat) != expected_length:
            continue

        # ----------------------------------------------------
        # Numerical validation
        # ----------------------------------------------------

        if not np.all(
            np.isfinite(beat)
        ):
            continue

        mean = float(
            np.mean(beat)
        )

        std = float(
            np.std(beat)
        )

        # Reject flat or nearly flat segments
        if std < 1e-8:
            continue

        normalized = beat.copy()

        if normalize:
            normalized = (
                normalized - mean
            ) / std

        # ----------------------------------------------------
        # RR interval
        # ----------------------------------------------------

        rr_interval_ms = None

        if peak_position > 0:

            try:
                previous_peak = int(
                    r_peaks[
                        peak_position - 1
                    ]
                )

                rr_samples = (
                    peak
                    - previous_peak
                )

                if rr_samples > 0:

                    rr_interval_ms = (
                        rr_samples
                        / fs
                    ) * 1000.0

            except (
                TypeError,
                ValueError,
                IndexError,
            ):
                rr_interval_ms = None

        beats.append(
            {
                "beat_index": len(beats),

                "r_peak_index": peak,

                "r_peak_position": pre_samples,

                "start_index": start_index,

                "end_index": end_index,

                "length": expected_length,

                "pre_r_peak_samples": pre_samples,

                "post_r_peak_samples": post_samples,

                "pre_r_peak_ms": BEAT_PRE_MS,

                "post_r_peak_ms": BEAT_POST_MS,

                "rr_interval_ms": (
                    round(
                        float(
                            rr_interval_ms
                        ),
                        2,
                    )
                    if rr_interval_ms
                    is not None
                    else None
                ),

                "raw_samples": [
                    float(x)
                    for x in beat
                ],

                "normalized_samples": [
                    float(x)
                    for x in normalized
                ],
            }
        )

    return beats


# ============================================================
# ML DATASET BUILDER
# ============================================================

def build_ml_dataset(
    samples: Sequence[Any],
    r_peaks: Sequence[int],
    fs: float = DEFAULT_FS,
) -> Dict[str, Any]:
    """
    Build a CNN-compatible ECG heartbeat dataset.

    Output:

        X.shape == (N, 200, 1)

    where:

        N = accepted heartbeat count
        200 = samples per heartbeat
        1 = ECG signal channel
    """

    fs = _safe_float(
        fs,
        DEFAULT_FS,
    )

    beats = segment_ecg_beats(
        samples=samples,
        r_peaks=r_peaks,
        fs=fs,
        normalize=True,
    )

    if not beats:

        return {
            "X": np.empty(
                (0, 0, 1),
                dtype=np.float32,
            ),
            "metadata": [],
            "samples_per_beat": 0,
            "beat_count": 0,
            "sampling_rate": fs,
            "pre_r_peak_samples": int(
                round(
                    fs
                    * BEAT_PRE_MS
                    / 1000.0
                )
            ),
            "post_r_peak_samples": int(
                round(
                    fs
                    * BEAT_POST_MS
                    / 1000.0
                )
            ),
        }

    X = np.asarray(
        [
            beat[
                "normalized_samples"
            ]
            for beat in beats
        ],
        dtype=np.float32,
    )

    # Add channel dimension for CNNs.
    X = X[..., np.newaxis]

    metadata: List[Dict[str, Any]] = []

    for beat in beats:

        metadata.append(
            {
                "beat_index": beat[
                    "beat_index"
                ],

                "r_peak_index": beat[
                    "r_peak_index"
                ],

                "r_peak_position": beat[
                    "r_peak_position"
                ],

                "start_index": beat[
                    "start_index"
                ],

                "end_index": beat[
                    "end_index"
                ],

                "length": beat[
                    "length"
                ],

                "rr_interval_ms": beat[
                    "rr_interval_ms"
                ],
            }
        )

    return {
        "X": X,

        "metadata": metadata,

        "samples_per_beat": int(
            X.shape[1]
        ),

        "beat_count": int(
            X.shape[0]
        ),

        "sampling_rate": fs,

        "pre_r_peak_samples": int(
            round(
                fs
                * BEAT_PRE_MS
                / 1000.0
            )
        ),

        "post_r_peak_samples": int(
            round(
                fs
                * BEAT_POST_MS
                / 1000.0
            )
        ),
    }


# ============================================================
# ML DATASET VALIDATION
# ============================================================

def validate_ml_dataset(
    dataset: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate the generated ML tensor.

    Checks:

    - shape
    - finite values
    - fixed length
    - normalization
    - channel dimension
    """

    X = dataset.get("X")

    if X is None:

        return {
            "valid": False,
            "errors": [
                "Dataset does not contain X"
            ],
        }

    X = np.asarray(X)

    errors: List[str] = []

    if X.ndim != 3:
        errors.append(
            f"Expected 3 dimensions, got {X.ndim}"
        )

    else:

        if X.shape[2] != 1:

            errors.append(
                "Expected one ECG channel"
            )

        if X.shape[1] != SAMPLES_PER_BEAT:

            errors.append(
                f"Expected {SAMPLES_PER_BEAT} "
                f"samples per beat, got "
                f"{X.shape[1]}"
            )

    if X.size > 0:

        if not np.all(
            np.isfinite(X)
        ):
            errors.append(
                "Dataset contains NaN or Infinity"
            )

        # Check Z-score normalization
        means = np.mean(
            X,
            axis=1,
        )

        stds = np.std(
            X,
            axis=1,
        )

        mean_error = float(
            np.max(
                np.abs(means)
            )
        )

        std_error = float(
            np.max(
                np.abs(
                    stds - 1.0
                )
            )
        )

        if mean_error > 0.05:

            errors.append(
                "Beat normalization mean "
                "is outside tolerance"
            )

        if std_error > 0.05:

            errors.append(
                "Beat normalization standard "
                "deviation is outside tolerance"
            )

    return {
        "valid": len(errors) == 0,

        "errors": errors,

        "shape": list(
            X.shape
        ),

        "beat_count": int(
            X.shape[0]
        )
        if X.ndim >= 1
        else 0,

        "samples_per_beat": int(
            X.shape[1]
        )
        if X.ndim >= 2
        else 0,

        "channels": int(
            X.shape[2]
        )
        if X.ndim >= 3
        else 0,

        "normalization": "z-score",
    }


# ============================================================
# PUBLIC CONSTANTS
# ============================================================

ML_BEAT_SPEC = {
    "sampling_rate_hz": DEFAULT_FS,
    "pre_r_peak_ms": BEAT_PRE_MS,
    "post_r_peak_ms": BEAT_POST_MS,
    "pre_r_peak_samples": BEAT_PRE_SAMPLES,
    "post_r_peak_samples": BEAT_POST_SAMPLES,
    "samples_per_beat": SAMPLES_PER_BEAT,
    "r_peak_position": BEAT_PRE_SAMPLES,
    "normalization": "z-score",
    "tensor_shape": "(N, 200, 1)",
}