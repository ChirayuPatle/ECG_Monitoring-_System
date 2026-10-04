from dataclasses import dataclass, asdict
from typing import Any, Dict, List

import numpy as np
from scipy import signal


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_SAMPLING_RATE = 200

LOW_CUTOFF_HZ = 0.5
HIGH_CUTOFF_HZ = 40.0
NOTCH_FREQUENCY_HZ = 50.0


# ============================================================
# RESULT TYPES
# ============================================================

@dataclass
class SignalQuality:
    score: float
    label: str

    baseline_wander: float
    noise_rms: float
    clipping_ratio: float
    peak_count: int

    notes: List[str]


@dataclass
class ECGMeasurement:
    heart_rate_bpm: float | None
    rr_interval_ms: float | None

    p_duration_ms: float | None
    pr_interval_ms: float | None
    qrs_duration_ms: float | None

    qt_interval_ms: float | None
    qtc_ms: float | None

    confidence: float

    status: str
    notes: List[str]


@dataclass
class ECGAnalysisResult:
    filtered_signal: List[float]

    r_peaks: List[int]

    rr_intervals_ms: List[float]

    heart_rate_bpm: float | None

    measurement: Dict[str, Any]

    signal_quality: Dict[str, Any]


# ============================================================
# SAFE NUMPY CONVERSION
# ============================================================

def to_array(samples) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64)

    values = values[np.isfinite(values)]

    return values


# ============================================================
# BANDPASS FILTER
# ============================================================

def bandpass_filter(
    samples,
    sampling_rate: int,
    low_cutoff: float = LOW_CUTOFF_HZ,
    high_cutoff: float = HIGH_CUTOFF_HZ,
):
    x = to_array(samples)

    if len(x) < 30:
        return x

    nyquist = sampling_rate / 2.0

    low = max(0.001, low_cutoff / nyquist)
    high = min(0.99, high_cutoff / nyquist)

    if low >= high:
        return x

    b, a = signal.butter(
        3,
        [low, high],
        btype="bandpass",
    )

    pad_length = min(
        len(x) - 1,
        3 * max(len(a), len(b)),
    )

    if pad_length < 3:
        return x

    return signal.filtfilt(
        b,
        a,
        x,
        padlen=pad_length,
    )


# ============================================================
# 50 Hz NOTCH
# ============================================================

def notch_filter(
    samples,
    sampling_rate: int,
    frequency: float = NOTCH_FREQUENCY_HZ,
):
    x = to_array(samples)

    if len(x) < 30:
        return x

    nyquist = sampling_rate / 2.0

    if frequency >= nyquist:
        return x

    quality_factor = 30.0

    b, a = signal.iirnotch(
        frequency,
        quality_factor,
        sampling_rate,
    )

    pad_length = min(
        len(x) - 1,
        3 * max(len(a), len(b)),
    )

    if pad_length < 3:
        return x

    return signal.filtfilt(
        b,
        a,
        x,
        padlen=pad_length,
    )


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_ecg(
    samples,
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
):
    x = to_array(samples)

    if len(x) < 30:
        return x

    # Remove DC offset first.
    x = x - np.median(x)

    # Remove power-line interference.
    x = notch_filter(
        x,
        sampling_rate,
    )

    # ECG band limiting.
    x = bandpass_filter(
        x,
        sampling_rate,
    )

    return x


# ============================================================
# R-PEAK DETECTION
# ============================================================

def detect_r_peaks(
    filtered_signal,
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
):
    x = to_array(filtered_signal)

    if len(x) < sampling_rate:
        return np.array([], dtype=int)

    # Derivative emphasizes rapid QRS transitions.
    derivative = np.diff(x, prepend=x[0])

    squared = derivative ** 2

    # Approximately 100 ms integration window.
    integration_window = max(
        1,
        int(0.10 * sampling_rate),
    )

    integrated = np.convolve(
        squared,
        np.ones(integration_window)
        / integration_window,
        mode="same",
    )

    if not np.any(np.isfinite(integrated)):
        return np.array([], dtype=int)

    # Adaptive threshold.
    threshold = (
        np.median(integrated)
        +
        0.5 * np.std(integrated)
    )

    if threshold <= 0:
        return np.array([], dtype=int)

    # Minimum distance between R peaks:
    # 250 ms → maximum theoretical HR around 240 bpm.
    minimum_distance = max(
        1,
        int(0.25 * sampling_rate),
    )

    candidates, _ = signal.find_peaks(
        integrated,
        distance=minimum_distance,
        prominence=threshold * 0.25,
    )

    if len(candidates) == 0:
        return np.array([], dtype=int)

    # Refine each candidate against the filtered ECG.
    search_radius = max(
        1,
        int(0.08 * sampling_rate),
    )

    refined = []

    for candidate in candidates:

        start = max(
            0,
            candidate - search_radius,
        )

        end = min(
            len(x),
            candidate + search_radius + 1,
        )

        local = np.abs(x[start:end])

        if len(local) == 0:
            continue

        local_peak = start + int(
            np.argmax(local)
        )

        refined.append(local_peak)

    if not refined:
        return np.array([], dtype=int)

    refined = np.asarray(
        sorted(set(refined)),
        dtype=int,
    )

    return refined


# ============================================================
# RR INTERVALS
# ============================================================

def calculate_rr_intervals(
    r_peaks,
    sampling_rate: int,
):
    peaks = np.asarray(
        r_peaks,
        dtype=int,
    )

    if len(peaks) < 2:
        return []

    differences = np.diff(peaks)

    rr_ms = (
        differences
        / sampling_rate
        * 1000.0
    )

    # Reject obviously impossible intervals.
    rr_ms = rr_ms[
        (rr_ms >= 250)
        &
        (rr_ms <= 2500)
    ]

    return [
        float(value)
        for value in rr_ms
    ]


# ============================================================
# HEART RATE
# ============================================================

def calculate_heart_rate(
    rr_intervals_ms,
):
    if not rr_intervals_ms:
        return None

    median_rr = float(
        np.median(rr_intervals_ms)
    )

    if median_rr <= 0:
        return None

    hr = 60000.0 / median_rr

    if hr < 20 or hr > 240:
        return None

    return round(hr, 2)


# ============================================================
# SIGNAL QUALITY
# ============================================================

def calculate_signal_quality(
    raw_signal,
    filtered_signal,
    r_peaks,
):
    raw = to_array(raw_signal)
    filtered = to_array(filtered_signal)

    notes = []

    if len(raw) == 0:
        return SignalQuality(
            score=0.0,
            label="invalid",
            baseline_wander=0.0,
            noise_rms=0.0,
            clipping_ratio=1.0,
            peak_count=0,
            notes=["No ECG samples available."],
        )

    # Baseline estimate.
    baseline = raw - filtered

    baseline_wander = float(
        np.std(baseline)
    )

    # Noise estimate.
    noise = raw - filtered

    noise_rms = float(
        np.sqrt(
            np.mean(
                np.square(noise)
            )
        )
    )

    # ADC clipping estimate.
    minimum = np.min(raw)
    maximum = np.max(raw)

    clipping_count = np.sum(
        (raw <= 1)
        |
        (raw >= 1022)
    )

    clipping_ratio = float(
        clipping_count / len(raw)
    )

    peak_count = len(r_peaks)

    score = 100.0

    # Baseline penalty.
    if baseline_wander > 100:
        score -= 25
        notes.append(
            "High baseline variation."
        )
    elif baseline_wander > 50:
        score -= 12
        notes.append(
            "Moderate baseline variation."
        )

    # Noise penalty.
    if noise_rms > 80:
        score -= 30
        notes.append(
            "High noise level."
        )
    elif noise_rms > 40:
        score -= 15
        notes.append(
            "Moderate noise level."
        )

    # Clipping penalty.
    if clipping_ratio > 0.05:
        score -= 30
        notes.append(
            "Significant ADC clipping."
        )
    elif clipping_ratio > 0:
        score -= 10
        notes.append(
            "Some ADC clipping detected."
        )

    if peak_count == 0:
        score -= 25
        notes.append(
            "No R-peaks detected."
        )

    score = max(
        0.0,
        min(100.0, score),
    )

    if score >= 80:
        label = "good"
    elif score >= 60:
        label = "acceptable"
    elif score >= 40:
        label = "poor"
    else:
        label = "very_poor"

    return SignalQuality(
        score=round(score, 2),
        label=label,
        baseline_wander=round(
            baseline_wander,
            4,
        ),
        noise_rms=round(
            noise_rms,
            4,
        ),
        clipping_ratio=round(
            clipping_ratio,
            6,
        ),
        peak_count=peak_count,
        notes=notes,
    )


# ============================================================
# RESEARCH MORPHOLOGY ESTIMATES
# ============================================================

def estimate_morphology(
    filtered_signal,
    r_peaks,
    sampling_rate,
):
    """
    Research-only estimates.

    These values are intentionally conservative.
    They must NOT be interpreted as clinically validated
    ECG interval measurements.
    """

    if len(r_peaks) == 0:
        return {
            "p_duration_ms": None,
            "pr_interval_ms": None,
            "qrs_duration_ms": None,
            "qt_interval_ms": None,
            "qtc_ms": None,
            "confidence": 0.0,
            "status": "insufficient_data",
            "notes": [
                "No reliable R-peaks available."
            ],
        }

    rr_intervals = calculate_rr_intervals(
        r_peaks,
        sampling_rate,
    )

    rr = (
        float(np.median(rr_intervals))
        if rr_intervals
        else None
    )

    # We deliberately don't invent morphology values.
    return {
        "p_duration_ms": None,
        "pr_interval_ms": None,
        "qrs_duration_ms": None,
        "qt_interval_ms": None,
        "qtc_ms": None,
        "confidence": 0.35 if rr else 0.15,
        "status": "research_estimate",
        "notes": [
            "P/QRS/T delineation is not yet clinically validated.",
            "Use R-peak and RR analysis for the current stage.",
        ],
    }


# ============================================================
# COMPLETE ANALYSIS
# ============================================================

def analyze_ecg(
    samples,
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
):
    raw = to_array(samples)

    if len(raw) < 30:
        quality = SignalQuality(
            score=0.0,
            label="insufficient_data",
            baseline_wander=0.0,
            noise_rms=0.0,
            clipping_ratio=0.0,
            peak_count=0,
            notes=[
                "Not enough samples for ECG analysis."
            ],
        )

        measurement = ECGMeasurement(
            heart_rate_bpm=None,
            rr_interval_ms=None,
            p_duration_ms=None,
            pr_interval_ms=None,
            qrs_duration_ms=None,
            qt_interval_ms=None,
            qtc_ms=None,
            confidence=0.0,
            status="insufficient_data",
            notes=[
                "At least 30 valid samples are required."
            ],
        )

        return ECGAnalysisResult(
            filtered_signal=raw.tolist(),
            r_peaks=[],
            rr_intervals_ms=[],
            heart_rate_bpm=None,
            measurement=asdict(measurement),
            signal_quality=asdict(quality),
        )

    filtered = preprocess_ecg(
        raw,
        sampling_rate,
    )

    r_peaks = detect_r_peaks(
        filtered,
        sampling_rate,
    )

    rr_intervals = calculate_rr_intervals(
        r_peaks,
        sampling_rate,
    )

    heart_rate = calculate_heart_rate(
        rr_intervals
    )

    quality = calculate_signal_quality(
        raw,
        filtered,
        r_peaks,
    )

    morphology = estimate_morphology(
        filtered,
        r_peaks,
        sampling_rate,
    )

    measurement = ECGMeasurement(
        heart_rate_bpm=heart_rate,
        rr_interval_ms=(
            round(
                float(np.median(rr_intervals)),
                2,
            )
            if rr_intervals
            else None
        ),
        p_duration_ms=morphology[
            "p_duration_ms"
        ],
        pr_interval_ms=morphology[
            "pr_interval_ms"
        ],
        qrs_duration_ms=morphology[
            "qrs_duration_ms"
        ],
        qt_interval_ms=morphology[
            "qt_interval_ms"
        ],
        qtc_ms=morphology[
            "qtc_ms"
        ],
        confidence=morphology[
            "confidence"
        ],
        status=morphology[
            "status"
        ],
        notes=morphology[
            "notes"
        ],
    )

    return ECGAnalysisResult(
        filtered_signal=[
            round(float(v), 6)
            for v in filtered
        ],
        r_peaks=[
            int(v)
            for v in r_peaks
        ],
        rr_intervals_ms=rr_intervals,
        heart_rate_bpm=heart_rate,
        measurement=asdict(measurement),
        signal_quality=asdict(quality),
    )   