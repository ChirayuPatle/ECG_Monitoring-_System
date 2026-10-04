from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import math

import numpy as np

from scipy.signal import (
    butter,
    filtfilt,
    find_peaks,
)


# =========================================================
# ECG MEASUREMENT MODEL
# =========================================================

@dataclass
class ECGMeasurement:

    heart_rate_bpm: Optional[float] = None

    rr_interval_ms: Optional[float] = None

    p_duration_ms: Optional[float] = None

    pr_interval_ms: Optional[float] = None

    qrs_duration_ms: Optional[float] = None

    qt_interval_ms: Optional[float] = None

    qtc_ms: Optional[float] = None

    # Wave locations
    p_onset_sample: Optional[int] = None

    p_offset_sample: Optional[int] = None

    qrs_onset_sample: Optional[int] = None

    r_peak_sample: Optional[int] = None

    qrs_offset_sample: Optional[int] = None

    t_onset_sample: Optional[int] = None

    t_offset_sample: Optional[int] = None

    confidence: float = 0.0

    status: str = "insufficient_data"

    notes: list[str] | None = None

    def to_dict(self):

        return asdict(self)


# =========================================================
# SIGNAL QUALITY MODEL
# =========================================================

@dataclass
class SignalQuality:

    score: float

    label: str

    baseline_wander: float

    noise_rms: float

    clipping_ratio: float

    peak_count: int

    notes: list[str]

    def to_dict(self):

        return asdict(self)


# =========================================================
# SAFE FLOAT
# =========================================================

def _safe_float(value):

    if value is None:
        return None

    value = float(value)

    return (
        value
        if math.isfinite(value)
        else None
    )


# =========================================================
# BANDPASS FILTER
# =========================================================

def bandpass(
    signal: np.ndarray,
    fs: int,
    low=0.5,
    high=40.0,
):

    nyquist = fs / 2.0

    high = min(
        high,
        nyquist - 1.0,
    )

    if high <= low:

        return signal.copy()

    b, a = butter(
        3,
        [
            low / nyquist,
            high / nyquist,
        ],
        btype="band",
    )

    return filtfilt(
        b,
        a,
        signal,
    )


# =========================================================
# R-PEAK DETECTION
# =========================================================

def detect_r_peaks(
    signal: np.ndarray,
    fs: int,
):

    filtered = bandpass(
        signal,
        fs,
        5.0,
        min(
            25.0,
            fs / 2.0 - 1.0,
        ),
    )

    # Derivative
    derivative = np.diff(
        filtered,
        prepend=filtered[0],
    )

    # Squared derivative
    energy = derivative * derivative

    # Moving integration window
    integration_window = max(
        1,
        int(0.10 * fs),
    )

    kernel = (
        np.ones(integration_window)
        / integration_window
    )

    envelope = np.convolve(
        energy,
        kernel,
        mode="same",
    )

    # Minimum R-R distance:
    # 250 ms -> maximum theoretical HR
    # around 240 BPM.
    distance = max(
        1,
        int(0.25 * fs),
    )

    median_env = np.median(
        envelope
    )

    mad_env = (
        np.median(
            np.abs(
                envelope
                - median_env
            )
        )
        + 1e-12
    )

    prominence = (
        median_env
        + 2.5 * mad_env
    )

    peaks, _ = find_peaks(

        envelope,

        distance=distance,

        prominence=max(
            prominence,
            np.percentile(
                envelope,
                75,
            )
            * 0.05,
        ),
    )

    # -----------------------------------------------------
    # Refine each candidate against ECG waveform
    # -----------------------------------------------------

    search_radius = max(
        1,
        int(0.08 * fs),
    )

    refined = []

    for peak in peaks:

        left = max(
            0,
            peak - search_radius,
        )

        right = min(
            len(filtered),
            peak + search_radius + 1,
        )

        if right <= left:
            continue

        local = filtered[
            left:right
        ]

        local_index = int(
            np.argmax(
                np.abs(local)
            )
        )

        refined.append(
            left + local_index
        )

    # -----------------------------------------------------
    # Remove duplicates
    # -----------------------------------------------------

    refined = sorted(
        set(refined)
    )

    result = []

    for peak in refined:

        if (
            not result
            or peak - result[-1]
            >= distance
        ):

            result.append(peak)

    return (
        np.asarray(
            result,
            dtype=int,
        ),
        filtered,
    )


# =========================================================
# SIGNAL QUALITY
# =========================================================

def estimate_signal_quality(
    signal: np.ndarray,
    fs: int,
    peak_count: int,
):

    if len(signal) == 0:

        return SignalQuality(

            0.0,

            "poor",

            0.0,

            0.0,

            0.0,

            0,

            [
                "No ECG samples available."
            ],
        )

    x = np.asarray(
        signal,
        dtype=float,
    )

    # -----------------------------------------------------
    # Baseline wander
    # -----------------------------------------------------

    if len(x) > fs * 2:

        baseline = bandpass(
            x,
            fs,
            0.05,
            min(
                0.5,
                fs / 2 - 1,
            ),
        )

    else:

        baseline = (
            x - np.median(x)
        )

    baseline_wander = float(
        np.std(baseline)
    )

    # -----------------------------------------------------
    # Noise
    # -----------------------------------------------------

    lowpassed = bandpass(
        x,
        fs,
        0.5,
        min(
            40.0,
            fs / 2 - 1,
        ),
    )

    residual = (
        x - lowpassed
    )

    noise_rms = float(
        np.sqrt(
            np.mean(
                residual ** 2
            )
        )
    )

    # -----------------------------------------------------
    # Clipping
    # -----------------------------------------------------

    amplitude = np.ptp(x)

    if amplitude <= 1e-9:

        clipping_ratio = 1.0

    else:

        # ESP8266 ADC range is normally
        # 0-1023 in this project.

        if (
            np.nanmax(x) <= 1023
            and np.nanmin(x) >= 0
        ):

            clipping_ratio = float(
                np.mean(
                    (x <= 2)
                    | (x >= 1021)
                )
            )

        else:

            clipping_ratio = 0.0

    # -----------------------------------------------------
    # Score
    # -----------------------------------------------------

    notes = []

    score = 100.0

    if (
        baseline_wander
        > max(
            amplitude * 0.20,
            1.0,
        )
    ):

        score -= 25

        notes.append(
            "Significant baseline movement detected."
        )

    if amplitude > 0:

        noise_ratio = (
            noise_rms
            / amplitude
        )

        if noise_ratio > 0.15:

            score -= 30

            notes.append(
                "Elevated high-frequency/noise component."
            )

    if clipping_ratio > 0.01:

        score -= 30

        notes.append(
            "Possible ADC clipping/saturation."
        )

    duration = (
        len(x) / fs
    )

    if (
        duration >= 5
        and peak_count
        < max(
            2,
            int(
                duration
                * 30
                / 60
            ),
        )
    ):

        score -= 15

        notes.append(
            "Few reliable beat candidates detected."
        )

    score = max(
        0.0,
        min(
            100.0,
            score,
        ),
    )

    if score >= 80:

        label = "good"

    elif score >= 60:

        label = "fair"

    else:

        label = "poor"

    return SignalQuality(

        score=round(
            score,
            1,
        ),

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
            5,
        ),

        peak_count=int(
            peak_count
        ),

        notes=notes,
    )


# =========================================================
# WAVE BOUNDARY ESTIMATION
# =========================================================

def _find_boundary(
    signal: np.ndarray,
    center: int,
    direction: int,
    fs: int,
    window_ms: float,
    threshold_ratio: float = 0.15,
):

    window = max(
        1,
        int(
            window_ms
            / 1000.0
            * fs
        ),
    )

    if direction < 0:

        start = max(
            0,
            center - window,
        )

        region = signal[
            start:center + 1
        ]

        if len(region) < 3:
            return None

        peak = abs(
            signal[center]
            - np.median(region)
        )

        if peak <= 1e-9:
            return None

        threshold = (
            peak
            * threshold_ratio
        )

        for i in range(
            len(region) - 1,
            0,
            -1,
        ):

            if (
                abs(
                    region[i]
                    - np.median(region)
                )
                <= threshold
            ):

                return start + i

        return start

    end = min(
        len(signal),
        center + window + 1,
    )

    region = signal[
        center:end
    ]

    if len(region) < 3:
        return None

    peak = abs(
        signal[center]
        - np.median(region)
    )

    if peak <= 1e-9:
        return None

    threshold = (
        peak
        * threshold_ratio
    )

    for i in range(
        len(region)
    ):

        if (
            abs(
                region[i]
                - np.median(region)
            )
            <= threshold
        ):

            return center + i

    return end - 1


# =========================================================
# SINGLE-BEAT P/QRS/T DELINEATION
# =========================================================

def delineate_single_beat(
    signal: np.ndarray,
    fs: int,
    r_peak: int,
):

    x = bandpass(
        signal,
        fs,
        0.5,
        min(
            40.0,
            fs / 2 - 1,
        ),
    )

    # -----------------------------------------------------
    # QRS
    # -----------------------------------------------------

    qrs_left = max(
        0,
        r_peak
        - int(
            0.12 * fs
        ),
    )

    qrs_right = min(
        len(x),
        r_peak
        + int(
            0.12 * fs
        ),
    )

    qrs_region = x[
        qrs_left:qrs_right
    ]

    if len(qrs_region) < 5:

        return ECGMeasurement(

            r_peak_sample=int(
                r_peak
            ),

            status="insufficient_data",

            notes=[
                "QRS region too short."
            ],
        )

    qrs_amp = max(

        np.ptp(
            qrs_region
        ),

        np.std(
            qrs_region
        ) * 4,

        1e-6,
    )

    threshold = (
        qrs_amp * 0.08
    )

    qrs_onset = r_peak

    for i in range(
        r_peak,
        qrs_left,
        -1,
    ):

        if (
            abs(
                x[i]
                - x[r_peak]
            )
            < threshold
        ):

            qrs_onset = i

            break

    qrs_offset = r_peak

    for i in range(
        r_peak,
        qrs_right,
    ):

        if (
            abs(
                x[i]
                - x[r_peak]
            )
            < threshold
        ):

            qrs_offset = i

            break

    qrs_duration = (
        qrs_offset
        - qrs_onset
    ) * 1000 / fs

    # -----------------------------------------------------
    # P wave
    # -----------------------------------------------------

    p_left = max(
        0,
        r_peak
        - int(
            0.32 * fs
        ),
    )

    p_right = max(
        p_left + 1,
        r_peak
        - int(
            0.08 * fs
        ),
    )

    p_region = x[
        p_left:p_right
    ]

    p_peak = None

    if len(p_region) >= 5:

        prominence = max(
            np.std(
                p_region
            ) * 0.35,
            1e-6,
        )

        p_candidates, _ = find_peaks(

            np.abs(
                p_region
                - np.median(
                    p_region
                )
            ),

            distance=max(
                1,
                int(
                    0.08 * fs
                ),
            ),

            prominence=prominence,
        )

        if len(p_candidates):

            candidate = (
                p_candidates[-1]
            )

            p_peak = (
                p_left
                + int(candidate)
            )

    p_onset = None
    p_offset = None
    p_duration = None
    pr_interval = None

    if p_peak is not None:

        p_onset = _find_boundary(
            x,
            p_peak,
            -1,
            fs,
            100,
        )

        p_offset = _find_boundary(
            x,
            p_peak,
            1,
            fs,
            100,
        )

        if (
            p_onset is not None
            and p_offset is not None
        ):

            p_duration = (
                p_offset
                - p_onset
            ) * 1000 / fs

        if p_onset is not None:

            pr_interval = (
                qrs_onset
                - p_onset
            ) * 1000 / fs

    # -----------------------------------------------------
    # T wave
    # -----------------------------------------------------

    t_left = min(
        len(x) - 1,
        r_peak
        + int(
            0.12 * fs
        ),
    )

    t_right = min(
        len(x),
        r_peak
        + int(
            0.55 * fs
        ),
    )

    t_region = x[
        t_left:t_right
    ]

    t_peak = None

    if len(t_region) >= 5:

        candidates, _ = find_peaks(

            np.abs(
                t_region
                - np.median(
                    t_region
                )
            ),

            distance=max(
                1,
                int(
                    0.12 * fs
                ),
            ),

            prominence=max(
                np.std(
                    t_region
                ) * 0.30,
                1e-6,
            ),
        )

        if len(candidates):

            strengths = [

                abs(
                    t_region[c]
                    - np.median(
                        t_region
                    )
                )

                for c in candidates
            ]

            t_peak = (
                t_left
                + int(
                    candidates[
                        int(
                            np.argmax(
                                strengths
                            )
                        )
                    ]
                )
            )

    t_onset = None
    t_offset = None
    qt_interval = None
    qtc = None

    if t_peak is not None:

        t_onset = _find_boundary(
            x,
            t_peak,
            -1,
            fs,
            180,
        )

        t_offset = _find_boundary(
            x,
            t_peak,
            1,
            fs,
            220,
        )

        if t_offset is not None:

            qt_interval = (
                t_offset
                - qrs_onset
            ) * 1000 / fs

    # -----------------------------------------------------
    # Confidence
    # -----------------------------------------------------

    confidence_components = []

    if (
        qrs_duration is not None
        and 40
        <= qrs_duration
        <= 180
    ):

        confidence_components.append(
            1.0
        )

    else:

        confidence_components.append(
            0.0
        )

    if p_peak is not None:

        confidence_components.append(
            0.7
        )

    else:

        confidence_components.append(
            0.0
        )

    if t_peak is not None:

        confidence_components.append(
            0.7
        )

    else:

        confidence_components.append(
            0.0
        )

    confidence = float(
        np.mean(
            confidence_components
        )
    )

    return ECGMeasurement(

        qrs_duration_ms=round(
            float(
                qrs_duration
            ),
            1,
        ),

        p_duration_ms=(

            round(
                float(
                    p_duration
                ),
                1,
            )

            if p_duration is not None

            else None
        ),

        pr_interval_ms=(

            round(
                float(
                    pr_interval
                ),
                1,
            )

            if pr_interval is not None

            else None
        ),

        qt_interval_ms=(

            round(
                float(
                    qt_interval
                ),
                1,
            )

            if qt_interval is not None

            else None
        ),

        p_onset_sample=p_onset,

        p_offset_sample=p_offset,

        qrs_onset_sample=qrs_onset,

        r_peak_sample=int(
            r_peak
        ),

        qrs_offset_sample=qrs_offset,

        t_onset_sample=t_onset,

        t_offset_sample=t_offset,

        confidence=round(
            confidence,
            3,
        ),

        status="research_estimate",

        notes=[

            "Single-lead experimental "
            "P/QRS/T delineation.",

            "Do not use these values "
            "for clinical decisions.",
        ],
    )


# =========================================================
# COMPLETE ECG ANALYSIS
# =========================================================

def analyze_ecg(
    samples: list[float],
    sampling_rate: int,
):

    if sampling_rate <= 0:

        raise ValueError(
            "sampling_rate must be positive."
        )

    x = np.asarray(
        samples,
        dtype=float,
    )

    minimum_samples = max(
        400,
        sampling_rate * 3,
    )

    # -----------------------------------------------------
    # Not enough data
    # -----------------------------------------------------

    if len(x) < minimum_samples:

        return {

            "status":
                "insufficient_data",

            "sampling_rate":
                sampling_rate,

            "samples_analyzed":
                int(len(x)),

            "measurements":

                ECGMeasurement(

                    status=
                        "insufficient_data",

                    notes=[

                        f"At least "
                        f"{minimum_samples} "
                        f"samples are recommended "
                        f"for the initial analysis "
                        f"window."
                    ],
                ).to_dict(),

            "signal_quality":

                estimate_signal_quality(
                    x,
                    sampling_rate,
                    0,
                ).to_dict(),

            "r_peaks": [],
        }

    # -----------------------------------------------------
    # R peaks
    # -----------------------------------------------------

    peaks, _ = detect_r_peaks(
        x,
        sampling_rate,
    )

    # -----------------------------------------------------
    # RR
    # -----------------------------------------------------

    rr = (
        np.diff(peaks)
        / sampling_rate
    )

    valid_rr = rr[
        (rr >= 0.30)
        &
        (rr <= 2.50)
    ]

    heart_rate = None
    rr_ms = None

    if len(valid_rr):

        rr_seconds = float(
            np.median(
                valid_rr
            )
        )

        rr_ms = (
            rr_seconds
            * 1000.0
        )

        heart_rate = (
            60.0
            / rr_seconds
        )

    # -----------------------------------------------------
    # Signal quality
    # -----------------------------------------------------

    quality = (
        estimate_signal_quality(
            x,
            sampling_rate,
            len(peaks),
        )
    )

    # -----------------------------------------------------
    # Base measurement object
    # -----------------------------------------------------

    measurement = ECGMeasurement(

        heart_rate_bpm=(

            round(
                heart_rate,
                1,
            )

            if heart_rate is not None

            else None
        ),

        rr_interval_ms=(

            round(
                rr_ms,
                1,
            )

            if rr_ms is not None

            else None
        ),

        confidence=min(

            1.0,

            max(
                0.0,
                quality.score
                / 100.0,
            ),
        ),

        status=(

            "research_estimate"

            if heart_rate is not None

            else "insufficient_data"
        ),

        notes=[],
    )

    # -----------------------------------------------------
    # Representative recent beat
    # -----------------------------------------------------

    if len(peaks):

        representative = int(
            peaks[-1]
        )

        beat = (
            delineate_single_beat(
                x,
                sampling_rate,
                representative,
            )
        )

        fields = [

            "p_duration_ms",

            "pr_interval_ms",

            "qrs_duration_ms",

            "qt_interval_ms",

            "p_onset_sample",

            "p_offset_sample",

            "qrs_onset_sample",

            "r_peak_sample",

            "qrs_offset_sample",

            "t_onset_sample",

            "t_offset_sample",
        ]

        for field in fields:

            value = getattr(
                beat,
                field,
            )

            if value is not None:

                setattr(
                    measurement,
                    field,
                    value,
                )

        # -------------------------------------------------
        # QTc - Fridericia
        # -------------------------------------------------

        if (

            measurement.qt_interval_ms
            is not None

            and rr_ms is not None
        ):

            qt_seconds = (
                measurement.qt_interval_ms
                / 1000.0
            )

            rr_seconds = (
                rr_ms
                / 1000.0
            )

            qtc = (
                qt_seconds
                / (
                    rr_seconds
                    ** (1.0 / 3.0)
                )
            )

            measurement.qtc_ms = round(
                qtc * 1000.0,
                1,
            )

        measurement.confidence = round(

            min(

                measurement.confidence,

                beat.confidence

                if beat.confidence > 0

                else measurement.confidence,
            ),

            3,
        )

        measurement.notes.extend(
            beat.notes or []
        )

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    return {

        "status":
            "ok",

        "sampling_rate":
            sampling_rate,

        "samples_analyzed":
            int(len(x)),

        "duration_seconds":
            round(
                len(x)
                / sampling_rate,
                3,
            ),

        "measurements":
            measurement.to_dict(),

        "signal_quality":
            quality.to_dict(),

        "r_peaks":
            peaks.tolist(),
    }