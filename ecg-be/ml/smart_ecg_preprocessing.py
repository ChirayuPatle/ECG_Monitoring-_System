"""
SmartECG-HD compatible preprocessing for the live hardware ECG stream.

This module intentionally mirrors the team's preprocessing pipeline:

    200 Hz ADC
        -> resample to 360 Hz
        -> Butterworth 0.5-40 Hz bandpass
        -> full-signal z-score normalization
        -> NeuroKit2 R-peak detection at 360 Hz
        -> 300-sample beat windows centered on the R peak
        -> per-beat z-score normalization

Expected model input:
    (N, 300, 1)

Important:
- Raw ADC values are not modified in the database/acquisition pipeline.
- This preprocessing is only for the SmartECG ML path.
- Do not run this pipeline independently on tiny ESP packets; use a
  sufficiently long signal/rolling buffer so filtfilt and R-peak detection
  have enough context.
"""

from __future__ import annotations

from typing import Sequence

import neurokit2 as nk
import numpy as np
from scipy.signal import butter, filtfilt, resample_poly


DEFAULT_ORIG_SR = 200
DEFAULT_TARGET_SR = 360
DEFAULT_LOW_CUT_HZ = 0.5
DEFAULT_HIGH_CUT_HZ = 40.0
DEFAULT_FILTER_ORDER = 4
DEFAULT_BEAT_SIZE = 300
DEFAULT_R_PEAK_POSITION = 150


def resample_signal(
    signal_200hz: Sequence[float] | np.ndarray,
    orig_sr: int = DEFAULT_ORIG_SR,
    target_sr: int = DEFAULT_TARGET_SR,
) -> np.ndarray:
    """
    Resample the acquisition signal to the SmartECG model's 360 Hz rate.

    The team's implementation uses polyphase resampling with up=9/down=5
    for the 200 Hz -> 360 Hz conversion.
    """
    signal = np.asarray(signal_200hz, dtype=np.float64)

    if signal.ndim != 1:
        raise ValueError(f"Expected a 1-D ECG signal, got shape {signal.shape}.")

    if signal.size == 0:
        return signal.copy()

    if orig_sr == target_sr:
        return signal

    if orig_sr == 200 and target_sr == 360:
        return resample_poly(signal, up=9, down=5, axis=0)

    # General fallback while preserving the team's exact 200 -> 360 path.
    from math import gcd

    divisor = gcd(int(orig_sr), int(target_sr))
    up = int(target_sr) // divisor
    down = int(orig_sr) // divisor
    return resample_poly(signal, up=up, down=down, axis=0)


def bandpass_filter(
    signal: Sequence[float] | np.ndarray,
    sampling_rate: int = DEFAULT_TARGET_SR,
    lowcut: float = DEFAULT_LOW_CUT_HZ,
    highcut: float = DEFAULT_HIGH_CUT_HZ,
    filter_order: int = DEFAULT_FILTER_ORDER,
) -> np.ndarray:
    """Apply the team's 4th-order Butterworth 0.5-40 Hz bandpass filter."""
    signal = np.asarray(signal, dtype=np.float64)

    if signal.ndim != 1:
        raise ValueError(f"Expected a 1-D ECG signal, got shape {signal.shape}.")

    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist

    if not 0 < low < high < 1:
        raise ValueError(
            f"Invalid bandpass frequencies: lowcut={lowcut}, "
            f"highcut={highcut}, sampling_rate={sampling_rate}."
        )

    b, a = butter(filter_order, [low, high], btype="band")
    filtered = filtfilt(b, a, signal)
    return filtered


def normalize_signal(signal: Sequence[float] | np.ndarray) -> np.ndarray:
    """Apply z-score normalization exactly as in the team's pipeline."""
    signal = np.asarray(signal, dtype=np.float64)

    if signal.size == 0:
        return signal.copy()

    std = np.std(signal)

    if std == 0:
        return signal - np.mean(signal)

    return (signal - np.mean(signal)) / std


def detect_rpeaks(
    signal: Sequence[float] | np.ndarray,
    sampling_rate: int = DEFAULT_TARGET_SR,
) -> np.ndarray:
    """
    Detect R peaks using NeuroKit2, matching the team's implementation.
    """
    signal = np.asarray(signal, dtype=np.float64)

    if signal.ndim != 1:
        raise ValueError(f"Expected a 1-D ECG signal, got shape {signal.shape}.")

    try:
        _, info = nk.ecg_peaks(signal, sampling_rate=sampling_rate)
        return np.asarray(info["ECG_R_Peaks"], dtype=int)
    except Exception as exc:
        print(f"R-peak detection error: {exc}")
        return np.array([], dtype=int)


def segment_beats(
    signal: Sequence[float] | np.ndarray,
    rpeaks: Sequence[int] | np.ndarray,
    window_size: int = DEFAULT_BEAT_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract 300-sample beat windows centered on each R peak.

    For window_size=300:
        start = peak - 150
        end   = peak + 150
        signal[start:end] -> exactly 300 samples
        R peak             -> index 150

    Boundary peaks are skipped, matching the team's implementation.
    Each accepted beat is z-score normalized independently.
    """
    signal = np.asarray(signal, dtype=np.float64)
    rpeaks = np.asarray(rpeaks, dtype=int)

    if signal.ndim != 1:
        raise ValueError(f"Expected a 1-D ECG signal, got shape {signal.shape}.")

    if window_size <= 0 or window_size % 2 != 0:
        raise ValueError("window_size must be a positive even integer.")

    half_window = window_size // 2
    beats: list[np.ndarray] = []
    valid_peaks: list[int] = []

    for peak in rpeaks:
        start = int(peak) - half_window
        end = int(peak) + half_window

        if start >= 0 and end < len(signal):
            beat = signal[start:end]

            # Keep the exact team behavior: normalize every beat separately.
            beats.append(normalize_signal(beat))
            valid_peaks.append(int(peak))

    if not beats:
        return (
            np.empty((0, window_size), dtype=np.float64),
            np.array([], dtype=int),
        )

    return np.asarray(beats, dtype=np.float64), np.asarray(valid_peaks, dtype=int)


def preprocess_hardware_stream(
    raw_200hz_adc_signal: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the complete SmartECG-HD preprocessing pipeline.

    Returns:
        beats:
            Shape (N, 300), ready to reshape to (N, 300, 1).
        normalized_signal:
            Full normalized 360 Hz signal.
        valid_peaks:
            R-peak indices in the 360 Hz normalized signal.
    """
    signal_200hz = np.asarray(raw_200hz_adc_signal, dtype=np.float64)

    if signal_200hz.ndim != 1:
        raise ValueError(
            f"Expected a 1-D 200 Hz ADC signal, got shape {signal_200hz.shape}."
        )

    if signal_200hz.size == 0:
        return (
            np.empty((0, DEFAULT_BEAT_SIZE), dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=int),
        )

    if not np.all(np.isfinite(signal_200hz)):
        raise ValueError("Input ECG signal contains NaN or infinite values.")

    signal_360hz = resample_signal(
        signal_200hz,
        orig_sr=DEFAULT_ORIG_SR,
        target_sr=DEFAULT_TARGET_SR,
    )

    filtered = bandpass_filter(
        signal_360hz,
        sampling_rate=DEFAULT_TARGET_SR,
        lowcut=DEFAULT_LOW_CUT_HZ,
        highcut=DEFAULT_HIGH_CUT_HZ,
        filter_order=DEFAULT_FILTER_ORDER,
    )

    normalized = normalize_signal(filtered)

    rpeaks = detect_rpeaks(
        normalized,
        sampling_rate=DEFAULT_TARGET_SR,
    )

    beats, valid_peaks = segment_beats(
        normalized,
        rpeaks,
        window_size=DEFAULT_BEAT_SIZE,
    )

    return beats, normalized, valid_peaks


def prepare_model_input(
    beats: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """
    Convert segmented beats from (N, 300) to the model's (N, 300, 1) input.

    This function does not alter beat values.
    """
    beats = np.asarray(beats, dtype=np.float32)

    if beats.ndim != 2 or beats.shape[1] != DEFAULT_BEAT_SIZE:
        raise ValueError(
            f"Expected beats with shape (N, {DEFAULT_BEAT_SIZE}), "
            f"got {beats.shape}."
        )

    return beats.reshape(-1, DEFAULT_BEAT_SIZE, 1)


def validate_model_input(model_input: np.ndarray) -> dict:
    """Return basic validation information before sending data to the model."""
    array = np.asarray(model_input)

    expected_shape = DEFAULT_BEAT_SIZE

    valid = (
        array.ndim == 3
        and array.shape[1] == expected_shape
        and array.shape[2] == 1
        and np.all(np.isfinite(array))
    )

    return {
        "valid": bool(valid),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "expected_shape": [None, DEFAULT_BEAT_SIZE, 1],
    }
