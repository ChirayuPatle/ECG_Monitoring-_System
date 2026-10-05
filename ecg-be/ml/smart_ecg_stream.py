from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from ml.smart_ecg_inference import predict_beat
from ml.smart_ecg_preprocessing import (
    resample_signal,
    bandpass_filter,
    normalize_signal,
    detect_rpeaks,
    segment_beats,
)


# ============================================================
# Configuration
# ============================================================

HARDWARE_FS = 200
MODEL_FS = 360

BUFFER_SECONDS = 20
BUFFER_SIZE = HARDWARE_FS * BUFFER_SECONDS

BEAT_SIZE = 300

# Do not classify peaks too close to the newest data.
#
# This gives filtering + R-peak detection enough future context
# before a beat becomes eligible for ML inference.
STABILITY_SECONDS = 2
STABILITY_SAMPLES = HARDWARE_FS * STABILITY_SECONDS

# Minimum physiological distance between newly emitted peaks.
# 250 ms = 240 BPM maximum.
MIN_NEW_PEAK_DISTANCE = int(0.25 * HARDWARE_FS)


class SmartECGStream:
    """
    Real-time ECG -> SmartECG inference pipeline.

    Input:
        200 Hz ECG samples from ESP8266.

    Processing:
        200 Hz
          ↓
        360 Hz resampling
          ↓
        0.5–40 Hz bandpass
          ↓
        full-signal Z-score
          ↓
        R-peak detection
          ↓
        300-sample beat extraction
          ↓
        CNN + BiLSTM
          ↓
        N / S / V / F / Q

    Important:
        This class does NOT modify the existing ECG acquisition
        pipeline. It only consumes a copy of incoming samples.
    """

    def __init__(
        self,
        buffer_seconds: int = BUFFER_SECONDS,
    ):
        self.hardware_fs = HARDWARE_FS
        self.model_fs = MODEL_FS

        self.buffer_size = self.hardware_fs * buffer_seconds

        self.buffer = deque(maxlen=self.buffer_size)

        # Total number of raw 200 Hz samples received by this stream.
        self.total_samples_received = 0

        # Absolute 200 Hz sample positions that have already
        # been classified.
        self.processed_peaks: set[int] = set()

        # Last emitted absolute sample.
        self.last_processed_absolute_sample = -1

    # ========================================================
    # Public API
    # ========================================================

    def add_samples(
        self,
        samples: list[float] | np.ndarray,
    ) -> list[dict[str, Any]]:
        """
        Add a new packet of 200 Hz ECG samples.

        The existing ESP8266 packet size is 50 samples,
        corresponding to 250 ms of ECG.

        Returns:
            A list of newly generated ML predictions.
        """

        samples = np.asarray(samples, dtype=np.float64)

        if samples.ndim != 1:
            raise ValueError(
                f"Expected 1-D ECG samples, got shape {samples.shape}."
            )

        if len(samples) == 0:
            return []

        # Add incoming samples to rolling buffer.
        for sample in samples:
            self.buffer.append(float(sample))

        self.total_samples_received += len(samples)

        # Wait until enough historical context exists.
        if len(self.buffer) < self.buffer_size:
            return []

        return self._process_buffer()

    # ========================================================
    # Processing
    # ========================================================

    def _process_buffer(self) -> list[dict[str, Any]]:
        """
        Process the current rolling ECG buffer.

        Only stable R-peaks are allowed to reach the ML model.
        """

        signal_200 = np.asarray(
            self.buffer,
            dtype=np.float64,
        )

        if len(signal_200) < self.buffer_size:
            return []

        # ----------------------------------------------------
        # Step 1: 200 Hz -> 360 Hz
        # ----------------------------------------------------

        signal_360 = resample_signal(
            signal_200,
            orig_sr=self.hardware_fs,
            target_sr=self.model_fs,
        )

        # ----------------------------------------------------
        # Step 2: Bandpass filter
        # ----------------------------------------------------

        filtered = bandpass_filter(
            signal_360,
            sampling_rate=self.model_fs,
            lowcut=0.5,
            highcut=40.0,
        )

        # ----------------------------------------------------
        # Step 3: Full-signal Z-score
        # ----------------------------------------------------

        normalized = normalize_signal(filtered)

        # ----------------------------------------------------
        # Step 4: Detect R-peaks
        # ----------------------------------------------------

        rpeaks_360 = detect_rpeaks(
            normalized,
            sampling_rate=self.model_fs,
        )

        if len(rpeaks_360) == 0:
            return []

        # ----------------------------------------------------
        # Step 5: Extract 300-sample beats
        # ----------------------------------------------------

        beats, valid_peaks_360 = segment_beats(
            normalized,
            rpeaks_360,
            window_size=BEAT_SIZE,
        )

        if len(beats) == 0:
            return []

        # ----------------------------------------------------
        # Convert current rolling-buffer coordinates into
        # absolute 200 Hz sample coordinates.
        # ----------------------------------------------------

        buffer_start_absolute = (
            self.total_samples_received
            - len(signal_200)
        )

        buffer_end_absolute = (
            self.total_samples_received - 1
        )

        # ----------------------------------------------------
        # Stability boundary
        #
        # We don't classify the newest ~2 seconds.
        #
        # Example:
        #
        # buffer:
        #
        # |--------------------20 sec--------------------|
        # |------------------stable----------------|2 sec|
        #                                      ^
        #                              newest data
        #
        # This prevents edge effects from producing
        # unstable/late R-peaks.
        # ----------------------------------------------------

        stable_end_absolute = (
            buffer_end_absolute
            - STABILITY_SAMPLES
        )

        results: list[dict[str, Any]] = []

        # ----------------------------------------------------
        # Process peaks in chronological order.
        # ----------------------------------------------------

        for beat, peak_360 in zip(
            beats,
            valid_peaks_360,
        ):
            peak_360 = int(peak_360)

            # Convert 360 Hz position to 200 Hz position.
            peak_200_relative = int(
                round(
                    peak_360
                    * self.hardware_fs
                    / self.model_fs
                )
            )

            absolute_peak = (
                buffer_start_absolute
                + peak_200_relative
            )

            # ------------------------------------------------
            # Ignore peaks outside the rolling buffer.
            # ------------------------------------------------

            if absolute_peak < buffer_start_absolute:
                continue

            if absolute_peak > buffer_end_absolute:
                continue

            # ------------------------------------------------
            # Do not classify unstable/newest data.
            # ------------------------------------------------

            if absolute_peak > stable_end_absolute:
                continue

            # ------------------------------------------------
            # Already classified?
            # ------------------------------------------------

            if absolute_peak in self.processed_peaks:
                continue

            # ------------------------------------------------
            # Enforce chronological ordering.
            #
            # This is an additional safety check against
            # edge-related R-peak movement.
            # ------------------------------------------------

            if (
                self.last_processed_absolute_sample >= 0
                and absolute_peak
                <= self.last_processed_absolute_sample
            ):
                continue

            # ------------------------------------------------
            # Physiological minimum distance.
            # ------------------------------------------------

            if (
                self.last_processed_absolute_sample >= 0
                and (
                    absolute_peak
                    - self.last_processed_absolute_sample
                    < MIN_NEW_PEAK_DISTANCE
                )
            ):
                continue

            # ------------------------------------------------
            # ML inference
            # ------------------------------------------------

            prediction = predict_beat(beat)

            result = {
                "r_peak_sample": absolute_peak,

                "r_peak_time_seconds": round(
                    absolute_peak / self.hardware_fs,
                    3,
                ),

                "predicted_class": prediction[
                    "predicted_class"
                ],

                "confidence_pct": prediction[
                    "confidence"
                ],

                "probabilities": prediction[
                    "probabilities"
                ],
            }

            results.append(result)

            # Mark as processed.
            self.processed_peaks.add(
                absolute_peak
            )

            self.last_processed_absolute_sample = (
                absolute_peak
            )

        # ----------------------------------------------------
        # Remove old processed peaks that can no longer appear
        # in the rolling buffer.
        # ----------------------------------------------------

        self.processed_peaks = {
            peak
            for peak in self.processed_peaks
            if peak >= buffer_start_absolute
        }

        return results

    # ========================================================
    # Status
    # ========================================================

    def get_status(self) -> dict[str, Any]:
        """
        Return current stream-processing status.
        """

        return {
            "buffer_samples": len(self.buffer),

            "buffer_capacity": self.buffer_size,

            "buffer_seconds": round(
                len(self.buffer)
                / self.hardware_fs,
                2,
            ),

            "total_samples_received": (
                self.total_samples_received
            ),

            "processed_beats": len(
                self.processed_peaks
            ),

            "last_processed_sample": (
                self.last_processed_absolute_sample
            ),

            "stability_seconds": STABILITY_SECONDS,

            "ready": (
                len(self.buffer)
                >= self.buffer_size
            ),
        }

    # ========================================================
    # Reset
    # ========================================================

    def reset(self):
        """
        Completely reset the rolling ML stream.
        """

        self.buffer.clear()

        self.total_samples_received = 0

        self.processed_peaks.clear()

        self.last_processed_absolute_sample = -1