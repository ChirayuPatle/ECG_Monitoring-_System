import json
from pathlib import Path

import numpy as np

from database import SessionLocal, ECGSession, ECGRawPacket
from ml.smart_ecg_inference import load_model, CLASSES
from ml.smart_ecg_preprocessing import (
    resample_signal,
    bandpass_filter,
    normalize_signal,
    detect_rpeaks,
    segment_beats,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "mitbih"
    / "best_model.keras"
)

HARDWARE_FS = 200
MODEL_FS = 360

# Use only the latest 20 seconds for this first test.
TEST_DURATION_SECONDS = 20
TEST_SAMPLES = HARDWARE_FS * TEST_DURATION_SECONDS

BEAT_SIZE = 300
MAX_TEST_BEATS = 10


# ============================================================
# LOAD REAL ECG FROM DATABASE
# ============================================================

def load_real_ecg_from_latest_session():
    db = SessionLocal()

    try:
        session = (
            db.query(ECGSession)
            .order_by(ECGSession.id.desc())
            .first()
        )

        if session is None:
            raise RuntimeError("No ECG session found in database.")

        print("=" * 70)
        print("REAL ECG → SMART ECG ML TEST")
        print("=" * 70)

        print(f"Session ID:       {session.session_id}")
        print(f"Session samples:  {session.samples_received}")
        print(f"Sampling rate:    {session.sampling_rate} Hz")

        packets = (
            db.query(ECGRawPacket)
            .filter(
                ECGRawPacket.session_id == session.session_id,
                ECGRawPacket.sampling_rate == HARDWARE_FS,
            )
            .order_by(ECGRawPacket.id.asc())
            .all()
        )

        if not packets:
            raise RuntimeError(
                "No raw ECG packets found for the latest session."
            )

        samples = []

        for packet in packets:
            if not packet.samples_json:
                continue

            try:
                packet_samples = json.loads(packet.samples_json)
            except json.JSONDecodeError:
                print(
                    f"Warning: Could not decode packet {packet.id}. "
                    "Skipping."
                )
                continue

            if isinstance(packet_samples, list):
                samples.extend(packet_samples)

        if len(samples) < TEST_SAMPLES:
            raise RuntimeError(
                f"Not enough ECG data. "
                f"Need {TEST_SAMPLES} samples but found {len(samples)}."
            )

        # Take the latest continuous segment.
        samples = samples[-TEST_SAMPLES:]

        signal = np.asarray(samples, dtype=np.float64)

        print(f"Raw samples used: {len(signal)}")
        print(
            f"Raw duration:     "
            f"{len(signal) / HARDWARE_FS:.2f} seconds"
        )

        return signal

    finally:
        db.close()


# ============================================================
# MAIN ML PIPELINE
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Load real hardware ECG
    # --------------------------------------------------------

    raw_signal = load_real_ecg_from_latest_session()

    print("\n" + "-" * 70)
    print("STEP 1: RAW ECG")
    print("-" * 70)

    print(f"Shape: {raw_signal.shape}")
    print(f"Min:   {np.min(raw_signal):.3f}")
    print(f"Max:   {np.max(raw_signal):.3f}")
    print(f"Mean:  {np.mean(raw_signal):.3f}")
    print(f"Std:   {np.std(raw_signal):.3f}")

    # --------------------------------------------------------
    # 2. Resample 200 Hz → 360 Hz
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("STEP 2: RESAMPLE 200 Hz → 360 Hz")
    print("-" * 70)

    resampled = resample_signal(
        raw_signal,
        orig_sr=HARDWARE_FS,
        target_sr=MODEL_FS,
    )

    print(f"Resampled samples: {len(resampled)}")
    print(
        f"Expected approximately: "
        f"{TEST_DURATION_SECONDS * MODEL_FS}"
    )

    # --------------------------------------------------------
    # 3. Bandpass filter 0.5–40 Hz
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("STEP 3: BANDPASS FILTER")
    print("-" * 70)

    filtered = bandpass_filter(
        resampled,
        sampling_rate=MODEL_FS,
        lowcut=0.5,
        highcut=40.0,
    )

    print("Filter: Butterworth 4th order")
    print("Range:  0.5–40 Hz")

    # --------------------------------------------------------
    # 4. Full-signal Z-score normalization
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("STEP 4: FULL-SIGNAL Z-SCORE")
    print("-" * 70)

    normalized = normalize_signal(filtered)

    print(
        f"Mean after normalization: "
        f"{np.mean(normalized):.6f}"
    )
    print(
        f"Std after normalization:  "
        f"{np.std(normalized):.6f}"
    )

    # --------------------------------------------------------
    # 5. Detect R-peaks
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("STEP 5: R-PEAK DETECTION")
    print("-" * 70)

    rpeaks = detect_rpeaks(
        normalized,
        sampling_rate=MODEL_FS,
    )

    print(f"R-peaks detected: {len(rpeaks)}")

    if len(rpeaks) == 0:
        raise RuntimeError(
            "No R-peaks detected. "
            "The signal cannot currently be passed to the model."
        )

    print(f"First peaks: {rpeaks[:10]}")

    # --------------------------------------------------------
    # 6. Extract 300-sample beats
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("STEP 6: BEAT SEGMENTATION")
    print("-" * 70)

    # segment_beats returns:
    #   beats       -> shape (N, 300)
    #   valid_peaks -> corresponding R-peak positions
    beats, valid_peaks = segment_beats(
        normalized,
        rpeaks,
        window_size=BEAT_SIZE,
    )

    print(f"Valid beats: {len(beats)}")
    print(f"Beat array shape: {beats.shape}")
    print(f"Valid R-peaks: {valid_peaks[:10].tolist()}")

    if len(beats) == 0:
        raise RuntimeError(
            "R-peaks were detected, but no valid 300-sample "
            "beat windows could be created."
        )

    # --------------------------------------------------------
    # 7. Load trained model
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("STEP 7: LOAD TRAINED MODEL")
    print("-" * 70)

    print(f"Model: {MODEL_PATH}")

    model = load_model()

    print("Model loaded successfully.")
    print(f"Model input shape:  {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")

    # --------------------------------------------------------
    # 8. Run inference
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("STEP 8: MODEL INFERENCE")
    print("-" * 70)

    results = []

    # Limit first test to first 10 beats.
    test_beats = beats[:MAX_TEST_BEATS]
    test_peaks = valid_peaks[:MAX_TEST_BEATS]

    for index, (beat, peak) in enumerate(
        zip(test_beats, test_peaks),
        start=1,
    ):
        beat = np.asarray(
            beat,
            dtype=np.float32,
        )

        # Exact model input:
        # (300,) → (1, 300, 1)
        x = beat.reshape(1, BEAT_SIZE, 1)

        probabilities = model.predict(
            x,
            verbose=0,
        )[0]

        predicted_index = int(
            np.argmax(probabilities)
        )

        predicted_class = CLASSES[predicted_index]

        confidence = float(
            probabilities[predicted_index] * 100
        )

        probability_map = {
            CLASSES[i]: round(
                float(probabilities[i] * 100),
                2,
            )
            for i in range(len(CLASSES))
        }

        result = {
            "beat": index,
            "r_peak": int(peak),
            "predicted_class": predicted_class,
            "confidence_pct": round(confidence, 2),
            "probabilities": probability_map,
        }

        results.append(result)

        print(
            f"Beat {index:02d} | "
            f"R-peak {int(peak):04d} | "
            f"{predicted_class} "
            f"({confidence:.2f}%) | "
            f"{probability_map}"
        )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ML INTEGRATION TEST COMPLETE")
    print("=" * 70)

    print(f"Raw sampling rate:       {HARDWARE_FS} Hz")
    print(f"Model sampling rate:     {MODEL_FS} Hz")
    print(f"Raw samples used:        {len(raw_signal)}")
    print(f"R-peaks detected:        {len(rpeaks)}")
    print(f"Valid beats extracted:   {len(beats)}")
    print(f"Beats classified:        {len(test_beats)}")

    print("\nPredictions:")

    for result in results:
        print(
            f"  Beat {result['beat']:02d}: "
            f"{result['predicted_class']} "
            f"({result['confidence_pct']:.2f}%)"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()