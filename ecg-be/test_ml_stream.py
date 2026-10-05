import json

from database import (
    SessionLocal,
    ECGSession,
    ECGRawPacket,
)

from ml.smart_ecg_stream import SmartECGStream


HARDWARE_FS = 200
TEST_DURATION_SECONDS = 30
TEST_SAMPLES = HARDWARE_FS * TEST_DURATION_SECONDS


def load_latest_ecg():
    db = SessionLocal()

    try:
        session = (
            db.query(ECGSession)
            .order_by(ECGSession.id.desc())
            .first()
        )

        if session is None:
            raise RuntimeError(
                "No ECG session found."
            )

        packets = (
            db.query(ECGRawPacket)
            .filter(
                ECGRawPacket.session_id
                == session.session_id,
                ECGRawPacket.sampling_rate
                == HARDWARE_FS,
            )
            .order_by(
                ECGRawPacket.id.asc()
            )
            .all()
        )

        samples = []

        for packet in packets:
            if not packet.samples_json:
                continue

            try:
                packet_samples = json.loads(
                    packet.samples_json
                )
            except json.JSONDecodeError:
                continue

            if isinstance(packet_samples, list):
                samples.extend(packet_samples)

        if len(samples) < TEST_SAMPLES:
            raise RuntimeError(
                f"Need {TEST_SAMPLES} samples, "
                f"found {len(samples)}."
            )

        return samples[-TEST_SAMPLES:]

    finally:
        db.close()


def main():

    print("=" * 70)
    print("SMART ECG STREAM TEST")
    print("=" * 70)

    samples = load_latest_ecg()

    print(
        f"Loaded {len(samples)} samples "
        f"({len(samples) / HARDWARE_FS:.1f} seconds)"
    )

    stream = SmartECGStream(
        buffer_seconds=20
    )

    print()
    print("Feeding ECG in 50-sample packets...")
    print("-" * 70)

    total_predictions = 0

    packet_size = 50

    prediction_samples = []

    duplicate_predictions = 0

    non_monotonic_predictions = 0

    too_close_predictions = 0

    previous_sample = None

    for start in range(
        0,
        len(samples),
        packet_size,
    ):

        packet = samples[
            start:start + packet_size
        ]

        predictions = stream.add_samples(
            packet
        )

        for prediction in predictions:

            sample = prediction[
                "r_peak_sample"
            ]

            # ------------------------------------------------
            # Check duplicate
            # ------------------------------------------------

            if sample in prediction_samples:
                duplicate_predictions += 1

            # ------------------------------------------------
            # Check chronological order
            # ------------------------------------------------

            if (
                previous_sample is not None
                and sample <= previous_sample
            ):
                non_monotonic_predictions += 1

            # ------------------------------------------------
            # Check minimum distance
            # ------------------------------------------------

            if (
                previous_sample is not None
                and sample - previous_sample < 50
            ):
                too_close_predictions += 1

            prediction_samples.append(
                sample
            )

            previous_sample = sample

            print(
                f"Prediction | "
                f"sample={sample:5d} | "
                f"class={prediction['predicted_class']} | "
                f"confidence="
                f"{prediction['confidence_pct']:.2f}%"
            )

            total_predictions += 1

    print()
    print("=" * 70)
    print("STREAM TEST COMPLETE")
    print("=" * 70)

    print(
        f"Total samples received: "
        f"{stream.total_samples_received}"
    )

    print(
        f"Final buffer size: "
        f"{len(stream.buffer)} samples"
    )

    print(
        f"ML predictions generated: "
        f"{total_predictions}"
    )

    print(
        f"Unique prediction samples: "
        f"{len(set(prediction_samples))}"
    )

    print()
    print("VALIDATION")
    print("-" * 70)

    print(
        f"Duplicate predictions: "
        f"{duplicate_predictions}"
    )

    print(
        f"Non-monotonic predictions: "
        f"{non_monotonic_predictions}"
    )

    print(
        f"Too-close predictions: "
        f"{too_close_predictions}"
    )

    if (
        duplicate_predictions == 0
        and non_monotonic_predictions == 0
        and too_close_predictions == 0
    ):
        print()
        print(
            "RESULT: PASS"
        )
        print(
            "Real-time ML stream ordering and "
            "deduplication are working correctly."
        )
    else:
        print()
        print(
            "RESULT: REVIEW REQUIRED"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()