"""
SmartECG-HD model inference.

Model:
    models/mitbih/best_model.keras

Expected input:
    (N, 300, 1)

Expected output:
    (N, 5)

Classes:
    N - Normal
    S - Supraventricular Ectopic Beat
    V - Premature Ventricular Contraction
    F - Fusion
    Q - Unknown / Unclassifiable
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "mitbih"
    / "best_model.keras"
)

CLASSES = [
    "N",
    "S",
    "V",
    "F",
    "Q",
]

EXPECTED_BEAT_SIZE = 300
EXPECTED_CHANNELS = 1
EXPECTED_CLASS_COUNT = 5


# ============================================================
# MODEL LOADING
# ============================================================

_model = None


def load_model():
    """
    Load the SmartECG model once.

    Subsequent calls return the already-loaded model.
    """

    global _model

    if _model is None:

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"SmartECG model not found: {MODEL_PATH}"
            )

        print(
            f"Loading SmartECG model from: {MODEL_PATH}"
        )

        _model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
        )

        validate_model(_model)

        print("SmartECG model loaded successfully.")

    return _model


# ============================================================
# MODEL VALIDATION
# ============================================================

def validate_model(model) -> None:
    """
    Verify that the loaded model matches the expected
    SmartECG architecture interface.
    """

    input_shape = model.input_shape
    output_shape = model.output_shape

    print(f"Model input shape:  {input_shape}")
    print(f"Model output shape: {output_shape}")

    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------

    if len(input_shape) != 3:
        raise ValueError(
            f"Expected 3D model input, got {input_shape}"
        )

    if input_shape[1] != EXPECTED_BEAT_SIZE:
        raise ValueError(
            "Unexpected model beat size: "
            f"{input_shape[1]}. "
            f"Expected {EXPECTED_BEAT_SIZE}."
        )

    if input_shape[2] != EXPECTED_CHANNELS:
        raise ValueError(
            "Unexpected model channel count: "
            f"{input_shape[2]}. "
            f"Expected {EXPECTED_CHANNELS}."
        )

    # --------------------------------------------------------
    # Output validation
    # --------------------------------------------------------

    if len(output_shape) != 2:
        raise ValueError(
            f"Expected 2D model output, got {output_shape}"
        )

    if output_shape[1] != EXPECTED_CLASS_COUNT:
        raise ValueError(
            "Unexpected number of output classes: "
            f"{output_shape[1]}. "
            f"Expected {EXPECTED_CLASS_COUNT}."
        )


# ============================================================
# INPUT PREPARATION
# ============================================================

def prepare_input(
    beat: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """
    Convert one ECG beat into model input shape:

        (300,)
            ↓
        (1, 300, 1)

    The beat is assumed to have already gone through
    SmartECG preprocessing and per-beat normalization.
    """

    beat = np.asarray(
        beat,
        dtype=np.float32,
    )

    if beat.shape == (300,):
        beat = beat.reshape(
            1,
            EXPECTED_BEAT_SIZE,
            EXPECTED_CHANNELS,
        )

    elif beat.shape == (300, 1):
        beat = beat.reshape(
            1,
            EXPECTED_BEAT_SIZE,
            EXPECTED_CHANNELS,
        )

    elif beat.shape == (1, 300, 1):
        pass

    else:
        raise ValueError(
            "Invalid ECG beat shape. "
            f"Expected (300,), (300, 1), or (1, 300, 1), "
            f"got {beat.shape}."
        )

    if not np.all(np.isfinite(beat)):
        raise ValueError(
            "ECG beat contains NaN or infinite values."
        )

    return beat


# ============================================================
# SINGLE BEAT PREDICTION
# ============================================================

def predict_beat(
    beat: Sequence[float] | np.ndarray,
) -> dict:
    """
    Predict the arrhythmia class for one ECG beat.

    Returns:

    {
        "predicted_class": "N",
        "confidence": 96.42,
        "probabilities": {
            "N": 96.42,
            "S": 1.21,
            "V": 1.85,
            "F": 0.31,
            "Q": 0.21
        }
    }
    """

    model = load_model()

    model_input = prepare_input(beat)

    predictions = model.predict(
        model_input,
        verbose=0,
    )

    probabilities = np.asarray(
        predictions,
        dtype=np.float64,
    )[0]

    # --------------------------------------------------------
    # Validate output
    # --------------------------------------------------------

    if probabilities.shape != (
        EXPECTED_CLASS_COUNT,
    ):
        raise ValueError(
            "Unexpected model prediction shape: "
            f"{probabilities.shape}"
        )

    # --------------------------------------------------------
    # Convert logits to probabilities if necessary
    # --------------------------------------------------------

    probability_sum = np.sum(probabilities)

    if (
        np.any(probabilities < 0)
        or not np.isclose(
            probability_sum,
            1.0,
            atol=1e-3,
        )
    ):
        probabilities = tf.nn.softmax(
            probabilities
        ).numpy()

    # --------------------------------------------------------
    # Predicted class
    # --------------------------------------------------------

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = CLASSES[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index] * 100
    )

    probability_dict = {
        class_name: round(
            float(probability * 100),
            2,
        )
        for class_name, probability
        in zip(
            CLASSES,
            probabilities,
        )
    }

    return {
        "predicted_class": predicted_class,
        "confidence": round(
            confidence,
            2,
        ),
        "probabilities": probability_dict,
    }


# ============================================================
# BATCH PREDICTION
# ============================================================

def predict_beats(
    beats: np.ndarray,
) -> list[dict]:
    """
    Predict multiple preprocessed ECG beats.

    Expected input:
        (N, 300)
        or
        (N, 300, 1)
    """

    model = load_model()

    beats = np.asarray(
        beats,
        dtype=np.float32,
    )

    if beats.ndim == 2:

        if beats.shape[1] != EXPECTED_BEAT_SIZE:
            raise ValueError(
                f"Expected (N, 300), got {beats.shape}"
            )

        beats = beats.reshape(
            -1,
            EXPECTED_BEAT_SIZE,
            EXPECTED_CHANNELS,
        )

    elif beats.ndim == 3:

        if beats.shape[1:] != (
            EXPECTED_BEAT_SIZE,
            EXPECTED_CHANNELS,
        ):
            raise ValueError(
                "Expected (N, 300, 1), "
                f"got {beats.shape}"
            )

    else:
        raise ValueError(
            "Expected ECG beats with shape "
            "(N, 300) or (N, 300, 1)."
        )

    if not np.all(np.isfinite(beats)):
        raise ValueError(
            "ECG beats contain NaN or infinite values."
        )

    predictions = model.predict(
        beats,
        verbose=0,
    )

    predictions = np.asarray(
        predictions,
        dtype=np.float64,
    )

    # Convert logits to probabilities if required.
    row_sums = np.sum(
        predictions,
        axis=1,
    )

    if (
        np.any(predictions < 0)
        or not np.allclose(
            row_sums,
            1.0,
            atol=1e-3,
        )
    ):
        predictions = tf.nn.softmax(
            predictions,
            axis=1,
        ).numpy()

    results = []

    for probabilities in predictions:

        predicted_index = int(
            np.argmax(probabilities)
        )

        predicted_class = CLASSES[
            predicted_index
        ]

        confidence = float(
            probabilities[predicted_index]
            * 100
        )

        probability_dict = {
            class_name: round(
                float(probability * 100),
                2,
            )
            for class_name, probability
            in zip(
                CLASSES,
                probabilities,
            )
        }

        results.append(
            {
                "predicted_class": predicted_class,
                "confidence": round(
                    confidence,
                    2,
                ),
                "probabilities": probability_dict,
            }
        )

    return results


# ============================================================
# MODEL INFORMATION
# ============================================================

def get_model_info() -> dict:
    """
    Return basic SmartECG model information.
    """

    model = load_model()

    return {
        "model": "SmartECG-HD",
        "model_path": str(MODEL_PATH),
        "input_shape": list(
            model.input_shape
        ),
        "output_shape": list(
            model.output_shape
        ),
        "classes": CLASSES,
        "parameter_count": model.count_params(),
    }


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SmartECG Model Test")
    print("=" * 60)

    model_info = get_model_info()

    print(
        f"Input shape:  "
        f"{model_info['input_shape']}"
    )

    print(
        f"Output shape: "
        f"{model_info['output_shape']}"
    )

    print(
        f"Parameters:    "
        f"{model_info['parameter_count']}"
    )

    print(
        f"Classes:       "
        f"{model_info['classes']}"
    )

    # --------------------------------------------------------
    # Temporary synthetic test beat
    #
    # This verifies that TensorFlow + model inference works.
    # It does NOT validate ECG classification accuracy.
    # --------------------------------------------------------

    test_beat = np.zeros(
        EXPECTED_BEAT_SIZE,
        dtype=np.float32,
    )

    result = predict_beat(
        test_beat
    )

    print()
    print("Test prediction:")
    print(result)

    print("=" * 60)
    print("Model inference test completed.")
    print("=" * 60)