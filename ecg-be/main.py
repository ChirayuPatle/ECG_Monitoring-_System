import asyncio
import json
import math
import uuid

from datetime import datetime, timezone
from collections import defaultdict, deque
from typing import Optional

from fastapi import (
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field

from sqlalchemy import desc

from ecg_analysis import (
    SAMPLES_PER_BEAT,
    MIN_ANALYSIS_SAMPLES,
    filter_ecg,
    detect_r_peaks,
    segment_ecg_beats,
    analyze_window,
    build_ml_dataset,
    validate_ml_dataset,
)

from database import (
    SessionLocal,
    ECGSession,
    ECGMeasurement,
    ECGSignalQuality,
    ECGRawPacket,
    ECGBeat,
    ECGMLPrediction,
    create_tables,
    get_database_info,
)

from ml.smart_ecg_stream import SmartECGStream


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_SAMPLING_RATE = 200

ANALYSIS_WINDOW_SECONDS = 10

ANALYSIS_WINDOW_SAMPLES = (
    DEFAULT_SAMPLING_RATE
    * ANALYSIS_WINDOW_SECONDS
)

DEVICE_DEFAULT = "ecg_esp8266_01"


# Run normal ECG analysis approximately once per second.
#
# Raw ECG streaming continues at the full 200 Hz.
#
# This controls ONLY the expensive derived-metric analysis.
ANALYSIS_INTERVAL_SECONDS = 1.0


# SmartECG ML queue capacity.
#
# ESP8266 normally sends:
#
# 50 samples / 200 Hz = 250 ms
#
# of ECG per packet.
ML_QUEUE_MAXSIZE = 20


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="ECG Monitoring Backend",
    version="3.5.0",
    description=(
        "ECG acquisition, analysis, session storage "
        "and non-blocking SmartECG ML inference backend."
    ),
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# RUNTIME STATE
# ============================================================

connected_clients: set[WebSocket] = set()


# ------------------------------------------------------------
# Recent raw ECG samples.
#
# 10 seconds × 200 Hz = 2000 samples.
# ------------------------------------------------------------

live_buffers: dict[str, deque] = defaultdict(
    lambda: deque(
        maxlen=ANALYSIS_WINDOW_SAMPLES
    )
)


# ------------------------------------------------------------
# Active session per device.
# ------------------------------------------------------------

active_sessions: dict[str, str] = {}


# ============================================================
# SMART ECG ML RUNTIME
# ============================================================

# device_id -> SmartECGStream
ml_streams: dict[str, SmartECGStream] = {}


# device_id -> ML queue
ml_queues: dict[
    str,
    asyncio.Queue[list[float]],
] = {}


# device_id -> ML worker task
ml_workers: dict[
    str,
    asyncio.Task,
] = {}


# ============================================================
# ECG ANALYSIS RUNTIME
# ============================================================

# device_id -> analysis worker task
analysis_workers: dict[
    str,
    asyncio.Task,
] = {}


# device_id -> event used to wake the analysis worker
analysis_events: dict[
    str,
    asyncio.Event,
] = {}


# device_id -> last time an analysis was scheduled
analysis_last_scheduled: dict[
    str,
    float,
] = {}


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> datetime:

    return datetime.now(
        timezone.utc
    ).replace(
        tzinfo=None
    )


def generate_session_id() -> str:

    return f"ecg_{uuid.uuid4().hex[:12]}"


def clean_samples(samples) -> list[float]:

    cleaned = []

    for value in samples:

        try:

            value = float(value)

            if math.isfinite(value):

                cleaned.append(
                    value
                )

        except (
            TypeError,
            ValueError,
        ):

            continue

    return cleaned


def get_session_by_public_id(
    db,
    session_id: str,
) -> Optional[ECGSession]:

    return (
        db.query(
            ECGSession
        )
        .filter(
            ECGSession.session_id
            == session_id
        )
        .first()
    )


def create_or_get_session(
    db,
    device_id: str,
    sampling_rate: int,
) -> ECGSession:

    existing_id = active_sessions.get(
        device_id
    )

    if existing_id:

        existing = get_session_by_public_id(
            db,
            existing_id,
        )

        if (
            existing
            and existing.status == "active"
        ):

            return existing

    existing = (
        db.query(
            ECGSession
        )
        .filter(
            ECGSession.device_id
            == device_id,

            ECGSession.status
            == "active",
        )
        .order_by(
            desc(
                ECGSession.started_at
            )
        )
        .first()
    )

    if existing:

        active_sessions[
            device_id
        ] = existing.session_id

        return existing

    now = utc_now()

    session = ECGSession(
        session_id=generate_session_id(),

        device_id=device_id,

        started_at=now,

        last_packet_at=now,

        sampling_rate=sampling_rate,

        samples_received=0,

        packets_received=0,

        status="active",

        lead="single-channel",
    )

    db.add(
        session
    )

    db.commit()

    db.refresh(
        session
    )

    active_sessions[
        device_id
    ] = session.session_id

    return session


async def broadcast(
    payload: dict,
):
    """
    Send a payload to all connected WebSocket clients.

    A slow or broken client must not hold the ECG pipeline
    indefinitely.
    """

    if not connected_clients:

        return

    message = json.dumps(
        payload
    )

    disconnected = []

    for websocket in list(
        connected_clients
    ):

        try:

            await asyncio.wait_for(
                websocket.send_text(
                    message
                ),
                timeout=0.05,
            )

        except Exception:

            disconnected.append(
                websocket
            )

    for websocket in disconnected:

        connected_clients.discard(
            websocket
        )


def serialize_analysis(
    analysis: dict,
) -> dict:

    return {
        "heart_rate_bpm": analysis.get(
            "heart_rate_bpm"
        ),

        "rr_interval_ms": analysis.get(
            "rr_interval_ms"
        ),

        "r_peak_count": analysis.get(
            "r_peak_count",
            0,
        ),

        "signal_quality": analysis.get(
            "signal_quality"
        ),

        "quality_score": analysis.get(
            "quality_score"
        ),

        "analysis_confidence": analysis.get(
            "analysis_confidence"
        ),

        "measurement_status": analysis.get(
            "measurement_status",
            "Normal monitoring",
        ),
    }


def serialize_quality(
    quality: dict,
) -> dict:

    return {
        "score": quality.get(
            "score"
        ),

        "label": quality.get(
            "label"
        ),

        "baseline_wander": quality.get(
            "baseline_wander"
        ),

        "noise_rms": quality.get(
            "noise_rms"
        ),

        "clipping_ratio": quality.get(
            "clipping_ratio"
        ),

        "notes": quality.get(
            "notes",
            [],
        ),
    }


# ============================================================
# SMART ECG ML
# ============================================================

def get_ml_stream(
    device_id: str,
) -> SmartECGStream:
    """
    Get or create the SmartECG rolling inference
    stream for a specific ECG device.
    """

    if device_id not in ml_streams:

        ml_streams[
            device_id
        ] = SmartECGStream(
            buffer_seconds=20
        )

    return ml_streams[
        device_id
    ]


def get_ml_queue(
    device_id: str,
) -> asyncio.Queue[list[float]]:
    """
    Get or create the per-device ML queue.
    """

    if device_id not in ml_queues:

        ml_queues[
            device_id
        ] = asyncio.Queue(
            maxsize=ML_QUEUE_MAXSIZE
        )

    return ml_queues[
        device_id
    ]


# ============================================================
# SMART ECG ML DATABASE PERSISTENCE
# ============================================================

def save_ml_predictions(
    device_id: str,
    predictions: list[dict],
):
    """
    Persist SmartECG predictions for the active ECG session.

    This function is called from the ML background worker.
    It does not run inside the /api/ecg request path.
    """

    if not predictions:

        return

    session_id = active_sessions.get(
        device_id
    )

    if not session_id:

        print(
            f"SmartECG ML persistence skipped: "
            f"no active session for {device_id}"
        )

        return

    db = SessionLocal()

    try:

        session = get_session_by_public_id(
            db,
            session_id,
        )

        if not session:

            print(
                f"SmartECG ML persistence skipped: "
                f"session {session_id} not found"
            )

            return

        # ----------------------------------------------------
        # Determine the next prediction index.
        # ----------------------------------------------------

        latest_prediction = (
            db.query(
                ECGMLPrediction
            )
            .filter(
                ECGMLPrediction.session_id
                == session.session_id
            )
            .order_by(
                desc(
                    ECGMLPrediction.prediction_index
                )
            )
            .first()
        )

        if latest_prediction:

            next_index = (
                latest_prediction.prediction_index
                + 1
            )

        else:

            next_index = 1

        stored_count = 0

        for prediction in predictions:

            try:

                r_peak_sample = int(
                    prediction.get(
                        "r_peak_sample",
                        -1,
                    )
                )

            except (
                TypeError,
                ValueError,
            ):

                continue

            if r_peak_sample < 0:

                continue

            # ------------------------------------------------
            # Protect against duplicate predictions.
            # ------------------------------------------------

            existing = (
                db.query(
                    ECGMLPrediction
                )
                .filter(
                    ECGMLPrediction.session_id
                    == session.session_id,

                    ECGMLPrediction.r_peak_sample
                    == r_peak_sample,
                )
                .first()
            )

            if existing:

                continue

            probabilities = prediction.get(
                "probabilities",
                {},
            )

            if not isinstance(
                probabilities,
                dict,
            ):

                probabilities = {}

            confidence_value = prediction.get(
                "confidence_pct"
            )

            if confidence_value is None:

                confidence_value = prediction.get(
                    "confidence",
                    0.0,
                )

            try:

                confidence = float(
                    confidence_value
                    or 0.0
                )

            except (
                TypeError,
                ValueError,
            ):

                confidence = 0.0

            try:

                r_peak_time = float(
                    prediction.get(
                        "r_peak_time_seconds",
                        0.0,
                    )
                    or 0.0
                )

            except (
                TypeError,
                ValueError,
            ):

                r_peak_time = 0.0

            predicted_class = str(
                prediction.get(
                    "predicted_class",
                    "Q",
                )
            )

            prediction_record = (
                ECGMLPrediction(

                    session_id=(
                        session.session_id
                    ),

                    device_id=(
                        device_id
                    ),

                    prediction_index=(
                        next_index
                    ),

                    r_peak_sample=(
                        r_peak_sample
                    ),

                    r_peak_time_seconds=(
                        r_peak_time
                    ),

                    predicted_class=(
                        predicted_class
                    ),

                    confidence=(
                        confidence
                    ),

                    probabilities_json=(
                        json.dumps(
                            probabilities
                        )
                    ),

                    model_name=(
                        "SmartECG-HD"
                    ),

                    recorded_at=utc_now(),
                )
            )

            db.add(
                prediction_record
            )

            next_index += 1

            stored_count += 1

        if stored_count > 0:

            db.commit()

            print(
                f"SmartECG ML DB | "
                f"{device_id} | "
                f"stored {stored_count} "
                f"prediction(s)"
            )

    except Exception as exc:

        db.rollback()

        print(
            f"SmartECG ML database error "
            f"for {device_id}: "
            f"{exc}"
        )

    finally:

        db.close()


# ============================================================
# SMART ECG ML WORKER
# ============================================================

async def ml_worker(
    device_id: str,
):
    """
    Background SmartECG worker.

    Heavy preprocessing, TensorFlow inference and ML
    persistence run outside the FastAPI event loop.
    """

    queue = get_ml_queue(
        device_id
    )

    print(
        f"SmartECG background worker started "
        f"for {device_id}"
    )

    try:

        while True:

            samples = await queue.get()

            try:

                ml_stream = get_ml_stream(
                    device_id
                )

                predictions = (
                    await asyncio.to_thread(
                        ml_stream.add_samples,
                        samples,
                    )
                )

                if predictions:

                    # ----------------------------------------
                    # Persist ML predictions.
                    #
                    # This is also moved to a background
                    # thread so SQLite operations do not block
                    # the FastAPI event loop.
                    # ----------------------------------------

                    await asyncio.to_thread(
                        save_ml_predictions,
                        device_id,
                        predictions,
                    )

                    # ----------------------------------------
                    # Send predictions to frontend.
                    # ----------------------------------------

                    await broadcast(
                        {
                            "type": (
                                "ml_prediction"
                            ),

                            "device_id": (
                                device_id
                            ),

                            "timestamp": (
                                utc_now()
                                .isoformat()
                            ),

                            "ml": {
                                "predictions": (
                                    predictions
                                ),

                                "prediction_count": (
                                    len(
                                        predictions
                                    )
                                ),
                            },
                        }
                    )

                    print(
                        f"SmartECG ML | "
                        f"{device_id} | "
                        f"{len(predictions)} "
                        f"prediction(s)"
                    )

            except asyncio.CancelledError:

                raise

            except Exception as exc:

                print(
                    f"SmartECG background ML error "
                    f"for {device_id}: "
                    f"{exc}"
                )

            finally:

                queue.task_done()

    except asyncio.CancelledError:

        print(
            f"SmartECG background worker stopped "
            f"for {device_id}"
        )

        raise


def ensure_ml_worker(
    device_id: str,
):
    """
    Ensure one ML worker exists for a device.
    """

    existing_worker = ml_workers.get(
        device_id
    )

    if (
        existing_worker is not None
        and not existing_worker.done()
    ):

        return

    get_ml_queue(
        device_id
    )

    ml_workers[
        device_id
    ] = asyncio.create_task(
        ml_worker(
            device_id
        )
    )


async def queue_ml_samples(
    device_id: str,
    samples: list[float],
):
    """
    Queue samples for SmartECG without blocking
    the ECG ingestion path.
    """

    ensure_ml_worker(
        device_id
    )

    queue = get_ml_queue(
        device_id
    )

    try:

        queue.put_nowait(
            samples
        )

    except asyncio.QueueFull:

        print(
            f"SmartECG ML queue full for "
            f"{device_id}; "
            f"skipping one ML packet."
        )


async def stop_ml_worker(
    device_id: str,
):
    """
    Stop and clean up the ML worker for a device.
    """

    worker = ml_workers.pop(
        device_id,
        None,
    )

    if worker is not None:

        worker.cancel()

        try:

            await worker

        except asyncio.CancelledError:

            pass

    ml_queues.pop(
        device_id,
        None,
    )

    ml_streams.pop(
        device_id,
        None,
    )


# ============================================================
# NORMAL ECG ANALYSIS
# ============================================================

def get_analysis_event(
    device_id: str,
) -> asyncio.Event:

    if device_id not in analysis_events:

        analysis_events[
            device_id
        ] = asyncio.Event()

    return analysis_events[
        device_id
    ]


async def run_analysis_once(
    device_id: str,
):
    """
    Run existing ECG analysis on a snapshot of the current
    10-second live buffer.

    CPU-heavy processing runs in a background thread.
    """

    buffer = live_buffers.get(
        device_id
    )

    if buffer is None:

        return

    if (
        len(buffer)
        < MIN_ANALYSIS_SAMPLES
    ):

        return

    # --------------------------------------------------------
    # Snapshot the deque on the event-loop thread.
    # --------------------------------------------------------

    samples_snapshot = list(
        buffer
    )

    try:

        analysis = await asyncio.to_thread(
            analyze_window,
            samples_snapshot,
            fs=DEFAULT_SAMPLING_RATE,
        )

    except Exception as exc:

        print(
            f"Background ECG analysis error "
            f"for {device_id}: "
            f"{exc}"
        )

        return

    if not isinstance(
        analysis,
        dict,
    ):

        return

    quality = {}

    if isinstance(
        analysis.get(
            "signal_quality"
        ),
        dict,
    ):

        quality = analysis.get(
            "signal_quality"
        )

    if not quality:

        if isinstance(
            analysis.get(
                "quality"
            ),
            dict,
        ):

            quality = analysis.get(
                "quality"
            )

    # --------------------------------------------------------
    # Store derived analysis.
    # --------------------------------------------------------

    session_id = active_sessions.get(
        device_id
    )

    if not session_id:

        return

    db = SessionLocal()

    try:

        session = get_session_by_public_id(
            db,
            session_id,
        )

        if not session:

            return

        now = utc_now()

        measurement = ECGMeasurement(
            session_id=(
                session.session_id
            ),

            recorded_at=now,

            hr=analysis.get(
                "heart_rate_bpm"
            ),

            rr=analysis.get(
                "rr_interval_ms"
            ),

            confidence=analysis.get(
                "analysis_confidence"
            ),

            measurement_status=analysis.get(
                "measurement_status",
                "Normal monitoring",
            ),
        )

        db.add(
            measurement
        )

        if quality:

            quality_record = (
                ECGSignalQuality(

                    session_id=(
                        session.session_id
                    ),

                    recorded_at=now,

                    score=quality.get(
                        "score"
                    ),

                    label=quality.get(
                        "label"
                    ),

                    baseline_wander=(
                        quality.get(
                            "baseline_wander"
                        )
                    ),

                    noise_rms=quality.get(
                        "noise_rms"
                    ),

                    clipping_ratio=(
                        quality.get(
                            "clipping_ratio"
                        )
                    ),

                    peak_count=(
                        analysis.get(
                            "r_peak_count",
                            0,
                        )
                    ),

                    notes=json.dumps(
                        quality.get(
                            "notes",
                            [],
                        )
                    ),
                )
            )

            db.add(
                quality_record
            )

        db.commit()

    except Exception as exc:

        db.rollback()

        print(
            f"Background ECG database error "
            f"for {device_id}: "
            f"{exc}"
        )

    finally:

        db.close()

    # --------------------------------------------------------
    # Send latest derived analysis separately.
    # --------------------------------------------------------

    await broadcast(
        {
            "type": "analysis",

            "device_id": (
                device_id
            ),

            "timestamp": (
                utc_now().isoformat()
            ),

            "analysis": (
                serialize_analysis(
                    analysis
                )
            ),

            "quality": (
                serialize_quality(
                    quality
                )
            ),
        }
    )


async def analysis_worker(
    device_id: str,
):
    """
    Background worker for normal ECG analysis.

    Event-driven and rate-limited to approximately one
    expensive analysis operation per second.
    """

    event = get_analysis_event(
        device_id
    )

    print(
        f"ECG analysis worker started "
        f"for {device_id}"
    )

    try:

        while True:

            await event.wait()

            event.clear()

            now = (
                asyncio
                .get_running_loop()
                .time()
            )

            last_scheduled = (
                analysis_last_scheduled.get(
                    device_id,
                    0.0,
                )
            )

            elapsed = (
                now
                - last_scheduled
            )

            if (
                elapsed
                < ANALYSIS_INTERVAL_SECONDS
            ):

                await asyncio.sleep(
                    ANALYSIS_INTERVAL_SECONDS
                    - elapsed
                )

            analysis_last_scheduled[
                device_id
            ] = (
                asyncio
                .get_running_loop()
                .time()
            )

            await run_analysis_once(
                device_id
            )

    except asyncio.CancelledError:

        print(
            f"ECG analysis worker stopped "
            f"for {device_id}"
        )

        raise


def ensure_analysis_worker(
    device_id: str,
):
    """
    Ensure one ECG analysis worker exists.
    """

    existing_worker = (
        analysis_workers.get(
            device_id
        )
    )

    if (
        existing_worker is not None
        and not existing_worker.done()
    ):

        return

    get_analysis_event(
        device_id
    )

    analysis_workers[
        device_id
    ] = asyncio.create_task(
        analysis_worker(
            device_id
        )
    )


def schedule_analysis(
    device_id: str,
):
    """
    Wake the background analysis worker.

    This function does not perform analysis.
    """

    ensure_analysis_worker(
        device_id
    )

    event = get_analysis_event(
        device_id
    )

    event.set()


async def stop_analysis_worker(
    device_id: str,
):
    """
    Stop and clean up the ECG analysis worker.
    """

    worker = analysis_workers.pop(
        device_id,
        None,
    )

    if worker is not None:

        worker.cancel()

        try:

            await worker

        except asyncio.CancelledError:

            pass

    analysis_events.pop(
        device_id,
        None,
    )

    analysis_last_scheduled.pop(
        device_id,
        None,
    )


# ============================================================
# REQUEST MODEL
# ============================================================

class ECGPayload(BaseModel):

    device_id: str = Field(
        default=DEVICE_DEFAULT
    )

    sampling_rate: int = Field(
        default=DEFAULT_SAMPLING_RATE,
        ge=50,
        le=1000,
    )

    samples: list[float] = Field(
        min_length=1
    )

    lead_off: bool = False

    timestamp: Optional[str] = None


# ============================================================
# ROOT / HEALTH
# ============================================================

@app.get("/")
def root():

    return {
        "status": "running",

        "service": (
            "ECG Monitoring Backend"
        ),

        "sampling_rate": (
            DEFAULT_SAMPLING_RATE
        ),

        "analysis": {
            "mode": (
                "background_worker"
            ),

            "window_seconds": (
                ANALYSIS_WINDOW_SECONDS
            ),

            "interval_seconds": (
                ANALYSIS_INTERVAL_SECONDS
            ),
        },

        "ml_integration": {
            "status": "active",

            "mode": (
                "background_worker"
            ),

            "target_model": (
                "SmartECG-HD"
            ),

            "input_shape": [
                300,
                1,
            ],

            "model_sampling_rate": 360,

            "hardware_sampling_rate": (
                DEFAULT_SAMPLING_RATE
            ),

            "rolling_buffer_seconds": 20,

            "ml_queue_max_packets": (
                ML_QUEUE_MAXSIZE
            ),

            "persistence": (
                "sqlite"
            ),

            "classes": [
                "N",
                "S",
                "V",
                "F",
                "Q",
            ],
        },
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",

        "service": (
            "ECG Monitoring Backend"
        ),

        "sampling_rate": (
            DEFAULT_SAMPLING_RATE
        ),

        "timestamp": (
            utc_now().isoformat()
        ),
    }


@app.get("/api/database")
def database_info():

    return get_database_info()


# ============================================================
# ECG INGESTION
# ============================================================

@app.post("/api/ecg")
async def receive_ecg(
    payload: ECGPayload,
):

    # ========================================================
    # 1. CLEAN INPUT
    # ========================================================

    samples = clean_samples(
        payload.samples
    )

    if not samples:

        raise HTTPException(
            status_code=400,
            detail=(
                "No valid ECG samples received."
            ),
        )

    sampling_rate = (
        payload.sampling_rate
        or DEFAULT_SAMPLING_RATE
    )

    if (
        sampling_rate
        != DEFAULT_SAMPLING_RATE
    ):

        print(
            f"WARNING: received sampling rate "
            f"{sampling_rate} Hz; expected "
            f"{DEFAULT_SAMPLING_RATE} Hz"
        )

    # ========================================================
    # 2. DATABASE SESSION
    # ========================================================

    db = SessionLocal()

    try:

        # ====================================================
        # CREATE / REUSE SESSION
        # ====================================================

        session = create_or_get_session(
            db=db,
            device_id=payload.device_id,
            sampling_rate=sampling_rate,
        )

        now = utc_now()

        # ====================================================
        # STORE RAW PACKET
        # ====================================================

        raw_packet = ECGRawPacket(
            session_id=session.session_id,

            device_id=payload.device_id,

            received_at=now,

            sampling_rate=sampling_rate,

            lead_off=(
                1
                if payload.lead_off
                else 0
            ),

            samples_json=json.dumps(
                samples
            ),

            sample_count=len(
                samples
            ),
        )

        db.add(
            raw_packet
        )

        # ====================================================
        # UPDATE SESSION COUNTERS
        # ====================================================

        session.last_packet_at = now

        session.samples_received = (
            (
                session.samples_received
                or 0
            )
            + len(samples)
        )

        session.packets_received = (
            (
                session.packets_received
                or 0
            )
            + 1
        )

        # ====================================================
        # UPDATE LIVE BUFFER
        # ====================================================

        buffer = live_buffers[
            payload.device_id
        ]

        buffer.extend(
            samples
        )

        # ====================================================
        # COMMIT RAW DATA
        #
        # No expensive ECG analysis or ML occurs before
        # this commit.
        # ====================================================

        db.commit()

        # ====================================================
        # START BACKGROUND WORKERS
        # ====================================================

        if (
            not payload.lead_off
            and sampling_rate
            == DEFAULT_SAMPLING_RATE
        ):

            # ------------------------------------------------
            # Queue packet for SmartECG ML.
            # ------------------------------------------------

            await queue_ml_samples(
                payload.device_id,
                samples,
            )

            # ------------------------------------------------
            # Schedule normal ECG analysis.
            # ------------------------------------------------

            schedule_analysis(
                payload.device_id
            )

        # ====================================================
        # IMMEDIATE WEBSOCKET PAYLOAD
        # ====================================================

        response = {

            "type": "ecg",

            "device_id": (
                payload.device_id
            ),

            "session_id": (
                session.session_id
            ),

            "sampling_rate": (
                sampling_rate
            ),

            "samples": samples,

            "lead_off": (
                payload.lead_off
            ),

            "timestamp": (
                now.isoformat()
            ),

            "analysis": None,

            "quality": None,

            "buffer": {

                "samples": len(
                    buffer
                ),

                "seconds": (
                    len(buffer)
                    / sampling_rate
                ),
            },

            "ml": {

                "status": (
                    "processing"
                    if (
                        not payload.lead_off
                        and sampling_rate
                        == DEFAULT_SAMPLING_RATE
                    )
                    else "disabled"
                ),
            },
        }

        # ====================================================
        # BROADCAST RAW ECG IMMEDIATELY
        # ====================================================

        await broadcast(
            response
        )

        # ====================================================
        # HTTP RESPONSE
        # ====================================================

        return {

            "status": "ok",

            "session_id": (
                session.session_id
            ),

            "samples_received": (
                len(samples)
            ),

            "total_samples": (
                session.samples_received
            ),

            "sampling_rate": (
                sampling_rate
            ),

            "ml_status": (
                "queued"
                if (
                    not payload.lead_off
                    and sampling_rate
                    == DEFAULT_SAMPLING_RATE
                )
                else "disabled"
            ),

            "analysis_status": (
                "queued"
                if (
                    not payload.lead_off
                    and sampling_rate
                    == DEFAULT_SAMPLING_RATE
                )
                else "disabled"
            ),
        }

    except Exception as exc:

        db.rollback()

        print(
            "ERROR /api/ecg:",
            repr(exc),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"ECG ingestion failed: "
                f"{exc}"
            ),
        )

    finally:

        db.close()


# ============================================================
# SESSIONS
# ============================================================

@app.get("/api/sessions")
def get_sessions(
    limit: int = 50,
):

    db = SessionLocal()

    try:

        sessions = (
            db.query(
                ECGSession
            )
            .order_by(
                desc(
                    ECGSession.started_at
                )
            )
            .limit(limit)
            .all()
        )

        return [

            {
                "id": (
                    session.session_id
                ),

                "session_id": (
                    session.session_id
                ),

                "device_id": (
                    session.device_id
                ),

                "started_at": (
                    session.started_at.isoformat()
                    if session.started_at
                    else None
                ),

                "last_packet_at": (
                    session.last_packet_at.isoformat()
                    if session.last_packet_at
                    else None
                ),

                "ended_at": (
                    session.ended_at.isoformat()
                    if session.ended_at
                    else None
                ),

                "sampling_rate": (
                    session.sampling_rate
                ),

                "samples_received": (
                    session.samples_received
                ),

                "packets_received": (
                    session.packets_received
                ),

                "status": (
                    session.status
                ),

                "lead": (
                    session.lead
                ),
            }

            for session in sessions

        ]

    finally:

        db.close()


@app.get(
    "/api/session/device/{device_id}"
)
def get_device_sessions(
    device_id: str,
):

    db = SessionLocal()

    try:

        sessions = (
            db.query(
                ECGSession
            )
            .filter(
                ECGSession.device_id
                == device_id
            )
            .order_by(
                desc(
                    ECGSession.started_at
                )
            )
            .all()
        )

        return [

            {
                "id": (
                    session.session_id
                ),

                "session_id": (
                    session.session_id
                ),

                "device_id": (
                    session.device_id
                ),

                "started_at": (
                    session.started_at.isoformat()
                    if session.started_at
                    else None
                ),

                "last_packet_at": (
                    session.last_packet_at.isoformat()
                    if session.last_packet_at
                    else None
                ),

                "ended_at": (
                    session.ended_at.isoformat()
                    if session.ended_at
                    else None
                ),

                "sampling_rate": (
                    session.sampling_rate
                ),

                "samples_received": (
                    session.samples_received
                ),

                "packets_received": (
                    session.packets_received
                ),

                "status": (
                    session.status
                ),

                "lead": (
                    session.lead
                ),
            }

            for session in sessions

        ]

    finally:

        db.close()


# ============================================================
# SESSION MEASUREMENTS
# ============================================================

@app.get(
    "/api/session/{session_id}/measurements"
)
def get_measurements(
    session_id: str,
    limit: int = 500,
):

    db = SessionLocal()

    try:

        session = get_session_by_public_id(
            db,
            session_id,
        )

        if not session:

            raise HTTPException(
                status_code=404,
                detail="Session not found.",
            )

        measurements = (
            db.query(
                ECGMeasurement
            )
            .filter(
                ECGMeasurement.session_id
                == session_id
            )
            .order_by(
                ECGMeasurement.recorded_at
            )
            .limit(limit)
            .all()
        )

        return [

            {
                "id": item.id,

                "session_id": (
                    item.session_id
                ),

                "recorded_at": (
                    item.recorded_at.isoformat()
                    if item.recorded_at
                    else None
                ),

                "heart_rate_bpm": (
                    item.hr
                ),

                "rr_interval_ms": (
                    item.rr
                ),

                "confidence": (
                    item.confidence
                ),

                "measurement_status": (
                    item.measurement_status
                ),
            }

            for item in measurements

        ]

    finally:

        db.close()


# ============================================================
# SMART ECG ML PREDICTIONS
# ============================================================

@app.get(
    "/api/session/{session_id}/ml-predictions"
)
def get_ml_predictions(
    session_id: str,
    limit: int = 1000,
):

    db = SessionLocal()

    try:

        session = get_session_by_public_id(
            db,
            session_id,
        )

        if not session:

            raise HTTPException(
                status_code=404,
                detail="Session not found.",
            )

        predictions = (
            db.query(
                ECGMLPrediction
            )
            .filter(
                ECGMLPrediction.session_id
                == session_id
            )
            .order_by(
                ECGMLPrediction.prediction_index
            )
            .limit(limit)
            .all()
        )

        result = []

        for prediction in predictions:

            try:

                probabilities = json.loads(
                    prediction.probabilities_json
                )

            except (
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ):

                probabilities = {}

            result.append(
                {
                    "id": (
                        prediction.id
                    ),

                    "session_id": (
                        prediction.session_id
                    ),

                    "device_id": (
                        prediction.device_id
                    ),

                    "prediction_index": (
                        prediction.prediction_index
                    ),

                    "r_peak_sample": (
                        prediction.r_peak_sample
                    ),

                    "r_peak_time_seconds": (
                        prediction.r_peak_time_seconds
                    ),

                    "predicted_class": (
                        prediction.predicted_class
                    ),

                    "confidence": (
                        prediction.confidence
                    ),

                    "probabilities": (
                        probabilities
                    ),

                    "model_name": (
                        prediction.model_name
                    ),

                    "recorded_at": (
                        prediction.recorded_at.isoformat()
                        if prediction.recorded_at
                        else None
                    ),
                }
            )

        return {

            "session_id": (
                session_id
            ),

            "prediction_count": (
                len(result)
            ),

            "model": (
                "SmartECG-HD"
            ),

            "classes": [
                "N",
                "S",
                "V",
                "F",
                "Q",
            ],

            "predictions": result,
        }

    finally:

        db.close()


# ============================================================
# END SESSION
# ============================================================

@app.post(
    "/api/session/{session_id}/end"
)
async def end_session(
    session_id: str,
):

    db = SessionLocal()

    try:

        session = get_session_by_public_id(
            db,
            session_id,
        )

        if not session:

            raise HTTPException(
                status_code=404,
                detail="Session not found.",
            )

        session.status = "completed"

        session.ended_at = utc_now()

        session.last_packet_at = (
            session.last_packet_at
            or session.ended_at
        )

        db.commit()

        # ----------------------------------------------------
        # Remove active session.
        # ----------------------------------------------------

        if (
            active_sessions.get(
                session.device_id
            )
            == session.session_id
        ):

            del active_sessions[
                session.device_id
            ]

        # ----------------------------------------------------
        # Stop background workers.
        # ----------------------------------------------------

        await stop_ml_worker(
            session.device_id
        )

        await stop_analysis_worker(
            session.device_id
        )

        # ----------------------------------------------------
        # Broadcast session end.
        # ----------------------------------------------------

        await broadcast(
            {
                "type": "session_end",

                "session_id": (
                    session.session_id
                ),

                "device_id": (
                    session.device_id
                ),
            }
        )

        return {

            "status": "completed",

            "session_id": (
                session.session_id
            ),
        }

    finally:

        db.close()


# ============================================================
# LIVE SESSION
# ============================================================

@app.get(
    "/api/session/{session_id}/live"
)
def get_live_session(
    session_id: str,
):

    db = SessionLocal()

    try:

        session = get_session_by_public_id(
            db,
            session_id,
        )

        if not session:

            raise HTTPException(
                status_code=404,
                detail="Session not found.",
            )

        buffer = live_buffers.get(
            session.device_id,
            deque(),
        )

        return {

            "session_id": (
                session.session_id
            ),

            "device_id": (
                session.device_id
            ),

            "sampling_rate": (
                session.sampling_rate
            ),

            "samples": list(
                buffer
            ),

            "sample_count": len(
                buffer
            ),

            "status": (
                session.status
            ),
        }

    finally:

        db.close()


# ============================================================
# BEAT GENERATION
# ============================================================

@app.post(
    "/api/session/{session_id}/generate-beats"
)
def generate_beats(
    session_id: str,
):

    db = SessionLocal()

    try:

        # ----------------------------------------------------
        # 1. Find session.
        # ----------------------------------------------------

        session = get_session_by_public_id(
            db,
            session_id,
        )

        if not session:

            raise HTTPException(
                status_code=404,
                detail="Session not found.",
            )

        # ----------------------------------------------------
        # 2. Load raw ECG packets.
        # ----------------------------------------------------

        packets = (
            db.query(
                ECGRawPacket
            )
            .filter(
                ECGRawPacket.session_id
                == session_id
            )
            .order_by(
                ECGRawPacket.received_at.asc()
            )
            .all()
        )

        if not packets:

            raise HTTPException(
                status_code=404,
                detail=(
                    "No raw ECG packets found "
                    "for this session."
                ),
            )

        # ----------------------------------------------------
        # 3. Reconstruct complete ECG.
        # ----------------------------------------------------

        raw_samples = []

        for packet in packets:

            try:

                packet_samples = json.loads(
                    packet.samples_json
                )

                if isinstance(
                    packet_samples,
                    list,
                ):

                    raw_samples.extend(
                        clean_samples(
                            packet_samples
                        )
                    )

            except Exception as exc:

                print(
                    f"Skipping malformed "
                    f"packet {packet.id}: "
                    f"{exc}"
                )

        if len(raw_samples) < 1000:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Not enough ECG samples. "
                    f"Received "
                    f"{len(raw_samples)}; "
                    f"minimum required is 1000."
                ),
            )

        sampling_rate = float(
            session.sampling_rate
            or DEFAULT_SAMPLING_RATE
        )

        # ----------------------------------------------------
        # 4. Filter ECG.
        # ----------------------------------------------------

        filtered_samples = filter_ecg(
            raw_samples,
            fs=sampling_rate,
        )

        # ----------------------------------------------------
        # 5. Detect R-peaks.
        # ----------------------------------------------------

        r_peaks = detect_r_peaks(
            filtered_samples,
            fs=sampling_rate,
        )

        if not r_peaks:

            raise HTTPException(
                status_code=422,
                detail={
                    "message": (
                        "No R-peaks detected "
                        "in the ECG session."
                    ),

                    "raw_samples": (
                        len(raw_samples)
                    ),

                    "sampling_rate": (
                        sampling_rate
                    ),
                },
            )

        # ----------------------------------------------------
        # 6. Segment beats.
        # ----------------------------------------------------

        beats = segment_ecg_beats(
            filtered_samples,
            r_peaks,
            fs=sampling_rate,
            normalize=True,
        )

        if not beats:

            raise HTTPException(
                status_code=422,
                detail={
                    "message": (
                        "R-peaks were detected, "
                        "but no valid heartbeat "
                        "segments were created."
                    ),

                    "raw_samples": (
                        len(raw_samples)
                    ),

                    "r_peaks_detected": (
                        len(r_peaks)
                    ),

                    "sampling_rate": (
                        sampling_rate
                    ),
                },
            )

        # ----------------------------------------------------
        # 7. Remove old generated beats.
        # ----------------------------------------------------

        db.query(
            ECGBeat
        ).filter(
            ECGBeat.session_id
            == session_id
        ).delete(
            synchronize_session=False
        )

        # ----------------------------------------------------
        # 8. Store accepted beats.
        # ----------------------------------------------------

        stored_beats = 0

        for beat in beats:

            beat_samples = beat.get(
                "normalized_samples",
                [],
            )

            if not beat_samples:

                continue

            if (
                len(beat_samples)
                != SAMPLES_PER_BEAT
            ):

                continue

            rr_interval_ms = beat.get(
                "rr_interval_ms"
            )

            heart_rate_bpm = None

            if (
                rr_interval_ms is not None
                and float(
                    rr_interval_ms
                ) > 0
            ):

                heart_rate_bpm = (
                    60000.0
                    / float(
                        rr_interval_ms
                    )
                )

            db_beat = ECGBeat(

                session_id=session_id,

                beat_index=int(
                    beat.get(
                        "beat_index",
                        stored_beats,
                    )
                ),

                r_peak_sample=int(
                    beat.get(
                        "r_peak_index"
                    )
                ),

                start_sample=int(
                    beat.get(
                        "start_index"
                    )
                ),

                end_sample=int(
                    beat.get(
                        "end_index"
                    )
                ),

                rr_interval_ms=(
                    rr_interval_ms
                ),

                heart_rate_bpm=(
                    heart_rate_bpm
                ),

                quality_score=None,

                sampling_rate=(
                    sampling_rate
                ),

                lead=(
                    session.lead
                    or "single-channel"
                ),

                sample_count=len(
                    beat_samples
                ),

                samples_json=json.dumps(
                    beat_samples
                ),

                created_at=utc_now(),
            )

            db.add(
                db_beat
            )

            stored_beats += 1

        if stored_beats == 0:

            db.rollback()

            raise HTTPException(
                status_code=422,
                detail={
                    "message": (
                        "Valid beat segments "
                        "were generated, but "
                        "none could be stored."
                    ),

                    "r_peaks_detected": (
                        len(r_peaks)
                    ),

                    "segments_generated": (
                        len(beats)
                    ),

                    "expected_samples_per_beat": (
                        SAMPLES_PER_BEAT
                    ),
                },
            )

        db.commit()

        # ----------------------------------------------------
        # 9. Return preprocessing summary.
        # ----------------------------------------------------

        return {

            "status": "success",

            "session_id": (
                session_id
            ),

            "sampling_rate": (
                sampling_rate
            ),

            "raw_samples": (
                len(raw_samples)
            ),

            "r_peaks_detected": (
                len(r_peaks)
            ),

            "segments_generated": (
                len(beats)
            ),

            "accepted_beats": (
                stored_beats
            ),

            "samples_per_beat": (
                SAMPLES_PER_BEAT
            ),

            "r_peak_position": 60,

            "pre_r_peak_ms": 300,

            "post_r_peak_ms": 700,

            "normalization": (
                "z-score"
            ),
        }

    except HTTPException:

        raise

    except Exception as exc:

        db.rollback()

        print(
            "ERROR generate-beats:",
            repr(exc),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Beat generation failed: "
                f"{exc}"
            ),
        )

    finally:

        db.close()


# ============================================================
# STORED BEATS
# ============================================================

@app.get(
    "/api/session/{session_id}/beats"
)
def get_beats(
    session_id: str,
    limit: int = 1000,
):

    db = SessionLocal()

    try:

        session = get_session_by_public_id(
            db,
            session_id,
        )

        if not session:

            raise HTTPException(
                status_code=404,
                detail="Session not found.",
            )

        beats = (
            db.query(
                ECGBeat
            )
            .filter(
                ECGBeat.session_id
                == session_id
            )
            .order_by(
                ECGBeat.beat_index
            )
            .limit(limit)
            .all()
        )

        return [

            {
                "id": beat.id,

                "session_id": (
                    beat.session_id
                ),

                "beat_index": (
                    beat.beat_index
                ),

                "r_peak_sample": (
                    beat.r_peak_sample
                ),

                "start_sample": (
                    beat.start_sample
                ),

                "end_sample": (
                    beat.end_sample
                ),

                "rr_interval_ms": (
                    beat.rr_interval_ms
                ),

                "heart_rate_bpm": (
                    beat.heart_rate_bpm
                ),

                "quality_score": (
                    beat.quality_score
                ),

                "sampling_rate": (
                    beat.sampling_rate
                ),

                "lead": (
                    beat.lead
                ),

                "sample_count": (
                    beat.sample_count
                ),

                "samples": (
                    json.loads(
                        beat.samples_json
                    )
                    if beat.samples_json
                    else []
                ),

                "created_at": (
                    beat.created_at.isoformat()
                    if beat.created_at
                    else None
                ),
            }

            for beat in beats

        ]

    finally:

        db.close()


# ============================================================
# LEGACY ML DATASET
#
# Retained for the existing 200 Hz / 200-sample
# preprocessing path.
#
# This endpoint is NOT the SmartECG-HD pipeline.
# ============================================================

@app.get(
    "/api/session/{session_id}/ml-dataset"
)
def get_ml_dataset(
    session_id: str,
):

    db = SessionLocal()

    try:

        session = get_session_by_public_id(
            db,
            session_id,
        )

        if session is None:

            raise HTTPException(
                status_code=404,
                detail=(
                    "ECG session not found"
                ),
            )

        # ----------------------------------------------------
        # Load raw ECG packets.
        # ----------------------------------------------------

        packets = (
            db.query(
                ECGRawPacket
            )
            .filter(
                ECGRawPacket.session_id
                == session_id
            )
            .order_by(
                ECGRawPacket.received_at.asc()
            )
            .all()
        )

        if not packets:

            raise HTTPException(
                status_code=404,
                detail=(
                    "No raw ECG packets found "
                    "for this session"
                ),
            )

        # ----------------------------------------------------
        # Reconstruct raw ECG signal.
        # ----------------------------------------------------

        raw_samples = []

        for packet in packets:

            try:

                packet_samples = json.loads(
                    packet.samples_json
                )

                if isinstance(
                    packet_samples,
                    list,
                ):

                    raw_samples.extend(
                        packet_samples
                    )

            except (
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ):

                continue

        raw_samples = clean_samples(
            raw_samples
        )

        if (
            len(raw_samples)
            < MIN_ANALYSIS_SAMPLES
        ):

            raise HTTPException(
                status_code=422,
                detail=(
                    f"Not enough ECG samples. "
                    f"Received "
                    f"{len(raw_samples)}, "
                    f"minimum required "
                    f"{MIN_ANALYSIS_SAMPLES}."
                ),
            )

        # ----------------------------------------------------
        # Sampling rate.
        # ----------------------------------------------------

        sampling_rate = float(
            session.sampling_rate
            or DEFAULT_SAMPLING_RATE
        )

        # ----------------------------------------------------
        # Filter ECG.
        # ----------------------------------------------------

        filtered_samples = filter_ecg(
            raw_samples,
            fs=sampling_rate,
        )

        filtered_samples = clean_samples(
            filtered_samples
        )

        if (
            len(filtered_samples)
            < MIN_ANALYSIS_SAMPLES
        ):

            raise HTTPException(
                status_code=422,
                detail=(
                    "Filtered ECG signal does "
                    "not contain enough samples "
                    "for ML dataset generation."
                ),
            )

        # ----------------------------------------------------
        # Detect R-peaks.
        # ----------------------------------------------------

        r_peaks = detect_r_peaks(
            filtered_samples,
            fs=sampling_rate,
        )

        if not r_peaks:

            raise HTTPException(
                status_code=422,
                detail=(
                    "No R-peaks detected in "
                    "the ECG session. "
                    "ML dataset cannot "
                    "be generated."
                ),
            )

        # ----------------------------------------------------
        # Build ML dataset.
        # ----------------------------------------------------

        dataset = build_ml_dataset(
            samples=filtered_samples,
            r_peaks=r_peaks,
            fs=sampling_rate,
        )

        # ----------------------------------------------------
        # Validate dataset.
        # ----------------------------------------------------

        validation = (
            validate_ml_dataset(
                dataset
            )
        )

        X = dataset.get(
            "X"
        )

        shape = (
            list(X.shape)
            if X is not None
            else [0, 0, 1]
        )

        # ----------------------------------------------------
        # Return metadata.
        # ----------------------------------------------------

        return {

            "status": "success",

            "session_id": (
                session_id
            ),

            "sampling_rate": (
                sampling_rate
            ),

            "raw_samples": (
                len(raw_samples)
            ),

            "filtered_samples": (
                len(filtered_samples)
            ),

            "r_peaks_detected": (
                len(r_peaks)
            ),

            "accepted_beats": int(
                dataset.get(
                    "beat_count",
                    0,
                )
            ),

            "samples_per_beat": int(
                dataset.get(
                    "samples_per_beat",
                    0,
                )
            ),

            "r_peak_position": int(
                dataset.get(
                    "pre_r_peak_samples",
                    0,
                )
            ),

            "pre_r_peak_ms": 300,

            "post_r_peak_ms": 700,

            "normalization": (
                "z-score"
            ),

            "tensor_shape": shape,

            "channels": (
                int(shape[2])
                if len(shape) == 3
                else 0
            ),

            "validation": (
                validation
            ),
        }

    except HTTPException:

        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"ML dataset generation "
                f"failed: {exc}"
            ),
        )

    finally:

        db.close()


# ============================================================
# WEBSOCKET
# ============================================================

@app.websocket("/ws/ecg")
async def websocket_ecg(
    websocket: WebSocket,
):

    await websocket.accept()

    connected_clients.add(
        websocket
    )

    print(
        "WebSocket client connected. "
        f"Clients: "
        f"{len(connected_clients)}"
    )

    try:

        await websocket.send_text(
            json.dumps(
                {
                    "type": "connection",

                    "status": "connected",

                    "sampling_rate": (
                        DEFAULT_SAMPLING_RATE
                    ),

                    "analysis": {
                        "status": "active",

                        "interval_seconds": (
                            ANALYSIS_INTERVAL_SECONDS
                        ),
                    },

                    "ml": {
                        "status": "active",

                        "mode": (
                            "background_worker"
                        ),

                        "model": (
                            "SmartECG-HD"
                        ),

                        "model_sampling_rate": 360,

                        "input_shape": [
                            300,
                            1,
                        ],

                        "persistence": (
                            "sqlite"
                        ),

                        "classes": [
                            "N",
                            "S",
                            "V",
                            "F",
                            "Q",
                        ],
                    },

                    "timestamp": (
                        utc_now().isoformat()
                    ),
                }
            )
        )

        while True:

            await websocket.receive_text()

    except WebSocketDisconnect:

        connected_clients.discard(
            websocket
        )

        print(
            "WebSocket client disconnected. "
            f"Clients: "
            f"{len(connected_clients)}"
        )

    except Exception as exc:

        connected_clients.discard(
            websocket
        )

        print(
            "WebSocket error:",
            repr(exc),
        )


# ============================================================
# STARTUP / SHUTDOWN
# ============================================================

@app.on_event("startup")
async def startup_event():

    create_tables()

    print(
        "ECG Monitoring Backend started"
    )

    print(
        f"Sampling rate: "
        f"{DEFAULT_SAMPLING_RATE} Hz"
    )

    print(
        "Raw ECG streaming: "
        "IMMEDIATE"
    )

    print(
        "ECG analysis: "
        "BACKGROUND WORKER"
    )

    print(
        "ECG analysis interval: "
        f"{ANALYSIS_INTERVAL_SECONDS} seconds"
    )

    print(
        "SmartECG ML integration: ACTIVE"
    )

    print(
        "SmartECG ML mode: "
        "BACKGROUND WORKER"
    )

    print(
        "SmartECG input: "
        "(300, 1) at 360 Hz"
    )

    print(
        "SmartECG classes: "
        "N / S / V / F / Q"
    )

    print(
        "SmartECG rolling buffer: "
        "20 seconds"
    )

    print(
        "SmartECG queue capacity: "
        f"{ML_QUEUE_MAXSIZE} packets"
    )

    print(
        "SmartECG ML persistence: "
        "SQLITE"
    )


@app.on_event("shutdown")
async def shutdown_event():

    print(
        "ECG Monitoring Backend shutting down"
    )

    # --------------------------------------------------------
    # Stop ML workers.
    # --------------------------------------------------------

    ml_device_ids = list(
        ml_workers.keys()
    )

    for device_id in ml_device_ids:

        await stop_ml_worker(
            device_id
        )

    # --------------------------------------------------------
    # Stop ECG analysis workers.
    # --------------------------------------------------------

    analysis_device_ids = list(
        analysis_workers.keys()
    )

    for device_id in analysis_device_ids:

        await stop_analysis_worker(
            device_id
        )

    print(
        "SmartECG ML workers stopped"
    )

    print(
        "ECG analysis workers stopped"
    )


# ============================================================
# LOCAL RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=9000,
        reload=False,
    )