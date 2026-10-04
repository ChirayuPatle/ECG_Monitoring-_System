import json
import math
import uuid
from datetime import datetime, timezone
from collections import defaultdict, deque
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import desc

from ecg_analysis import (
    DEFAULT_FS,
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
    create_tables,
    get_database_info,
)

from ecg_analysis import (
    DEFAULT_FS,
    SAMPLES_PER_BEAT,
    analyze_window,
    filter_ecg,
    detect_r_peaks,
    segment_ecg_beats,
    build_ml_dataset,
    validate_ml_dataset,
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_SAMPLING_RATE = 200
ANALYSIS_WINDOW_SECONDS = 10
ANALYSIS_WINDOW_SAMPLES = (
    DEFAULT_SAMPLING_RATE * ANALYSIS_WINDOW_SECONDS
)

DEVICE_DEFAULT = "ecg_esp8266_01"


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="ECG Monitoring Backend",
    version="3.0.0",
    description=(
        "ECG acquisition, analysis, session storage "
        "and ML dataset preparation backend."
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

# device_id -> recent raw samples
live_buffers: dict[str, deque] = defaultdict(
    lambda: deque(maxlen=ANALYSIS_WINDOW_SAMPLES)
)

# device_id -> active session_id
active_sessions: dict[str, str] = {}


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def generate_session_id() -> str:
    return f"ecg_{uuid.uuid4().hex[:12]}"


def clean_samples(samples) -> list[float]:
    cleaned = []

    for value in samples:
        try:
            value = float(value)

            if math.isfinite(value):
                cleaned.append(value)

        except (TypeError, ValueError):
            continue

    return cleaned


def get_session_by_public_id(
    db,
    session_id: str,
) -> Optional[ECGSession]:

    return (
        db.query(ECGSession)
        .filter(
            ECGSession.session_id == session_id
        )
        .first()
    )


def create_or_get_session(
    db,
    device_id: str,
    sampling_rate: int,
) -> ECGSession:

    existing_id = active_sessions.get(device_id)

    if existing_id:
        existing = get_session_by_public_id(
            db,
            existing_id,
        )

        if existing and existing.status == "active":
            return existing

    existing = (
        db.query(ECGSession)
        .filter(
            ECGSession.device_id == device_id,
            ECGSession.status == "active",
        )
        .order_by(
            desc(ECGSession.started_at)
        )
        .first()
    )

    if existing:
        active_sessions[device_id] = (
            existing.session_id
        )
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
        lead="II",
    )

    db.add(session)
    db.commit()
    db.refresh(session)

    active_sessions[device_id] = (
        session.session_id
    )

    return session


async def broadcast(payload: dict):

    if not connected_clients:
        return

    message = json.dumps(payload)

    disconnected = []

    for websocket in list(connected_clients):

        try:
            await websocket.send_text(message)

        except Exception:
            disconnected.append(websocket)

    for websocket in disconnected:
        connected_clients.discard(websocket)


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
        "score": quality.get("score"),
        "label": quality.get("label"),
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
# REQUEST MODEL
# ============================================================

class ECGPayload(BaseModel):

    device_id: str = Field(
        default=DEVICE_DEFAULT,
    )

    sampling_rate: int = Field(
        default=DEFAULT_SAMPLING_RATE,
        ge=50,
        le=1000,
    )

    samples: list[float] = Field(
        min_length=1,
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
        "service": "ECG Monitoring Backend",
        "sampling_rate": DEFAULT_SAMPLING_RATE,
        "ml_beat_size": SAMPLES_PER_BEAT,
        "ml_tensor": "(N, 200, 1)",
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "ECG Monitoring Backend",
        "sampling_rate": DEFAULT_SAMPLING_RATE,
        "timestamp": utc_now().isoformat(),
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

    samples = clean_samples(
        payload.samples
    )

    if not samples:
        raise HTTPException(
            status_code=400,
            detail="No valid ECG samples received.",
        )

    sampling_rate = (
        payload.sampling_rate
        or DEFAULT_SAMPLING_RATE
    )

    if sampling_rate != DEFAULT_SAMPLING_RATE:

        print(
            f"WARNING: received sampling rate "
            f"{sampling_rate} Hz; expected "
            f"{DEFAULT_SAMPLING_RATE} Hz"
        )

    db = SessionLocal()

    try:

        # ----------------------------------------------------
        # 1. Create / reuse session
        # ----------------------------------------------------

        session = create_or_get_session(
            db=db,
            device_id=payload.device_id,
            sampling_rate=sampling_rate,
        )

        now = utc_now()

        # ----------------------------------------------------
        # 2. Store RAW packet
        # ----------------------------------------------------

        raw_packet = ECGRawPacket(
            session_id=session.session_id,
            device_id=payload.device_id,
            received_at=now,
            sampling_rate=sampling_rate,
            lead_off=1 if payload.lead_off else 0,
            samples_json=json.dumps(samples),
            sample_count=len(samples),
        )

        db.add(raw_packet)

        # ----------------------------------------------------
        # 3. Update session counters
        # ----------------------------------------------------

        session.last_packet_at = now

        session.samples_received = (
            (session.samples_received or 0)
            + len(samples)
        )

        session.packets_received = (
            (session.packets_received or 0)
            + 1
        )

        # ----------------------------------------------------
        # 4. Update live buffer
        # ----------------------------------------------------

        buffer = live_buffers[
            payload.device_id
        ]

        buffer.extend(samples)

        analysis = {}

        # ----------------------------------------------------
        # 5. Analyze current window
        # ----------------------------------------------------

        try:

            analysis = analyze_window(
                list(buffer),
                fs=sampling_rate,
            )

        except Exception as exc:

            print(
                f"Analysis error for "
                f"{payload.device_id}: {exc}"
            )

            analysis = {}

        # ----------------------------------------------------
        # 6. Extract signal quality
        # ----------------------------------------------------

        quality = {}

        if isinstance(
            analysis.get("signal_quality"),
            dict,
        ):

            quality = analysis.get(
                "signal_quality"
            )

        if not quality:

            if isinstance(
                analysis.get("quality"),
                dict,
            ):

                quality = analysis.get(
                    "quality"
                )

        # ----------------------------------------------------
        # 7. Store measurement
        # ----------------------------------------------------

        measurement = ECGMeasurement(
            session_id=session.session_id,
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

        db.add(measurement)

        # ----------------------------------------------------
        # 8. Store signal quality
        # ----------------------------------------------------

        if quality:

            quality_record = ECGSignalQuality(
                session_id=session.session_id,
                recorded_at=now,
                score=quality.get(
                    "score"
                ),
                label=quality.get(
                    "label"
                ),
                baseline_wander=quality.get(
                    "baseline_wander"
                ),
                noise_rms=quality.get(
                    "noise_rms"
                ),
                clipping_ratio=quality.get(
                    "clipping_ratio"
                ),
                peak_count=analysis.get(
                    "r_peak_count",
                    0,
                ),
                notes=json.dumps(
                    quality.get(
                        "notes",
                        [],
                    )
                ),
            )

            db.add(quality_record)

        # ----------------------------------------------------
        # 9. Commit everything
        # ----------------------------------------------------

        db.commit()

        # ----------------------------------------------------
        # 10. WebSocket payload
        # ----------------------------------------------------

        response = {
            "type": "ecg",
            "device_id": payload.device_id,
            "session_id": session.session_id,
            "sampling_rate": sampling_rate,
            "samples": samples,
            "lead_off": payload.lead_off,
            "timestamp": now.isoformat(),
            "analysis": serialize_analysis(
                analysis
            ),
            "quality": serialize_quality(
                quality
            ),
            "buffer": {
                "samples": len(buffer),
                "seconds": (
                    len(buffer)
                    / sampling_rate
                ),
            },
        }

        await broadcast(response)

        return {
            "status": "ok",
            "session_id": session.session_id,
            "samples_received": len(samples),
            "total_samples": session.samples_received,
            "sampling_rate": sampling_rate,
        }

    except Exception as exc:

        db.rollback()

        print(
            "ERROR /api/ecg:",
            repr(exc),
        )

        raise HTTPException(
            status_code=500,
            detail=f"ECG ingestion failed: {exc}",
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
            db.query(ECGSession)
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
                "id": session.session_id,
                "session_id": session.session_id,
                "device_id": session.device_id,
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
                "status": session.status,
                "lead": session.lead,
            }
            for session in sessions
        ]

    finally:

        db.close()


@app.get("/api/session/device/{device_id}")
def get_device_sessions(
    device_id: str,
):

    db = SessionLocal()

    try:

        sessions = (
            db.query(ECGSession)
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
                "id": session.session_id,
                "session_id": session.session_id,
                "device_id": session.device_id,
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
                "status": session.status,
                "lead": session.lead,
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
            db.query(ECGMeasurement)
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
                "session_id": item.session_id,
                "recorded_at": (
                    item.recorded_at.isoformat()
                    if item.recorded_at
                    else None
                ),
                "heart_rate_bpm": item.hr,
                "rr_interval_ms": item.rr,
                "confidence": item.confidence,
                "measurement_status": (
                    item.measurement_status
                ),
            }
            for item in measurements
        ]

    finally:

        db.close()


# ============================================================
# END SESSION
# ============================================================

@app.post("/api/session/{session_id}/end")
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

        if (
            active_sessions.get(
                session.device_id
            )
            == session.session_id
        ):

            del active_sessions[
                session.device_id
            ]

        await broadcast(
            {
                "type": "session_end",
                "session_id": session.session_id,
                "device_id": session.device_id,
            }
        )

        return {
            "status": "completed",
            "session_id": session.session_id,
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
            "session_id": session.session_id,
            "device_id": session.device_id,
            "sampling_rate": (
                session.sampling_rate
            ),
            "samples": list(buffer),
            "sample_count": len(buffer),
            "status": session.status,
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
        # 1. Find session
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
        # 2. Load raw ECG packets
        # ----------------------------------------------------

        packets = (
            db.query(ECGRawPacket)
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
        # 3. Reconstruct complete ECG signal
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
                    f"Skipping malformed packet "
                    f"{packet.id}: {exc}"
                )

        if len(raw_samples) < 1000:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Not enough ECG samples. "
                    f"Received {len(raw_samples)}; "
                    f"minimum required is 1000."
                ),
            )

        sampling_rate = float(
            session.sampling_rate
            or DEFAULT_SAMPLING_RATE
        )

        # ----------------------------------------------------
        # 4. Filter ECG
        # ----------------------------------------------------

        filtered_samples = filter_ecg(
            raw_samples,
            fs=sampling_rate,
        )

        # ----------------------------------------------------
        # 5. Detect R-peaks
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
                    "raw_samples": len(
                        raw_samples
                    ),
                    "sampling_rate": sampling_rate,
                },
            )

        # ----------------------------------------------------
        # 6. Segment beats
        #
        # segment_ecg_beats() returns a LIST.
        #
        # Each beat contains:
        # normalized_samples
        # raw_samples
        # r_peak_index
        # start_index
        # end_index
        # rr_interval_ms
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
                    "raw_samples": len(
                        raw_samples
                    ),
                    "r_peaks_detected": len(
                        r_peaks
                    ),
                    "sampling_rate": sampling_rate,
                },
            )

        # ----------------------------------------------------
        # 7. Remove old generated beats
        # ----------------------------------------------------

        db.query(ECGBeat).filter(
            ECGBeat.session_id
            == session_id
        ).delete(
            synchronize_session=False
        )

        # ----------------------------------------------------
        # 8. Store accepted beats
        # ----------------------------------------------------

        stored_beats = 0

        for beat in beats:

            # segment_ecg_beats() provides the
            # normalized ML-ready samples here.
            beat_samples = beat.get(
                "normalized_samples",
                [],
            )

            if not beat_samples:
                continue

            # Ensure exactly 200 samples.
            if len(beat_samples) != SAMPLES_PER_BEAT:
                continue

            rr_interval_ms = beat.get(
                "rr_interval_ms"
            )

            # Calculate heart rate when RR
            # interval is available.
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
                        "r_peak_index",
                    )
                ),

                start_sample=int(
                    beat.get(
                        "start_index",
                    )
                ),

                end_sample=int(
                    beat.get(
                        "end_index",
                    )
                ),

                rr_interval_ms=(
                    rr_interval_ms
                ),

                heart_rate_bpm=(
                    heart_rate_bpm
                ),

                # Current segment_ecg_beats()
                # does not calculate per-beat quality.
                quality_score=None,

                sampling_rate=sampling_rate,

                lead=session.lead or "II",

                sample_count=len(
                    beat_samples
                ),

                samples_json=json.dumps(
                    beat_samples
                ),

                created_at=utc_now(),
            )

            db.add(db_beat)

            stored_beats += 1

        if stored_beats == 0:

            db.rollback()

            raise HTTPException(
                status_code=422,
                detail={
                    "message": (
                        "Valid beat segments were "
                        "generated, but none could "
                        "be stored."
                    ),
                    "r_peaks_detected": len(
                        r_peaks
                    ),
                    "segments_generated": len(
                        beats
                    ),
                    "expected_samples_per_beat": (
                        SAMPLES_PER_BEAT
                    ),
                },
            )

        db.commit()

        # ----------------------------------------------------
        # 9. Return preprocessing summary
        # ----------------------------------------------------

        return {
            "status": "success",
            "session_id": session_id,
            "sampling_rate": sampling_rate,
            "raw_samples": len(
                raw_samples
            ),
            "r_peaks_detected": len(
                r_peaks
            ),
            "segments_generated": len(
                beats
            ),
            "accepted_beats": stored_beats,
            "samples_per_beat": (
                SAMPLES_PER_BEAT
            ),
            "r_peak_position": 60,
            "pre_r_peak_ms": 300,
            "post_r_peak_ms": 700,
            "normalization": "z-score",
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
            db.query(ECGBeat)
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
                "session_id": beat.session_id,
                "beat_index": beat.beat_index,
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
                "lead": beat.lead,
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
# ML DATASET
# ============================================================

@app.get("/api/session/{session_id}/ml-dataset")
def get_ml_dataset(session_id: str):
    """
    Generate and validate the ML-ready ECG dataset for a session.

    Pipeline:
        Raw ECG
        -> Filtering
        -> R-peak detection
        -> Beat segmentation
        -> Z-score normalization
        -> (N, 200, 1) tensor
    """

    db = SessionLocal()

    try:
        session = get_session_by_public_id(
            db,
            session_id,
        )

        if session is None:
            raise HTTPException(
                status_code=404,
                detail="ECG session not found",
            )

        # ----------------------------------------------------
        # Load raw ECG packets
        # ----------------------------------------------------

        packets = (
            db.query(ECGRawPacket)
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
                detail="No raw ECG packets found for this session",
            )

        # ----------------------------------------------------
        # Reconstruct raw ECG signal
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

        if len(raw_samples) < MIN_ANALYSIS_SAMPLES:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Not enough ECG samples. "
                    f"Received {len(raw_samples)}, "
                    f"minimum required "
                    f"{MIN_ANALYSIS_SAMPLES}."
                ),
            )

        # ----------------------------------------------------
        # Sampling rate
        # ----------------------------------------------------

        sampling_rate = float(
            session.sampling_rate
            or DEFAULT_SAMPLING_RATE
        )

        # ----------------------------------------------------
        # Filter ECG
        # ----------------------------------------------------

        filtered_samples = filter_ecg(
            raw_samples,
            fs=sampling_rate,
        )

        filtered_samples = clean_samples(
            filtered_samples
        )

        if len(filtered_samples) < MIN_ANALYSIS_SAMPLES:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Filtered ECG signal does not contain "
                    "enough samples for ML dataset generation."
                ),
            )

        # ----------------------------------------------------
        # Detect R-peaks
        #
        # IMPORTANT:
        # build_ml_dataset() requires BOTH:
        #     samples
        #     r_peaks
        # ----------------------------------------------------

        r_peaks = detect_r_peaks(
            filtered_samples,
            fs=sampling_rate,
        )

        if not r_peaks:
            raise HTTPException(
                status_code=422,
                detail=(
                    "No R-peaks detected in the ECG session. "
                    "ML dataset cannot be generated."
                ),
            )

        # ----------------------------------------------------
        # Build ML dataset
        # ----------------------------------------------------

        dataset = build_ml_dataset(
            samples=filtered_samples,
            r_peaks=r_peaks,
            fs=sampling_rate,
        )

        # ----------------------------------------------------
        # Validate dataset
        # ----------------------------------------------------

        validation = validate_ml_dataset(
            dataset
        )

        X = dataset.get("X")

        shape = (
            list(X.shape)
            if X is not None
            else [0, 0, 1]
        )

        # ----------------------------------------------------
        # Return metadata only
        #
        # Do NOT return the complete ECG tensor through the
        # API response. It can become very large.
        # ----------------------------------------------------

        return {
            "status": "success",

            "session_id": session_id,

            "sampling_rate": sampling_rate,

            "raw_samples": len(raw_samples),

            "filtered_samples": len(
                filtered_samples
            ),

            "r_peaks_detected": len(
                r_peaks
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

            "normalization": "z-score",

            "tensor_shape": shape,

            "channels": (
                int(shape[2])
                if len(shape) == 3
                else 0
            ),

            "validation": validation,
        }

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"ML dataset generation failed: {exc}"
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
        f"Clients: {len(connected_clients)}"
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
            f"Clients: {len(connected_clients)}"
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
        f"ML beat size: "
        f"{SAMPLES_PER_BEAT} samples"
    )

    print(
        "ML tensor: (N, 200, 1)"
    )


@app.on_event("shutdown")
async def shutdown_event():

    print(
        "ECG Monitoring Backend shutting down"
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