import json
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database import (
    ECGBeat,
    ECGMeasurement,
    ECGRawPacket,
    ECGSession,
    ECGSignalQuality,
    SessionLocal,
    create_tables,
    get_database_info,
)

from ecg_analysis import analyze_ecg

from beat_extraction import (
    build_ml_dataset,
    extract_beats,
)


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="ECG Monitoring Backend",
    version="5.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# CONFIGURATION
# ============================================================

ANALYSIS_WINDOW_SECONDS = 10

DEFAULT_SAMPLING_RATE = 200


# ============================================================
# RUNTIME STATE
# ============================================================

analysis_buffers = defaultdict(
    lambda: deque(
        maxlen=DEFAULT_SAMPLING_RATE
        * ANALYSIS_WINDOW_SECONDS
    )
)

active_sessions = {}

connected_clients = set()


# ============================================================
# REQUEST MODEL
# ============================================================

class ECGData(BaseModel):

    device_id: str = Field(
        min_length=1,
        max_length=100,
    )

    sampling_rate: int = Field(
        gt=0,
        le=1000,
    )

    samples: list[float] = Field(
        min_length=1,
        max_length=1000,
    )

    lead_off: bool = False


# ============================================================
# TIME
# ============================================================

def utc_now():
    return datetime.now(
        timezone.utc
    )


# ============================================================
# SESSION ID
# ============================================================

def generate_session_id():
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    return (
        f"ecg_{timestamp}_"
        f"{uuid.uuid4().hex[:8]}"
    )


# ============================================================
# JSON SERIALIZATION
# ============================================================

def json_safe(value):

    if isinstance(
        value,
        datetime,
    ):
        return value.isoformat()

    if isinstance(
        value,
        np.integer,
    ):
        return int(value)

    if isinstance(
        value,
        np.floating,
    ):
        return float(value)

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        dict,
    ):
        return {
            key: json_safe(val)
            for key, val in value.items()
        }

    if isinstance(
        value,
        list,
    ):
        return [
            json_safe(item)
            for item in value
        ]

    return value


# ============================================================
# SESSION MANAGEMENT
# ============================================================

def get_or_create_session(
    db,
    device_id,
    sampling_rate,
):

    session_id = active_sessions.get(
        device_id
    )

    if session_id:

        session = (
            db.query(ECGSession)
            .filter(
                ECGSession.session_id
                == session_id
            )
            .first()
        )

        if session:
            return session

    session_id = generate_session_id()

    session = ECGSession(
        session_id=session_id,
        device_id=device_id,
        lead="II",
        sampling_rate=sampling_rate,
        started_at=utc_now(),
        status="recording",
    )

    db.add(session)
    db.commit()
    db.refresh(session)

    active_sessions[
        device_id
    ] = session_id

    return session


# ============================================================
# BROADCAST
# ============================================================

async def broadcast(payload):

    disconnected = []

    for websocket in list(
        connected_clients
    ):

        try:

            await websocket.send_json(
                json_safe(payload)
            )

        except Exception:

            disconnected.append(
                websocket
            )

    for websocket in disconnected:

        connected_clients.discard(
            websocket
        )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "running",
        "service": "ECG Monitoring Backend",
        "version": "5.0.0",
        "database": "SQLite",
        "sampling_target_hz": 200,
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    info = get_database_info()

    return {
        "status": "healthy",
        "database": info,
    }


# ============================================================
# DATABASE INFO
# ============================================================

@app.get("/api/database")
def database_info():

    return get_database_info()


# ============================================================
# LIST SESSIONS
# ============================================================

@app.get("/api/sessions")
def list_sessions():

    db = SessionLocal()

    try:

        sessions = (
            db.query(ECGSession)
            .order_by(
                ECGSession.started_at.desc()
            )
            .all()
        )

        return [
            {
                "session_id": s.session_id,
                "device_id": s.device_id,
                "lead": s.lead,
                "sampling_rate": s.sampling_rate,
                "started_at": s.started_at,
                "last_packet_at": s.last_packet_at,
                "ended_at": s.ended_at,
                "samples_received": s.samples_received,
                "packets_received": s.packets_received,
                "status": s.status,
            }
            for s in sessions
        ]

    finally:

        db.close()


# ============================================================
# DEVICE SESSIONS
# ============================================================

@app.get(
    "/api/session/device/{device_id}"
)
def device_sessions(
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
                ECGSession.started_at.desc()
            )
            .all()
        )

        return [
            {
                "session_id": s.session_id,
                "device_id": s.device_id,
                "lead": s.lead,
                "sampling_rate": s.sampling_rate,
                "started_at": s.started_at,
                "ended_at": s.ended_at,
                "samples_received": s.samples_received,
                "packets_received": s.packets_received,
                "status": s.status,
            }
            for s in sessions
        ]

    finally:

        db.close()


# ============================================================
# MEASUREMENTS
# ============================================================

@app.get(
    "/api/session/{session_id}/measurements"
)
def session_measurements(
    session_id: str,
):

    db = SessionLocal()

    try:

        rows = (
            db.query(
                ECGMeasurement
            )
            .filter(
                ECGMeasurement.session_id
                == session_id
            )
            .order_by(
                ECGMeasurement.recorded_at.desc()
            )
            .all()
        )

        return [
            {
                "recorded_at": row.recorded_at,
                "heart_rate_bpm": row.heart_rate_bpm,
                "rr_interval_ms": row.rr_interval_ms,
                "p_duration_ms": row.p_duration_ms,
                "pr_interval_ms": row.pr_interval_ms,
                "qrs_duration_ms": row.qrs_duration_ms,
                "qt_interval_ms": row.qt_interval_ms,
                "qtc_ms": row.qtc_ms,
                "confidence": row.confidence,
                "measurement_status": row.measurement_status,
            }
            for row in rows
        ]

    finally:

        db.close()


# ============================================================
# SIGNAL QUALITY
# ============================================================

@app.get(
    "/api/session/{session_id}/signal-quality"
)
def session_signal_quality(
    session_id: str,
):

    db = SessionLocal()

    try:

        rows = (
            db.query(
                ECGSignalQuality
            )
            .filter(
                ECGSignalQuality.session_id
                == session_id
            )
            .order_by(
                ECGSignalQuality.recorded_at.desc()
            )
            .all()
        )

        return [
            {
                "recorded_at": row.recorded_at,
                "score": row.score,
                "label": row.label,
                "baseline_wander": row.baseline_wander,
                "noise_rms": row.noise_rms,
                "clipping_ratio": row.clipping_ratio,
                "peak_count": row.peak_count,
                "notes": row.notes,
            }
            for row in rows
        ]

    finally:

        db.close()


# ============================================================
# ECG PACKET
# ============================================================

@app.post("/api/ecg")
async def receive_ecg(
    data: ECGData,
):

    db = SessionLocal()

    try:

        session = get_or_create_session(
            db,
            data.device_id,
            data.sampling_rate,
        )

        now = utc_now()

        samples = [
            float(value)
            for value in data.samples
            if np.isfinite(value)
        ]

        if not samples:

            raise HTTPException(
                status_code=400,
                detail="No valid ECG samples received.",
            )

        # ----------------------------------------------------
        # RAW PACKET
        # ----------------------------------------------------

        raw_packet = ECGRawPacket(
            session_id=session.session_id,
            device_id=data.device_id,
            received_at=now,
            sampling_rate=data.sampling_rate,
            lead_off=data.lead_off,
            samples_json=json.dumps(
                samples
            ),
            sample_count=len(samples),
        )

        db.add(raw_packet)

        # ----------------------------------------------------
        # SESSION UPDATE
        # ----------------------------------------------------

        session.last_packet_at = now

        session.packets_received += 1

        session.samples_received += len(
            samples
        )

        # ----------------------------------------------------
        # ANALYSIS BUFFER
        # ----------------------------------------------------

        buffer = analysis_buffers[
            data.device_id
        ]

        buffer.extend(samples)

        analysis = analyze_ecg(
            list(buffer),
            data.sampling_rate,
        )

        measurement_data = (
            analysis.measurement
        )

        quality_data = (
            analysis.signal_quality
        )

        # ----------------------------------------------------
        # MEASUREMENT
        # ----------------------------------------------------

        measurement = ECGMeasurement(
            session_id=session.session_id,
            recorded_at=now,
            heart_rate_bpm=(
                analysis.heart_rate_bpm
            ),
            rr_interval_ms=(
                measurement_data.get(
                    "rr_interval_ms"
                )
            ),
            p_duration_ms=(
                measurement_data.get(
                    "p_duration_ms"
                )
            ),
            pr_interval_ms=(
                measurement_data.get(
                    "pr_interval_ms"
                )
            ),
            qrs_duration_ms=(
                measurement_data.get(
                    "qrs_duration_ms"
                )
            ),
            qt_interval_ms=(
                measurement_data.get(
                    "qt_interval_ms"
                )
            ),
            qtc_ms=(
                measurement_data.get(
                    "qtc_ms"
                )
            ),
            confidence=(
                measurement_data.get(
                    "confidence"
                )
            ),
            measurement_status=(
                measurement_data.get(
                    "status"
                )
            ),
        )

        db.add(measurement)

        # ----------------------------------------------------
        # SIGNAL QUALITY
        # ----------------------------------------------------

        quality = ECGSignalQuality(
            session_id=session.session_id,
            recorded_at=now,
            score=quality_data.get(
                "score"
            ),
            label=quality_data.get(
                "label"
            ),
            baseline_wander=quality_data.get(
                "baseline_wander"
            ),
            noise_rms=quality_data.get(
                "noise_rms"
            ),
            clipping_ratio=quality_data.get(
                "clipping_ratio"
            ),
            peak_count=quality_data.get(
                "peak_count"
            ),
            notes=json.dumps(
                quality_data.get(
                    "notes",
                    [],
                )
            ),
        )

        db.add(quality)

        db.commit()

        # ----------------------------------------------------
        # WEBSOCKET
        # ----------------------------------------------------

        websocket_payload = {
            "type": "ecg",

            "session_id": (
                session.session_id
            ),

            "device_id": (
                data.device_id
            ),

            "sampling_rate": (
                data.sampling_rate
            ),

            "lead": "II",

            "lead_off": data.lead_off,

            "samples": samples,

            "analysis": {
                "r_peaks": (
                    analysis.r_peaks
                ),

                "rr_intervals_ms": (
                    analysis.rr_intervals_ms
                ),

                "heart_rate_bpm": (
                    analysis.heart_rate_bpm
                ),

                "measurement": (
                    measurement_data
                ),

                "signal_quality": (
                    quality_data
                ),
            },
        }

        await broadcast(
            websocket_payload
        )

        return {
            "status": "received",

            "session_id": (
                session.session_id
            ),

            "samples_received": len(
                samples
            ),

            "total_samples": (
                session.samples_received
            ),

            "total_packets": (
                session.packets_received
            ),

            "heart_rate_bpm": (
                analysis.heart_rate_bpm
            ),

            "signal_quality": (
                quality_data.get(
                    "score"
                )
            ),
        }

    except HTTPException:

        db.rollback()
        raise

    except Exception as exc:

        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    finally:

        db.close()


# ============================================================
# GENERATE BEATS FOR SESSION
# ============================================================

@app.post(
    "/api/session/{session_id}/generate-beats"
)
def generate_session_beats(
    session_id: str,
):

    db = SessionLocal()

    try:

        session = (
            db.query(ECGSession)
            .filter(
                ECGSession.session_id
                == session_id
            )
            .first()
        )

        if not session:

            raise HTTPException(
                status_code=404,
                detail="Session not found.",
            )

        # ----------------------------------------------------
        # LOAD RAW PACKETS IN CHRONOLOGICAL ORDER
        # ----------------------------------------------------

        packets = (
            db.query(ECGRawPacket)
            .filter(
                ECGRawPacket.session_id
                == session_id
            )
            .order_by(
                ECGRawPacket.received_at.asc(),
                ECGRawPacket.id.asc(),
            )
            .all()
        )

        if not packets:

            raise HTTPException(
                status_code=404,
                detail="No raw ECG packets found.",
            )

        all_samples = []

        for packet in packets:

            try:

                values = json.loads(
                    packet.samples_json
                )

                all_samples.extend(
                    float(value)
                    for value in values
                    if np.isfinite(value)
                )

            except Exception:

                continue

        if len(all_samples) < (
            session.sampling_rate
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    "At least one second of "
                    "ECG data is required."
                ),
            )

        # ----------------------------------------------------
        # COMPLETE SESSION ANALYSIS
        # ----------------------------------------------------

        analysis = analyze_ecg(
            all_samples,
            session.sampling_rate,
        )

        if not analysis.r_peaks:

            return {
                "status": "no_beats_detected",
                "session_id": session_id,
                "beats_created": 0,
            }

        # ----------------------------------------------------
        # EXTRACT BEATS
        # ----------------------------------------------------

        beats = extract_beats(
            analysis.filtered_signal,
            analysis.r_peaks,
            sampling_rate=session.sampling_rate,
            quality_threshold=40.0,
        )

        # ----------------------------------------------------
        # REMOVE OLD GENERATED BEATS
        # ----------------------------------------------------

        (
            db.query(ECGBeat)
            .filter(
                ECGBeat.session_id
                == session_id
            )
            .delete(
                synchronize_session=False
            )
        )

        # ----------------------------------------------------
        # SAVE NEW BEATS
        # ----------------------------------------------------

        for beat in beats:

            db.add(
                ECGBeat(
                    session_id=session_id,
                    beat_index=beat.beat_index,
                    r_peak_sample=beat.r_peak_sample,
                    start_sample=beat.start_sample,
                    end_sample=beat.end_sample,
                    rr_interval_ms=beat.rr_interval_ms,
                    heart_rate_bpm=beat.heart_rate_bpm,
                    quality_score=beat.quality_score,
                    sampling_rate=session.sampling_rate,
                    lead=session.lead,
                    sample_count=len(
                        beat.samples
                    ),
                    samples_json=json.dumps(
                        beat.samples
                    ),
                )
            )

        db.commit()

        return {
            "status": "success",
            "session_id": session_id,
            "sampling_rate": session.sampling_rate,
            "lead": session.lead,
            "total_raw_samples": len(
                all_samples
            ),
            "r_peaks_detected": len(
                analysis.r_peaks
            ),
            "beats_created": len(
                beats
            ),
        }

    except HTTPException:

        db.rollback()
        raise

    except Exception as exc:

        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    finally:

        db.close()


# ============================================================
# GET EXTRACTED BEATS
# ============================================================

@app.get(
    "/api/session/{session_id}/beats"
)
def get_session_beats(
    session_id: str,
    limit: int = 100,
    offset: int = 0,
):

    limit = max(
        1,
        min(limit, 1000),
    )

    offset = max(
        0,
        offset,
    )

    db = SessionLocal()

    try:

        session = (
            db.query(ECGSession)
            .filter(
                ECGSession.session_id
                == session_id
            )
            .first()
        )

        if not session:

            raise HTTPException(
                status_code=404,
                detail="Session not found.",
            )

        query = (
            db.query(ECGBeat)
            .filter(
                ECGBeat.session_id
                == session_id
            )
            .order_by(
                ECGBeat.beat_index.asc()
            )
        )

        total = query.count()

        rows = (
            query
            .offset(offset)
            .limit(limit)
            .all()
        )

        return {
            "session_id": session_id,
            "sampling_rate": (
                session.sampling_rate
            ),
            "lead": session.lead,
            "total_beats": total,
            "limit": limit,
            "offset": offset,
            "beats": [
                {
                    "beat_index": (
                        row.beat_index
                    ),

                    "r_peak_sample": (
                        row.r_peak_sample
                    ),

                    "start_sample": (
                        row.start_sample
                    ),

                    "end_sample": (
                        row.end_sample
                    ),

                    "rr_interval_ms": (
                        row.rr_interval_ms
                    ),

                    "heart_rate_bpm": (
                        row.heart_rate_bpm
                    ),

                    "quality_score": (
                        row.quality_score
                    ),

                    "sample_count": (
                        row.sample_count
                    ),

                    "samples": json.loads(
                        row.samples_json
                    ),
                }
                for row in rows
            ],
        }

    finally:

        db.close()


# ============================================================
# ML DATASET
# ============================================================

@app.get(
    "/api/session/{session_id}/ml-dataset"
)
def get_ml_dataset(
    session_id: str,
):

    db = SessionLocal()

    try:

        session = (
            db.query(ECGSession)
            .filter(
                ECGSession.session_id
                == session_id
            )
            .first()
        )

        if not session:

            raise HTTPException(
                status_code=404,
                detail="Session not found.",
            )

        rows = (
            db.query(ECGBeat)
            .filter(
                ECGBeat.session_id
                == session_id
            )
            .order_by(
                ECGBeat.beat_index.asc()
            )
            .all()
        )

        beats = []

        for row in rows:

            # Reconstruct the lightweight
            # ExtractedBeat-like structure.
            class Beat:
                pass

            beat = Beat()

            beat.samples = json.loads(
                row.samples_json
            )

            beats.append(beat)

        dataset = build_ml_dataset(
            beats,
            sampling_rate=session.sampling_rate,
        )

        return {
            "session_id": session_id,
            "lead": session.lead,
            "device_id": session.device_id,
            **dataset,
        }

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

    try:

        while True:

            await websocket.receive_text()

    except Exception:

        pass

    finally:

        connected_clients.discard(
            websocket
        )


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup():

    print()
    print("=" * 50)
    print("ECG MONITORING BACKEND")
    print("=" * 50)

    print(
        "Initializing SQLite database..."
    )

    create_tables()

    info = get_database_info()

    print(
        f"Database: {info['database']}"
    )

    print(
        f"Sessions: {info['sessions']}"
    )

    print(
        f"Measurements: "
        f"{info['measurements']}"
    )

    print(
        f"Signal quality: "
        f"{info['signal_quality']}"
    )

    print(
        f"Raw packets: "
        f"{info['raw_packets']}"
    )

    print(
        f"ECG beats: "
        f"{info['beats']}"
    )

    print()
    print(
        "FastAPI: http://0.0.0.0:9000"
    )

    print(
        "ECG POST: /api/ecg"
    )

    print(
        "WebSocket: /ws/ecg"
    )

    print("=" * 50)
    print()


# ============================================================
# SHUTDOWN
# ============================================================

@app.on_event("shutdown")
def shutdown():

    print()
    print(
        "ECG backend shutting down."
    )