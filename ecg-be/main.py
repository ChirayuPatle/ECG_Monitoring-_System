from __future__ import annotations

from collections import deque
from datetime import datetime, timezone

from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel, Field
from sqlalchemy import desc

from database import (
    ECGMeasurement,
    ECGSession,
    ECGSignalQuality,
    SessionLocal,
    create_tables,
    get_database_path,
)

from ecg_analysis import analyze_ecg


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="ECG Monitoring Backend",
    description="IoMT ECG receiver, analysis and persistence backend",
    version="3.0.0",
)


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

create_tables()


# ============================================================
# ECG CONFIGURATION
# ============================================================

ANALYSIS_SECONDS = 10

DEFAULT_SAMPLE_RATE = 200

MAX_ANALYSIS_SAMPLES = (
    DEFAULT_SAMPLE_RATE *
    ANALYSIS_SECONDS
)


# ============================================================
# REQUEST MODEL
# ============================================================

class ECGData(BaseModel):

    device_id: str

    sampling_rate: int = Field(
        gt=0
    )

    samples: list[float]

    timestamp_ms: int | None = None

    sequence: int | None = None

    lead_off: bool | None = None


# ============================================================
# WEBSOCKET MANAGER
# ============================================================

class ConnectionManager:

    def __init__(self):

        self.active_connections = []


    async def connect(
        self,
        websocket: WebSocket,
    ):

        await websocket.accept()

        self.active_connections.append(
            websocket
        )

        print(
            f"[WebSocket] Client connected "
            f"({len(self.active_connections)} active)"
        )


    def disconnect(
        self,
        websocket: WebSocket,
    ):

        if websocket in self.active_connections:

            self.active_connections.remove(
                websocket
            )

        print(
            f"[WebSocket] Client disconnected "
            f"({len(self.active_connections)} active)"
        )


    async def broadcast(
        self,
        data: dict,
    ):

        disconnected = []

        for websocket in self.active_connections:

            try:

                await websocket.send_json(
                    data
                )

            except Exception:

                disconnected.append(
                    websocket
                )

        for websocket in disconnected:

            self.disconnect(
                websocket
            )


manager = ConnectionManager()


# ============================================================
# ANALYSIS BUFFERS
# ============================================================

analysis_buffers: dict[
    str,
    deque
] = {}


def get_analysis_buffer(
    device_id: str,
    sampling_rate: int,
):

    max_samples = max(
        1,
        int(
            sampling_rate *
            ANALYSIS_SECONDS
        ),
    )

    if device_id not in analysis_buffers:

        analysis_buffers[
            device_id
        ] = deque(
            maxlen=max_samples
        )

    return analysis_buffers[
        device_id
    ]


# ============================================================
# ACTIVE SESSION CACHE
# ============================================================

active_sessions: dict[
    str,
    str
] = {}


# ============================================================
# TIME HELPERS
# ============================================================

def utc_now():
    return datetime.now(
        timezone.utc
    ).replace(
        tzinfo=None
    )


# ============================================================
# SESSION CREATION
# ============================================================

def get_or_create_session(
    device_id: str,
    sampling_rate: int,
):

    db = SessionLocal()

    try:

        session_id = active_sessions.get(
            device_id
        )

        if session_id:

            existing = (
                db.query(ECGSession)
                .filter(
                    ECGSession.session_id
                    == session_id
                )
                .first()
            )

            if existing:

                return existing


        # ----------------------------------------------------
        # Look for an existing active session.
        # ----------------------------------------------------

        existing = (
            db.query(ECGSession)
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


        # ----------------------------------------------------
        # Create new session.
        # ----------------------------------------------------

        now = utc_now()

        session_id = (
            f"{device_id}-"
            f"{now.strftime('%Y%m%dT%H%M%S')}"
        )

        session = ECGSession(

            session_id=session_id,

            device_id=device_id,

            started_at=now,

            last_packet_at=now,

            sampling_rate=sampling_rate,

            samples_received=0,

            packets_received=0,

            status="active",
        )

        db.add(session)

        db.commit()

        db.refresh(session)

        active_sessions[
            device_id
        ] = session.session_id

        print(
            f"[SESSION] Created: "
            f"{session.session_id}"
        )

        return session

    except Exception:

        db.rollback()

        raise

    finally:

        db.close()


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "running",
        "service": "ECG Monitoring Backend",
        "version": "3.0.0",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    db = SessionLocal()

    try:

        session_count = (
            db.query(ECGSession)
            .count()
        )

        measurement_count = (
            db.query(ECGMeasurement)
            .count()
        )

        quality_count = (
            db.query(ECGSignalQuality)
            .count()
        )

        return {

            "status": "healthy",

            "database": "connected",

            "sessions": session_count,

            "measurements":
                measurement_count,

            "signal_quality_records":
                quality_count,

            "websocket_clients":
                len(
                    manager.active_connections
                ),

            "active_devices":
                len(
                    active_sessions
                ),
        }

    finally:

        db.close()


# ============================================================
# DATABASE DIAGNOSTIC
# ============================================================

@app.get("/api/database")
def database_info():

    db = SessionLocal()

    try:

        sessions = (
            db.query(ECGSession)
            .count()
        )

        measurements = (
            db.query(ECGMeasurement)
            .count()
        )

        quality = (
            db.query(
                ECGSignalQuality
            )
            .count()
        )

        return {

            "database_path":
                get_database_path(),

            "sessions":
                sessions,

            "measurements":
                measurements,

            "signal_quality":
                quality,
        }

    finally:

        db.close()


# ============================================================
# GET ALL SESSIONS
# ============================================================

@app.get("/api/sessions")
def get_sessions(
    limit: int = 20,
):

    limit = max(
        1,
        min(limit, 100),
    )

    db = SessionLocal()

    try:

        rows = (
            db.query(ECGSession)
            .order_by(
                desc(
                    ECGSession.started_at
                )
            )
            .limit(limit)
            .all()
        )

        result = []

        for row in rows:

            result.append({

                "session_id":
                    row.session_id,

                "device_id":
                    row.device_id,

                "started_at":
                    row.started_at.isoformat(),

                "last_packet_at":
                    (
                        row.last_packet_at.isoformat()
                        if row.last_packet_at
                        else None
                    ),

                "ended_at":
                    (
                        row.ended_at.isoformat()
                        if row.ended_at
                        else None
                    ),

                "sampling_rate":
                    row.sampling_rate,

                "samples_received":
                    row.samples_received,

                "packets_received":
                    row.packets_received,

                "status":
                    row.status,
            })

        return {

            "status": "success",

            "count":
                len(result),

            "sessions":
                result,
        }

    finally:

        db.close()


# ============================================================
# GET SESSION BY DEVICE
# ============================================================

@app.get("/api/session/device/{device_id}")
def get_latest_device_session(
    device_id: str,
):

    db = SessionLocal()

    try:

        session = (
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
            .first()
        )

        if not session:

            return {
                "status": "not_found",
                "device_id": device_id,
            }

        return {

            "status": "success",

            "session": {

                "session_id":
                    session.session_id,

                "device_id":
                    session.device_id,

                "started_at":
                    session.started_at.isoformat(),

                "last_packet_at":
                    (
                        session.last_packet_at.isoformat()
                        if session.last_packet_at
                        else None
                    ),

                "ended_at":
                    (
                        session.ended_at.isoformat()
                        if session.ended_at
                        else None
                    ),

                "sampling_rate":
                    session.sampling_rate,

                "samples_received":
                    session.samples_received,

                "packets_received":
                    session.packets_received,

                "status":
                    session.status,
            },
        }

    finally:

        db.close()


# ============================================================
# GET SESSION MEASUREMENTS
# ============================================================

@app.get(
    "/api/session/{session_id}/measurements"
)
def get_session_measurements(
    session_id: str,
    limit: int = 100,
):

    limit = max(
        1,
        min(limit, 500),
    )

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
                desc(
                    ECGMeasurement.recorded_at
                )
            )
            .limit(limit)
            .all()
        )

        result = []

        for row in reversed(rows):

            result.append({

                "recorded_at":
                    row.recorded_at.isoformat(),

                "heart_rate_bpm":
                    row.heart_rate_bpm,

                "rr_interval_ms":
                    row.rr_interval_ms,

                "p_duration_ms":
                    row.p_duration_ms,

                "pr_interval_ms":
                    row.pr_interval_ms,

                "qrs_duration_ms":
                    row.qrs_duration_ms,

                "qt_interval_ms":
                    row.qt_interval_ms,

                "qtc_ms":
                    row.qtc_ms,

                "confidence":
                    row.confidence,

                "measurement_status":
                    row.measurement_status,
            })

        return {

            "status": "success",

            "session_id":
                session_id,

            "count":
                len(result),

            "measurements":
                result,
        }

    finally:

        db.close()


# ============================================================
# RECEIVE ECG
# ============================================================

@app.post("/api/ecg")
async def receive_ecg(
    data: ECGData,
):

    if not data.samples:

        return {
            "status": "ignored",
            "message":
                "No ECG samples received.",
        }


    # --------------------------------------------------------
    # Get/create persistent session
    # --------------------------------------------------------

    session = get_or_create_session(
        data.device_id,
        data.sampling_rate,
    )


    now = utc_now()


    # --------------------------------------------------------
    # Update persistent session
    # --------------------------------------------------------

    db = SessionLocal()

    try:

        database_session = (
            db.query(ECGSession)
            .filter(
                ECGSession.session_id
                == session.session_id
            )
            .first()
        )

        if not database_session:

            raise RuntimeError(
                "ECG session disappeared from database."
            )


        database_session.last_packet_at = now

        database_session.sampling_rate = (
            data.sampling_rate
        )

        database_session.samples_received += (
            len(data.samples)
        )

        database_session.packets_received += 1

        database_session.status = "active"

        db.commit()


    except Exception:

        db.rollback()

        raise

    finally:

        db.close()


    # --------------------------------------------------------
    # Analysis buffer
    # --------------------------------------------------------

    buffer = get_analysis_buffer(
        data.device_id,
        data.sampling_rate,
    )

    buffer.extend(
        data.samples
    )

    analysis_samples = list(
        buffer
    )


    # --------------------------------------------------------
    # ECG analysis
    # --------------------------------------------------------

    analysis = analyze_ecg(
        analysis_samples,
        data.sampling_rate,
    )


    measurements = (
        analysis.get(
            "measurements",
            {}
        )
    )

    quality = (
        analysis.get(
            "signal_quality",
            {}
        )
    )


    # --------------------------------------------------------
    # Persist calculated ECG measurements
    # --------------------------------------------------------

    if (
        measurements.get(
            "heart_rate_bpm"
        )
        is not None
    ):

        db = SessionLocal()

        try:

            measurement = ECGMeasurement(

                session_id=
                    session.session_id,

                recorded_at=now,

                heart_rate_bpm=
                    measurements.get(
                        "heart_rate_bpm"
                    ),

                rr_interval_ms=
                    measurements.get(
                        "rr_interval_ms"
                    ),

                p_duration_ms=
                    measurements.get(
                        "p_duration_ms"
                    ),

                pr_interval_ms=
                    measurements.get(
                        "pr_interval_ms"
                    ),

                qrs_duration_ms=
                    measurements.get(
                        "qrs_duration_ms"
                    ),

                qt_interval_ms=
                    measurements.get(
                        "qt_interval_ms"
                    ),

                qtc_ms=
                    measurements.get(
                        "qtc_ms"
                    ),

                confidence=
                    measurements.get(
                        "confidence"
                    ),

                measurement_status=
                    measurements.get(
                        "status"
                    ),
            )

            db.add(
                measurement
            )

            db.commit()

        except Exception:

            db.rollback()

            raise

        finally:

            db.close()


    # --------------------------------------------------------
    # Persist signal quality
    # --------------------------------------------------------

    db = SessionLocal()

    try:

        quality_record = ECGSignalQuality(

            session_id=
                session.session_id,

            recorded_at=now,

            score=
                quality.get(
                    "score"
                ),

            label=
                quality.get(
                    "label"
                ),

            baseline_wander=
                quality.get(
                    "baseline_wander"
                ),

            noise_rms=
                quality.get(
                    "noise_rms"
                ),

            clipping_ratio=
                quality.get(
                    "clipping_ratio"
                ),

            peak_count=
                quality.get(
                    "peak_count"
                ),

            notes="\n".join(
                quality.get(
                    "notes",
                    []
                )
            ),
        )

        db.add(
            quality_record
        )

        db.commit()

    except Exception:

        db.rollback()

        raise

    finally:

        db.close()


    # --------------------------------------------------------
    # WebSocket payload
    # --------------------------------------------------------

    websocket_payload = {

        "type": "ecg",

        "device_id":
            data.device_id,

        "session_id":
            session.session_id,

        "sampling_rate":
            data.sampling_rate,

        "samples":
            data.samples,

        "timestamp_ms":
            data.timestamp_ms,

        "sequence":
            data.sequence,

        "lead_off":
            data.lead_off,

        "lead":
            "II",

        "analysis":
            analysis,
    }


    await manager.broadcast(
        websocket_payload
    )


    print(
        f"[ECG] "
        f"device={data.device_id} "
        f"session={session.session_id} "
        f"samples={len(data.samples)} "
        f"total={database_session.samples_received if 'database_session' in locals() else '?'} "
        f"HR={measurements.get('heart_rate_bpm')} "
        f"quality={quality.get('label')}"
    )


    return {

        "status":
            "success",

        "device_id":
            data.device_id,

        "session_id":
            session.session_id,

        "samples_received":
            len(data.samples),

        "analysis_status":
            analysis.get(
                "status"
            ),
    }


# ============================================================
# WEBSOCKET
# ============================================================

@app.websocket("/ws/ecg")
async def websocket_ecg(
    websocket: WebSocket,
):

    await manager.connect(
        websocket
    )

    try:

        while True:

            await websocket.receive()

    except WebSocketDisconnect:

        manager.disconnect(
            websocket
        )

    except Exception as error:

        print(
            f"[WebSocket] Error: {error}"
        )

        manager.disconnect(
            websocket
        )