from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    desc,
)
from sqlalchemy.orm import declarative_base, sessionmaker


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "ecg_monitor.db"

DATABASE_URL = f"sqlite:///{DATABASE_PATH.as_posix()}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


# ============================================================
# HELPERS
# ============================================================

def utc_now():
    return datetime.now(timezone.utc)


# ============================================================
# ECG SESSION
# ============================================================

class ECGSession(Base):
    __tablename__ = "ecg_sessions"

    id = Column(Integer, primary_key=True, index=True)

    session_id = Column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
    )

    device_id = Column(
        String(100),
        nullable=False,
        index=True,
    )

    lead = Column(
        String(20),
        default="II",
        nullable=False,
    )

    started_at = Column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    last_packet_at = Column(
        DateTime(timezone=True),
        nullable=True,
    )

    ended_at = Column(
        DateTime(timezone=True),
        nullable=True,
    )

    sampling_rate = Column(
        Integer,
        nullable=False,
    )

    samples_received = Column(
        Integer,
        default=0,
        nullable=False,
    )

    packets_received = Column(
        Integer,
        default=0,
        nullable=False,
    )

    status = Column(
        String(30),
        default="recording",
        nullable=False,
    )


# ============================================================
# ECG MEASUREMENTS
# ============================================================

class ECGMeasurement(Base):
    __tablename__ = "ecg_measurements"

    id = Column(Integer, primary_key=True, index=True)

    session_id = Column(
        String(100),
        ForeignKey("ecg_sessions.session_id"),
        nullable=False,
        index=True,
    )

    recorded_at = Column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    heart_rate_bpm = Column(Float, nullable=True)
    rr_interval_ms = Column(Float, nullable=True)

    p_duration_ms = Column(Float, nullable=True)
    pr_interval_ms = Column(Float, nullable=True)
    qrs_duration_ms = Column(Float, nullable=True)

    qt_interval_ms = Column(Float, nullable=True)
    qtc_ms = Column(Float, nullable=True)

    confidence = Column(Float, nullable=True)

    measurement_status = Column(
        String(50),
        nullable=True,
    )


# ============================================================
# SIGNAL QUALITY
# ============================================================

class ECGSignalQuality(Base):
    __tablename__ = "ecg_signal_quality"

    id = Column(Integer, primary_key=True, index=True)

    session_id = Column(
        String(100),
        ForeignKey("ecg_sessions.session_id"),
        nullable=False,
        index=True,
    )

    recorded_at = Column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    score = Column(Float, nullable=True)

    label = Column(
        String(50),
        nullable=True,
    )

    baseline_wander = Column(Float, nullable=True)
    noise_rms = Column(Float, nullable=True)
    clipping_ratio = Column(Float, nullable=True)
    peak_count = Column(Integer, nullable=True)

    notes = Column(
        Text,
        nullable=True,
    )


# ============================================================
# RAW ECG PACKETS
# ============================================================

class ECGRawPacket(Base):
    __tablename__ = "ecg_raw_packets"

    id = Column(Integer, primary_key=True, index=True)

    session_id = Column(
        String(100),
        ForeignKey("ecg_sessions.session_id"),
        nullable=False,
        index=True,
    )

    device_id = Column(
        String(100),
        nullable=False,
    )

    received_at = Column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    sampling_rate = Column(
        Integer,
        nullable=False,
    )

    lead_off = Column(
        Boolean,
        default=False,
        nullable=False,
    )

    samples_json = Column(
        Text,
        nullable=False,
    )

    sample_count = Column(
        Integer,
        nullable=False,
    )


# ============================================================
# ML-READY ECG BEATS
# ============================================================

class ECGBeat(Base):
    __tablename__ = "ecg_beats"

    id = Column(Integer, primary_key=True, index=True)

    session_id = Column(
        String(100),
        ForeignKey("ecg_sessions.session_id"),
        nullable=False,
        index=True,
    )

    beat_index = Column(
        Integer,
        nullable=False,
    )

    r_peak_sample = Column(
        Integer,
        nullable=False,
    )

    start_sample = Column(
        Integer,
        nullable=False,
    )

    end_sample = Column(
        Integer,
        nullable=False,
    )

    rr_interval_ms = Column(
        Float,
        nullable=True,
    )

    heart_rate_bpm = Column(
        Float,
        nullable=True,
    )

    quality_score = Column(
        Float,
        nullable=True,
    )

    sampling_rate = Column(
        Integer,
        nullable=False,
    )

    lead = Column(
        String(20),
        default="II",
        nullable=False,
    )

    sample_count = Column(
        Integer,
        nullable=False,
    )

    # JSON encoded normalized ECG samples.
    samples_json = Column(
        Text,
        nullable=False,
    )

    created_at = Column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )


# ============================================================
# CREATE TABLES
# ============================================================

def create_tables():
    Base.metadata.create_all(bind=engine)


# ============================================================
# DATABASE INFO
# ============================================================

def get_database_info():
    db = SessionLocal()

    try:
        return {
            "database": str(DATABASE_PATH),
            "exists": DATABASE_PATH.exists(),
            "size_bytes": (
                DATABASE_PATH.stat().st_size
                if DATABASE_PATH.exists()
                else 0
            ),
            "sessions": db.query(ECGSession).count(),
            "measurements": db.query(ECGMeasurement).count(),
            "signal_quality": db.query(ECGSignalQuality).count(),
            "raw_packets": db.query(ECGRawPacket).count(),
            "beats": db.query(ECGBeat).count(),
        }

    finally:
        db.close()