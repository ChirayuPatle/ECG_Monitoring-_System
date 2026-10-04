from pathlib import Path
from datetime import datetime

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Text,
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

Base = declarative_base()

# IMPORTANT:
# Main.py imports this directly.
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


# ============================================================
# ECG SESSION
# ============================================================

class ECGSession(Base):
    __tablename__ = "ecg_sessions"

    id = Column(Integer, primary_key=True)

    session_id = Column(
        String(64),
        unique=True,
        index=True,
        nullable=False,
    )

    device_id = Column(
        String(100),
        index=True,
        nullable=False,
    )

    started_at = Column(
        DateTime,
        nullable=False,
    )

    last_packet_at = Column(DateTime)

    ended_at = Column(DateTime)

    sampling_rate = Column(
        Integer,
        default=200,
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
        default="active",
        nullable=False,
    )

    lead = Column(
        String(20),
        default="II",
        nullable=False,
    )


# ============================================================
# ECG MEASUREMENT / ANALYSIS SNAPSHOT
# ============================================================

class ECGMeasurement(Base):
    __tablename__ = "ecg_measurements"

    id = Column(Integer, primary_key=True)

    session_id = Column(
        String(64),
        index=True,
        nullable=False,
    )

    recorded_at = Column(
        DateTime,
        nullable=False,
    )

    hr = Column(Float)

    rr = Column(Float)

    p_duration = Column(Float)

    pr = Column(Float)

    qrs = Column(Float)

    qt = Column(Float)

    qtc = Column(Float)

    confidence = Column(Float)

    measurement_status = Column(
        String(50),
    )


# ============================================================
# ECG SIGNAL QUALITY
# ============================================================

class ECGSignalQuality(Base):
    __tablename__ = "ecg_signal_quality"

    id = Column(Integer, primary_key=True)

    session_id = Column(
        String(64),
        index=True,
        nullable=False,
    )

    recorded_at = Column(
        DateTime,
        nullable=False,
    )

    score = Column(Float)

    label = Column(String(30))

    baseline_wander = Column(Float)

    noise_rms = Column(Float)

    clipping_ratio = Column(Float)

    peak_count = Column(Integer)

    notes = Column(Text)


# ============================================================
# RAW ECG PACKETS
# ============================================================

class ECGRawPacket(Base):
    __tablename__ = "ecg_raw_packets"

    id = Column(Integer, primary_key=True)

    session_id = Column(
        String(64),
        index=True,
        nullable=False,
    )

    device_id = Column(
        String(100),
        index=True,
        nullable=False,
    )

    received_at = Column(
        DateTime,
        nullable=False,
    )

    sampling_rate = Column(
        Integer,
        nullable=False,
    )

    lead_off = Column(
        Integer,
        default=0,
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
# ECG BEATS
# ============================================================

class ECGBeat(Base):
    __tablename__ = "ecg_beats"

    id = Column(Integer, primary_key=True)

    session_id = Column(
        String(64),
        index=True,
        nullable=False,
    )

    beat_index = Column(Integer)

    r_peak_sample = Column(Integer)

    start_sample = Column(Integer)

    end_sample = Column(Integer)

    rr_interval_ms = Column(Float)

    heart_rate_bpm = Column(Float)

    quality_score = Column(Float)

    sampling_rate = Column(
        Integer,
        nullable=False,
    )

    lead = Column(
        String(20),
        default="II",
        nullable=False,
    )

    sample_count = Column(Integer)

    samples_json = Column(Text)

    created_at = Column(DateTime)


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def create_tables():
    Base.metadata.create_all(bind=engine)


# ============================================================
# DATABASE INFORMATION
# ============================================================

def get_database_info():
    create_tables()

    db = SessionLocal()

    try:
        return {
            "path": str(DATABASE_PATH),
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