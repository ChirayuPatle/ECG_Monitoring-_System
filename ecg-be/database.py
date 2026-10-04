from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    sessionmaker,
)


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATABASE_PATH = BASE_DIR / "ecg_monitor.db"

DATABASE_URL = (
    f"sqlite:///{DATABASE_PATH.as_posix()}"
)


engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False,
    },
)


SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


# ============================================================
# BASE
# ============================================================

class Base(DeclarativeBase):
    pass


# ============================================================
# ECG SESSION
# ============================================================

class ECGSession(Base):
    __tablename__ = "ecg_sessions"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    session_id: Mapped[str] = mapped_column(
        String(150),
        unique=True,
        index=True,
        nullable=False,
    )

    device_id: Mapped[str] = mapped_column(
        String(100),
        index=True,
        nullable=False,
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
    )

    last_packet_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    sampling_rate: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    samples_received: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    packets_received: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(30),
        default="active",
        nullable=False,
    )


# ============================================================
# ECG MEASUREMENTS
# ============================================================

class ECGMeasurement(Base):
    __tablename__ = "ecg_measurements"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    session_id: Mapped[str] = mapped_column(
        String(150),
        index=True,
        nullable=False,
    )

    recorded_at: Mapped[datetime] = mapped_column(
        DateTime,
        index=True,
        nullable=False,
    )

    heart_rate_bpm: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    rr_interval_ms: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    p_duration_ms: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    pr_interval_ms: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    qrs_duration_ms: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    qt_interval_ms: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    qtc_ms: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    measurement_status: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )


# ============================================================
# SIGNAL QUALITY
# ============================================================

class ECGSignalQuality(Base):
    __tablename__ = "ecg_signal_quality"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    session_id: Mapped[str] = mapped_column(
        String(150),
        index=True,
        nullable=False,
    )

    recorded_at: Mapped[datetime] = mapped_column(
        DateTime,
        index=True,
        nullable=False,
    )

    score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    label: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True,
    )

    baseline_wander: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    noise_rms: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    clipping_ratio: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    peak_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )


# ============================================================
# CREATE TABLES
# ============================================================

def create_tables():
    Base.metadata.create_all(
        bind=engine
    )


# ============================================================
# DATABASE INFO
# ============================================================

def get_database_path() -> str:
    return str(DATABASE_PATH)