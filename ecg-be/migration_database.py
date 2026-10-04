import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "ecg_monitor.db"


def table_exists(cursor, table_name):
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    return cursor.fetchone() is not None


def column_exists(cursor, table_name, column_name):
    if not table_exists(cursor, table_name):
        return False

    cursor.execute(f"PRAGMA table_info({table_name})")
    return any(row[1] == column_name for row in cursor.fetchall())


def add_column(cursor, table_name, column_name, definition):
    if not table_exists(cursor, table_name):
        print(f"[SKIP] Table does not exist: {table_name}")
        return

    if column_exists(cursor, table_name, column_name):
        print(f"[OK] {table_name}.{column_name}")
        return

    cursor.execute(
        f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}"
    )

    print(f"[ADDED] {table_name}.{column_name}")


if __name__ == "__main__":
    print(f"Database: {DATABASE_PATH}")

    if not DATABASE_PATH.exists():
        raise SystemExit("ERROR: ecg_monitor.db not found")

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    try:
        # --------------------------------------------------
        # ECG SESSIONS
        # --------------------------------------------------

        add_column(
            cursor,
            "ecg_sessions",
            "lead",
            "TEXT NOT NULL DEFAULT 'II'",
        )

        # --------------------------------------------------
        # ECG MEASUREMENTS
        # --------------------------------------------------

        add_column(
            cursor,
            "ecg_measurements",
            "hr",
            "REAL",
        )

        add_column(
            cursor,
            "ecg_measurements",
            "rr",
            "REAL",
        )

        add_column(
            cursor,
            "ecg_measurements",
            "p_duration",
            "REAL",
        )

        add_column(
            cursor,
            "ecg_measurements",
            "pr",
            "REAL",
        )

        add_column(
            cursor,
            "ecg_measurements",
            "qrs",
            "REAL",
        )

        add_column(
            cursor,
            "ecg_measurements",
            "qt",
            "REAL",
        )

        add_column(
            cursor,
            "ecg_measurements",
            "qtc",
            "REAL",
        )

        add_column(
            cursor,
            "ecg_measurements",
            "confidence",
            "REAL",
        )

        add_column(
            cursor,
            "ecg_measurements",
            "measurement_status",
            "TEXT",
        )

        # --------------------------------------------------
        # ECG SIGNAL QUALITY
        # --------------------------------------------------

        add_column(
            cursor,
            "ecg_signal_quality",
            "score",
            "REAL",
        )

        add_column(
            cursor,
            "ecg_signal_quality",
            "label",
            "TEXT",
        )

        add_column(
            cursor,
            "ecg_signal_quality",
            "baseline_wander",
            "REAL",
        )

        add_column(
            cursor,
            "ecg_signal_quality",
            "noise_rms",
            "REAL",
        )

        add_column(
            cursor,
            "ecg_signal_quality",
            "clipping_ratio",
            "REAL",
        )

        add_column(
            cursor,
            "ecg_signal_quality",
            "peak_count",
            "INTEGER",
        )

        add_column(
            cursor,
            "ecg_signal_quality",
            "notes",
            "TEXT",
        )

        # --------------------------------------------------
        # RAW PACKETS
        # --------------------------------------------------

        add_column(
            cursor,
            "ecg_raw_packets",
            "device_id",
            "TEXT",
        )

        add_column(
            cursor,
            "ecg_raw_packets",
            "sampling_rate",
            "INTEGER",
        )

        add_column(
            cursor,
            "ecg_raw_packets",
            "lead_off",
            "INTEGER DEFAULT 0",
        )

        add_column(
            cursor,
            "ecg_raw_packets",
            "samples_json",
            "TEXT",
        )

        add_column(
            cursor,
            "ecg_raw_packets",
            "sample_count",
            "INTEGER",
        )

        # --------------------------------------------------
        # ECG BEATS
        # --------------------------------------------------

        add_column(
            cursor,
            "ecg_beats",
            "beat_index",
            "INTEGER",
        )

        add_column(
            cursor,
            "ecg_beats",
            "r_peak_sample",
            "INTEGER",
        )

        add_column(
            cursor,
            "ecg_beats",
            "start_sample",
            "INTEGER",
        )

        add_column(
            cursor,
            "ecg_beats",
            "end_sample",
            "INTEGER",
        )

        add_column(
            cursor,
            "ecg_beats",
            "rr_interval_ms",
            "REAL",
        )

        add_column(
            cursor,
            "ecg_beats",
            "heart_rate_bpm",
            "REAL",
        )

        add_column(
            cursor,
            "ecg_beats",
            "quality_score",
            "REAL",
        )

        add_column(
            cursor,
            "ecg_beats",
            "sampling_rate",
            "INTEGER",
        )

        add_column(
            cursor,
            "ecg_beats",
            "lead",
            "TEXT DEFAULT 'II'",
        )

        add_column(
            cursor,
            "ecg_beats",
            "sample_count",
            "INTEGER",
        )

        add_column(
            cursor,
            "ecg_beats",
            "samples_json",
            "TEXT",
        )

        add_column(
            cursor,
            "ecg_beats",
            "created_at",
            "DATETIME",
        )

        connection.commit()

        print()
        print("========================================")
        print("DATABASE MIGRATION COMPLETED")
        print("Existing data has been preserved.")
        print("========================================")

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()