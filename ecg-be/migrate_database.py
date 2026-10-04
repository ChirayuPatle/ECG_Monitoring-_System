import sqlite3
from pathlib import Path


# ============================================================
# DATABASE LOCATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "ecg_monitor.db"


# ============================================================
# MIGRATION
# ============================================================

def column_exists(
    cursor,
    table_name,
    column_name,
):
    cursor.execute(
        f"PRAGMA table_info({table_name})"
    )

    columns = cursor.fetchall()

    return any(
        column[1] == column_name
        for column in columns
    )


def add_column_if_missing(
    cursor,
    table_name,
    column_name,
    column_definition,
):
    if column_exists(
        cursor,
        table_name,
        column_name,
    ):
        print(
            f"[OK] {table_name}.{column_name} already exists"
        )
        return

    cursor.execute(
        f"""
        ALTER TABLE {table_name}
        ADD COLUMN {column_name}
        {column_definition}
        """
    )

    print(
        f"[ADDED] {table_name}.{column_name}"
    )


def main():

    print()
    print("=" * 60)
    print("ECG SQLite DATABASE MIGRATION")
    print("=" * 60)

    print(
        f"Database: {DATABASE_PATH}"
    )

    if not DATABASE_PATH.exists():

        print()
        print(
            "ERROR: ecg_monitor.db was not found."
        )

        print(
            "Make sure this script is inside:"
        )

        print(
            str(BASE_DIR)
        )

        return

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    try:

        # ----------------------------------------------------
        # EXISTING SESSION TABLE
        # ----------------------------------------------------

        print()
        print(
            "Checking ecg_sessions..."
        )

        add_column_if_missing(
            cursor,
            "ecg_sessions",
            "lead",
            "TEXT NOT NULL DEFAULT 'II'",
        )

        # ----------------------------------------------------
        # COMMIT
        # ----------------------------------------------------

        connection.commit()

        print()
        print(
            "Migration completed successfully."
        )

        # ----------------------------------------------------
        # VERIFY
        # ----------------------------------------------------

        cursor.execute(
            "PRAGMA table_info(ecg_sessions)"
        )

        columns = cursor.fetchall()

        print()
        print(
            "Current ecg_sessions columns:"
        )

        for column in columns:
            print(
                f"  - {column[1]} "
                f"({column[2]})"
            )

        print()
        print(
            "Existing ECG data was preserved."
        )

    except Exception as error:

        connection.rollback()

        print()
        print(
            "MIGRATION FAILED:"
        )

        print(error)

        raise

    finally:

        connection.close()


if __name__ == "__main__":
    main()