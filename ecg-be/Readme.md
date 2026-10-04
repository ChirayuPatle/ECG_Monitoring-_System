# ECG signal fix

Replace the backend `main.py`, `ecg_analysis.py`, and `migrate_database.py` with these files. Keep your existing `database.py` and `ecg_monitor.db`.

1. Stop Uvicorn.
2. Run `python migrate_database.py`.
3. Start: `python -m uvicorn main:app --host 0.0.0.0 --port 9000`.
4. Verify `/`, `/api/health`, `/api/sessions`.
5. Upload `esp8266_ecg.ino` after checking Wi-Fi/server IP.

The live filter is stateful 0.5–40 Hz + 50 Hz notch. HR/RR are calculated from a rolling 12-second window, not each 250-ms packet. Implausible R-R intervals are rejected. P/PR/QRS/QT/QTc remain null until a validated morphology detector is implemented.

`/api/session` alone is intentionally not a route. Use `/api/sessions` to list sessions or `/api/session/{session_id}/...` for a specific session.
