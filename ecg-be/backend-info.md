# ECG Backend v2

This is the first backend step toward the final ECG monitoring product.

## Current pipeline

ESP8266/AD8232
    -> POST /api/ecg
    -> analysis buffer
    -> ECG analysis
    -> WebSocket /ws/ecg
    -> React dashboard

## Current measurements

- Heart rate
- RR interval
- Research P/QRS/T delineation
- PR interval estimate
- QRS duration estimate
- QT interval estimate
- QTc using Fridericia correction
- Signal-quality score
- Baseline/noise/clipping indicators
- R-peak locations

## Important limitation

P/QRS/T, PR, QRS, QT and QTc values are explicitly marked
`research_estimate`.

The current device is a single-lead academic IoMT prototype.
These measurements must be validated against annotated ECG data
and expert measurements before any clinical use.

The backend intentionally preserves the raw ECG samples.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 9000
```

## Endpoints

GET /
GET /api/health
GET /api/session/{device_id}
POST /api/ecg
WebSocket /ws/ecg

## Next step

Connect the React dashboard to the `analysis` object already
included in the WebSocket payload. Then add persistent storage
for sessions and historical ECG recordings.


## Important routes
`http://localhost:9000`
`http://localhost:9000/api/session`
`http://localhost:9000/api/database`
`http://localhost:9000/api/session/Enter_session_ID/measurements`
