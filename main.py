from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="ECG Monitoring Backend",
    description="Backend for receiving and streaming ECG data",
    version="1.0.0"
)


# ============================================================
# ECG DATA MODEL
# ============================================================

class ECGData(BaseModel):
    device_id: str
    sampling_rate: int = Field(gt=0)
    samples: list[float]


# ============================================================
# WEBSOCKET CONNECTION MANAGER
# ============================================================

class ConnectionManager:

    def __init__(self):
        self.active_connections: list[WebSocket] = []


    async def connect(self, websocket: WebSocket):

        await websocket.accept()

        self.active_connections.append(websocket)

        print(
            f"[WebSocket] Client connected "
            f"({len(self.active_connections)} active)"
        )


    def disconnect(self, websocket: WebSocket):

        if websocket in self.active_connections:

            self.active_connections.remove(websocket)

        print(
            f"[WebSocket] Client disconnected "
            f"({len(self.active_connections)} active)"
        )


    async def broadcast(self, data: dict):

        disconnected = []

        for websocket in self.active_connections:

            try:

                await websocket.send_json(data)

            except Exception:

                disconnected.append(websocket)


        for websocket in disconnected:

            self.disconnect(websocket)


# Create WebSocket manager

manager = ConnectionManager()


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "running",
        "service": "ECG Monitoring Backend"
    }


# ============================================================
# ECG HTTP ENDPOINT
# ============================================================

@app.post("/api/ecg")
async def receive_ecg(data: ECGData):

    print()
    print("========== ECG DATA RECEIVED ==========")

    print(f"Device ID     : {data.device_id}")
    print(f"Sampling Rate : {data.sampling_rate} Hz")
    print(f"Samples       : {len(data.samples)}")


    if data.samples:

        print(f"First Sample  : {data.samples[0]}")
        print(f"Last Sample   : {data.samples[-1]}")


    print("=======================================")


    # ========================================================
    # BROADCAST ECG DATA TO FRONTEND
    # ========================================================

    await manager.broadcast({

        "type": "ecg",

        "device_id": data.device_id,

        "sampling_rate": data.sampling_rate,

        "samples": data.samples

    })


    # ========================================================
    # RESPONSE TO ESP8266
    # ========================================================

    return {

        "status": "success",

        "message": "ECG data received successfully",

        "device_id": data.device_id,

        "samples_received": len(data.samples)

    }


# ============================================================
# ECG WEBSOCKET ENDPOINT
# ============================================================

@app.websocket("/ws/ecg")
async def websocket_ecg(websocket: WebSocket):

    await manager.connect(websocket)


    try:

        # Keep WebSocket connection alive.
        #
        # The frontend does not need to send ECG data.
        # ECG data is pushed from /api/ecg using broadcast().

        while True:

            await websocket.receive()


    except WebSocketDisconnect:

        manager.disconnect(websocket)


    except Exception as error:

        print(
            f"[WebSocket] Connection error: {error}"
        )

        manager.disconnect(websocket)