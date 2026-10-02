import { useEffect, useRef, useState } from "react";
import "./App.css";

const WS_URL = "ws://10.132.32.10:8000/ws/ecg";

const SAMPLE_RATE = 250;
const DISPLAY_SECONDS = 10;
const MAX_SAMPLES = SAMPLE_RATE * DISPLAY_SECONDS;

function App() {
  const [connectionStatus, setConnectionStatus] = useState("Connecting...");

  const [deviceId, setDeviceId] = useState("--");
  const [samplingRate, setSamplingRate] = useState("--");
  const [samplesReceived, setSamplesReceived] = useState(0);
  const [bufferStatus, setBufferStatus] = useState("Waiting");

  const canvasRef = useRef(null);

  const incomingBufferRef = useRef([]);
  const displayBufferRef = useRef([]);

  const websocketRef = useRef(null);

  const playbackAccumulatorRef = useRef(0);
  const lastFrameTimeRef = useRef(null);

  /*
   * ================================================
   * WEBSOCKET
   * ================================================
   */

  useEffect(() => {
    let reconnectTimer;
    let shouldReconnect = true;

    const connectWebSocket = () => {
      console.log("[WebSocket] Connecting:", WS_URL);

      setConnectionStatus("Connecting...");

      const websocket = new WebSocket(WS_URL);

      websocketRef.current = websocket;

      websocket.onopen = () => {
        console.log("[WebSocket] Connected");

        setConnectionStatus("Connected");
      };

      websocket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type !== "ecg") {
            return;
          }

          setDeviceId(data.device_id ?? "--");

          setSamplingRate(data.sampling_rate ?? "--");

          const samples = Array.isArray(data.samples) ? data.samples : [];

          if (!samples.length) {
            return;
          }

          /*
           * Add the entire batch to
           * the playback queue.
           */
          incomingBufferRef.current.push(...samples);

          setSamplesReceived((previous) => previous + samples.length);
        } catch (error) {
          console.error("[WebSocket] Invalid data:", error);
        }
      };

      websocket.onerror = () => {
        setConnectionStatus("Error");
      };

      websocket.onclose = () => {
        setConnectionStatus("Disconnected");

        websocketRef.current = null;

        if (shouldReconnect) {
          reconnectTimer = setTimeout(connectWebSocket, 2000);
        }
      };
    };

    connectWebSocket();

    return () => {
      shouldReconnect = false;

      clearTimeout(reconnectTimer);

      if (websocketRef.current) {
        websocketRef.current.close();
      }
    };
  }, []);

  /*
   * ================================================
   * CONTINUOUS PLAYBACK + DRAWING
   * ================================================
   */

  useEffect(() => {
    let animationFrame;

    const render = (timestamp) => {
      /*
       * Calculate elapsed time.
       */
      if (lastFrameTimeRef.current === null) {
        lastFrameTimeRef.current = timestamp;
      }

      let delta = (timestamp - lastFrameTimeRef.current) / 1000;

      lastFrameTimeRef.current = timestamp;

      /*
       * Avoid a huge jump when
       * browser becomes inactive.
       */
      delta = Math.min(delta, 0.1);

      /*
       * At 250 Hz, determine how
       * many samples should advance.
       */
      playbackAccumulatorRef.current += delta * SAMPLE_RATE;

      let samplesToConsume = Math.floor(playbackAccumulatorRef.current);

      playbackAccumulatorRef.current -= samplesToConsume;

      /*
       * Consume incoming samples
       * at a fixed 250 Hz.
       */
      while (samplesToConsume > 0 && incomingBufferRef.current.length > 0) {
        const sample = incomingBufferRef.current.shift();

        displayBufferRef.current.push(sample);

        if (displayBufferRef.current.length > MAX_SAMPLES) {
          displayBufferRef.current.shift();
        }

        samplesToConsume--;
      }

      /*
       * Buffer status.
       */
      const queueSize = incomingBufferRef.current.length;

      if (queueSize >= SAMPLE_RATE) {
        setBufferStatus("Stable");
      } else if (queueSize > 0) {
        setBufferStatus("Streaming");
      } else {
        setBufferStatus("Buffering");
      }

      /*
       * Draw current ECG.
       */
      drawECG(canvasRef.current, displayBufferRef.current);

      animationFrame = requestAnimationFrame(render);
    };

    animationFrame = requestAnimationFrame(render);

    return () => {
      cancelAnimationFrame(animationFrame);
    };
  }, []);

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>ECG Monitoring System</h1>

          <p>Real-time cardiac signal monitoring</p>
        </div>

        <div
          className={`status ${connectionStatus
            .toLowerCase()
            .replace(" ", "-")}`}
        >
          <span className="status-dot" />
          {connectionStatus}
        </div>
      </header>

      <section className="info-grid">
        <div className="info-card">
          <span>Device</span>
          <strong>{deviceId}</strong>
        </div>

        <div className="info-card">
          <span>Sampling Rate</span>
          <strong>{samplingRate} Hz</strong>
        </div>

        <div className="info-card">
          <span>Display Window</span>
          <strong>{DISPLAY_SECONDS} sec</strong>
        </div>

        <div className="info-card">
          <span>Samples Received</span>
          <strong>{samplesReceived.toLocaleString()}</strong>
        </div>
      </section>

      <section className="ecg-card">
        <div className="ecg-header">
          <div>
            <h2>Live ECG</h2>

            <span>Continuous waveform • 250 Hz</span>
          </div>

          <div className="ecg-right-status">
            <div className="buffer-status">
              <span />
              {bufferStatus}
            </div>

            <div className="live-indicator">
              <span />
              LIVE
            </div>
          </div>
        </div>

        <div className="canvas-container">
          <canvas ref={canvasRef} width={1400} height={560} />
        </div>
      </section>

      <footer>
        <span>AD8232 → ESP8266 → FastAPI → WebSocket → ECG Monitor</span>
      </footer>
    </div>
  );
}

/*
 * ====================================================
 * DRAW ECG
 * ====================================================
 */

function drawECG(canvas, samples) {
  if (!canvas) {
    return;
  }

  const ctx = canvas.getContext("2d");

  const width = canvas.width;

  const height = canvas.height;

  /*
   * Background
   */

  ctx.fillStyle = "#ffffff";

  ctx.fillRect(0, 0, width, height);

  /*
   * Determine amplitude range.
   */

  let min = Infinity;
  let max = -Infinity;

  for (let i = 0; i < samples.length; i++) {
    const value = samples[i];

    if (value < min) {
      min = value;
    }

    if (value > max) {
      max = value;
    }
  }

  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    min = -100;
    max = 100;
  }

  let range = max - min;

  if (range < 1) {
    range = 1;
  }

  const padding = range * 0.15;

  min -= padding;
  max += padding;

  /*
   * Grid.
   */

  drawGrid(ctx, width, height);

  /*
   * Axes.
   */

  drawXAxis(ctx, width, height);

  drawYAxis(ctx, width, height, min, max);

  /*
   * Waveform.
   */

  if (samples.length > 1) {
    ctx.beginPath();

    for (let i = 0; i < samples.length; i++) {
      const x = (i / (MAX_SAMPLES - 1)) * width;

      const normalized = (samples[i] - min) / (max - min);

      const y = height - normalized * (height - 30);

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }

    ctx.strokeStyle = "#059669";

    ctx.lineWidth = 2.2;

    ctx.lineJoin = "round";

    ctx.lineCap = "round";

    ctx.stroke();
  }

  /*
   * Axis titles.
   */

  ctx.fillStyle = "#475569";

  ctx.font = "600 12px Inter, system-ui, sans-serif";

  ctx.textAlign = "center";

  ctx.fillText("Time (seconds)", width / 2, height - 4);

  ctx.save();

  ctx.translate(15, height / 2);

  ctx.rotate(-Math.PI / 2);

  ctx.fillText("Amplitude (ADC units)", 0, 0);

  ctx.restore();
}

/*
 * ====================================================
 * GRID
 * ====================================================
 */

function drawGrid(ctx, width, height) {
  const smallGrid = 25;
  const largeGrid = 125;

  /*
   * Small grid
   */

  ctx.lineWidth = 1;

  ctx.strokeStyle = "rgba(16, 185, 129, 0.10)";

  for (let x = 0; x <= width; x += smallGrid) {
    ctx.beginPath();

    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);

    ctx.stroke();
  }

  for (let y = 0; y <= height; y += smallGrid) {
    ctx.beginPath();

    ctx.moveTo(0, y);
    ctx.lineTo(width, y);

    ctx.stroke();
  }

  /*
   * Major grid
   */

  ctx.strokeStyle = "rgba(15, 118, 110, 0.18)";

  for (let x = 0; x <= width; x += largeGrid) {
    ctx.beginPath();

    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);

    ctx.stroke();
  }

  for (let y = 0; y <= height; y += largeGrid) {
    ctx.beginPath();

    ctx.moveTo(0, y);
    ctx.lineTo(width, y);

    ctx.stroke();
  }
}

/*
 * ====================================================
 * X AXIS
 * ====================================================
 */

function drawXAxis(ctx, width, height) {
  ctx.fillStyle = "#64748b";

  ctx.font = "12px Inter, system-ui, sans-serif";

  ctx.textAlign = "center";

  for (let second = 0; second <= DISPLAY_SECONDS; second++) {
    const x = (second / DISPLAY_SECONDS) * width;

    ctx.fillText(`${second}s`, x, height - 17);
  }
}

/*
 * ====================================================
 * Y AXIS
 * ====================================================
 */

function drawYAxis(ctx, width, height, min, max) {
  ctx.fillStyle = "#64748b";

  ctx.font = "12px Inter, system-ui, sans-serif";

  ctx.textAlign = "right";

  const divisions = 5;

  for (let i = 0; i <= divisions; i++) {
    const ratio = i / divisions;

    const value = max - ratio * (max - min);

    const y = ratio * (height - 30);

    ctx.fillText(value.toFixed(0), width - 8, y + 4);
  }
}

export default App;
