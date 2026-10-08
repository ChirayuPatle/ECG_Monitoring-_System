import React, { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const WS_URL = "ws://10.132.32.10:9000/ws/ecg";

const DEFAULT_SAMPLE_RATE = 200;
const DEFAULT_WINDOW_SECONDS = 10;
const MAX_BUFFER_SECONDS = 30;

const MAX_BUFFER_SAMPLES = DEFAULT_SAMPLE_RATE * MAX_BUFFER_SECONDS;

const DISPLAY_SAMPLES = DEFAULT_SAMPLE_RATE * DEFAULT_WINDOW_SECONDS;

/* =========================================================
   Signal Processing
   Display-only processing.
   RAW IoT samples are never modified before reaching backend.
   ========================================================= */

function createLowPassCoefficients(sampleRate, cutoff, q = 0.7071) {
  const omega = (2 * Math.PI * cutoff) / sampleRate;

  const alpha = Math.sin(omega) / (2 * q);

  const cos = Math.cos(omega);

  const b0 = (1 - cos) / 2;
  const b1 = 1 - cos;
  const b2 = (1 - cos) / 2;

  const a0 = 1 + alpha;
  const a1 = -2 * cos;
  const a2 = 1 - alpha;

  return {
    b0: b0 / a0,
    b1: b1 / a0,
    b2: b2 / a0,
    a1: a1 / a0,
    a2: a2 / a0,
  };
}

function createHighPassCoefficients(sampleRate, cutoff, q = 0.7071) {
  const omega = (2 * Math.PI * cutoff) / sampleRate;

  const alpha = Math.sin(omega) / (2 * q);

  const cos = Math.cos(omega);

  const b0 = (1 + cos) / 2;
  const b1 = -(1 + cos);
  const b2 = (1 + cos) / 2;

  const a0 = 1 + alpha;
  const a1 = -2 * cos;
  const a2 = 1 - alpha;

  return {
    b0: b0 / a0,
    b1: b1 / a0,
    b2: b2 / a0,
    a1: a1 / a0,
    a2: a2 / a0,
  };
}

function createNotchCoefficients(sampleRate, frequency, q = 30) {
  const omega = (2 * Math.PI * frequency) / sampleRate;

  const alpha = Math.sin(omega) / (2 * q);

  const cos = Math.cos(omega);

  const b0 = 1;
  const b1 = -2 * cos;
  const b2 = 1;

  const a0 = 1 + alpha;
  const a1 = -2 * cos;
  const a2 = 1 - alpha;

  return {
    b0: b0 / a0,
    b1: b1 / a0,
    b2: b2 / a0,
    a1: a1 / a0,
    a2: a2 / a0,
  };
}

function applyBiquad(signal, coefficients) {
  const output = new Float64Array(signal.length);

  let x1 = 0;
  let x2 = 0;
  let y1 = 0;
  let y2 = 0;

  const { b0, b1, b2, a1, a2 } = coefficients;

  for (let i = 0; i < signal.length; i++) {
    const x0 = signal[i];

    const y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;

    output[i] = y0;

    x2 = x1;
    x1 = x0;

    y2 = y1;
    y1 = y0;
  }

  return output;
}

function zeroPhaseFilter(signal, coefficients) {
  if (signal.length < 10) {
    return signal;
  }

  const forward = applyBiquad(signal, coefficients);

  const reversed = new Float64Array(forward.length);

  for (let i = 0; i < forward.length; i++) {
    reversed[i] = forward[forward.length - 1 - i];
  }

  const backward = applyBiquad(reversed, coefficients);

  const output = new Float64Array(backward.length);

  for (let i = 0; i < backward.length; i++) {
    output[i] = backward[backward.length - 1 - i];
  }

  return output;
}

function median(values) {
  if (!values.length) return 0;

  const sorted = [...values].sort((a, b) => a - b);

  const middle = Math.floor(sorted.length / 2);

  if (sorted.length % 2 === 0) {
    return (sorted[middle - 1] + sorted[middle]) / 2;
  }

  return sorted[middle];
}

function percentile(values, percentileValue) {
  if (!values.length) return 1;

  const sorted = [...values].sort((a, b) => a - b);

  const index = (percentileValue / 100) * (sorted.length - 1);

  const lower = Math.floor(index);
  const upper = Math.ceil(index);

  if (lower === upper) {
    return sorted[lower];
  }

  const weight = index - lower;

  return sorted[lower] + (sorted[upper] - sorted[lower]) * weight;
}

function movingAverage(signal, windowSize = 3) {
  if (signal.length < windowSize) {
    return signal;
  }

  const output = new Float64Array(signal.length);

  const half = Math.floor(windowSize / 2);

  for (let i = 0; i < signal.length; i++) {
    let sum = 0;
    let count = 0;

    for (let j = i - half; j <= i + half; j++) {
      if (j >= 0 && j < signal.length) {
        sum += signal[j];
        count++;
      }
    }

    output[i] = sum / count;
  }

  return output;
}

/**
 * Converts the actual captured ADC waveform
 * into a cleaner visualization.
 *
 * IMPORTANT:
 * This function is ONLY for the screen.
 * Backend receives/stores original raw samples.
 */
function prepareDisplaySignal(rawSamples, sampleRate) {
  if (!rawSamples || rawSamples.length < 20) {
    return rawSamples || [];
  }

  const values = rawSamples.map(Number).filter(Number.isFinite);

  if (values.length < 20) {
    return values;
  }

  /* Remove ADC/DC offset */
  const dcOffset = median(values);

  const centered = new Float64Array(values.length);

  for (let i = 0; i < values.length; i++) {
    centered[i] = values[i] - dcOffset;
  }

  /*
   * Display conditioning:
   *
   * High-pass  : 0.5 Hz
   * Low-pass   : 35 Hz
   * Notch      : 50 Hz
   *
   * This is visualization conditioning only.
   */

  const highPass = createHighPassCoefficients(sampleRate, 0.5);

  const lowPass = createLowPassCoefficients(sampleRate, 35);

  const notch = createNotchCoefficients(sampleRate, 50);

  let filtered = zeroPhaseFilter(centered, highPass);

  filtered = zeroPhaseFilter(filtered, lowPass);

  /*
   * Apply notch only when sampling
   * rate can represent 50 Hz safely.
   */

  if (sampleRate > 100) {
    filtered = zeroPhaseFilter(filtered, notch);
  }

  /* Gentle smoothing */
  filtered = movingAverage(filtered, 3);

  /*
   * Robust amplitude scaling.
   * Prevents one abnormal spike from
   * stretching the entire ECG waveform.
   */

  const absoluteValues = Array.from(filtered)
    .map(Math.abs)
    .filter(Number.isFinite);

  const amplitude = percentile(absoluteValues, 98.5) || 1;

  const scale = amplitude * 1.15;

  const normalized = new Float64Array(filtered.length);

  for (let i = 0; i < filtered.length; i++) {
    normalized[i] = Math.max(-1.25, Math.min(1.25, filtered[i] / scale));
  }

  return Array.from(normalized);
}

/* =========================================================
   ECG Canvas
   ========================================================= */

function ECGCanvas({ samples, sampleRate, width = 1400, height = 380 }) {
  const canvasRef = useRef(null);
  const [canvasWidth, setCanvasWidth] = useState(width);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = canvas?.parentElement;

    if (!container) return;

    const resizeObserver = new ResizeObserver(() => {
      setCanvasWidth(Math.max(240, Math.floor(container.clientWidth)));
    });

    resizeObserver.observe(container);
    setCanvasWidth(Math.max(240, Math.floor(container.clientWidth)));

    return () => resizeObserver.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;

    if (!canvas) return;

    const context = canvas.getContext("2d");

    if (!context) return;

    const dpr = window.devicePixelRatio || 1;

    canvas.width = canvasWidth * dpr;

    canvas.height = height * dpr;

    canvas.style.width = `${canvasWidth}px`;

    canvas.style.height = `${height}px`;

    context.setTransform(dpr, 0, 0, dpr, 0, 0);

    /* Background */
    context.fillStyle = "#fbfdfb";

    context.fillRect(0, 0, canvasWidth, height);

    const paddingLeft = 58;
    const paddingRight = 20;
    const paddingTop = 22;
    const paddingBottom = 38;

    const graphWidth = canvasWidth - paddingLeft - paddingRight;

    const graphHeight = height - paddingTop - paddingBottom;

    const graphBottom = paddingTop + graphHeight;

    const centerY = paddingTop + graphHeight / 2;

    /* =====================================================
       ECG Grid
       ===================================================== */

    const minorSeconds = 0.2;
    const majorSeconds = 1;

    const minorSpacing = (minorSeconds * graphWidth) / DEFAULT_WINDOW_SECONDS;

    const majorSpacing = (majorSeconds * graphWidth) / DEFAULT_WINDOW_SECONDS;

    context.lineWidth = 1;

    for (
      let x = paddingLeft;
      x <= paddingLeft + graphWidth;
      x += minorSpacing
    ) {
      context.strokeStyle = "rgba(54, 139, 88, 0.12)";

      context.beginPath();

      context.moveTo(x, paddingTop);

      context.lineTo(x, graphBottom);

      context.stroke();
    }

    for (let y = paddingTop; y <= graphBottom; y += 20) {
      context.strokeStyle = "rgba(54, 139, 88, 0.12)";

      context.beginPath();

      context.moveTo(paddingLeft, y);

      context.lineTo(paddingLeft + graphWidth, y);

      context.stroke();
    }

    for (
      let x = paddingLeft;
      x <= paddingLeft + graphWidth;
      x += majorSpacing
    ) {
      context.strokeStyle = "rgba(54, 139, 88, 0.22)";

      context.beginPath();

      context.moveTo(x, paddingTop);

      context.lineTo(x, graphBottom);

      context.stroke();
    }

    for (let y = paddingTop; y <= graphBottom; y += 100) {
      context.strokeStyle = "rgba(54, 139, 88, 0.22)";

      context.beginPath();

      context.moveTo(paddingLeft, y);

      context.lineTo(paddingLeft + graphWidth, y);

      context.stroke();
    }

    /* Baseline */

    context.strokeStyle = "rgba(47, 125, 77, 0.32)";

    context.lineWidth = 1;

    context.beginPath();

    context.moveTo(paddingLeft, centerY);

    context.lineTo(paddingLeft + graphWidth, centerY);

    context.stroke();

    /* =====================================================
       Axis
       ===================================================== */

    context.strokeStyle = "rgba(47, 125, 77, 0.45)";

    context.lineWidth = 1;

    context.beginPath();

    context.moveTo(paddingLeft, paddingTop);

    context.lineTo(paddingLeft, graphBottom);

    context.lineTo(paddingLeft + graphWidth, graphBottom);

    context.stroke();

    /* Y-axis labels */

    context.fillStyle = "rgba(67, 95, 77, 0.82)";

    context.font = "11px Inter, Arial, sans-serif";

    context.textAlign = "right";

    const yLabels = [
      {
        text: "+1",
        y: centerY - graphHeight * 0.35,
      },
      {
        text: "0",
        y: centerY + 4,
      },
      {
        text: "-1",
        y: centerY + graphHeight * 0.35,
      },
    ];

    yLabels.forEach((label) => {
      context.fillText(label.text, paddingLeft - 10, label.y);
    });

    /* X-axis labels */

    context.textAlign = "center";

    const labelInterval = canvasWidth < 600 ? 2 : 1;

    for (
      let second = 0;
      second <= DEFAULT_WINDOW_SECONDS;
      second += labelInterval
    ) {
      const x = paddingLeft + (second / DEFAULT_WINDOW_SECONDS) * graphWidth;

      context.fillText(`${second}s`, x, graphBottom + 22);
    }

    /* =====================================================
       Waveform
       ===================================================== */

    if (!samples || samples.length < 2) {
      context.fillStyle = "rgba(67, 95, 77, 0.65)";

      context.font = "14px Inter, Arial, sans-serif";

      context.textAlign = "center";

      context.fillText("Waiting for ECG signal...", canvasWidth / 2, centerY);

      return;
    }

    context.beginPath();

    context.lineWidth = 2;

    context.strokeStyle = "#258a50";

    context.lineJoin = "round";

    context.lineCap = "round";

    const count = samples.length;

    for (let i = 0; i < count; i++) {
      const x = paddingLeft + (i / Math.max(count - 1, 1)) * graphWidth;

      const value = samples[i];

      const y = centerY - value * graphHeight * 0.38;

      if (i === 0) {
        context.moveTo(x, y);
      } else {
        context.lineTo(x, y);
      }
    }

    context.stroke();

    /* X-axis title */

    context.fillStyle = "rgba(67, 95, 77, 0.72)";

    context.font = "11px Inter, Arial, sans-serif";

    context.textAlign = "center";

    context.fillText("Time", paddingLeft + graphWidth / 2, height - 8);

    /* Y-axis title */

    context.save();

    context.translate(14, paddingTop + graphHeight / 2);

    context.rotate(-Math.PI / 2);

    context.fillText("Relative amplitude", 0, 0);

    context.restore();
  }, [samples, sampleRate, canvasWidth, height]);

  return <canvas ref={canvasRef} className="ecg-canvas" />;
}

/* =========================================================
   SmartECG Helpers
   ========================================================= */

const ML_CLASS_NAMES = {
  N: "Normal",
  S: "Supraventricular",
  V: "Ventricular",
  F: "Fusion",
  Q: "Unknown",
};

function getMLClassName(predictedClass) {
  return ML_CLASS_NAMES[predictedClass] || predictedClass || "Unknown";
}

/* =========================================================
   Main Application
   ========================================================= */

export default function App() {
  const [samples, setSamples] = useState([]);

  const [analysis, setAnalysis] = useState(null);

  const [quality, setQuality] = useState(null);

  /* =====================================================
     NEW:
     SmartECG ML state
     ===================================================== */

  const [mlPrediction, setMlPrediction] = useState(null);

  const [mlStatus, setMlStatus] = useState("Waiting");

  const [mlPredictionTime, setMlPredictionTime] = useState(null);

  const [samplingRate, setSamplingRate] = useState(DEFAULT_SAMPLE_RATE);

  const [connectionStatus, setConnectionStatus] = useState("Connecting");

  const [isPlaying, setIsPlaying] = useState(true);

  const [recordingSeconds, setRecordingSeconds] = useState(0);

  const [bufferSeconds, setBufferSeconds] = useState(0);

  const wsRef = useRef(null);

  const rawBufferRef = useRef([]);

  const receivedSamplesRef = useRef(0);

  const isPlayingRef = useRef(true);

  const lastTimestampRef = useRef(null);

  useEffect(() => {
    isPlayingRef.current = isPlaying;
  }, [isPlaying]);

  /* =====================================================
     WebSocket
     ===================================================== */

  useEffect(() => {
    let reconnectTimer;

    let destroyed = false;

    const connect = () => {
      if (destroyed) return;

      setConnectionStatus("Connecting");

      /*
       * ML state belongs to the current
       * WebSocket/session lifecycle.
       */

      setMlStatus("Waiting");

      setMlPrediction(null);

      setMlPredictionTime(null);

      const ws = new WebSocket(WS_URL);

      wsRef.current = ws;

      ws.onopen = () => {
        setConnectionStatus("Connected");
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);

          /* =================================================
             CONNECTION MESSAGE
             ================================================= */

          if (message.type === "connection") {
            if (message.sampling_rate) {
              setSamplingRate(
                Number(message.sampling_rate) || DEFAULT_SAMPLE_RATE,
              );
            }

            return;
          }

          /* =================================================
             NORMAL ECG PACKET
             ================================================= */

          if (message.type === "ecg") {
            const incomingSamples = Array.isArray(message.samples)
              ? message.samples.map(Number).filter(Number.isFinite)
              : [];

            if (incomingSamples.length === 0) {
              return;
            }

            const rate = Number(message.sampling_rate) || DEFAULT_SAMPLE_RATE;

            setSamplingRate(rate);

            rawBufferRef.current.push(...incomingSamples);

            if (rawBufferRef.current.length > MAX_BUFFER_SAMPLES) {
              rawBufferRef.current =
                rawBufferRef.current.slice(-MAX_BUFFER_SAMPLES);
            }

            receivedSamplesRef.current += incomingSamples.length;

            const bufferLength = rawBufferRef.current.length;

            setBufferSeconds(bufferLength / rate);

            /*
             * The optimized backend sends
             * derived analysis separately.
             *
             * Keep support for analysis/quality
             * if an older backend sends them
             * inside the ECG packet.
             */

            if (message.analysis) {
              setAnalysis(message.analysis);
            }

            if (message.quality) {
              setQuality(message.quality);
            }

            if (message.ml) {
              if (message.ml.status) {
                setMlStatus(message.ml.status);
              }
            }

            if (message.timestamp) {
              lastTimestampRef.current = message.timestamp;
            }

            /*
             * Continue receiving and buffering
             * while paused. Only stop drawing.
             */

            if (isPlayingRef.current) {
              const visible = rawBufferRef.current.slice(
                -Math.min(DISPLAY_SAMPLES, rawBufferRef.current.length),
              );

              setSamples(visible);
            }

            return;
          }

          /* =================================================
             ECG ANALYSIS UPDATE
             ================================================= */

          if (message.type === "analysis") {
            if (message.analysis) {
              setAnalysis(message.analysis);
            }

            if (message.quality) {
              setQuality(message.quality);
            }

            return;
          }

          /* =================================================
             SMART ECG ML PREDICTION
             ================================================= */

          if (message.type === "ml_prediction") {
            const predictions = message.ml?.predictions;

            if (!Array.isArray(predictions) || predictions.length === 0) {
              return;
            }

            /*
             * The backend can return more than
             * one prediction in a single message.
             *
             * Because predictions are ordered
             * chronologically, the final item
             * represents the newest prediction.
             */

            const latestPrediction = predictions[predictions.length - 1];

            if (!latestPrediction) {
              return;
            }

            setMlPrediction(latestPrediction);

            setMlStatus("Live");

            setMlPredictionTime(message.timestamp || null);

            return;
          }

          /* =================================================
             SESSION END
             ================================================= */

          if (message.type === "session_end") {
            setMlStatus("Session ended");

            return;
          }
        } catch (error) {
          console.error("ECG WebSocket message error:", error);
        }
      };

      ws.onerror = () => {
        setConnectionStatus("Error");
      };

      ws.onclose = () => {
        setConnectionStatus("Disconnected");

        setMlStatus("Disconnected");

        if (!destroyed) {
          reconnectTimer = setTimeout(connect, 2000);
        }
      };
    };

    connect();

    return () => {
      destroyed = true;

      clearTimeout(reconnectTimer);

      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  /* =====================================================
     Recording Timer
     ===================================================== */

  useEffect(() => {
    const interval = setInterval(() => {
      const totalSamples = receivedSamplesRef.current;

      const seconds = totalSamples / Math.max(samplingRate, 1);

      setRecordingSeconds(seconds);
    }, 1000);

    return () => clearInterval(interval);
  }, [samplingRate]);

  /* =====================================================
     Display waveform
     ===================================================== */

  const displaySamples = useMemo(() => {
    return prepareDisplaySignal(samples, samplingRate);
  }, [samples, samplingRate]);

  /* =====================================================
     ECG Values
     ===================================================== */

  const heartRate = analysis?.heart_rate_bpm;

  const rrInterval = analysis?.rr_interval_ms;

  const qualityLabel = quality?.label || analysis?.signal_quality || "Waiting";

  const qualityScore = quality?.score ?? analysis?.quality_score;

  const leadOff = analysis?.lead_off;

  /* =====================================================
     ML Values
     ===================================================== */

  const predictedClass = mlPrediction?.predicted_class;

  const predictedClassName = getMLClassName(predictedClass);

  const mlConfidence = mlPrediction?.confidence_pct;

  const mlConfidenceText =
    mlConfidence !== undefined &&
    mlConfidence !== null &&
    Number.isFinite(Number(mlConfidence))
      ? `${Number(mlConfidence).toFixed(1)}%`
      : "--";

  /* =====================================================
     Recording Time
     ===================================================== */

  const formattedRecordingTime = useMemo(() => {
    const total = Math.floor(recordingSeconds);

    const minutes = Math.floor(total / 60);

    const seconds = total % 60;

    return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(
      2,
      "0",
    )}`;
  }, [recordingSeconds]);

  /* =====================================================
     RR Formatting
     ===================================================== */

  const formattedRR = useMemo(() => {
    if (
      rrInterval === null ||
      rrInterval === undefined ||
      !Number.isFinite(Number(rrInterval))
    ) {
      return "--";
    }

    return `${Number(rrInterval).toFixed(0)} ms`;
  }, [rrInterval]);

  /* =====================================================
     HR Formatting
     ===================================================== */

  const formattedHR = useMemo(() => {
    if (
      heartRate === null ||
      heartRate === undefined ||
      !Number.isFinite(Number(heartRate))
    ) {
      return "--";
    }

    return Number(heartRate).toFixed(1);
  }, [heartRate]);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <div className="brand">ECG Monitor</div>
          <div className="subtitle">Your personal heart monitoring view</div>
        </div>

        <div className="connection-area">
          <span
            className={`status-dot ${connectionStatus
              .toLowerCase()
              .replace(" ", "-")}`}
          />

          <span>{connectionStatus}</span>
        </div>
      </header>

      <main className="main-content">
        <section className="welcome-panel">
          <div className="welcome-copy">
            <div className="welcome-eyebrow">Your monitoring overview</div>
            <h1 className="welcome-title">Your ECG, clearly in view.</h1>
            <p className="welcome-description">
              Follow your live recording and signal readings in one place.
              This information is for monitoring and is not a medical diagnosis.
            </p>
          </div>
          <div className="recording-badge">
            <span className={`button-indicator ${isPlaying ? "active" : ""}`} />
            <span className="recording-badge-copy">
              <span className="recording-badge-label">
                {isPlaying ? "LIVE VIEW" : "VIEW PAUSED"}
              </span>
              <span className="recording-badge-time">
                Session {formattedRecordingTime}
              </span>
            </span>
          </div>
        </section>

        <section className="readings-section" aria-labelledby="readings-title">
          <div className="section-heading">
            <div>
              <h2 className="section-title" id="readings-title">
                Your readings
              </h2>
              <p className="section-description">
                Key information from your current monitoring session
              </p>
            </div>
            <div className="connection-chip">
              <span
                className={`status-dot ${connectionStatus
                  .toLowerCase()
                  .replace(" ", "-")}`}
              />
              {connectionStatus === "Connected"
                ? "Monitor connected"
                : connectionStatus}
            </div>
          </div>
          <div className="metrics-grid">
            <div className="metric-card primary-metric">
              <div className="metric-label">HEART RATE</div>
              <div className="metric-value">
                {formattedHR}
                <span className="metric-unit">BPM</span>
              </div>
              <div className="metric-caption">Current reading</div>
            </div>

            <div className="metric-card">
              <div className="metric-label">RR INTERVAL</div>
              <div className="metric-value">{formattedRR}</div>
              <div className="metric-caption">Time between heartbeats</div>
            </div>

            <div className="metric-card">
              <div className="metric-label">SIGNAL QUALITY</div>
              <div className="metric-value metric-quality">{qualityLabel}</div>
              <div className="metric-caption">
                {qualityScore !== undefined && qualityScore !== null
                  ? `Signal reading: ${Number(qualityScore).toFixed(0)}%`
                  : "Waiting for a signal reading"}
              </div>
            </div>

            <div className="metric-card">
              <div className="metric-label">ELECTRODE CONTACT</div>
              <div
                className={`metric-value ${
                  leadOff === true
                    ? "status-warning"
                    : leadOff === false
                      ? "status-ok"
                      : ""
                }`}
              >
                {leadOff === true
                  ? "Check contact"
                  : leadOff === false
                    ? "Connected"
                    : "Waiting"}
              </div>
              <div className="metric-caption">Contact with the ECG monitor</div>
            </div>
          </div>
        </section>

        <section className="ecg-panel">
          <div className="panel-header">
            <div>
              <div className="panel-heading">
                <span className="panel-title">Live ECG waveform</span>
                <span className="waveform-live-label">
                  <span className={`button-indicator ${isPlaying ? "active" : ""}`} />
                  {isPlaying ? "LIVE" : "PAUSED"}
                </span>
              </div>
              <div className="panel-subtitle">
                Your heart signal as it is received by the monitor
              </div>
            </div>
            <div className="waveform-meta">
              <span>{samplingRate} Hz</span>
              <span>{DEFAULT_WINDOW_SECONDS}s window</span>
            </div>
          </div>

          <div className="waveform-container">
            <ECGCanvas samples={displaySamples} sampleRate={samplingRate} />
          </div>

          <div className="waveform-footer">
            <div className="legend-item">
              <span className="legend-line" />
              <span>Live signal</span>
            </div>
            <div className="waveform-note">
              The display is adjusted for viewing; recorded signal data is
              unchanged.
            </div>
          </div>
        </section>

        <section className="control-row">
          <div className="recording-info">
            <div className="control-label">RECORDING TIME</div>
            <div className="recording-time">{formattedRecordingTime}</div>
          </div>
          <div className="buffer-info">
            <div className="control-label">RECENT SIGNAL</div>
            <div className="buffer-time">{bufferSeconds.toFixed(1)} seconds</div>
          </div>
          <button
            className="live-button"
            onClick={() => setIsPlaying((previous) => !previous)}
            aria-pressed={isPlaying}
          >
            <span
              className={
                isPlaying ? "button-indicator active" : "button-indicator"
              }
            />

            {isPlaying ? "Live" : "Paused"}
          </button>
        </section>

        <details className="technical-details">
          <summary>Technical details</summary>
          <div className="details-content">
            <div className="pipeline-title">SIGNAL ANALYSIS</div>
            <div className="metrics-grid">
              <div className="metric-card">
                <div className="metric-label">BEAT CLASSIFICATION</div>
                <div className="metric-value">
                  {mlPrediction ? predictedClassName : "--"}
                </div>
                <div className="metric-caption">
                  {mlPrediction ? `Class: ${predictedClass}` : "Waiting for prediction"}
                </div>
              </div>
              <div className="metric-card">
                <div className="metric-label">MODEL CONFIDENCE</div>
                <div className="metric-value">{mlConfidenceText}</div>
                <div className="metric-caption">Automated analysis estimate</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">ANALYSIS STATUS</div>
                <div className={`metric-value ${mlStatus === "Live" ? "status-ok" : ""}`}>
                  {mlStatus}
                </div>
                <div className="metric-caption">Most recent analysis state</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">MODEL INPUT</div>
                <div className="metric-value">360 Hz</div>
                <div className="metric-caption">300 samples · normalized</div>
              </div>
            </div>
            {mlPrediction && (
              <>
                <div className="pipeline-title">LATEST AUTOMATED RESULT</div>
                <div className="pipeline">
                  <div className="pipeline-step">
                    <span className="step-number">{predictedClass}</span>
                    <span>{predictedClassName}</span>
                  </div>
                  <div className="pipeline-step">
                    <span className="step-number">{mlConfidenceText}</span>
                    <span>Model confidence</span>
                  </div>
                </div>
              </>
            )}
            <div className="pipeline-title pipeline-title-spaced">
              SIGNAL PATH
            </div>
            <div className="pipeline">
              <div className="pipeline-step">
                <span className="step-number">01</span>
                <span>AD8232</span>
              </div>
              <div className="pipeline-arrow">→</div>
              <div className="pipeline-step">
                <span className="step-number">02</span>
                <span>ESP8266</span>
              </div>
              <div className="pipeline-arrow">→</div>
              <div className="pipeline-step">
                <span className="step-number">03</span>
                <span>FastAPI</span>
              </div>
              <div className="pipeline-arrow">→</div>
              <div className="pipeline-step">
                <span className="step-number">04</span>
                <span>SmartECG ML</span>
              </div>
              <div className="pipeline-arrow">→</div>
              <div className="pipeline-step">
                <span className="step-number">05</span>
                <span>ECG Monitor</span>
              </div>
            </div>
            <div className="technical-note">
              <strong>Important:</strong> Automated classifications are model
              outputs and are not a medical diagnosis. The monitor uses three
              electrodes to capture a single-channel ECG signal.
            </div>
          </div>
        </details>
      </main>

      <footer className="footer">
        <span>ECG Monitoring System</span>
        <span>Live heart monitoring</span>
      </footer>
    </div>
  );
}
