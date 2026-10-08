from pathlib import Path

app = '''
import React, { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const WS_URL = "ws://10.132.32.10:9000/ws/ecg";

const DEFAULT_SAMPLE_RATE = 200;
const DEFAULT_WINDOW_SECONDS = 10;
const MAX_BUFFER_SECONDS = 30;

const MAX_BUFFER_SAMPLES = DEFAULT_SAMPLE_RATE * MAX_BUFFER_SECONDS;
const DISPLAY_SAMPLES = DEFAULT_SAMPLE_RATE * DEFAULT_WINDOW_SECONDS;

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
        count += 1;
      }
    }

    output[i] = sum / count;
  }

  return output;
}

function prepareDisplaySignal(rawSamples, sampleRate) {
  if (!rawSamples || rawSamples.length < 20) {
    return rawSamples || [];
  }

  const values = rawSamples.map(Number).filter(Number.isFinite);
  if (values.length < 20) {
    return values;
  }

  const dcOffset = median(values);
  const centered = new Float64Array(values.length);

  for (let i = 0; i < values.length; i++) {
    centered[i] = values[i] - dcOffset;
  }

  const highPass = createHighPassCoefficients(sampleRate, 0.5);
  const lowPass = createLowPassCoefficients(sampleRate, 35);
  const notch = createNotchCoefficients(sampleRate, 50);

  let filtered = zeroPhaseFilter(centered, highPass);
  filtered = zeroPhaseFilter(filtered, lowPass);

  if (sampleRate > 100) {
    filtered = zeroPhaseFilter(filtered, notch);
  }

  filtered = movingAverage(filtered, 3);

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

function ECGCanvas({ samples, sampleRate, width = 1200, height = 340 }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const context = canvas.getContext("2d");
    if (!context) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.fillStyle = "#F7FAF9";
    context.fillRect(0, 0, width, height);

    const paddingLeft = 48;
    const paddingRight = 12;
    const paddingTop = 18;
    const paddingBottom = 28;
    const graphWidth = width - paddingLeft - paddingRight;
    const graphHeight = height - paddingTop - paddingBottom;
    const graphBottom = paddingTop + graphHeight;
    const centerY = paddingTop + graphHeight / 2;

    const minorSeconds = 0.2;
    const majorSeconds = 1;
    const minorSpacing = (minorSeconds * graphWidth) / DEFAULT_WINDOW_SECONDS;
    const majorSpacing = (majorSeconds * graphWidth) / DEFAULT_WINDOW_SECONDS;

    context.lineWidth = 1;

    for (let x = paddingLeft; x <= paddingLeft + graphWidth; x += minorSpacing) {
      context.strokeStyle = "rgba(21, 154, 120, 0.10)";
      context.beginPath();
      context.moveTo(x, paddingTop);
      context.lineTo(x, graphBottom);
      context.stroke();
    }

    for (let y = paddingTop; y <= graphBottom; y += 22) {
      context.strokeStyle = "rgba(21, 154, 120, 0.10)";
      context.beginPath();
      context.moveTo(paddingLeft, y);
      context.lineTo(paddingLeft + graphWidth, y);
      context.stroke();
    }

    for (let x = paddingLeft; x <= paddingLeft + graphWidth; x += majorSpacing) {
      context.strokeStyle = "rgba(21, 154, 120, 0.18)";
      context.beginPath();
      context.moveTo(x, paddingTop);
      context.lineTo(x, graphBottom);
      context.stroke();
    }

    context.strokeStyle = "rgba(15, 118, 96, 0.34)";
    context.beginPath();
    context.moveTo(paddingLeft, centerY);
    context.lineTo(paddingLeft + graphWidth, centerY);
    context.stroke();

    context.strokeStyle = "rgba(23, 33, 38, 0.35)";
    context.beginPath();
    context.moveTo(paddingLeft, paddingTop);
    context.lineTo(paddingLeft, graphBottom);
    context.lineTo(paddingLeft + graphWidth, graphBottom);
    context.stroke();

    context.fillStyle = "rgba(102, 117, 124, 0.7)";
    context.font = "11px Inter, Arial, sans-serif";
    context.textAlign = "right";

    const yLabels = [
      { text: "+1", y: centerY - graphHeight * 0.35 },
      { text: "0", y: centerY + 3 },
      { text: "-1", y: centerY + graphHeight * 0.35 },
    ];

    yLabels.forEach((label) => {
      context.fillText(label.text, paddingLeft - 8, label.y);
    });

    context.textAlign = "center";
    for (let second = 0; second <= DEFAULT_WINDOW_SECONDS; second += 1) {
      const x = paddingLeft + (second / DEFAULT_WINDOW_SECONDS) * graphWidth;
      context.fillText(`${second}s`, x, graphBottom + 20);
    }

    if (!samples || samples.length < 2) {
      context.fillStyle = "rgba(102, 117, 124, 0.75)";
      context.font = "14px Inter, Arial, sans-serif";
      context.textAlign = "center";
      context.fillText("Waiting for ECG signal...", width / 2, centerY);
      return;
    }

    context.beginPath();
    context.lineWidth = 2.2;
    context.strokeStyle = "#159A78";
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
  }, [samples, sampleRate, width, height]);

  return <canvas ref={canvasRef} className="ecg-canvas" aria-label="Live ECG waveform" />;
}

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

export default function App() {
  const [samples, setSamples] = useState([]);
  const [analysis, setAnalysis] = useState(null);
  const [quality, setQuality] = useState(null);
  const [mlPrediction, setMlPrediction] = useState(null);
  const [mlStatus, setMlStatus] = useState("Waiting");
  const [samplingRate, setSamplingRate] = useState(DEFAULT_SAMPLE_RATE);
  const [connectionStatus, setConnectionStatus] = useState("Connecting");
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [bufferSeconds, setBufferSeconds] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);

  const wsRef = useRef(null);
  const rawBufferRef = useRef([]);
  const receivedSamplesRef = useRef(0);
  const isPlayingRef = useRef(true);

  useEffect(() => {
    isPlayingRef.current = isPlaying;
  }, [isPlaying]);

  useEffect(() => {
    let reconnectTimer;
    let destroyed = false;

    const connect = () => {
      if (destroyed) return;

      setConnectionStatus("Connecting");
      setMlStatus("Waiting");

      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnectionStatus("Connected");
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);

          if (message.type === "connection") {
            if (message.sampling_rate) {
              setSamplingRate(Number(message.sampling_rate) || DEFAULT_SAMPLE_RATE);
            }
            return;
          }

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
              rawBufferRef.current = rawBufferRef.current.slice(-MAX_BUFFER_SAMPLES);
            }

            receivedSamplesRef.current += incomingSamples.length;
            setBufferSeconds(rawBufferRef.current.length / rate);

            if (message.analysis) setAnalysis(message.analysis);
            if (message.quality) setQuality(message.quality);
            if (message.ml && message.ml.status) setMlStatus(message.ml.status);

            if (isPlayingRef.current) {
              const visible = rawBufferRef.current.slice(
                -Math.min(DISPLAY_SAMPLES, rawBufferRef.current.length),
              );
              setSamples(visible);
            }

            return;
          }

          if (message.type === "analysis") {
            if (message.analysis) setAnalysis(message.analysis);
            if (message.quality) setQuality(message.quality);
            return;
          }

          if (message.type === "ml_prediction") {
            const predictions = message.ml?.predictions;
            if (!Array.isArray(predictions) || predictions.length === 0) {
              return;
            }

            const latestPrediction = predictions[predictions.length - 1];
            if (!latestPrediction) {
              return;
            }

            setMlPrediction(latestPrediction);
            setMlStatus("Live");
            return;
          }

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
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      const seconds = receivedSamplesRef.current / Math.max(samplingRate, 1);
      setRecordingSeconds(seconds);
    }, 1000);

    return () => clearInterval(interval);
  }, [samplingRate]);

  const displaySamples = useMemo(() => prepareDisplaySignal(samples, samplingRate), [samples, samplingRate]);

  const heartRate = analysis?.heart_rate_bpm;
  const rrInterval = analysis?.rr_interval_ms;
  const qualityLabel = quality?.label || analysis?.signal_quality || "Waiting";
  const qualityScore = quality?.score ?? analysis?.quality_score;
  const leadOff = analysis?.lead_off ?? false;
  const predictedClass = mlPrediction?.predicted_class;
  const predictedClassName = getMLClassName(predictedClass);

  const formattedRecordingTime = useMemo(() => {
    const total = Math.floor(recordingSeconds);
    const minutes = Math.floor(total / 60);
    const seconds = total % 60;
    return `${String(minutes).padStart(2, "0")} : ${String(seconds).padStart(2, "0")}`;
  }, [recordingSeconds]);

  const formattedRR = useMemo(() => {
    if (rrInterval === null || rrInterval === undefined || !Number.isFinite(Number(rrInterval))) {
      return "--";
    }
    return `${Number(rrInterval).toFixed(0)} ms`;
  }, [rrInterval]);

  const formattedHR = useMemo(() => {
    if (heartRate === null || heartRate === undefined || !Number.isFinite(Number(heartRate))) {
      return "--";
    }
    return Number(heartRate).toFixed(1);
  }, [heartRate]);

  const patientStatus = useMemo(() => {
    if (connectionStatus === "Connecting") {
      return {
        tone: "stable",
        badge: "Recording",
        title: "Connecting to ECG device...",
        description: "The monitor is establishing a live connection to the ECG signal stream.",
      };
    }

    if (connectionStatus === "Disconnected" || connectionStatus === "Error") {
      return {
        tone: "signal",
        badge: "Connection issue",
        title: "Connection interrupted",
        description: "The ECG signal is temporarily unavailable. Please check the device connection.",
      };
    }

    const lowerQuality = String(qualityLabel || "").toLowerCase();
    const signalScore = Number(qualityScore);

    if (leadOff || lowerQuality.includes("poor") || lowerQuality.includes("noisy")) {
      return {
        tone: "signal",
        badge: "Signal issue",
        title: "Signal Quality Needs Attention",
        description: "The ECG signal is currently too noisy for reliable analysis. Please check electrode placement.",
      };
    }

    if (predictedClass && ["S", "V", "F", "Q"].includes(String(predictedClass).toUpperCase())) {
      return {
        tone: "attention",
        badge: "Attention",
        title: "Potential ECG Rhythm Abnormality",
        description: "An unusual rhythm pattern was detected during this recording. Medical evaluation is recommended.",
      };
    }

    if (!Number.isNaN(signalScore) && signalScore < 0.5) {
      return {
        tone: "signal",
        badge: "Signal issue",
        title: "Signal Quality Needs Attention",
        description: "The ECG signal is currently too noisy for reliable analysis. Please check electrode placement.",
      };
    }

    return {
      tone: "stable",
      badge: "Stable",
      title: "ECG Recording Looks Stable",
      description: "No significant ECG rhythm abnormality detected during this recording.",
    };
  }, [connectionStatus, leadOff, predictedClass, qualityLabel, qualityScore]);

  const connectionLabel =
    connectionStatus === "Connected"
      ? "Device connected"
      : connectionStatus === "Connecting"
        ? "Connecting to ECG device..."
        : connectionStatus === "Disconnected"
          ? "Connection lost"
          : connectionStatus === "Error"
            ? "Connection issue"
            : connectionStatus;

  const signalSummary = useMemo(() => {
    const normalized = String(qualityLabel || "").trim();

    if (!normalized || normalized === "Waiting") {
      return "Waiting";
    }

    const lowered = normalized.toLowerCase();
    if (lowered.includes("good")) return "Good";
    if (lowered.includes("fair")) return "Fair";
    if (lowered.includes("poor")) return "Poor";

    return normalized;
  }, [qualityLabel]);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <div className="brand">ECG Monitor</div>
          <div className="subtitle">Single-channel ECG monitoring</div>
        </div>

        <div className="connection-area" aria-live="polite">
          <span className={`status-dot ${connectionStatus.toLowerCase().replace(/\s+/g, "-")}`} />
          <span>{connectionLabel}</span>
        </div>
      </header>

      <main className="main-content">
        <section className="status-card" aria-live="polite">
          <div className="status-head">
            <div>
              <div className="status-kicker">Current ECG status</div>
              <h1 className="status-title">{patientStatus.title}</h1>
            </div>

            <div className={`status-badge ${patientStatus.tone}`}>
              <span className="status-badge-dot" />
              {patientStatus.badge}
            </div>
          </div>

          <p className="status-description">{patientStatus.description}</p>
        </section>

        <section className="metrics-grid" aria-label="ECG summary metrics">
          <div className="metric-card">
            <div className="metric-label">Heart rate</div>
            <div className="metric-value">
              {formattedHR}
              <span className="metric-unit">BPM</span>
            </div>
            <div className="metric-caption">Current heart rate</div>
          </div>

          <div className="metric-card">
            <div className="metric-label">RR interval</div>
            <div className="metric-value">{formattedRR}</div>
            <div className="metric-caption">Average beat interval</div>
          </div>

          <div className="metric-card">
            <div className="metric-label">Signal quality</div>
            <div className={`metric-value ${signalSummary.toLowerCase() === "good" ? "status-ok" : "status-warning"}`}>
              {signalSummary}
            </div>
            <div className="metric-caption">Current signal clarity</div>
          </div>

          <div className="metric-card">
            <div className="metric-label">Recording</div>
            <div className="metric-value">{formattedRecordingTime}</div>
            <div className="metric-caption">Duration</div>
          </div>
        </section>

        <section className="ecg-panel" aria-label="Live ECG waveform">
          <div className="panel-header">
            <div>
              <div className="panel-title">Live ECG</div>
              <div className="panel-subtitle">Real-time single-channel ECG</div>
            </div>

            <div className="waveform-meta">
              <span>{samplingRate} Hz</span>
              <span>{DEFAULT_WINDOW_SECONDS}s window</span>
            </div>
          </div>

          <div className="waveform-container">
            <ECGCanvas samples={displaySamples} sampleRate={samplingRate} />
          </div>
        </section>

        <div className="lower-grid">
          <section className="info-card" aria-labelledby="heartbeat-title">
            <div className="card-heading" id="heartbeat-title">Heartbeat</div>
            <div className="heartbeat-list">
              <div className="heartbeat-row">
                <span>Heart rate</span>
                <strong>{formattedHR} BPM</strong>
              </div>

              <div className="heartbeat-row">
                <span>RR interval</span>
                <strong>{formattedRR}</strong>
              </div>

              <div className="heartbeat-row">
                <span>Monitoring</span>
                <strong>{mlStatus}</strong>
              </div>
            </div>
          </section>

          <section className="info-card" aria-labelledby="recording-info-title">
            <div className="card-heading" id="recording-info-title">Recording information</div>

            <div className="recording-details">
              <div className="detail-row">
                <span>Device</span>
                <strong>{connectionStatus === "Connected" ? "Connected" : "Connecting"}</strong>
              </div>

              <div className="detail-row">
                <span>Recording</span>
                <strong>{isPlaying ? "Active" : "Paused"}</strong>
              </div>

              <div className="detail-row">
                <span>Signal quality</span>
                <strong>{signalSummary}</strong>
              </div>

              <div className="detail-row">
                <span>Duration</span>
                <strong>{formattedRecordingTime}</strong>
              </div>

              <div className="detail-row">
                <span>Buffer</span>
                <strong>{bufferSeconds.toFixed(1)} s</strong>
              </div>
            </div>

            <button type="button" className="live-button" onClick={() => setIsPlaying((previous) => !previous)}>
              <span className={isPlaying ? "button-indicator active" : "button-indicator"} />
              {isPlaying ? "Live" : "Paused"}
            </button>
          </section>
        </div>

        <section className="info-card guidance-card" aria-labelledby="guidance-title">
          <div className="card-heading" id="guidance-title">For a better ECG recording</div>
          <ul className="guidance-list">
            <li>
              <span className="bullet" aria-hidden="true" />
              <span>Stay relaxed</span>
            </li>
            <li>
              <span className="bullet" aria-hidden="true" />
              <span>Avoid unnecessary movement</span>
            </li>
            <li>
              <span className="bullet" aria-hidden="true" />
              <span>Keep electrodes properly attached</span>
            </li>
            <li>
              <span className="bullet" aria-hidden="true" />
              <span>Remain still during measurement</span>
            </li>
          </ul>
        </section>

        <footer className="disclaimer">
          This system is a research/monitoring prototype and is not a substitute for professional medical evaluation or diagnosis.
        </footer>
      </main>
    </div>
  );
}
'''

css = '''
:root {
  font-family: "Inter", "Plus Jakarta Sans", ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: #172126;
  background: #f7faf9;
  line-height: 1.5;
  font-weight: 400;
  font-synthesis: none;
  text-rendering: optimizeLegibility;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  min-width: 320px;
  min-height: 100vh;
  background: #f7faf9;
  color: #172126;
}
button, input, textarea, select { font: inherit; }
img, canvas { max-width: 100%; display: block; }
#root { min-height: 100vh; }

.app-shell {
  min-height: 100vh;
  background: #f7faf9;
}

.topbar {
  position: sticky;
  top: 0;
  z-index: 10;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  min-height: 76px;
  padding: 18px 32px;
  background: rgba(255, 255, 255, 0.88);
  border-bottom: 1px solid #e3ebe8;
  backdrop-filter: blur(10px);
}

.brand-block { display: flex; flex-direction: column; gap: 4px; }
.brand {
  font-size: 1.65rem;
  font-weight: 700;
  letter-spacing: -0.04em;
  color: #172126;
}
.subtitle {
  font-size: 0.74rem;
  letter-spacing: 0.02em;
  color: #66757c;
}

.connection-area {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 8px 14px;
  border: 1px solid #dfeae6;
  border-radius: 999px;
  background: #ffffff;
  color: #33434b;
  font-size: 0.78rem;
  font-weight: 600;
}

.status-dot {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: #a8b4b8;
  display: inline-block;
}
.status-dot.connected { background: #159a78; box-shadow: 0 0 0 4px rgba(21, 154, 120, 0.12); }
.status-dot.connecting { background: #d97706; box-shadow: 0 0 0 4px rgba(217, 119, 6, 0.12); }
.status-dot.disconnected, .status-dot.error { background: #dc2626; box-shadow: 0 0 0 4px rgba(220, 38, 38, 0.12); }

.main-content {
  width: min(1200px, calc(100% - 40px));
  margin: 0 auto;
  padding: 28px 0 48px;
}

.status-card {
  background: #ffffff;
  border: 1px solid #e3ebe8;
  border-radius: 24px;
  box-shadow: 0 10px 28px rgba(23, 33, 38, 0.04);
  padding: 24px 28px;
  margin-bottom: 24px;
}

.status-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
}

.status-kicker {
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #66757c;
}

.status-title {
  margin: 10px 0 0;
  font-size: clamp(2rem, 2vw + 1rem, 2.8rem);
  line-height: 1.1;
  letter-spacing: -0.05em;
  color: #172126;
}

.status-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 10px 16px;
  border-radius: 999px;
  font-size: 0.82rem;
  font-weight: 700;
  white-space: nowrap;
}

.status-badge-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: currentColor;
}

.status-badge.stable { background: #e8f7f2; color: #159a78; }
.status-badge.attention { background: #fff4e5; color: #d97706; }
.status-badge.signal { background: #feeceb; color: #dc2626; }

.status-description {
  margin: 16px 0 0;
  font-size: 1.02rem;
  line-height: 1.6;
  color: #505f66;
  max-width: 760px;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 18px;
  margin-bottom: 24px;
}

.metric-card {
  background: #ffffff;
  border: 1px solid #e3ebe8;
  border-radius: 20px;
  padding: 20px 20px 18px;
  box-shadow: 0 10px 20px rgba(23, 33, 38, 0.02);
}

.metric-label {
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #66757c;
}

.metric-value {
  margin-top: 16px;
  font-size: clamp(1.8rem, 1.5vw + 1rem, 2.5rem);
  line-height: 1.1;
  letter-spacing: -0.05em;
  font-weight: 700;
  color: #172126;
  font-variant-numeric: tabular-nums;
}

.metric-unit {
  margin-left: 8px;
  font-size: 0.82rem;
  font-weight: 600;
  letter-spacing: 0.04em;
  color: #66757c;
  vertical-align: middle;
}

.metric-caption {
  margin-top: 12px;
  font-size: 0.82rem;
  color: #66757c;
}

.status-ok { color: #159a78; }
.status-warning { color: #d97706; }

.ecg-panel {
  border: 1px solid #dfeae6;
  border-radius: 26px;
  overflow: hidden;
  background: #ffffff;
  box-shadow: 0 18px 30px rgba(23, 33, 38, 0.04);
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
  padding: 18px 22px 16px;
  border-bottom: 1px solid #edf3f1;
  background: #fbfdfc;
}

.panel-title {
  font-size: 1.14rem;
  font-weight: 700;
  color: #172126;
}

.panel-subtitle {
  margin-top: 5px;
  font-size: 0.82rem;
  color: #66757c;
}

.waveform-meta {
  display: inline-flex;
  align-items: center;
  gap: 16px;
  padding: 8px 12px;
  border-radius: 999px;
  background: #eff9f5;
  border: 1px solid #dfeae6;
  font-size: 0.74rem;
  font-weight: 700;
  color: #0f7660;
}

.waveform-container {
  width: 100%;
  overflow-x: auto;
  background: #f7faf9;
}

.ecg-canvas {
  display: block;
  width: 100%;
  min-width: 720px;
  background: #f7faf9;
}

.lower-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 18px;
  margin-top: 24px;
}

.info-card {
  background: #ffffff;
  border: 1px solid #e3ebe8;
  border-radius: 20px;
  padding: 22px 20px;
  box-shadow: 0 10px 20px rgba(23, 33, 38, 0.02);
}

.card-heading {
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #66757c;
}

.heartbeat-list, .recording-details {
  margin-top: 18px;
  display: grid;
  gap: 14px;
}

.heartbeat-row, .detail-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding-bottom: 10px;
  border-bottom: 1px solid #edf3f1;
  color: #43555d;
  font-size: 0.96rem;
}

.heartbeat-row:last-child, .detail-row:last-child { border-bottom: 0; padding-bottom: 0; }
.heartbeat-row strong, .detail-row strong {
  color: #172126;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
}

.live-button {
  margin-top: 18px;
  width: 100%;
  min-height: 44px;
  border: 1px solid #dfeae6;
  border-radius: 12px;
  background: #eff9f5;
  color: #0f7660;
  font-weight: 700;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  cursor: pointer;
  transition: background 0.2s ease, border-color 0.2s ease;
}
.live-button:hover { background: #e8f7f2; border-color: #bfe5d9; }

.button-indicator {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #7e8d93;
}
.button-indicator.active {
  background: #159a78;
  box-shadow: 0 0 0 4px rgba(21, 154, 120, 0.12);
}

.guidance-card { margin-top: 24px; }
.guidance-list {
  margin: 18px 0 0;
  padding: 0;
  list-style: none;
  display: grid;
  gap: 12px;
}
.guidance-list li {
  display: flex;
  align-items: center;
  gap: 12px;
  color: #43555d;
  font-size: 0.98rem;
}
.bullet {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #159a78;
  flex-shrink: 0;
}

.disclaimer {
  margin-top: 24px;
  padding: 16px 18px;
  border-radius: 14px;
  background: #f3f8f6;
  border: 1px solid #e3ebe8;
  color: #5f6f75;
  font-size: 0.82rem;
  line-height: 1.5;
}

@media (max-width: 900px) {
  .metrics-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .lower-grid { grid-template-columns: 1fr; }
}

@media (max-width: 640px) {
  .topbar {
    padding: 14px 18px;
    min-height: 68px;
    align-items: flex-start;
  }
  .brand { font-size: 1.25rem; }
  .connection-area { font-size: 0.72rem; padding: 8px 10px; }
  .main-content { width: calc(100% - 20px); padding-top: 18px; }
  .status-card { padding: 20px 18px; }
  .status-head { flex-direction: column; align-items: flex-start; }
  .status-badge { margin-top: 6px; }
  .metrics-grid { grid-template-columns: 1fr; gap: 12px; }
  .metric-card { padding: 18px 18px 16px; }
  .panel-header { flex-direction: column; align-items: flex-start; }
  .waveform-meta { width: 100%; justify-content: space-between; }
  .ecg-canvas { min-width: 620px; }
  .heartbeat-row, .detail-row { font-size: 0.9rem; }
}
'''

Path(r"C:\Users\HP\Desktop\ecg_prototype\patient\src\App.jsx").write_text(app.strip() + "\n", encoding="utf-8")
Path(r"C:\Users\HP\Desktop\ecg_prototype\patient\src\App.css").write_text(css.strip() + "\n", encoding="utf-8")
print("updated")
