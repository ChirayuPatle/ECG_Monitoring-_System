import React, { useEffect, useRef, useState } from "react";
import "./App.css";

const API_URL = "http://10.132.32.10:9000";
const WS_URL = "ws://10.132.32.10:9000/ws/ecg";
const DEFAULT_SAMPLE_RATE = 250;
const DEFAULT_WINDOW_SECONDS = 10;

function toNumber(value, fallback = null) {
  if (value === null || value === undefined || value === "") return fallback;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function normalizeQuality(value) {
  if (value === null || value === undefined) return null;
  if (typeof value === "string") return value.trim();
  if (typeof value === "object") {
    return value.label || value.status || value.quality || value.name || value.text || null;
  }
  return null;
}

function qualityTone(value) {
  const label = String(value || "").toLowerCase();
  if (label.includes("excellent") || label.includes("good") || label.includes("stable")) return "good";
  if (label.includes("fair") || label.includes("moderate") || label.includes("buffering")) return "fair";
  if (label.includes("poor") || label.includes("bad") || label.includes("critical") || label.includes("lead")) return "poor";
  return "waiting";
}

function formatDuration(totalSeconds) {
  const safeSeconds = Math.max(0, Math.floor(toNumber(totalSeconds, 0) || 0));
  const minutes = Math.floor(safeSeconds / 60);
  const seconds = safeSeconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function MetricCard({ label, value, unit, status, tone = "neutral" }) {
  return (
    <div className="metric-card">
      <div className="metric-card__label">{label}</div>
      <div className="metric-card__value">
        {value ?? "--"}
        {unit ? <span className="metric-card__unit">{unit}</span> : null}
      </div>
      <div className={`metric-card__status ${tone}`}>{status}</div>
    </div>
  );
}

function Waveform({ samples, connected, windowSeconds, leadOff, sampleRate }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const context = canvas.getContext("2d");
    if (!context) return;

    const parent = canvas.parentElement;
    const width = Math.max(320, parent.clientWidth);
    const height = 330;
    const dpr = window.devicePixelRatio || 1;

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, width, height);

    const bg = "#071a17";
    const grid = "rgba(114, 153, 142, 0.18)";
    const gridStrong = "rgba(114, 153, 142, 0.26)";
    const waveformColor = "#6fe8b0";
    const muted = "#a6c7c0";

    context.fillStyle = bg;
    context.fillRect(0, 0, width, height);

    const paddingLeft = 52;
    const paddingRight = 14;
    const paddingTop = 18;
    const paddingBottom = 30;
    const plotWidth = width - paddingLeft - paddingRight;
    const plotHeight = height - paddingTop - paddingBottom;
    const centerY = paddingTop + plotHeight / 2;

    for (let x = paddingLeft; x <= width - paddingRight; x += 12) {
      context.strokeStyle = grid;
      context.beginPath();
      context.moveTo(x, paddingTop);
      context.lineTo(x, height - paddingBottom);
      context.stroke();
    }

    for (let y = paddingTop; y <= height - paddingBottom; y += 14) {
      context.strokeStyle = grid;
      context.beginPath();
      context.moveTo(paddingLeft, y);
      context.lineTo(width - paddingRight, y);
      context.stroke();
    }

    for (let x = paddingLeft; x <= width - paddingRight; x += 60) {
      context.strokeStyle = gridStrong;
      context.beginPath();
      context.moveTo(x, paddingTop);
      context.lineTo(x, height - paddingBottom);
      context.stroke();
    }

    context.beginPath();
    context.moveTo(paddingLeft, centerY);
    context.lineTo(width - paddingRight, centerY);
    context.strokeStyle = "rgba(136, 205, 182, 0.38)";
    context.stroke();

    context.font = "11px Inter, sans-serif";
    context.fillStyle = muted;
    context.textAlign = "right";
    context.fillText("+1 mV", paddingLeft - 8, paddingTop + 8);
    context.fillText("0", paddingLeft - 8, centerY);
    context.fillText("-1 mV", paddingLeft - 8, height - paddingBottom - 6);

    context.textAlign = "center";
    const majorX = plotWidth / Math.max(windowSeconds, 1);
    for (let i = 0; i <= windowSeconds; i += 1) {
      const x = paddingLeft + i * majorX;
      context.fillText(`${i}s`, x, height - paddingBottom + 12);
    }

    if (!Array.isArray(samples) || samples.length === 0) {
      context.fillStyle = muted;
      context.font = "600 15px Inter, sans-serif";
      context.textAlign = "center";
      context.fillText(connected ? "Waiting for ECG samples" : "Waiting for connection", width / 2, centerY);
      return;
    }

    const filtered = samples.filter((value) => Number.isFinite(Number(value)));
    if (filtered.length < 2) {
      context.fillStyle = muted;
      context.font = "600 15px Inter, sans-serif";
      context.textAlign = "center";
      context.fillText("Waiting for reliable ECG data", width / 2, centerY);
      return;
    }

    const displayLimit = Math.max(40, Math.floor((sampleRate || DEFAULT_SAMPLE_RATE) * windowSeconds));
    const visibleSamples = filtered.slice(-displayLimit);

    let min = Infinity;
    let max = -Infinity;
    visibleSamples.forEach((value) => {
      const numeric = Number(value);
      min = Math.min(min, numeric);
      max = Math.max(max, numeric);
    });

    const range = Math.max(max - min, 1);
    const pad = range * 0.2;
    const drawMin = min - pad;
    const drawMax = max + pad;

    context.beginPath();
    visibleSamples.forEach((value, index) => {
      const numeric = Number(value);
      const x = paddingLeft + (plotWidth * index) / Math.max(visibleSamples.length - 1, 1);
      const y = height - paddingBottom - ((numeric - drawMin) / Math.max(drawMax - drawMin, 1)) * plotHeight;
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });

    context.strokeStyle = waveformColor;
    context.lineWidth = 2;
    context.lineJoin = "round";
    context.lineCap = "round";
    context.stroke();

    if (leadOff) {
      context.fillStyle = "rgba(248, 187, 92, 0.08)";
      context.fillRect(paddingLeft, paddingTop, plotWidth, plotHeight);
      context.fillStyle = "#f5c366";
      context.font = "600 13px Inter, sans-serif";
      context.textAlign = "center";
      context.fillText("Lead connection may be unstable", width / 2, centerY);
    }
  }, [samples, connected, windowSeconds, leadOff, sampleRate]);

  return (
    <div className="waveform-surface">
      <canvas ref={canvasRef} />
      <div className="waveform-overlay">
        <span>Lead II</span>
        <span>•</span>
        <span>{windowSeconds}s</span>
      </div>
    </div>
  );
}

function App() {
  const [connection, setConnection] = useState("connecting");
  const [sampleRate, setSampleRate] = useState(DEFAULT_SAMPLE_RATE);
  const [deviceId, setDeviceId] = useState("ecg_esp8266_01");
  const [leadOff, setLeadOff] = useState(false);
  const [windowSeconds, setWindowSeconds] = useState(DEFAULT_WINDOW_SECONDS);
  const [isPlaying, setIsPlaying] = useState(true);
  const [samples, setSamples] = useState([]);
  const [analysis, setAnalysis] = useState(null);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [sessionStatus, setSessionStatus] = useState("Recording");

  const samplesRef = useRef([]);
  const socketRef = useRef(null);
  const sessionStartRef = useRef(null);

  useEffect(() => {
    const timer = window.setInterval(() => {
      if (sessionStartRef.current) {
        setRecordingSeconds(Math.floor((Date.now() - sessionStartRef.current) / 1000));
      }
    }, 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    let cancelled = false;
    let reconnectTimer = null;

    const connect = () => {
      if (cancelled) return;
      setConnection("connecting");

      const socket = new WebSocket(WS_URL);
      socketRef.current = socket;

      socket.onopen = () => {
        if (!cancelled) setConnection("connected");
      };

      socket.onmessage = (event) => {
        if (cancelled) return;
        try {
          const message = JSON.parse(event.data);
          if (message.type !== "ecg") return;

          const incomingSamples = Array.isArray(message.samples)
            ? message.samples.map((value) => Number(value)).filter(Number.isFinite)
            : [];

          if (incomingSamples.length > 0) {
            const nextSamples = [...samplesRef.current, ...incomingSamples].slice(-DEFAULT_SAMPLE_RATE * 20);
            samplesRef.current = nextSamples;
            if (isPlaying) setSamples(nextSamples);
          }

          if (message.sample_rate || message.sampling_rate) {
            setSampleRate(toNumber(message.sample_rate ?? message.sampling_rate, DEFAULT_SAMPLE_RATE));
          }

          if (message.device_id) setDeviceId(message.device_id);
          if (message.session_id !== undefined && message.session_id !== null) {
            if (!sessionStartRef.current) sessionStartRef.current = Date.now();
          }

          if (message.lead_off !== undefined) setLeadOff(Boolean(message.lead_off));
          if (message.analysis) setAnalysis(message.analysis);
          if (message.session_status) setSessionStatus(message.session_status);
          if (message.type === "ecg") setSessionStatus("Recording");
        } catch (error) {
          console.error("Invalid WebSocket message", error);
        }
      };

      socket.onerror = () => setConnection("disconnected");
      socket.onclose = () => {
        if (cancelled) return;
        setConnection("disconnected");
        reconnectTimer = window.setTimeout(connect, 2500);
      };
    };

    connect();
    return () => {
      cancelled = true;
      if (reconnectTimer) window.clearTimeout(reconnectTimer);
      if (socketRef.current) socketRef.current.close();
    };
  }, [isPlaying]);

  useEffect(() => {
    const fetchSessions = async () => {
      try {
        const response = await fetch(`${API_URL}/api/sessions`);
        if (!response.ok) return;
        const data = await response.json();
        const list = Array.isArray(data) ? data : data.sessions || data.items || [];
        if (list.length > 0 && list[0].device_id) {
          setDeviceId(list[0].device_id);
        }
      } catch (error) {
        console.error("Session fetch failed", error);
      }
    };
    fetchSessions();
  }, []);

  const heartRate = toNumber(analysis?.heart_rate_bpm ?? analysis?.heart_rate ?? analysis?.current_heart_rate, null);
  const rrInterval = toNumber(analysis?.rr_interval_ms ?? analysis?.rr_interval ?? analysis?.median_rr_interval, null);
  const confidence = toNumber(analysis?.confidence ?? analysis?.analysis_confidence ?? analysis?.confidence_percent, null);
  const qualityValue = normalizeQuality(analysis?.signal_quality ?? analysis?.quality ?? analysis?.quality_label) || "Waiting for signal";
  const signalScore = toNumber(analysis?.signal_quality_score ?? analysis?.quality_score ?? analysis?.score, null);
  const qualityLabel = qualityValue === "Waiting for signal" ? "Waiting for signal" : qualityValue;
  const signalTone = leadOff ? "poor" : qualityTone(qualityLabel);

  const statusLabel =
    connection === "connected" ? "Connected" : connection === "connecting" ? "Connecting" : "Disconnected";

  const deviceStatus = leadOff ? "Lead-off" : connection === "connected" ? "Connected" : "Waiting";

  const alertText = leadOff
    ? "⚠ Check electrode contact"
    : connection === "connected"
      ? "✓ No active monitoring alerts"
      : "⚠ Device connection interrupted";

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand-wrap">
          <div className="brand-mark" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
          <div>
            <div className="brand-title">ECG Monitor</div>
            <div className="brand-subtitle">Real-time cardiac monitoring</div>
          </div>
        </div>

        <div className="header-meta">
          <div className={`status-badge ${connection}`}>
            <span className="live-dot" aria-hidden="true" />
            {statusLabel}
          </div>
          <div className="device-tag">{deviceId}</div>
        </div>
      </header>

      <main className="dashboard">
        <section className="panel metric-panel">
          <div className="metrics-grid">
            <MetricCard
              label="Heart Rate"
              value={heartRate !== null ? `${heartRate}` : "--"}
              unit="BPM"
              status={heartRate !== null ? "Current rate" : "Waiting for analysis"}
              tone={heartRate !== null ? "good" : "waiting"}
            />
            <MetricCard
              label="RR Interval"
              value={rrInterval !== null ? `${rrInterval}` : "--"}
              unit="ms"
              status={rrInterval !== null ? "Median" : "Waiting for signal"}
              tone={rrInterval !== null ? "good" : "waiting"}
            />
            <MetricCard
              label="Signal"
              value={signalScore !== null ? `${Math.round(signalScore)}` : "--"}
              unit={signalScore !== null ? "/ 100" : ""}
              status={qualityLabel}
              tone={signalTone}
            />
            <MetricCard
              label="Confidence"
              value={confidence !== null ? `${Math.round(confidence)}%` : "--"}
              status={confidence !== null ? "Analysis ready" : "Waiting"}
              tone={confidence !== null ? "good" : "waiting"}
            />
          </div>
        </section>

        <section className="panel waveform-panel">
          <div className="waveform-toolbar">
            <div className="waveform-title-wrap">
              <span className="mini-label">Live ECG</span>
              <h2>Lead II waveform</h2>
            </div>

            <div className="waveform-controls" aria-label="Waveform controls">
              <button className={`control-button ${isPlaying ? "active" : ""}`} type="button" onClick={() => setIsPlaying(true)}>Live</button>
              <button className="control-button" type="button" onClick={() => setIsPlaying(false)}>Pause</button>
              {[5, 10, 20].map((value) => (
                <button
                  key={value}
                  className={`control-button ${windowSeconds === value ? "active" : ""}`}
                  type="button"
                  onClick={() => setWindowSeconds(value)}
                >
                  {value}s
                </button>
              ))}
            </div>
          </div>

          <Waveform
            samples={isPlaying ? samples : []}
            connected={connection === "connected"}
            windowSeconds={windowSeconds}
            leadOff={leadOff}
            sampleRate={sampleRate}
          />

          <div className="ecg-overlay">
            <div className="overlay-chip">
              <span className="overlay-label">Live</span>
              <strong>{heartRate !== null ? `${heartRate} BPM` : "-- BPM"}</strong>
            </div>
            <div className="overlay-chip">
              <span className="overlay-label">RR</span>
              <strong>{rrInterval !== null ? `${rrInterval} ms` : "-- ms"}</strong>
            </div>
            <div className="overlay-chip">
              <span className="overlay-label">Status</span>
              <strong>{qualityLabel}</strong>
            </div>
            <div className="overlay-chip">
              <span className="overlay-label">Confidence</span>
              <strong>{confidence !== null ? `${Math.round(confidence)}%` : "Unavailable"}</strong>
            </div>
          </div>
        </section>

        <section className="panel summary-strip">
          <div className="summary-item">
            <span>Sampling</span>
            <strong>{sampleRate} Hz</strong>
          </div>
          <div className="summary-item">
            <span>Lead</span>
            <strong>II</strong>
          </div>
          <div className="summary-item">
            <span>Session</span>
            <strong>{sessionStatus}</strong>
          </div>
          <div className="summary-item">
            <span>Duration</span>
            <strong>{formatDuration(recordingSeconds)}</strong>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
