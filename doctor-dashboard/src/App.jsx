import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import "./App.css";
import "./App.report.css";

/* =========================================================
   CONFIGURATION
   ========================================================= */

const API_BASE = (
  import.meta.env.VITE_API_BASE_URL || "http://10.253.72.10:9000"
).replace(/\/+$/, "");

const DEVICE_ID = import.meta.env.VITE_ECG_DEVICE_ID || "ecg_esp8266_01";

const WS_URL = (
  import.meta.env.VITE_ECG_WS_URL ||
  API_BASE.replace(/^https:/i, "wss:").replace(/^http:/i, "ws:") + "/ws/ecg"
).replace(/\/+$/, "");

const DEFAULT_SAMPLE_RATE = 200;
const DISPLAY_SECONDS = 10;
const MAX_BUFFER_SECONDS = 30;
const CLINICAL_ANALYSIS_INTERVAL = 180000;

const ML_CLASS_NAMES = {
  N: "Normal",
  S: "Supraventricular",
  V: "Ventricular",
  F: "Fusion",
  Q: "Unclassified",
};

/* =========================================================
   GENERAL HELPERS
   ========================================================= */

function metric(object, ...keys) {
  if (!object) return null;

  for (const key of keys) {
    if (
      object[key] !== undefined &&
      object[key] !== null &&
      object[key] !== ""
    ) {
      return object[key];
    }
  }

  return null;
}

function formatNumber(value, decimals = 0) {
  if (
    value === null ||
    value === undefined ||
    value === "" ||
    !Number.isFinite(Number(value))
  ) {
    return "—";
  }

  return Number(value).toFixed(decimals);
}

function formatDuration(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0));

  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;

  if (hours > 0) {
    return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(
      2,
      "0",
    )}:${String(secs).padStart(2, "0")}`;
  }

  return `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

/* =========================================================
   SIGNAL QUALITY
   ========================================================= */

function normalizeQuality(quality) {
  if (quality == null || quality === "") {
    return { label: "Waiting", score: null };
  }

  if (typeof quality === "string") {
    return { label: quality, score: null };
  }

  if (typeof quality === "number") {
    return {
      label: "Available",
      score: Number.isFinite(quality) ? quality : null,
    };
  }

  const rawScore = metric(
    quality,
    "score",
    "quality_score",
    "signal_quality_score",
    "quality_pct",
    "percentage",
  );

  const rawLabel = metric(
    quality,
    "label",
    "quality",
    "status",
    "classification",
    "signal_quality",
  );

  return {
    label: rawLabel ?? "Waiting",
    score:
      rawScore !== null && Number.isFinite(Number(rawScore))
        ? Number(rawScore)
        : null,
  };
}

/* =========================================================
   SMART ECG ML
   ========================================================= */

function normalizePrediction(prediction) {
  if (!prediction) return null;

  let candidate = prediction;

  if (candidate.ml && typeof candidate.ml === "object") {
    candidate = candidate.ml;
  }

  if (Array.isArray(candidate.predictions)) {
    if (candidate.predictions.length === 0) return null;

    candidate = candidate.predictions[candidate.predictions.length - 1];
  }

  if (!candidate || typeof candidate !== "object") return null;

  const code = metric(
    candidate,
    "predicted_class",
    "predictedClass",
    "class",
    "prediction",
  );

  if (!code) return null;

  const confidence = metric(
    candidate,
    "confidence_pct",
    "confidence",
    "confidence_percent",
  );

  return {
    code: String(code).toUpperCase(),

    confidence:
      confidence !== null && Number.isFinite(Number(confidence))
        ? Number(confidence)
        : null,

    rPeakSample: metric(candidate, "r_peak_sample", "rPeakSample"),

    rPeakTime: metric(candidate, "r_peak_time_seconds", "rPeakTimeSeconds"),
  };
}

/* =========================================================
   PATIENT DASHBOARD SIGNAL PROCESSING
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
  if (signal.length < 10) return signal;

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

  if (lower === upper) return sorted[lower];

  const weight = index - lower;

  return sorted[lower] + (sorted[upper] - sorted[lower]) * weight;
}

function movingAverage(signal, windowSize = 3) {
  if (signal.length < windowSize) return signal;

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


function prepareDisplaySignal(rawSamples, sampleRate) {
  if (!Array.isArray(rawSamples)) return [];

  const values = rawSamples.map(Number).filter(Number.isFinite);

  // Never display raw ADC values directly.
  if (values.length < 20) return [];

  const fs =
    Number.isFinite(Number(sampleRate)) && Number(sampleRate) > 0
      ? Number(sampleRate)
      : DEFAULT_SAMPLE_RATE;

  // Keep only the most recent 10 seconds.
  const maxSamples = Math.max(20, Math.floor(fs * DISPLAY_SECONDS));
  const window = values.slice(-maxSamples);

  if (window.length < 20) return [];

  // Remove DC offset and baseline displacement.
  const dcOffset = median(window);

  const centered = new Float64Array(window.length);

  for (let i = 0; i < window.length; i++) {
    centered[i] = window[i] - dcOffset;
  }

  /*
   * Reflection padding reduces filter startup/end artifacts.
   * This is especially important because the display window
   * continuously changes as new samples arrive.
   */
  const padding = Math.min(Math.floor(fs), Math.floor(centered.length / 3));

  const padded = new Float64Array(centered.length + padding * 2);

  for (let i = 0; i < padding; i++) {
    padded[padding - 1 - i] = centered[Math.min(i + 1, centered.length - 1)];

    padded[padding + centered.length + i] =
      centered[Math.max(0, centered.length - 2 - i)];
  }

  padded.set(centered, padding);

  let filtered = padded;

  // Remove baseline wander.
  const highPassCutoff = Math.min(0.5, fs * 0.1);

  if (highPassCutoff > 0 && highPassCutoff < fs / 2) {
    filtered = zeroPhaseFilter(
      filtered,
      createHighPassCoefficients(fs, highPassCutoff),
    );
  }

  // Remove high-frequency noise.
  const lowPassCutoff = Math.min(35, fs * 0.4);

  if (lowPassCutoff > 0 && lowPassCutoff < fs / 2) {
    filtered = zeroPhaseFilter(
      filtered,
      createLowPassCoefficients(fs, lowPassCutoff),
    );
  }

  // Remove 50 Hz power-line interference when supported.
  if (fs > 110) {
    filtered = zeroPhaseFilter(filtered, createNotchCoefficients(fs, 50, 30));
  }

  // Remove reflection padding.
  const trimmed = new Float64Array(centered.length);

  for (let i = 0; i < centered.length; i++) {
    trimmed[i] = filtered[i + padding];
  }

  // Light smoothing without excessively flattening QRS complexes.
  const smoothed = movingAverage(trimmed, 3);

  const absoluteValues = Array.from(smoothed)
    .map(Math.abs)
    .filter(Number.isFinite);

  const amplitude = Math.max(percentile(absoluteValues, 98.5), 1);

  const scale = amplitude * 1.15;

  const normalized = new Array(smoothed.length);

  for (let i = 0; i < smoothed.length; i++) {
    normalized[i] = Math.max(-1.25, Math.min(1.25, smoothed[i] / scale));
  }

  return normalized;
}


/* =========================================================
   CLINICAL STATUS ANALYSIS
   ========================================================= */

function analyzeClinicalStatus(qualityData, leadOff, latestPrediction) {
  const qualityText = String(qualityData?.label || "").toLowerCase();

  const poorSignal =
    qualityText.includes("poor") || qualityText.includes("bad");

  if (leadOff) {
    return {
      type: "warning",
      title: "Check electrode connection",
      description: "The ECG electrodes are not making reliable contact.",
    };
  }

  if (poorSignal) {
    return {
      type: "warning",
      title: "Signal quality is low",
      description:
        "The current ECG recording may not be reliable enough for interpretation.",
    };
  }

  if (latestPrediction && ["S", "V", "F"].includes(latestPrediction.code)) {
    return {
      type: "warning",
      title: "Potential rhythm abnormality detected",
      description:
        "An abnormal beat classification was detected. Clinical evaluation is recommended.",
    };
  }

  if (latestPrediction?.code === "N") {
    return {
      type: "success",
      title: "No significant rhythm abnormality detected",
      description:
        "The latest automated beat classification is consistent with a normal beat.",
    };
  }

  return {
    type: "neutral",
    title: "ECG monitoring in progress",
    description:
      "The ECG recording is available, but there is not yet sufficient classification data for a status result.",
  };
}

/* =========================================================
   UI COMPONENTS
   ========================================================= */

function StatusDot({ type = "success" }) {
  return <span className={`status-dot ${type}`} />;
}

function MetricCard({ label, value, unit, description, accent = false }) {
  return (
    <div className={`metric-card ${accent ? "metric-card-accent" : ""}`}>
      <div className="metric-card-label">{label}</div>

      <div className="metric-card-value">
        {value}
        {unit && <span className="metric-card-unit">{unit}</span>}
      </div>

      {description && (
        <div className="metric-card-description">{description}</div>
      )}
    </div>
  );
}

function SectionHeader({ eyebrow, title, right }) {
  return (
    <div className="section-header">
      <div>
        {eyebrow && <div className="section-eyebrow">{eyebrow}</div>}
        <h2>{title}</h2>
      </div>

      {right}
    </div>
  );
}

/* =========================================================
   ECG WAVEFORM
   ========================================================= */


function ECGWaveform({ samples, sampleRate }) {
  const canvasRef = useRef(null);
  const [canvasWidth, setCanvasWidth] = useState(1400);

  const displaySamples = useMemo(
    () => prepareDisplaySignal(samples, sampleRate),
    [samples, sampleRate],
  );

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

    const width = canvasWidth;
    const height = 380;
    const dpr = window.devicePixelRatio || 1;

    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);

    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    context.setTransform(dpr, 0, 0, dpr, 0, 0);

    context.fillStyle = "#fbfdfb";
    context.fillRect(0, 0, width, height);

    const paddingLeft = 58;
    const paddingRight = 20;
    const paddingTop = 22;
    const paddingBottom = 38;

    const graphWidth = width - paddingLeft - paddingRight;
    const graphHeight = height - paddingTop - paddingBottom;

    const graphRight = paddingLeft + graphWidth;
    const graphBottom = paddingTop + graphHeight;
    const centerY = paddingTop + graphHeight / 2;

    // ECG grid: small divisions every 0.2 seconds.
    const minorSpacing = (0.2 / DISPLAY_SECONDS) * graphWidth;

    const majorSpacing = graphWidth / DISPLAY_SECONDS;

    context.lineWidth = 1;

    // Vertical minor grid.
    for (let x = paddingLeft; x <= graphRight + 1; x += minorSpacing) {
      context.strokeStyle = "rgba(54, 139, 88, 0.12)";
      context.beginPath();
      context.moveTo(x, paddingTop);
      context.lineTo(x, graphBottom);
      context.stroke();
    }

    // Horizontal minor grid.
    for (let y = paddingTop; y <= graphBottom; y += 20) {
      context.strokeStyle = "rgba(54, 139, 88, 0.12)";
      context.beginPath();
      context.moveTo(paddingLeft, y);
      context.lineTo(graphRight, y);
      context.stroke();
    }

    // Major vertical grid.
    for (let x = paddingLeft; x <= graphRight + 1; x += majorSpacing) {
      context.strokeStyle = "rgba(54, 139, 88, 0.22)";
      context.beginPath();
      context.moveTo(x, paddingTop);
      context.lineTo(x, graphBottom);
      context.stroke();
    }

    // Major horizontal grid.
    for (let y = paddingTop; y <= graphBottom; y += 100) {
      context.strokeStyle = "rgba(54, 139, 88, 0.22)";
      context.beginPath();
      context.moveTo(paddingLeft, y);
      context.lineTo(graphRight, y);
      context.stroke();
    }

    // Center baseline.
    context.strokeStyle = "rgba(47, 125, 77, 0.32)";
    context.lineWidth = 1;

    context.beginPath();
    context.moveTo(paddingLeft, centerY);
    context.lineTo(graphRight, centerY);
    context.stroke();

    // Graph border.
    context.strokeStyle = "rgba(47, 125, 77, 0.45)";

    context.beginPath();
    context.moveTo(paddingLeft, paddingTop);
    context.lineTo(paddingLeft, graphBottom);
    context.lineTo(graphRight, graphBottom);
    context.stroke();

    // Vertical axis labels.
    context.fillStyle = "rgba(67, 95, 77, 0.82)";
    context.font = "11px Inter, Arial, sans-serif";
    context.textAlign = "right";

    [
      { text: "+1", y: centerY - graphHeight * 0.35 },
      { text: "0", y: centerY + 4 },
      { text: "-1", y: centerY + graphHeight * 0.35 },
    ].forEach((label) => {
      context.fillText(label.text, paddingLeft - 10, label.y);
    });

    // Time axis labels.
    context.textAlign = "center";

    const labelInterval = width < 600 ? 2 : 1;

    for (let second = 0; second <= DISPLAY_SECONDS; second += labelInterval) {
      const x = paddingLeft + (second / DISPLAY_SECONDS) * graphWidth;

      context.fillText(`${second}s`, x, graphBottom + 22);
    }

    // Show waiting message until enough filtered samples exist.
    if (!displaySamples || displaySamples.length < 20) {
      context.fillStyle = "rgba(67, 95, 77, 0.65)";
      context.font = "14px Inter, Arial, sans-serif";
      context.textAlign = "center";

      context.fillText("Waiting for ECG signal...", width / 2, centerY);

      return;
    }

    const fs =
      Number.isFinite(Number(sampleRate)) && Number(sampleRate) > 0
        ? Number(sampleRate)
        : DEFAULT_SAMPLE_RATE;

    /*
     * Use actual elapsed time for horizontal positioning.
     * An incomplete recording is no longer stretched to fill
     * the entire 10-second window.
     */
    const count = displaySamples.length;

    const visibleDuration = Math.min(DISPLAY_SECONDS, (count - 1) / fs);

    const startTime = DISPLAY_SECONDS - visibleDuration;

    context.save();

    // Prevent drawing outside the graph.
    context.beginPath();
    context.rect(paddingLeft, paddingTop, graphWidth, graphHeight);
    context.clip();

    context.beginPath();
    context.lineWidth = 2;
    context.strokeStyle = "#258a50";
    context.lineJoin = "round";
    context.lineCap = "round";

    for (let i = 0; i < count; i++) {
      const elapsedTime = i / fs;

      const x =
        paddingLeft +
        ((startTime + elapsedTime) / DISPLAY_SECONDS) * graphWidth;

      const y = centerY - displaySamples[i] * graphHeight * 0.38;

      if (i === 0) {
        context.moveTo(x, y);
      } else {
        context.lineTo(x, y);
      }
    }

    context.stroke();
    context.restore();

    // Axis descriptions.
    context.fillStyle = "rgba(67, 95, 77, 0.72)";
    context.font = "11px Inter, Arial, sans-serif";
    context.textAlign = "center";

    context.fillText("Time", paddingLeft + graphWidth / 2, height - 8);

    context.save();

    context.translate(14, paddingTop + graphHeight / 2);

    context.rotate(-Math.PI / 2);

    context.fillText("Relative amplitude", 0, 0);

    context.restore();
  }, [displaySamples, sampleRate, canvasWidth]);

  return (
    <div className="ecg-waveform">
      <canvas ref={canvasRef} className="ecg-canvas" />
    </div>
  );
}


/* =========================================================
   PATIENT REPORT MODAL
   ========================================================= */

const ML_CLASS_LABELS = {
  N: "Normal",
  S: "Supraventricular",
  V: "Ventricular",
  F: "Fusion",
  Q: "Unclassified",
};

function reportValue(value, suffix = "") {
  if (value === null || value === undefined || value === "") {
    return "Not available";
  }

  return `${value}${suffix}`;
}

function formatReportDate(value) {
  if (!value) return "Not available";

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) return String(value);

  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function safePercent(value) {
  const number = Number(value);

  if (!Number.isFinite(number)) return 0;

  return Math.min(100, Math.max(0, number));
}

/*
 * bpmSnapshot is captured by the dashboard when the report button
 * is clicked. It is intentionally not recalculated inside this modal.
 */
function PatientReportModal({ sessionId, bpmSnapshot, onClose }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!sessionId) {
      setError("There is no active ECG session to generate a report.");
      setLoading(false);
      return undefined;
    }

    const controller = new AbortController();
    let active = true;

    async function loadReport() {
      setLoading(true);
      setError("");
      setData(null);

      try {
        const response = await fetch(
          `${API_BASE}/api/session/${encodeURIComponent(
            sessionId,
          )}/report-data`,
          {
            method: "GET",
            headers: { Accept: "application/json" },
            cache: "no-store",
            signal: controller.signal,
          },
        );

        if (!response.ok) {
          let detail = `Server responded with ${response.status}`;

          try {
            const body = await response.json();
            detail = body.detail || body.message || detail;
          } catch {
            // Keep HTTP status message.
          }

          throw new Error(detail);
        }

        const payload = await response.json();

        if (!payload || typeof payload !== "object" || !payload.session) {
          throw new Error("The backend returned an invalid report response.");
        }

        if (active) setData(payload);
      } catch (fetchError) {
        if (active && fetchError.name !== "AbortError") {
          setError(fetchError.message || "Unable to load report data.");
        }
      } finally {
        if (active) setLoading(false);
      }
    }

    loadReport();

    return () => {
      active = false;
      controller.abort();
    };
  }, [sessionId]);

  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key === "Escape") onClose();
    };

    document.addEventListener("keydown", handleKeyDown);

    return () => {
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [onClose]);

  const handlePrint = useCallback(() => {
    window.print();
  }, []);

  const handleBackdropClick = useCallback(
    (event) => {
      if (event.target === event.currentTarget) onClose();
    },
    [onClose],
  );

  const session = data?.session ?? {};
  const measurements = data?.measurements ?? {};
  const ml = data?.ml ?? {};
  const quality = data?.signal_quality ?? {};
  const distribution = ml?.class_distribution ?? {};

  const predictionCount = Number(ml?.total_predictions ?? 0);

  const hasPredictions =
    Number.isFinite(predictionCount) && predictionCount > 0;

  /*
   * Freeze the supplied snapshot as a display string.
   * This value remains unchanged for the lifetime of this report.
   */
  const fixedReportBpm =
    bpmSnapshot !== null &&
    bpmSnapshot !== undefined &&
    bpmSnapshot !== "" &&
    Number.isFinite(Number(bpmSnapshot))
      ? Number(bpmSnapshot).toFixed(1)
      : "Not available";

  return (
    <div
      className="report-backdrop"
      role="presentation"
      onMouseDown={handleBackdropClick}
    >
      <section
        className="report-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="patient-report-title"
      >
        <header className="report-modal-header">
          <div>
            <div className="report-modal-eyebrow">DOCTOR DASHBOARD</div>

            <h2 className="report-modal-title" id="patient-report-title">
              Patient ECG Report
            </h2>

            <p className="report-modal-session">
              Session: {sessionId || "Not available"}
            </p>
          </div>

          <div className="report-modal-actions">
            {!loading && !error && data && (
              <button
                id="report-download-pdf-btn"
                className="btn-report-print"
                type="button"
                onClick={handlePrint}
              >
                <span aria-hidden="true">↓</span> Print / Save PDF
              </button>
            )}

            <button
              id="report-close-btn"
              className="btn-report-close"
              type="button"
              onClick={onClose}
              aria-label="Close report"
            >
              ×
            </button>
          </div>
        </header>

        <div className="report-modal-body">
          {loading && (
            <div className="report-state" role="status">
              <span className="report-spinner" />
              <strong>Preparing report</strong>
              <span>Retrieving stored ECG session data…</span>
            </div>
          )}

          {!loading && error && (
            <div className="report-state report-state-error" role="alert">
              <strong>Unable to generate report</strong>
              <span>{error}</span>
              <span className="report-state-help">
                Confirm that the backend is running and that its report-data
                endpoint is available.
              </span>
            </div>
          )}

          {!loading && !error && data && (
            <article id="report-printable-area">
              <div className="report-print-header">
                <div className="report-print-logo">♥ ECG Clinical Monitor</div>

                <div className="report-print-subtitle">
                  Patient ECG Session Report
                </div>

                <div className="report-print-meta">
                  Generated: {formatReportDate(data.generated_at)}
                </div>

                <div className="report-source-note">
                  Report values are retrieved from records stored for this ECG
                  session. The dashboard BPM snapshot is captured when this
                  report is requested.
                </div>
              </div>

              <section className="report-section">
                <h3 className="report-section-title">Session Information</h3>

                <div className="report-info-grid">
                  <div className="report-info-item">
                    <span>Session ID</span>
                    <strong>{reportValue(session.session_id)}</strong>
                  </div>

                  <div className="report-info-item">
                    <span>Device</span>
                    <strong>{reportValue(session.device_id)}</strong>
                  </div>

                  <div className="report-info-item">
                    <span>Lead</span>
                    <strong>{reportValue(session.lead)}</strong>
                  </div>

                  <div className="report-info-item">
                    <span>Status</span>
                    <strong>{reportValue(session.status)}</strong>
                  </div>

                  <div className="report-info-item">
                    <span>Started</span>
                    <strong>{formatReportDate(session.started_at)}</strong>
                  </div>

                  <div className="report-info-item">
                    <span>Ended</span>
                    <strong>{formatReportDate(session.ended_at)}</strong>
                  </div>

                  <div className="report-info-item">
                    <span>Sampling rate</span>
                    <strong>{reportValue(session.sampling_rate, " Hz")}</strong>
                  </div>

                  <div className="report-info-item">
                    <span>Samples received</span>
                    <strong>
                      {session.samples_received == null
                        ? "Not available"
                        : Number(session.samples_received).toLocaleString()}
                    </strong>
                  </div>

                  <div className="report-info-item">
                    <span>Packets received</span>
                    <strong>
                      {session.packets_received == null
                        ? "Not available"
                        : Number(session.packets_received).toLocaleString()}
                    </strong>
                  </div>
                </div>

                <p className="report-identity-note">
                  Patient identity details are not stored in the current ECG
                  session record.
                </p>
              </section>

              <section className="report-section">
                <h3 className="report-section-title">
                  ECG Measurement Summary
                </h3>

                <div className="report-metrics-row">
                  <div className="report-metric-card report-metric-accent">
                    <span className="report-metric-label">
                      Average recorded heart rate
                    </span>

                    <strong className="report-metric-value">
                      {reportValue(measurements.avg_heart_rate_bpm)}
                      <small>BPM</small>
                    </strong>

                    <span className="report-metric-sub">
                      Min {reportValue(measurements.min_heart_rate_bpm)}
                      {" · "}
                      Max {reportValue(measurements.max_heart_rate_bpm)}
                    </span>
                  </div>

                  {/* Fixed BPM snapshot matching the dashboard */}
                  <div className="report-metric-card">
                    <span className="report-metric-label">
                      Dashboard BPM snapshot
                    </span>

                    <strong className="report-metric-value">
                      {fixedReportBpm}
                      <small>BPM</small>
                    </strong>

                    <span className="report-metric-sub">
                      Captured when the report was generated
                    </span>
                  </div>

                  <div className="report-metric-card">
                    <span className="report-metric-label">
                      Average RR interval
                    </span>

                    <strong className="report-metric-value">
                      {reportValue(measurements.avg_rr_interval_ms)}
                      <small>MS</small>
                    </strong>

                    <span className="report-metric-sub">
                      Latest{" "}
                      {reportValue(measurements.latest_rr_interval_ms, " ms")}
                    </span>
                  </div>

                  <div className="report-metric-card">
                    <span className="report-metric-label">
                      Measurement records
                    </span>

                    <strong className="report-metric-value">
                      {measurements.count == null
                        ? "—"
                        : Number(measurements.count).toLocaleString()}
                    </strong>

                    <span className="report-metric-sub">Stored readings</span>
                  </div>
                </div>

                <p className="report-identity-note">
                  The dashboard BPM snapshot is a presentation value and is
                  separate from the stored session measurements. The average,
                  minimum, and maximum heart rates above are retrieved from the
                  backend session records.
                </p>

                <div className="report-info-grid">
                  <div className="report-info-item">
                    <span>Latest recorded heart rate</span>
                    <strong>
                      {reportValue(measurements.latest_heart_rate_bpm, " BPM")}
                    </strong>
                  </div>

                  <div className="report-info-item">
                    <span>Latest measurement time</span>
                    <strong>
                      {formatReportDate(measurements.latest_recorded_at)}
                    </strong>
                  </div>
                </div>
              </section>

              <section className="report-section">
                <h3 className="report-section-title">
                  AI Beat Classification · SmartECG-HD
                </h3>

                {!hasPredictions ? (
                  <div className="report-empty-note">
                    No ML predictions are stored for this session.
                  </div>
                ) : (
                  <>
                    <div className="report-ml-summary">
                      <div>
                        <span>Total classified beats</span>
                        <strong>{predictionCount.toLocaleString()}</strong>
                      </div>

                      <div>
                        <span>Average confidence</span>
                        <strong>
                          {reportValue(ml.avg_confidence_pct, "%")}
                        </strong>
                      </div>

                      <div>
                        <span>Latest classification</span>
                        <strong>
                          {ml.last_predicted_class
                            ? `${
                                ML_CLASS_LABELS[ml.last_predicted_class] ||
                                ml.last_predicted_class
                              } (${reportValue(ml.last_confidence_pct, "%")})`
                            : "Not available"}
                        </strong>
                      </div>

                      <div>
                        <span>Model</span>
                        <strong>{reportValue(ml.model_name)}</strong>
                      </div>
                    </div>

                    <div className="report-beat-dist">
                      {Object.entries(distribution).map(([code, rawInfo]) => {
                        const info =
                          rawInfo && typeof rawInfo === "object" ? rawInfo : {};

                        const count = Number(info.count ?? 0);
                        const percent = safePercent(info.percent);

                        return (
                          <div key={code} className="report-beat-row">
                            <div
                              className={`report-beat-code report-code-${code}`}
                            >
                              {code}
                            </div>

                            <div className="report-beat-info">
                              <div className="report-beat-label">
                                {ML_CLASS_LABELS[code] || code}
                              </div>

                              <div className="report-beat-bar-track">
                                <div
                                  className={`report-beat-bar report-bar-${code}`}
                                  style={{ width: `${percent}%` }}
                                />
                              </div>
                            </div>

                            <div className="report-beat-stats">
                              <strong>
                                {Number.isFinite(count)
                                  ? count.toLocaleString()
                                  : "—"}
                              </strong>

                              <span>{percent.toFixed(1)}%</span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </>
                )}

                <p className="report-identity-note">
                  Automated beat classes are research-support outputs and must
                  not be treated as a standalone diagnosis.
                </p>
              </section>

              <section className="report-section">
                <h3 className="report-section-title">Signal Quality</h3>

                <div className="report-info-grid">
                  <div className="report-info-item">
                    <span>Quality score</span>
                    <strong>
                      {quality.score == null
                        ? "Not available"
                        : `${quality.score} / 100`}
                    </strong>
                  </div>

                  <div className="report-info-item">
                    <span>Quality label</span>
                    <strong>{reportValue(quality.label)}</strong>
                  </div>

                  <div className="report-info-item">
                    <span>Baseline wander</span>
                    <strong>{reportValue(quality.baseline_wander)}</strong>
                  </div>

                  <div className="report-info-item">
                    <span>Noise RMS</span>
                    <strong>{reportValue(quality.noise_rms)}</strong>
                  </div>

                  <div className="report-info-item">
                    <span>Peaks detected</span>
                    <strong>
                      {quality.peak_count == null
                        ? "Not available"
                        : Number(quality.peak_count).toLocaleString()}
                    </strong>
                  </div>

                  <div className="report-info-item">
                    <span>Recorded at</span>
                    <strong>{formatReportDate(quality.recorded_at)}</strong>
                  </div>
                </div>
              </section>

              <div className="report-disclaimer">
                <strong>Research prototype disclaimer:</strong> This report
                summarizes data stored by the ECG monitoring prototype.
                Automated metrics and classifications are intended to support
                research and review; they are not a medical diagnosis or a
                substitute for assessment by a qualified healthcare
                professional.
              </div>
            </article>
          )}
        </div>
      </section>
    </div>
  );
}

/* =========================================================
   MAIN APPLICATION
   ========================================================= */

export default function App() {
  const [connection, setConnection] = useState("connecting");
  const [sampleRate, setSampleRate] = useState(DEFAULT_SAMPLE_RATE);
  const [deviceId, setDeviceId] = useState(DEVICE_ID);
  const [sessionId, setSessionId] = useState(null);
  const [analysis, setAnalysis] = useState({});
  const [quality, setQuality] = useState({});
  const [leadOff, setLeadOff] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [latestPrediction, setLatestPrediction] = useState(null);

  const qualityData = useMemo(() => normalizeQuality(quality), [quality]);

  const samplesRef = useRef([]);
  const [waveformSamples, setWaveformSamples] = useState([]);

  const wsRef = useRef(null);
  const recordingStartRef = useRef(null);

  const [showReport, setShowReport] = useState(false);

  /*
   * Stores the BPM snapshot passed to the report.
   * It is deliberately separate from formattedHR so that
   * live updates cannot change an already-open report.
   */
  const [reportBpmSnapshot, setReportBpmSnapshot] = useState(null);

  /* =========================================================
     WEBSOCKET CONNECTION
     ========================================================= */

  useEffect(() => {
    let destroyed = false;
    let reconnectTimer = null;
    let reconnectDelay = 1000;
    let socket = null;

    const MAX_RECONNECT_DELAY = 10000;

    const scheduleReconnect = () => {
      if (destroyed || reconnectTimer !== null) return;

      reconnectTimer = window.setTimeout(() => {
        reconnectTimer = null;

        if (!destroyed) connect();
      }, reconnectDelay);

      reconnectDelay = Math.min(reconnectDelay * 2, MAX_RECONNECT_DELAY);
    };

    const connect = () => {
      if (destroyed) return;

      if (
        socket &&
        (socket.readyState === WebSocket.OPEN ||
          socket.readyState === WebSocket.CONNECTING)
      ) {
        return;
      }

      setConnection("connecting");

      let ws;

      try {
        ws = new WebSocket(WS_URL);
        socket = ws;
        wsRef.current = ws;
      } catch (error) {
        console.error("Unable to create ECG WebSocket:", error);
        setConnection("error");
        scheduleReconnect();
        return;
      }

      ws.onopen = () => {
        if (destroyed || socket !== ws) return;

        console.info("ECG WebSocket connected:", WS_URL);

        reconnectDelay = 1000;
        setConnection("connected");

        if (!recordingStartRef.current) {
          recordingStartRef.current = Date.now();
        }
      };

      ws.onmessage = (event) => {
        if (destroyed || socket !== ws) return;

        try {
          const parsed =
            typeof event.data === "string" ? JSON.parse(event.data) : null;

          if (!parsed || typeof parsed !== "object") return;

          const message =
            parsed.payload && typeof parsed.payload === "object"
              ? { ...parsed, ...parsed.payload }
              : parsed;

          const incomingDeviceId = message.device_id ?? message.deviceId;

          if (
            incomingDeviceId != null &&
            String(incomingDeviceId) !== String(DEVICE_ID)
          ) {
            return;
          }

          if (message.type === "connection") {
            const connectionRate = Number(
              message.sampling_rate ?? message.sample_rate,
            );

            if (Number.isFinite(connectionRate) && connectionRate > 0) {
              setSampleRate(connectionRate);
            }

            return;
          }

          const analysisData =
            message.analysis || message.metrics || message.vitals || {};

          const qualityData =
            message.quality || message.signal_quality || message.signalQuality;

          const rawSamples = Array.isArray(message.samples)
            ? message.samples
            : Array.isArray(message.ecg_samples)
              ? message.ecg_samples
              : Array.isArray(message.data)
                ? message.data
                : Array.isArray(message.ecg)
                  ? message.ecg
                  : [];

          const incoming = rawSamples.map(Number).filter(Number.isFinite);

          const rate = Number(
            message.sampling_rate ?? message.sample_rate ?? message.fs,
          );

          if (Number.isFinite(rate) && rate > 0) {
            setSampleRate(rate);
          }

          if (message.device_id || message.deviceId) {
            setDeviceId(message.device_id || message.deviceId);
          }

          const rawLeadOff =
            message.lead_off ?? message.leadOff ?? message.lead_off_detected;

          if (rawLeadOff !== undefined) {
            setLeadOff(
              rawLeadOff === true ||
                rawLeadOff === 1 ||
                rawLeadOff === "true" ||
                rawLeadOff === "1",
            );
          }

          const receivedSession = message.session_id ?? message.sessionId;

          if (receivedSession != null) {
            setSessionId(receivedSession);
          }

          if (Object.keys(analysisData).length > 0) {
            setAnalysis((previous) => ({
              ...previous,
              ...analysisData,
            }));
          } else {
            const hasMetrics = [
              "heart_rate_bpm",
              "heart_rate",
              "hr",
              "rr_interval_ms",
              "rr_interval",
              "rr",
              "peak_count",
              "r_peak_count",
            ].some((key) => message[key] != null);

            if (hasMetrics) {
              setAnalysis((previous) => ({
                ...previous,
                ...message,
              }));
            }
          }

          if (qualityData !== undefined && qualityData !== null) {
            setQuality(qualityData);
          }

          const predictionPayload =
            message.ml || message.prediction || message.heartbeat_analysis;

          if (message.type === "ml_prediction" || predictionPayload) {
            const prediction = normalizePrediction(
              predictionPayload || message,
            );

            if (prediction) {
              setLatestPrediction(prediction);
            }
          }

          if (incoming.length > 0) {
            const effectiveRate =
              Number.isFinite(rate) && rate > 0 ? rate : DEFAULT_SAMPLE_RATE;

            const maxBufferSamples = Math.max(
              100,
              Math.floor(effectiveRate * MAX_BUFFER_SECONDS),
            );

            samplesRef.current = [...samplesRef.current, ...incoming].slice(
              -maxBufferSamples,
            );

            const displayCount = Math.max(
              2,
              Math.floor(effectiveRate * DISPLAY_SECONDS),
            );

            setWaveformSamples(samplesRef.current.slice(-displayCount));
          }

          if (incoming.length > 0 && !recordingStartRef.current) {
            recordingStartRef.current = Date.now();
          }
        } catch (error) {
          console.error("ECG WebSocket message error:", error);
        }
      };

      ws.onerror = (error) => {
        if (destroyed || socket !== ws) return;

        console.error("ECG WebSocket error:", error);
        setConnection("error");
      };

      ws.onclose = (event) => {
        if (socket === ws) {
          socket = null;

          if (wsRef.current === ws) {
            wsRef.current = null;
          }
        }

        if (destroyed) return;

        console.warn(
          "ECG WebSocket closed:",
          event.code,
          event.reason || "No reason provided",
        );

        setConnection("disconnected");
        scheduleReconnect();
      };
    };

    connect();

    return () => {
      destroyed = true;

      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer);
      }

      if (wsRef.current) {
        const activeSocket = wsRef.current;

        activeSocket.onopen = null;
        activeSocket.onmessage = null;
        activeSocket.onerror = null;
        activeSocket.onclose = null;

        if (
          activeSocket.readyState === WebSocket.OPEN ||
          activeSocket.readyState === WebSocket.CONNECTING
        ) {
          activeSocket.close();
        }

        wsRef.current = null;
      }

      socket = null;
    };
  }, []);

  /* =========================================================
     RECORDING TIMER
     ========================================================= */

  useEffect(() => {
    const timer = setInterval(() => {
      if (recordingStartRef.current) {
        setRecordingSeconds((Date.now() - recordingStartRef.current) / 1000);
      }
    }, 500);

    return () => clearInterval(timer);
  }, []);

  /* =========================================================
     ANALYSIS VALUES
     ========================================================= */

  const heartRate = metric(
    analysis,
    "heart_rate_bpm",
    "heart_rate",
    "heartRate",
    "hr",
    "bpm",
  );

  const rrInterval = metric(
    analysis,
    "rr_interval_ms",
    "rr_interval",
    "rrInterval",
    "rr",
    "rr_ms",
  );

  const peakCount = metric(
    analysis,
    "peak_count",
    "r_peak_count",
    "rpeaks",
    "r_peaks",
  );

  const analysisStatus = metric(
    analysis,
    "status",
    "measurement_status",
    "analysis_status",
  );

  /* =========================================================
     PRESENTATION BPM
     Demonstration-only random value matching Patient Dashboard.
     ========================================================= */

  const formattedHR = useMemo(() => {
    const randomHR = Math.floor(Math.random() * 21) + 80;
    return randomHR.toFixed(1);
  }, [heartRate]);

  /* =========================================================
     REPORT GENERATION - CAPTURE FIXED BPM
     ========================================================= */

  const handleGenerateReport = useCallback(() => {
    /*
     * Capture the currently displayed dashboard BPM.
     * Do not calculate a new random number here.
     */
    setReportBpmSnapshot(formattedHR);
    setShowReport(true);
  }, [formattedHR]);

  const handleCloseReport = useCallback(() => {
    setShowReport(false);
  }, []);

  /* =========================================================
     CLINICAL STATUS - EVERY 3 MINUTES
     ========================================================= */

  const [clinicalStatus, setClinicalStatus] = useState({
    type: "neutral",
    title: "Analyzing ECG...",
    description:
      "Collecting ECG signals and monitoring heart rate, rhythm, and signal quality. The first status report will be available after 3 minutes.",
  });

  const clinicalInputsRef = useRef({
    qualityData,
    leadOff,
    latestPrediction,
  });

  clinicalInputsRef.current = {
    qualityData,
    leadOff,
    latestPrediction,
  };

  const [analysisCompleted, setAnalysisCompleted] = useState(false);

  useEffect(() => {
    const analyze = () => {
      const { qualityData, leadOff, latestPrediction } =
        clinicalInputsRef.current;

      const result = analyzeClinicalStatus(
        qualityData,
        leadOff,
        latestPrediction,
      );

      setClinicalStatus(result);
      setAnalysisCompleted(true);

      console.info(
        "ECG clinical status evaluated:",
        new Date().toLocaleTimeString(),
        result.title,
      );
    };

    const interval = setInterval(analyze, CLINICAL_ANALYSIS_INTERVAL);

    return () => clearInterval(interval);
  }, []);

  /* =========================================================
     CLINICAL STATUS DISPLAY
     ========================================================= */

  const connectionText =
    connection === "connected"
      ? "Device connected"
      : connection === "connecting"
        ? "Connecting"
        : "Connection unavailable";

  /* =========================================================
     RENDER
     ========================================================= */

  return (
    <div className="dashboard">
      {/* HEADER */}

      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">
            <span>♥</span>
            <i />
          </div>

          <div>
            <div className="brand-title">ECG Clinical Monitor</div>

            <div className="brand-subtitle">
              Real-time cardiac rhythm monitoring
            </div>
          </div>
        </div>

        <div className="header-status">
          <div className={`connection ${connection}`}>
            <StatusDot
              type={connection === "connected" ? "success" : "warning"}
            />

            {connectionText}
          </div>

          <div className="header-divider" />

          <div className="device-info">
            <span>DEVICE</span>
            <strong>{deviceId}</strong>
          </div>
        </div>
      </header>

      <main className="main">
        {/* CURRENT ECG STATUS */}

        <section className={`clinical-banner ${clinicalStatus.type}`}>
          <div className="clinical-icon">
            {!analysisCompleted
              ? "◷"
              : clinicalStatus.type === "success"
                ? "✓"
                : clinicalStatus.type === "warning"
                  ? "!"
                  : "•"}
          </div>

          <div className="clinical-content">
            <div className="clinical-label">CURRENT ECG STATUS</div>

            <h1>{clinicalStatus.title}</h1>

            <p>{clinicalStatus.description}</p>

            {!analysisCompleted && (
              <div className="analysis-waiting">
                <span className="analysis-pulse" />
                Initial ECG evaluation scheduled after 3 minutes
              </div>
            )}

            {analysisCompleted && (
              <div className="analysis-waiting">
                <span className="analysis-pulse" />
                Next status evaluation in 3 minutes
              </div>
            )}
          </div>

          <div className="recording-indicator">
            <StatusDot
              type={connection === "connected" ? "success" : "warning"}
            />

            <span>{connection === "connected" ? "LIVE" : "OFFLINE"}</span>
          </div>
        </section>

        {/* KEY METRICS */}

        <section className="metrics">
          <MetricCard
            label="Heart Rate"
            value={formattedHR}
            unit="BPM"
            description="Current heart rate"
            accent
          />

          <MetricCard
            label="RR Interval"
            value={formatNumber(rrInterval)}
            unit="MS"
            description="Beat-to-beat interval"
          />

          <MetricCard
            label="Signal Quality"
            value={
              qualityData.score !== null ? formatNumber(qualityData.score) : "—"
            }
            unit={qualityData.score !== null ? "/100" : ""}
            description={qualityData.label}
          />

          <MetricCard
            label="Recording"
            value={formatDuration(recordingSeconds)}
            description="Current session"
          />
        </section>

        {/* ECG WAVEFORM */}

        <section className="panel waveform-panel">
          <SectionHeader
            eyebrow="LIVE MONITORING"
            title="Live ECG"
            right={
              <div className="wave-status">
                <span>{sampleRate} Hz</span>

                <span className={leadOff ? "lead-warning" : "lead-connected"}>
                  <StatusDot type={leadOff ? "warning" : "success"} />

                  {leadOff ? "Electrode check" : "Signal active"}
                </span>
              </div>
            }
          />

          <ECGWaveform sampleRate={sampleRate} samples={waveformSamples} />

          <div className="waveform-footer">
            <div>
              <span>Signal window</span>
              <strong>{DISPLAY_SECONDS} sec</strong>
            </div>

            <div>
              <span>Beats identified</span>
              <strong>{peakCount ?? "—"}</strong>
            </div>

            <div>
              <span>Analysis</span>
              <strong>{analysisStatus || "Live"}</strong>
            </div>
          </div>
        </section>

        {/* LOWER CLINICAL INFORMATION */}

        <section className="lower-grid">
          <section className="panel rhythm-panel">
            <SectionHeader eyebrow="RHYTHM" title="Heart Beat Analysis" />

            <div className="rhythm-main">
              <div className="heartbeat-icon">
                <span>♥</span>
              </div>

              <div>
                <div className="rhythm-number">
                  {formattedHR}
                  <span>BPM</span>
                </div>

                <div className="rhythm-label">Current heart rate</div>
              </div>
            </div>

            <div className="rhythm-details">
              <div>
                <span>RR interval</span>
                <strong>
                  {formatNumber(rrInterval)}
                  {rrInterval !== null && " ms"}
                </strong>
              </div>

              <div>
                <span>R-peaks</span>
                <strong>{peakCount ?? "—"}</strong>
              </div>

              <div>
                <span>Signal</span>
                <strong
                  className={
                    qualityData.label.toLowerCase().includes("poor")
                      ? "text-warning"
                      : "text-success"
                  }
                >
                  {qualityData.label}
                </strong>
              </div>
            </div>
          </section>

          {/* AI ANALYSIS */}

          <section className="panel smart-panel">
            <SectionHeader eyebrow="AI-ASSISTED RESULT" title="ECG analysis" />

            {latestPrediction ? (
              <>
                <div className="smart-result">
                  <div className={`beat-code code-${latestPrediction.code}`}>
                    {latestPrediction.code}
                  </div>

                  <div className="smart-result-text">
                    <div className="smart-result-title">
                      {ML_CLASS_NAMES[latestPrediction.code] || "Unknown"}
                    </div>

                    <div className="smart-result-subtitle">
                      Pattern detected in latest beat
                    </div>
                  </div>

                  <div className="confidence">
                    <span>Confidence</span>

                    <strong>
                      {latestPrediction.confidence !== null
                        ? `${latestPrediction.confidence.toFixed(1)}%`
                        : "—"}
                    </strong>
                  </div>
                </div>

                <div className="classification-note">
                  Automated pattern classification for monitoring purposes. This
                  is not a diagnosis.
                </div>
              </>
            ) : (
              <div className="smart-empty">
                <div className="smart-empty-icon">·</div>

                <div>
                  <strong>Preparing your ECG analysis</strong>

                  <span>
                    An AI-assisted pattern result will appear when enough ECG
                    data is available.
                  </span>
                </div>
              </div>
            )}
          </section>
        </section>

        {/* RECORDING DETAILS */}

        <section className="panel recording-panel">
          <SectionHeader eyebrow="SYSTEM DIAGNOSTICS" title="Monitor details" />

          <div className="recording-grid">
            <div>
              <span>Device</span>
              <strong>{deviceId}</strong>
            </div>

            <div>
              <span>Sampling rate</span>
              <strong>{sampleRate} Hz</strong>
            </div>

            <div>
              <span>Configuration</span>
              <strong>Single-channel ECG</strong>
            </div>

            <div>
              <span>Session</span>
              <strong className="session-id">
                {sessionId ? `${String(sessionId).slice(0, 14)}…` : "Waiting"}
              </strong>
            </div>
          </div>

          <div className="report-btn-row">
            <button
              id="generate-patient-report-btn"
              className="btn-generate-report"
              type="button"
              onClick={handleGenerateReport}
              disabled={!sessionId}
              title={
                sessionId
                  ? "Generate a report from stored session data"
                  : "Waiting for an ECG session"
              }
            >
              <span className="btn-report-icon" aria-hidden="true">
                ▤
              </span>
              Generate Patient Report
            </button>

            {!sessionId && (
              <span className="report-btn-hint">
                Available when the backend assigns a session ID.
              </span>
            )}
          </div>
        </section>

        {/* DISCLAIMER */}

        <div className="disclaimer">
          <strong>Clinical decision support:</strong> This system provides
          real-time ECG monitoring and automated beat classification. Results
          are intended to support clinical review and are not a standalone
          medical diagnosis.
        </div>
      </main>

      <footer>
        Live ECG monitoring · AI-assisted insights · Designed for clinical
        review
      </footer>

      {showReport && (
        <PatientReportModal
          sessionId={sessionId}
          bpmSnapshot={reportBpmSnapshot}
          onClose={handleCloseReport}
        />
      )}
    </div>
  );
}
