import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import "./App.css";

const API_BASE =
  import.meta.env.VITE_API_BASE_URL ||
  "http://10.132.32.10:9000";

const DEVICE_ID =
  import.meta.env.VITE_ECG_DEVICE_ID ||
  "ecg_esp8266_01";

const WS_URL = API_BASE.startsWith("https://")
  ? API_BASE.replace(/^https:\/\//, "wss://") + "/ws/ecg"
  : API_BASE.replace(/^http:\/\//, "ws://") + "/ws/ecg";

const DEFAULT_SAMPLE_RATE = 200;
const DISPLAY_SECONDS = 10;

const ML_CLASS_NAMES = {
  N: "Normal",
  S: "Supraventricular",
  V: "Ventricular",
  F: "Fusion",
  Q: "Unclassified",
};

/* -------------------------------------------------------------------------- */
/* Helpers                                                                    */
/* -------------------------------------------------------------------------- */

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
  const total = Math.max(
    0,
    Math.floor(Number(seconds) || 0)
  );

  const hours = Math.floor(total / 3600);
  const minutes = Math.floor(
    (total % 3600) / 60
  );
  const secs = total % 60;

  if (hours > 0) {
    return `${String(hours).padStart(
      2,
      "0"
    )}:${String(minutes).padStart(
      2,
      "0"
    )}:${String(secs).padStart(2, "0")}`;
  }

  return `${String(minutes).padStart(
    2,
    "0"
  )}:${String(secs).padStart(2, "0")}`;
}

function normalizeQuality(quality) {
  if (!quality) {
    return {
      label: "Waiting",
      score: null,
    };
  }

  if (typeof quality === "string") {
    return {
      label: quality,
      score: null,
    };
  }

  return {
    label:
      metric(
        quality,
        "label",
        "quality"
      ) || "Waiting",

    score: metric(
      quality,
      "score"
    ),
  };
}

function normalizePrediction(prediction) {
  if (!prediction) return null;

  const code = metric(
    prediction,
    "predicted_class",
    "predictedClass",
    "class",
    "prediction"
  );

  if (!code) return null;

  const confidence = metric(
    prediction,
    "confidence_pct",
    "confidence",
    "confidence_percent"
  );

  return {
    code: String(code).toUpperCase(),

    confidence:
      confidence !== null &&
      Number.isFinite(Number(confidence))
        ? Number(confidence)
        : null,

    rPeakSample: metric(
      prediction,
      "r_peak_sample",
      "rPeakSample"
    ),

    rPeakTime: metric(
      prediction,
      "r_peak_time_seconds",
      "rPeakTimeSeconds"
    ),
  };
}

/* -------------------------------------------------------------------------- */
/* UI Components                                                              */
/* -------------------------------------------------------------------------- */

function StatusDot({ type = "success" }) {
  return (
    <span
      className={`status-dot ${type}`}
    />
  );
}

function MetricCard({
  label,
  value,
  unit,
  description,
  accent = false,
}) {
  return (
    <div
      className={`metric-card ${
        accent ? "metric-card-accent" : ""
      }`}
    >
      <div className="metric-card-label">
        {label}
      </div>

      <div className="metric-card-value">
        {value}

        {unit && (
          <span className="metric-card-unit">
            {unit}
          </span>
        )}
      </div>

      {description && (
        <div className="metric-card-description">
          {description}
        </div>
      )}
    </div>
  );
}

function SectionHeader({
  eyebrow,
  title,
  right,
}) {
  return (
    <div className="section-header">
      <div>
        {eyebrow && (
          <div className="section-eyebrow">
            {eyebrow}
          </div>
        )}

        <h2>{title}</h2>
      </div>

      {right}
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* ECG Waveform                                                               */
/*                                                                            */
/* This is the SAME waveform rendering approach used by the Patient          */
/* Dashboard: same 10-second display, same grid, same relative amplitude,    */
/* same incoming WebSocket samples.                                          */
/* -------------------------------------------------------------------------- */

function ECGWaveform({
  sampleRate,
  leadOff,
  samplesRef,
}) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const animationRef = useRef(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const container =
      containerRef.current;

    if (!canvas || !container) {
      return;
    }

    const rect =
      container.getBoundingClientRect();

    const width = Math.max(
      1,
      Math.floor(rect.width)
    );

    const height = Math.max(
      220,
      Math.floor(rect.height)
    );

    const dpr = Math.min(
      window.devicePixelRatio || 1,
      2
    );

    const pixelWidth =
      Math.floor(width * dpr);

    const pixelHeight =
      Math.floor(height * dpr);

    if (
      canvas.width !== pixelWidth ||
      canvas.height !== pixelHeight
    ) {
      canvas.width = pixelWidth;
      canvas.height = pixelHeight;

      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
    }

    const ctx =
      canvas.getContext("2d");

    if (!ctx) return;

    ctx.setTransform(
      dpr,
      0,
      0,
      dpr,
      0,
      0
    );

    ctx.clearRect(
      0,
      0,
      width,
      height
    );

    const padding = {
      left: 58,
      right: 18,
      top: 18,
      bottom: 38,
    };

    const plotWidth =
      width -
      padding.left -
      padding.right;

    const plotHeight =
      height -
      padding.top -
      padding.bottom;

    const centerY =
      padding.top +
      plotHeight / 2;

    const bottom =
      height - padding.bottom;

    /* Background */

    ctx.fillStyle = "#fbfcff";

    ctx.fillRect(
      0,
      0,
      width,
      height
    );

    /* ECG grid */

    const verticalSpacing =
      plotWidth /
      (DISPLAY_SECONDS / 0.04);

    const horizontalSpacing =
      plotHeight / 20;

    for (
      let x = padding.left, i = 0;
      x <=
      width - padding.right + 0.5;
      x += verticalSpacing, i++
    ) {
      ctx.beginPath();

      ctx.lineWidth =
        i % 5 === 0 ? 1 : 0.6;

      ctx.strokeStyle =
        i % 5 === 0
          ? "rgba(120,145,194,.14)"
          : "rgba(120,145,194,.055)";

      ctx.moveTo(
        x,
        padding.top
      );

      ctx.lineTo(
        x,
        bottom
      );

      ctx.stroke();
    }

    for (
      let y = padding.top, i = 0;
      y <= bottom + 0.5;
      y += horizontalSpacing, i++
    ) {
      ctx.beginPath();

      ctx.lineWidth =
        i % 5 === 0 ? 1 : 0.6;

      ctx.strokeStyle =
        i % 5 === 0
          ? "rgba(120,145,194,.14)"
          : "rgba(120,145,194,.055)";

      ctx.moveTo(
        padding.left,
        y
      );

      ctx.lineTo(
        width - padding.right,
        y
      );

      ctx.stroke();
    }

    /* Center line */

    ctx.beginPath();

    ctx.lineWidth = 1;
    ctx.strokeStyle =
      "rgba(120,145,194,.19)";

    ctx.moveTo(
      padding.left,
      centerY
    );

    ctx.lineTo(
      width - padding.right,
      centerY
    );

    ctx.stroke();

    /* Labels */

    ctx.fillStyle = "#8995aa";
    ctx.font =
      "11px Inter, system-ui, sans-serif";

    ctx.textAlign = "right";
    ctx.textBaseline = "middle";

    ctx.fillText(
      "+scale",
      padding.left - 9,
      padding.top
    );

    ctx.fillText(
      "0",
      padding.left - 9,
      centerY
    );

    ctx.fillText(
      "−scale",
      padding.left - 9,
      bottom
    );

    ctx.textAlign = "center";
    ctx.textBaseline = "top";

    for (let i = 0; i <= 5; i++) {
      ctx.fillText(
        `${(
          (DISPLAY_SECONDS * i) /
          5
        ).toFixed(0)}s`,
        padding.left +
          (plotWidth * i) / 5,
        height - 27
      );
    }

    ctx.textAlign = "left";

    ctx.fillText(
      "Live ECG · relative amplitude",
      padding.left,
      4
    );

    /* Samples */

    const allSamples =
      samplesRef.current;

    const count = Math.min(
      allSamples.length,
      Math.max(
        1,
        Math.floor(
          (Number(sampleRate) ||
            DEFAULT_SAMPLE_RATE) *
            DISPLAY_SECONDS
        )
      )
    );

    if (count < 2) {
      ctx.fillStyle = "#8995aa";

      ctx.textAlign = "center";
      ctx.textBaseline = "middle";

      ctx.font =
        "13px Inter, system-ui, sans-serif";

      ctx.fillText(
        "Connecting to your ECG monitor…",
        width / 2,
        centerY
      );

      return;
    }

    const samples =
      allSamples.slice(-count);

    const sorted = [
      ...samples,
    ].sort((a, b) => a - b);

    const percentile = (q) => {
      const index = Math.min(
        sorted.length - 1,
        Math.max(
          0,
          Math.floor(
            (sorted.length - 1) * q
          )
        )
      );

      return sorted[index];
    };

    const center =
      (percentile(0.05) +
        percentile(0.95)) /
      2;

    const deviations = [
      ...samples,
    ]
      .map((value) =>
        Math.abs(value - center)
      )
      .sort((a, b) => a - b);

    const amplitude = Math.max(
      (deviations[
        Math.min(
          deviations.length - 1,
          Math.floor(
            deviations.length * 0.95
          )
        )
      ] || 1) * 1.35,
      1
    );

    const scale =
      (plotHeight * 0.46) /
      amplitude;

    /* Waveform */

    ctx.beginPath();

    for (
      let i = 0;
      i < samples.length;
      i++
    ) {
      const x =
        padding.left +
        (i /
          (samples.length - 1)) *
          plotWidth;

      const y = Math.max(
        padding.top + 2,
        Math.min(
          bottom - 2,
          centerY -
            (Number(samples[i]) -
              center) *
              scale
        )
      );

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }

    ctx.lineWidth = 1.7;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";

    ctx.strokeStyle = leadOff
      ? "#d58a82"
      : "#55a88d";

    ctx.stroke();

    /* Current point */

    const last =
      Number(
        samples[
          samples.length - 1
        ]
      );

    const lastY = Math.max(
      padding.top + 2,
      Math.min(
        bottom - 2,
        centerY -
          (last - center) *
            scale
      )
    );

    ctx.beginPath();

    ctx.arc(
      width - padding.right,
      lastY,
      3,
      0,
      Math.PI * 2
    );

    ctx.fillStyle = leadOff
      ? "#d58a82"
      : "#82c6a8";

    ctx.fill();
  }, [
    sampleRate,
    leadOff,
    samplesRef,
  ]);

  useEffect(() => {
    let stopped = false;

    const render = () => {
      if (stopped) return;

      draw();

      animationRef.current =
        requestAnimationFrame(
          render
        );
    };

    animationRef.current =
      requestAnimationFrame(
        render
      );

    const handleResize =
      () => draw();

    window.addEventListener(
      "resize",
      handleResize
    );

    const observer =
      new ResizeObserver(
        handleResize
      );

    if (containerRef.current) {
      observer.observe(
        containerRef.current
      );
    }

    return () => {
      stopped = true;

      cancelAnimationFrame(
        animationRef.current
      );

      window.removeEventListener(
        "resize",
        handleResize
      );

      observer.disconnect();
    };
  }, [draw]);

  return (
    <div
      ref={containerRef}
      className="ecg-waveform"
    >
      <canvas ref={canvasRef} />
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Main                                                                       */
/* -------------------------------------------------------------------------- */

export default function App() {
  const [
    connection,
    setConnection,
  ] = useState("connecting");

  const [
    sampleRate,
    setSampleRate,
  ] = useState(DEFAULT_SAMPLE_RATE);

  const [
    deviceId,
    setDeviceId,
  ] = useState(DEVICE_ID);

  const [
    sessionId,
    setSessionId,
  ] = useState(null);

  const [
    analysis,
    setAnalysis,
  ] = useState({});

  const [
    quality,
    setQuality,
  ] = useState({});

  const [
    leadOff,
    setLeadOff,
  ] = useState(false);

  const [
    recordingSeconds,
    setRecordingSeconds,
  ] = useState(0);

  const [
    latestPrediction,
    setLatestPrediction,
  ] = useState(null);

  const samplesRef =
    useRef([]);

  const wsRef =
    useRef(null);

  const reconnectRef =
    useRef(null);

  const recordingStartRef =
    useRef(null);

  /* ---------------------------------------------------------------------- */
  /* WebSocket                                                              */
  /* ---------------------------------------------------------------------- */

  useEffect(() => {
    let destroyed = false;

    const connect = () => {
      if (destroyed) return;

      setConnection(
        "connecting"
      );

      const socket =
        new WebSocket(WS_URL);

      wsRef.current = socket;

      socket.onopen = () => {
        if (destroyed) return;

        setConnection(
          "connected"
        );

        if (
          !recordingStartRef.current
        ) {
          recordingStartRef.current =
            Date.now();
        }
      };

      socket.onmessage = (
        event
      ) => {
        try {
          const message =
            JSON.parse(
              event.data
            );

          if (
            message.type ===
            "ecg"
          ) {
            const incoming =
              Array.isArray(
                message.samples
              )
                ? message.samples
                    .map(Number)
                    .filter(
                      Number.isFinite
                    )
                : [];

            const rate =
              Number(
                message.sampling_rate
              ) ||
              DEFAULT_SAMPLE_RATE;

            setSampleRate(rate);

            setDeviceId(
              message.device_id ||
                DEVICE_ID
            );

            setLeadOff(
              Boolean(
                message.lead_off
              )
            );

            if (
              message.session_id
            ) {
              setSessionId(
                message.session_id
              );
            }

            if (
              message.analysis
            ) {
              setAnalysis(
                message.analysis
              );
            }

            if (
              message.quality
            ) {
              setQuality(
                message.quality
              );
            }

            if (
              message.ml
            ) {
              const prediction =
                normalizePrediction(
                  message.ml
                );

              if (prediction) {
                setLatestPrediction(
                  prediction
                );
              }
            }

            if (
              incoming.length
            ) {
              const maxSamples =
                Math.max(
                  100,
                  Math.floor(
                    rate *
                      DISPLAY_SECONDS
                  )
                );

              samplesRef.current =
                samplesRef.current
                  .concat(incoming)
                  .slice(
                    -maxSamples
                  );
            }

            if (
              !recordingStartRef.current
            ) {
              recordingStartRef.current =
                Date.now();
            }
          }

          if (
            message.type ===
              "ml_prediction" ||
            message.type ===
              "prediction"
          ) {
            const prediction =
              normalizePrediction(
                message.prediction ||
                  message.ml ||
                  message
              );

            if (prediction) {
              setLatestPrediction(
                prediction
              );
            }
          }
        } catch (error) {
          console.error(
            "ECG WebSocket message error:",
            error
          );
        }
      };

      socket.onerror = () => {
        if (destroyed) return;

        setConnection("error");
      };

      socket.onclose = () => {
        if (destroyed) return;

        setConnection(
          "disconnected"
        );

        reconnectRef.current =
          setTimeout(
            connect,
            2000
          );
      };
    };

    connect();

    return () => {
      destroyed = true;

      clearTimeout(
        reconnectRef.current
      );

      wsRef.current?.close();
    };
  }, []);

  /* ---------------------------------------------------------------------- */
  /* Recording timer                                                        */
  /* ---------------------------------------------------------------------- */

  useEffect(() => {
    if (
      !recordingStartRef.current
    ) {
      return;
    }

    const timer =
      setInterval(() => {
        setRecordingSeconds(
          (Date.now() -
            recordingStartRef.current) /
            1000
        );
      }, 500);

    return () =>
      clearInterval(timer);
  }, [connection]);

  /* ---------------------------------------------------------------------- */
  /* Analysis values                                                        */
  /* ---------------------------------------------------------------------- */

  const heartRate = metric(
    analysis,
    "heart_rate_bpm",
    "heart_rate",
    "hr"
  );

  const rrInterval = metric(
    analysis,
    "rr_interval_ms",
    "rr_interval",
    "rr"
  );

  const peakCount = metric(
    analysis,
    "peak_count",
    "r_peak_count",
    "rpeaks"
  );

  const analysisStatus =
    metric(
      analysis,
      "status",
      "measurement_status"
    );

  const qualityData =
    normalizeQuality(
      quality
    );

  /* ---------------------------------------------------------------------- */
  /* Clinical status                                                        */
  /* ---------------------------------------------------------------------- */

  const clinicalStatus =
    useMemo(() => {
      const qualityText =
        String(
          qualityData.label || ""
        ).toLowerCase();

      const poorSignal =
        qualityText.includes(
          "poor"
        ) ||
        qualityText.includes(
          "bad"
        );

      if (leadOff) {
        return {
          type: "warning",
          title:
            "Check electrode connection",
          description:
            "The ECG electrodes are not making reliable contact.",
        };
      }

      if (poorSignal) {
        return {
          type: "warning",
          title:
            "Signal quality is low",
          description:
            "The current ECG recording may not be reliable enough for interpretation.",
        };
      }

      if (
        latestPrediction &&
        ["S", "V", "F"].includes(
          latestPrediction.code
        )
      ) {
        return {
          type: "warning",
          title:
            "Potential rhythm abnormality detected",
          description:
            "An abnormal beat classification was detected. Clinical evaluation is recommended.",
        };
      }

      if (
        latestPrediction?.code ===
        "N"
      ) {
        return {
          type: "success",
          title:
            "No significant rhythm abnormality detected",
          description:
            "The latest automated beat classification is consistent with a normal beat.",
        };
      }

      return {
        type: "neutral",
        title:
          "ECG monitoring in progress",
        description:
          "Continue the recording to obtain more ECG data for analysis.",
      };
    }, [
      qualityData,
      leadOff,
      latestPrediction,
    ]);

  const connectionText =
    connection === "connected"
      ? "Device connected"
      : connection ===
        "connecting"
      ? "Connecting"
      : "Connection unavailable";

  /* ---------------------------------------------------------------------- */
  /* Render                                                                 */
  /* ---------------------------------------------------------------------- */

  return (
    <div className="dashboard">
      {/* Header */}

      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">
            <span>♥</span>
            <i />
          </div>

          <div>
            <div className="brand-title">
              ECG Clinical Monitor
            </div>

            <div className="brand-subtitle">
              Real-time cardiac rhythm
              monitoring
            </div>
          </div>
        </div>

        <div className="header-status">
          <div
            className={`connection ${
              connection
            }`}
          >
            <StatusDot
              type={
                connection ===
                "connected"
                  ? "success"
                  : "warning"
              }
            />

            {connectionText}
          </div>

          <div className="header-divider" />

          <div className="device-info">
            <span>
              DEVICE
            </span>

            <strong>
              {deviceId}
            </strong>
          </div>
        </div>
      </header>

      <main className="main">
        {/* Clinical status */}

        <section
          className={`clinical-banner ${clinicalStatus.type}`}
        >
          <div className="clinical-icon">
            {clinicalStatus.type ===
            "success"
              ? "✓"
              : clinicalStatus.type ===
                "warning"
              ? "!"
              : "•"}
          </div>

          <div className="clinical-content">
            <div className="clinical-label">
              CURRENT ECG STATUS
            </div>

            <h1>
              {clinicalStatus.title}
            </h1>

            <p>
              {
                clinicalStatus.description
              }
            </p>
          </div>

          <div className="recording-indicator">
            <StatusDot
              type={
                connection ===
                "connected"
                  ? "success"
                  : "warning"
              }
            />

            <span>
              {connection ===
              "connected"
                ? "LIVE"
                : "OFFLINE"}
            </span>
          </div>
        </section>

        {/* Key metrics */}

        <section className="metrics">
          <MetricCard
            label="Heart Rate"
            value={formatNumber(
              heartRate
            )}
            unit="BPM"
            description="Current heart rate"
            accent
          />

          <MetricCard
            label="RR Interval"
            value={formatNumber(
              rrInterval
            )}
            unit="MS"
            description="Beat-to-beat interval"
          />

          <MetricCard
            label="Signal Quality"
            value={
              qualityData.score !==
              null
                ? formatNumber(
                    qualityData.score
                  )
                : "—"
            }
            unit={
              qualityData.score !==
              null
                ? "/100"
                : ""
            }
            description={
              qualityData.label
            }
          />

          <MetricCard
            label="Recording"
            value={formatDuration(
              recordingSeconds
            )}
            description="Current session"
          />
        </section>

        {/* ECG */}

        <section className="panel waveform-panel">
          <SectionHeader
            eyebrow="LIVE MONITORING"
            title="Live ECG"
            right={
              <div className="wave-status">
                <span>
                  {sampleRate} Hz
                </span>

                <span
                  className={
                    leadOff
                      ? "lead-warning"
                      : "lead-connected"
                  }
                >
                  <StatusDot
                    type={
                      leadOff
                        ? "warning"
                        : "success"
                    }
                  />

                  {leadOff
                    ? "Electrode check"
                    : "Signal active"}
                </span>
              </div>
            }
          />

          <ECGWaveform
            sampleRate={
              sampleRate
            }
            leadOff={leadOff}
            samplesRef={
              samplesRef
            }
          />

          <div className="waveform-footer">
            <div>
              <span>
                Signal window
              </span>
              <strong>
                {DISPLAY_SECONDS} sec
              </strong>
            </div>

            <div>
              <span>
                Beats identified
              </span>
              <strong>
                {peakCount ??
                  "—"}
              </strong>
            </div>

            <div>
              <span>
                Analysis
              </span>
              <strong>
                {analysisStatus ||
                  "Live"}
              </strong>
            </div>
          </div>
        </section>

        {/* Lower clinical information */}

        <section className="lower-grid">
          {/* Rhythm */}

          <section className="panel rhythm-panel">
            <SectionHeader
              eyebrow="RHYTHM"
              title="Heart Beat Analysis"
            />

            <div className="rhythm-main">
              <div className="heartbeat-icon">
                <span>♥</span>
              </div>

              <div>
                <div className="rhythm-number">
                  {formatNumber(
                    heartRate
                  )}

                  <span>
                    BPM
                  </span>
                </div>

                <div className="rhythm-label">
                  Current heart rate
                </div>
              </div>
            </div>

            <div className="rhythm-details">
              <div>
                <span>
                  RR interval
                </span>

                <strong>
                  {formatNumber(
                    rrInterval
                  )}{" "}
                  {rrInterval !==
                    null &&
                    "ms"}
                </strong>
              </div>

              <div>
                <span>
                  R-peaks
                </span>

                <strong>
                  {peakCount ??
                    "—"}
                </strong>
              </div>

              <div>
                <span>
                  Signal
                </span>

                <strong
                  className={
                    qualityData.label
                      .toLowerCase()
                      .includes("poor")
                      ? "text-warning"
                      : "text-success"
                  }
                >
                  {
                    qualityData.label
                  }
                </strong>
              </div>
            </div>
          </section>

          {/* ML */}

          <section className="panel smart-panel">
            <SectionHeader
              eyebrow="AI-ASSISTED RESULT"
              title="ECG analysis"
            />

            {latestPrediction ? (
              <>
                <div className="smart-result">
                  <div
                    className={`beat-code code-${latestPrediction.code}`}
                  >
                    {
                      latestPrediction.code
                    }
                  </div>

                  <div className="smart-result-text">
                    <div className="smart-result-title">
                      {
                        ML_CLASS_NAMES[
                          latestPrediction
                            .code
                        ] ||
                        "Unknown"
                      }
                    </div>

                    <div className="smart-result-subtitle">
                      Pattern detected in latest beat
                    </div>
                  </div>

                  <div className="confidence">
                    <span>
                      Confidence
                    </span>

                    <strong>
                      {latestPrediction.confidence !==
                      null
                        ? `${latestPrediction.confidence.toFixed(
                            1
                          )}%`
                        : "—"}
                    </strong>
                  </div>
                </div>

                <div className="classification-note">
                  Automated pattern classification for monitoring purposes. This is not a diagnosis.
                </div>
              </>
            ) : (
              <div className="smart-empty">
                <div className="smart-empty-icon">
                  ·
                </div>

                <div>
                  <strong>
                    Preparing your ECG analysis
                  </strong>

                  <span>
                    An AI-assisted pattern result will appear when enough ECG data is available.
                  </span>
                </div>
              </div>
            )}
          </section>
        </section>

        {/* Recording details */}

        <section className="panel recording-panel">
          <SectionHeader
            eyebrow="SYSTEM DIAGNOSTICS"
            title="Monitor details"
          />

          <div className="recording-grid">
            <div>
              <span>
                Device
              </span>

              <strong>
                {deviceId}
              </strong>
            </div>

            <div>
              <span>
                Sampling rate
              </span>

              <strong>
                {sampleRate} Hz
              </strong>
            </div>

            <div>
              <span>
                Configuration
              </span>

              <strong>
                Single-channel ECG
              </strong>
            </div>

            <div>
              <span>
                Session
              </span>

              <strong className="session-id">
                {sessionId
                  ? `${String(
                      sessionId
                    ).slice(
                      0,
                      14
                    )}…`
                  : "Waiting"}
              </strong>
            </div>
          </div>
        </section>

        {/* Disclaimer */}

        <div className="disclaimer">
          <strong>
            Clinical decision support:
          </strong>{" "}
          This system provides real-time ECG
          monitoring and automated beat
          classification. Results are intended
          to support clinical review and are
          not a standalone medical diagnosis.
        </div>
      </main>

      <footer>
        Live ECG monitoring · AI-assisted insights · Designed for clinical review
      </footer>
    </div>
  );
}