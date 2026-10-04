import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import "./App.css";

const API_URL = "http://10.132.32.10:9000";
const WS_URL = "ws://10.132.32.10:9000/ws/ecg";

const DEFAULT_SAMPLE_RATE = 200;
const DISPLAY_SECONDS = 10;
const MAX_SAMPLES = DEFAULT_SAMPLE_RATE * DISPLAY_SECONDS;

function formatNumber(value, digits = 0, fallback = "—") {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return fallback;
  }
  return Number(value).toFixed(digits);
}

function formatDate(value) {
  if (!value) return "—";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return String(value);
  return d.toLocaleString();
}

function formatDuration(seconds) {
  const total = Math.max(0, Math.floor(seconds || 0));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return h > 0
    ? `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`
    : `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function getMetric(obj, ...keys) {
  for (const key of keys) {
    if (obj && obj[key] !== undefined && obj[key] !== null) return obj[key];
  }
  return null;
}

function MetricCard({ label, value, unit, hint }) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="metric-value">
        {value}
        {unit && <span className="metric-unit">{unit}</span>}
      </div>
      {hint && <div className="metric-hint">{hint}</div>}
    </div>
  );
}

function SectionTitle({ eyebrow, title, right }) {
  return (
    <div className="section-title">
      <div>
        {eyebrow && <div className="eyebrow">{eyebrow}</div>}
        <h2>{title}</h2>
      </div>
      {right}
    </div>
  );
}

function Waveform({ samples, sampleRate, leadOff }) {
  const canvasRef = useRef(null);
  const wrapperRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrapper = wrapperRef.current;
    if (!canvas || !wrapper) return;

    const draw = () => {
      const rect = wrapper.getBoundingClientRect();
      const width = Math.max(320, Math.floor(rect.width));
      const height = Math.max(300, Math.floor(rect.height));
      const dpr = window.devicePixelRatio || 1;

      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;

      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);

      const pad = { left: 58, right: 18, top: 20, bottom: 34 };
      const plotW = width - pad.left - pad.right;
      const plotH = height - pad.top - pad.bottom;
      const midY = pad.top + plotH / 2;

      ctx.fillStyle = "#0b1118";
      ctx.fillRect(0, 0, width, height);

      // ECG-style grid.
      ctx.lineWidth = 1;
      for (
        let x = pad.left;
        x <= width - pad.right;
        x += Math.max(1, plotW / 50)
      ) {
        ctx.strokeStyle = "rgba(92, 116, 132, 0.10)";
        ctx.beginPath();
        ctx.moveTo(x, pad.top);
        ctx.lineTo(x, height - pad.bottom);
        ctx.stroke();
      }
      for (
        let y = pad.top;
        y <= height - pad.bottom;
        y += Math.max(1, plotH / 16)
      ) {
        ctx.strokeStyle = "rgba(92, 116, 132, 0.10)";
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(width - pad.right, y);
        ctx.stroke();
      }

      // Main axes.
      ctx.strokeStyle = "rgba(159, 178, 191, 0.32)";
      ctx.beginPath();
      ctx.moveTo(pad.left, pad.top);
      ctx.lineTo(pad.left, height - pad.bottom);
      ctx.lineTo(width - pad.right, height - pad.bottom);
      ctx.stroke();

      ctx.fillStyle = "#8293a0";
      ctx.font = "11px Inter, system-ui, sans-serif";
      ctx.textAlign = "center";

      const duration = DISPLAY_SECONDS;
      for (let i = 0; i <= 5; i++) {
        const x = pad.left + (plotW * i) / 5;
        ctx.fillText(`${((duration * i) / 5).toFixed(0)}s`, x, height - 12);
      }

      ctx.save();
      ctx.translate(15, midY);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("Filtered ECG amplitude (ADC-relative)", 0, 0);
      ctx.restore();

      ctx.textAlign = "left";
      ctx.fillText("0", 34, midY + 4);
      ctx.fillText("time", width - 48, height - 12);

      if (!samples.length) {
        ctx.fillStyle = "#6f808c";
        ctx.textAlign = "center";
        ctx.font = "13px Inter, system-ui, sans-serif";
        ctx.fillText("Waiting for ECG samples…", width / 2, midY);
        return;
      }

      const visible = samples.slice(
        -Math.max(1, Math.floor(sampleRate * DISPLAY_SECONDS)),
      );
      const min = Math.min(...visible);
      const max = Math.max(...visible);
      const center = (min + max) / 2;
      const span = Math.max(1, max - min);
      const scale = (plotH * 0.4) / span;

      ctx.beginPath();
      visible.forEach((sample, index) => {
        const x = pad.left + (index / Math.max(1, visible.length - 1)) * plotW;
        const y = midY - (sample - center) * scale;
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });

      ctx.lineWidth = 1.8;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.strokeStyle = leadOff ? "#8a6262" : "#73c69b";
      ctx.stroke();
    };

    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(wrapper);
    window.addEventListener("resize", draw);

    return () => {
      observer.disconnect();
      window.removeEventListener("resize", draw);
    };
  }, [samples, sampleRate, leadOff]);

  return (
    <div ref={wrapperRef} className="waveform-shell">
      <canvas ref={canvasRef} />
    </div>
  );
}

export default function App() {
  const [connection, setConnection] = useState("connecting");
  const [samples, setSamples] = useState([]);
  const [sampleRate, setSampleRate] = useState(DEFAULT_SAMPLE_RATE);
  const [deviceId, setDeviceId] = useState("ecg_esp8266_01");
  const [sessionId, setSessionId] = useState(null);
  const [analysis, setAnalysis] = useState({});
  const [quality, setQuality] = useState({});
  const [leadOff, setLeadOff] = useState(false);

  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const recordingStartRef = useRef(null);
  const wsRef = useRef(null);
  const reconnectTimerRef = useRef(null);

  const [sessions, setSessions] = useState([]);
  const [selectedSession, setSelectedSession] = useState(null);
  const [selectedMeasurements, setSelectedMeasurements] = useState([]);
  const [loadingSessions, setLoadingSessions] = useState(false);
  const [loadingMeasurements, setLoadingMeasurements] = useState(false);
  const [dbInfo, setDbInfo] = useState(null);

  const fetchSessions = useCallback(async () => {
    try {
      setLoadingSessions(true);
      const response = await fetch(`${API_URL}/api/sessions`);
      if (!response.ok)
        throw new Error(`Sessions request failed: ${response.status}`);
      const data = await response.json();
      setSessions(Array.isArray(data) ? data : data.sessions || []);
    } catch (error) {
      console.error("Failed to load sessions:", error);
    } finally {
      setLoadingSessions(false);
    }
  }, []);

  const fetchDatabaseInfo = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/api/database`);
      if (!response.ok) return;
      setDbInfo(await response.json());
    } catch (error) {
      console.error("Failed to load database info:", error);
    }
  }, []);

  const fetchMeasurements = useCallback(async (id) => {
    if (!id) return;
    try {
      setLoadingMeasurements(true);
      const response = await fetch(
        `${API_URL}/api/session/${encodeURIComponent(id)}/measurements`,
      );
      if (!response.ok)
        throw new Error(`Measurements request failed: ${response.status}`);
      const data = await response.json();
      setSelectedMeasurements(
        Array.isArray(data) ? data : data.measurements || [],
      );
    } catch (error) {
      console.error("Failed to load measurements:", error);
      setSelectedMeasurements([]);
    } finally {
      setLoadingMeasurements(false);
    }
  }, []);

  const openSession = useCallback(
    async (session) => {
      setSelectedSession(session);
      await fetchMeasurements(session.session_id);
    },
    [fetchMeasurements],
  );

  useEffect(() => {
    fetchSessions();
    fetchDatabaseInfo();
  }, [fetchSessions, fetchDatabaseInfo]);

  // Refresh persistent history while a recording is active.
  useEffect(() => {
    if (!sessionId) return;
    const timer = setInterval(() => {
      fetchSessions();
      fetchDatabaseInfo();
      if (selectedSession?.session_id === sessionId) {
        fetchMeasurements(sessionId);
      }
    }, 5000);
    return () => clearInterval(timer);
  }, [
    sessionId,
    selectedSession,
    fetchSessions,
    fetchDatabaseInfo,
    fetchMeasurements,
  ]);

  useEffect(() => {
    if (connection !== "connected" || !recordingStartRef.current) return;
    const timer = setInterval(() => {
      setRecordingSeconds((Date.now() - recordingStartRef.current) / 1000);
    }, 1000);
    return () => clearInterval(timer);
  }, [connection]);

  useEffect(() => {
    let disposed = false;

    const connect = () => {
      if (disposed) return;

      setConnection("connecting");
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        if (disposed) return;
        setConnection("connected");
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          if (message.type !== "ecg") return;

          const incoming = Array.isArray(message.samples)
            ? message.samples.map(Number).filter(Number.isFinite)
            : [];

          const rate = Number(message.sampling_rate) || DEFAULT_SAMPLE_RATE;
          setSampleRate(rate);
          setDeviceId(message.device_id || "ecg_esp8266_01");
          setLeadOff(Boolean(message.lead_off));

          if (message.session_id) {
            setSessionId((previous) => {
              if (previous !== message.session_id) {
                recordingStartRef.current =
                  recordingStartRef.current || Date.now();
              }
              return message.session_id;
            });
          }

          if (message.analysis) {
            setAnalysis(message.analysis || {});
          }
          if (message.quality) {
            setQuality(message.quality || {});
          }

          setSamples((previous) => {
            const merged = previous.concat(incoming);
            const max = Math.max(1000, Math.floor(rate * DISPLAY_SECONDS));
            return merged.slice(-max);
          });

          if (!recordingStartRef.current) {
            recordingStartRef.current = Date.now();
          }
        } catch (error) {
          console.error("Invalid WebSocket message:", error);
        }
      };

      ws.onerror = () => {
        if (!disposed) setConnection("error");
      };

      ws.onclose = () => {
        if (disposed) return;
        setConnection("disconnected");
        reconnectTimerRef.current = setTimeout(connect, 2000);
      };
    };

    connect();

    return () => {
      disposed = true;
      clearTimeout(reconnectTimerRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const normalizedAnalysis = useMemo(() => {
    const a = analysis || {};
    return {
      heartRate: getMetric(a, "heart_rate_bpm", "heart_rate", "hr"),
      rr: getMetric(a, "rr_interval_ms", "rr"),
      pDuration: getMetric(a, "p_duration_ms", "p_duration"),
      pr: getMetric(a, "pr_interval_ms", "pr_interval_ms", "pr"),
      qrs: getMetric(a, "qrs_duration_ms", "qrs_duration"),
      qt: getMetric(a, "qt_interval_ms", "qt_interval"),
      qtc: getMetric(a, "qtc_ms", "qtc"),
      confidence: getMetric(a, "confidence"),
      status: getMetric(a, "status", "measurement_status"),
    };
  }, [analysis]);

  const qualityScore = getMetric(quality, "score");
  const qualityLabel = getMetric(quality, "label") || "Waiting";
  const packets = dbInfo?.packets ?? dbInfo?.packets_received ?? "—";
  const storedSamples = dbInfo?.samples ?? dbInfo?.samples_received ?? "—";

  const connectionLabel =
    {
      connected: "Connected",
      connecting: "Connecting",
      disconnected: "Disconnected",
      error: "Connection error",
    }[connection] || connection;

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <div className="brand-kicker">ECG / IoMT MONITOR</div>
          <h1>Single-Channel ECG Dashboard</h1>
          <p>
            3-electrode configuration · Lead II monitoring · FastAPI persistence
          </p>
        </div>
        <div className={`connection-pill ${connection}`}>
          <span className="status-dot" />
          {connectionLabel}
        </div>
      </header>

      <main className="content">
        <section className="overview-grid">
          <MetricCard
            label="Heart Rate"
            value={formatNumber(normalizedAnalysis.heartRate)}
            unit="bpm"
            hint="R-peak based estimate"
          />
          <MetricCard
            label="RR Interval"
            value={formatNumber(normalizedAnalysis.rr)}
            unit="ms"
            hint="Beat-to-beat interval"
          />
          <MetricCard
            label="Signal Quality"
            value={qualityScore !== null ? formatNumber(qualityScore, 0) : "—"}
            unit={qualityScore !== null ? "/100" : ""}
            hint={String(qualityLabel)}
          />
          <MetricCard
            label="Recording Time"
            value={formatDuration(recordingSeconds)}
            hint="Current live session"
          />
        </section>

        <section className="panel waveform-panel">
          <SectionTitle
            eyebrow="LIVE SIGNAL"
            title="ECG Waveform"
            right={
              <div className="wave-meta">
                <span>Lead II</span>
                <span>{sampleRate} Hz</span>
                <span>{leadOff ? "Lead-off detected" : "Lead connected"}</span>
              </div>
            }
          />
          <Waveform
            samples={samples}
            sampleRate={sampleRate}
            leadOff={leadOff}
          />
          <div className="wave-footer">
            <span>Display window: {DISPLAY_SECONDS}s</span>
            <span>Samples in view: {samples.length.toLocaleString()}</span>
            <span>Session: {sessionId || "Waiting"}</span>
          </div>
        </section>

        <div className="two-column">
          <section className="panel">
            <SectionTitle
              eyebrow="ECG INTERVALS"
              title="Essential 3-Electrode Parameters"
            />
            <div className="parameter-grid">
              <MetricCard
                label="P Duration"
                value={formatNumber(normalizedAnalysis.pDuration)}
                unit="ms"
              />
              <MetricCard
                label="PR / PQ"
                value={formatNumber(normalizedAnalysis.pr)}
                unit="ms"
              />
              <MetricCard
                label="QRS Duration"
                value={formatNumber(normalizedAnalysis.qrs)}
                unit="ms"
              />
              <MetricCard
                label="QT Interval"
                value={formatNumber(normalizedAnalysis.qt)}
                unit="ms"
              />
              <MetricCard
                label="QTc"
                value={formatNumber(normalizedAnalysis.qtc)}
                unit="ms"
              />
              <MetricCard
                label="Confidence"
                value={
                  normalizedAnalysis.confidence !== null
                    ? formatNumber(normalizedAnalysis.confidence, 2)
                    : "—"
                }
              />
            </div>
            <div className="research-note">
              These interval values are algorithmic research estimates from the
              available single-channel waveform. They are not clinical
              measurements or a diagnosis.
            </div>
          </section>

          <section className="panel">
            <SectionTitle
              eyebrow="SIGNAL / DEVICE"
              title="Acquisition Status"
            />
            <div className="status-list">
              <div>
                <span>Device</span>
                <strong>{deviceId}</strong>
              </div>
              <div>
                <span>Configuration</span>
                <strong>3-electrode · Lead II</strong>
              </div>
              <div>
                <span>Sampling rate</span>
                <strong>{sampleRate} Hz</strong>
              </div>
              <div>
                <span>Lead status</span>
                <strong className={leadOff ? "bad" : "good"}>
                  {leadOff ? "Lead-off" : "Connected"}
                </strong>
              </div>
              <div>
                <span>Analysis status</span>
                <strong>{normalizedAnalysis.status || "Streaming"}</strong>
              </div>
              <div>
                <span>Quality label</span>
                <strong>{qualityLabel}</strong>
              </div>
              <div>
                <span>Baseline wander</span>
                <strong>
                  {formatNumber(getMetric(quality, "baseline_wander"), 2)}
                </strong>
              </div>
              <div>
                <span>Noise RMS</span>
                <strong>
                  {formatNumber(getMetric(quality, "noise_rms"), 2)}
                </strong>
              </div>
              <div>
                <span>Clipping ratio</span>
                <strong>
                  {formatNumber(getMetric(quality, "clipping_ratio"), 4)}
                </strong>
              </div>
            </div>
          </section>
        </div>

        <section className="panel">
          <SectionTitle
            eyebrow="PERSISTENCE"
            title="Saved ECG Sessions"
            right={
              <button
                className="secondary-btn"
                onClick={() => {
                  fetchSessions();
                  fetchDatabaseInfo();
                }}
              >
                Refresh
              </button>
            }
          />

          <div className="database-strip">
            <span>
              Stored sessions:{" "}
              <strong>{dbInfo?.sessions ?? sessions.length}</strong>
            </span>
            <span>
              Stored measurements:{" "}
              <strong>{dbInfo?.measurements ?? "—"}</strong>
            </span>
            <span>
              Stored quality records:{" "}
              <strong>{dbInfo?.quality_records ?? "—"}</strong>
            </span>
            <span className="db-path">
              {dbInfo?.database_path || "SQLite database"}
            </span>
          </div>

          {loadingSessions ? (
            <div className="empty-state">Loading saved sessions…</div>
          ) : sessions.length === 0 ? (
            <div className="empty-state">
              No persisted sessions yet. Start the ECG stream and records will
              appear here.
            </div>
          ) : (
            <div className="session-table-wrap">
              <table className="session-table">
                <thead>
                  <tr>
                    <th>Session</th>
                    <th>Device</th>
                    <th>Started</th>
                    <th>Last packet</th>
                    <th>Rate</th>
                    <th>Samples</th>
                    <th>Packets</th>
                    <th>Status</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {sessions.map((session) => (
                    <tr key={session.session_id}>
                      <td className="mono">
                        {String(session.session_id).slice(0, 16)}…
                      </td>
                      <td>{session.device_id || "—"}</td>
                      <td>{formatDate(session.started_at)}</td>
                      <td>{formatDate(session.last_packet_at)}</td>
                      <td>{session.sampling_rate || "—"} Hz</td>
                      <td>
                        {Number(session.samples_received || 0).toLocaleString()}
                      </td>
                      <td>
                        {Number(session.packets_received || 0).toLocaleString()}
                      </td>
                      <td>
                        <span
                          className={`session-status ${session.status || "active"}`}
                        >
                          {session.status || "active"}
                        </span>
                      </td>
                      <td>
                        <button
                          className="view-btn"
                          onClick={() => openSession(session)}
                        >
                          View
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        {selectedSession && (
          <section className="panel detail-panel">
            <SectionTitle
              eyebrow="PERSISTED RECORD"
              title={`Session ${String(selectedSession.session_id).slice(0, 20)}`}
              right={
                <button
                  className="secondary-btn"
                  onClick={() => setSelectedSession(null)}
                >
                  Close
                </button>
              }
            />
            <div className="detail-summary">
              <div>
                <span>Device</span>
                <strong>{selectedSession.device_id}</strong>
              </div>
              <div>
                <span>Started</span>
                <strong>{formatDate(selectedSession.started_at)}</strong>
              </div>
              <div>
                <span>Last packet</span>
                <strong>{formatDate(selectedSession.last_packet_at)}</strong>
              </div>
              <div>
                <span>Sampling</span>
                <strong>{selectedSession.sampling_rate} Hz</strong>
              </div>
              <div>
                <span>Samples</span>
                <strong>
                  {Number(
                    selectedSession.samples_received || 0,
                  ).toLocaleString()}
                </strong>
              </div>
              <div>
                <span>Packets</span>
                <strong>
                  {Number(
                    selectedSession.packets_received || 0,
                  ).toLocaleString()}
                </strong>
              </div>
            </div>

            <h3 className="subheading">Stored ECG Measurements</h3>
            {loadingMeasurements ? (
              <div className="empty-state">Loading measurement history…</div>
            ) : selectedMeasurements.length === 0 ? (
              <div className="empty-state">
                No measurement rows were stored for this session yet.
              </div>
            ) : (
              <div className="measurement-table-wrap">
                <table className="measurement-table">
                  <thead>
                    <tr>
                      <th>Recorded</th>
                      <th>HR</th>
                      <th>RR</th>
                      <th>P</th>
                      <th>PR/PQ</th>
                      <th>QRS</th>
                      <th>QT</th>
                      <th>QTc</th>
                      <th>Confidence</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {selectedMeasurements
                      .slice()
                      .reverse()
                      .map((m, index) => (
                        <tr key={m.id || `${m.recorded_at}-${index}`}>
                          <td>{formatDate(m.recorded_at)}</td>
                          <td>
                            {formatNumber(
                              getMetric(m, "heart_rate_bpm", "heart_rate"),
                            )}
                          </td>
                          <td>
                            {formatNumber(getMetric(m, "rr_interval_ms", "rr"))}
                          </td>
                          <td>
                            {formatNumber(
                              getMetric(m, "p_duration_ms", "p_duration"),
                            )}
                          </td>
                          <td>
                            {formatNumber(getMetric(m, "pr_interval_ms", "pr"))}
                          </td>
                          <td>
                            {formatNumber(
                              getMetric(m, "qrs_duration_ms", "qrs"),
                            )}
                          </td>
                          <td>
                            {formatNumber(getMetric(m, "qt_interval_ms", "qt"))}
                          </td>
                          <td>{formatNumber(getMetric(m, "qtc_ms", "qtc"))}</td>
                          <td>{formatNumber(getMetric(m, "confidence"), 2)}</td>
                          <td>{m.measurement_status || m.status || "—"}</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        )}

        <section className="panel limitations">
          <SectionTitle
            eyebrow="SCOPE"
            title="What this 3-electrode system reports"
          />
          <div className="scope-grid">
            <div>
              <h3>Included</h3>
              <p>
                HR, RR, P duration, PR/PQ, QRS, QT, QTc, P/QRS/T waveform
                analysis when detectable, lead-off state, signal quality,
                sampling rate and recording/session statistics.
              </p>
            </div>
            <div>
              <h3>Not fabricated</h3>
              <p>
                No 12-lead axis values, RV5/SV1, or 12-lead interpretation are
                shown because the current AD8232 setup provides a single ECG
                channel.
              </p>
            </div>
          </div>
        </section>
      </main>

      <footer>
        AD8232 → ESP8266 → FastAPI :9000 → WebSocket → React ECG Monitor →
        SQLite persistence
      </footer>
    </div>
  );
}
