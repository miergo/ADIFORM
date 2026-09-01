/*
 * First-version frontend snapshot (pre-ADIFORM Figma UI).
 * Archived for documenting the evolution of the software.
 * Not imported by the running app.
 *
 * Restored from git HEAD: frontend/src/App.jsx
 * Screens: select (WebCam / Upload Video), live workout + SVG chart, analytics/replay.
 */
import { useEffect, useRef, useState } from "react";

const EMPTY = {
  reps: 0,
  stage: "-",
  side: null,
  status: "idle",
  elbow: null,
  eye_height: null,
  speed: null,
  series: [],
  running: false,
  error: null,
  touch_y0: false,
  posture_angle: null,
  posture_status: null,
};

function WorkoutChart({ series }) {
  if (!series || series.length < 2) {
    return <p className="hint">Not enough samples for a chart yet.</p>;
  }

  const w = 640;
  const h = 220;
  const pad = { top: 16, right: 48, bottom: 28, left: 48 };
  const iw = w - pad.left - pad.right;
  const ih = h - pad.top - pad.bottom;

  const t0 = series[0].t;
  const t1 = series[series.length - 1].t;
  const tSpan = Math.max(t1 - t0, 1e-6);

  const speeds = series.map((p) => p.speed).filter((v) => v != null);
  const elbows = series.map((p) => p.elbow).filter((v) => v != null);
  const reps = series.map((p) => p.reps ?? 0);
  const angles = series.map((p) => p.posture_angle).filter((v) => v != null);

  const speedMin = Math.min(0, ...speeds, -0.1);
  const speedMax = Math.max(0.1, ...speeds);
  const elbowMin = 0;
  const elbowMax = Math.max(180, ...elbows, 1);
  const repsMax = Math.max(1, ...reps);
  const angleMin = angles.length ? Math.min(150, ...angles) : 150;
  const angleMax = angles.length ? Math.max(180, ...angles) : 180;

  const xOf = (t) => pad.left + ((t - t0) / tSpan) * iw;
  const ySpeed = (v) =>
    pad.top + ih - ((v - speedMin) / (speedMax - speedMin || 1)) * ih;
  const yElbow = (v) =>
    pad.top + ih - ((v - elbowMin) / (elbowMax - elbowMin || 1)) * ih;
  const yReps = (v) => pad.top + ih - (v / repsMax) * ih;
  const yAngle = (v) =>
    pad.top + ih - ((v - angleMin) / (angleMax - angleMin || 1)) * ih;

  function polyline(points) {
    return points.map(([x, y]) => `${x},${y}`).join(" ");
  }

  const speedPts = series
    .filter((p) => p.speed != null)
    .map((p) => [xOf(p.t), ySpeed(p.speed)]);
  const elbowPts = series
    .filter((p) => p.elbow != null)
    .map((p) => [xOf(p.t), yElbow(p.elbow)]);
  const posturePts = series
    .filter((p) => p.posture_angle != null)
    .map((p) => [xOf(p.t), yAngle(p.posture_angle)]);

  // Step line for reps
  const repPts = [];
  series.forEach((p, i) => {
    const x = xOf(p.t);
    const y = yReps(p.reps ?? 0);
    if (i === 0) {
      repPts.push([x, y]);
    } else {
      const prevY = yReps(series[i - 1].reps ?? 0);
      repPts.push([x, prevY], [x, y]);
    }
  });

  return (
    <div className="chart-wrap">
      <div className="chart-legend">
        <span className="leg speed">Speed (m/s)</span>
        <span className="leg elbow">Elbow (°)</span>
        <span className="leg reps">Reps</span>
        <span className="leg posture">Posture (°)</span>
      </div>
      <svg className="workout-chart" viewBox={`0 0 ${w} ${h}`} role="img">
        <line
          x1={pad.left}
          y1={pad.top}
          x2={pad.left}
          y2={pad.top + ih}
          stroke="#888"
        />
        <line
          x1={pad.left + iw}
          y1={pad.top}
          x2={pad.left + iw}
          y2={pad.top + ih}
          stroke="#888"
        />
        <line
          x1={pad.left}
          y1={pad.top + ih}
          x2={pad.left + iw}
          y2={pad.top + ih}
          stroke="#888"
        />
        <text x={pad.left} y={12} className="axis-label" fill="#2a7">
          {speedMax.toFixed(2)}
        </text>
        <text x={pad.left} y={pad.top + ih} className="axis-label" fill="#2a7">
          {speedMin.toFixed(2)}
        </text>
        <text x={pad.left + iw + 4} y={12} className="axis-label" fill="#a2a">
          {elbowMax.toFixed(0)}°
        </text>
        <text
          x={pad.left + iw + 4}
          y={pad.top + ih}
          className="axis-label"
          fill="#a2a"
        >
          0°
        </text>
        {speedPts.length > 1 ? (
          <polyline
            fill="none"
            stroke="#2a7a3a"
            strokeWidth="2"
            points={polyline(speedPts)}
          />
        ) : null}
        {elbowPts.length > 1 ? (
          <polyline
            fill="none"
            stroke="#9b2d9b"
            strokeWidth="2"
            points={polyline(elbowPts)}
          />
        ) : null}
        {repPts.length > 1 ? (
          <polyline
            fill="none"
            stroke="#c45c1a"
            strokeWidth="2"
            points={polyline(repPts)}
          />
        ) : null}
        {posturePts.length > 1 ? (
          <polyline
            fill="none"
            stroke="#1a6fc4"
            strokeWidth="2"
            points={polyline(posturePts)}
          />
        ) : null}
        <text x={pad.left} y={h - 6} className="axis-label" fill="#555">
          0s
        </text>
        <text
          x={pad.left + iw - 24}
          y={h - 6}
          className="axis-label"
          fill="#555"
        >
          {tSpan.toFixed(1)}s
        </text>
      </svg>
    </div>
  );
}

export default function App() {
  const [view, setView] = useState("select");
  const [status, setStatus] = useState(EMPTY);
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState("");
  const [streamKey, setStreamKey] = useState(0);
  const [recordingKey, setRecordingKey] = useState(0);
  const [mjpegReplay, setMjpegReplay] = useState(false);
  const fileRef = useRef(null);
  const replayRef = useRef(null);

  useEffect(() => {
    if (view !== "workout") {
      return undefined;
    }
    const id = setInterval(async () => {
      try {
        const res = await fetch("/api/status");
        setStatus(await res.json());
      } catch {
        setError("Cannot reach the backend. Start it with uvicorn.");
      }
    }, 250);
    return () => clearInterval(id);
  }, [view]);

  async function startWebcam() {
    setError("");
    const body = new FormData();
    body.append("camera_id", "0");
    const res = await fetch("/api/start/webcam", { method: "POST", body });
    if (!res.ok) {
      setError("Could not start webcam.");
      return;
    }
    setStreamKey(Date.now());
    setView("workout");
  }

  async function startVideo(file) {
    if (!file) {
      return;
    }
    setError("");
    const body = new FormData();
    body.append("file", file);
    const res = await fetch("/api/start/video", { method: "POST", body });
    if (!res.ok) {
      setError("Could not start video.");
      return;
    }
    setStreamKey(Date.now());
    setView("workout");
  }

  async function finishWorkout() {
    const res = await fetch("/api/stop", { method: "POST" });
    const data = await res.json();
    setSummary(data);
    setRecordingKey(Date.now());
    setMjpegReplay(false);
    setView("analytics");
  }

  function replayVideo() {
    const video = replayRef.current;
    if (video && !mjpegReplay) {
      video.currentTime = 0;
      video.play();
      return;
    }
    setRecordingKey(Date.now());
  }

  if (view === "select") {
    return (
      <div className="page">
        <div className="select-card">
          <h1>Select Option</h1>
          <div className="select-actions">
            <button className="outline-btn" type="button" onClick={startWebcam}>
              WebCam
            </button>
            <button
              className="outline-btn"
              type="button"
              onClick={() => fileRef.current?.click()}
            >
              Upload Video
            </button>
            <input
              ref={fileRef}
              className="hidden-file"
              type="file"
              accept="video/*"
              onChange={(e) => startVideo(e.target.files?.[0])}
            />
          </div>
        </div>
        {error ? <p className="error">{error}</p> : null}
      </div>
    );
  }

  if (view === "analytics") {
    const data = summary || status;
    const recordingUrl = `/api/recording?t=${recordingKey}`;
    return (
      <div className="page">
        <div className="analytics-page">
          <div className="analytics-btn">Show Rep Analytics</div>
          <div className="stats">
            <span>Reps: {data.reps ?? 0}</span>
            <span>Side: {data.side || "-"}</span>
            <span>Good frames: {data.good_frames ?? 0}</span>
            <span>Issue frames: {data.issue_frames ?? 0}</span>
          </div>
          <WorkoutChart series={data.series} />
          <div className="analytics-video">
            {data.has_recording ? (
              mjpegReplay ? (
                <img src={`/api/replay/stream?t=${recordingKey}`} alt="Replay" />
              ) : (
                <video
                  ref={replayRef}
                  src={recordingUrl}
                  controls
                  playsInline
                  onError={() => setMjpegReplay(true)}
                />
              )
            ) : (
              <p>No recording available.</p>
            )}
          </div>
          <div className="select-actions">
            <button className="outline-btn" type="button" onClick={replayVideo}>
              Replay
            </button>
            {data.source === "webcam" ? (
              <a className="outline-btn" href="/api/recording/download" download>
                Save video
              </a>
            ) : null}
          </div>
          {data.source === "webcam" ? (
            <p className="hint">
              Webcam clips are kept temporarily. Replay without saving, or click Save video to download.
            </p>
          ) : null}
          <button className="back-link" type="button" onClick={() => setView("select")}>
            Back to select
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="page">
      <div className="workout-page">
        <div className="video-stage">
          <img src={`/api/stream?t=${streamKey}`} alt="Workout video" />
          <div className="overlay left">Rep Count: {status.reps}</div>
          <div className="overlay right">
            <h2>Analysis Details</h2>
            <p>Stage: {status.stage}</p>
            <p>Side: {status.side || "detecting..."}</p>
            <p>{status.status}</p>
            {status.elbow != null ? <p>Elbow: {status.elbow}°</p> : null}
            {status.eye_height != null ? (
              <p>Eye height: {status.eye_height}</p>
            ) : null}
            {status.speed != null ? <p>Speed: {status.speed} m/s</p> : null}
            {status.error ? <p className="error">{status.error}</p> : null}
          </div>
        </div>
        <WorkoutChart series={status.series} />
        <button className="finish-btn" type="button" onClick={finishWorkout}>
          Finish Workout
        </button>
      </div>
    </div>
  );
}
