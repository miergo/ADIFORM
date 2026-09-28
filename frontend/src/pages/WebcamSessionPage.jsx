import { useEffect, useRef, useState } from "react";
import SessionStage from "../components/SessionStage.jsx";
import { playSfx } from "../sounds.js";

const COUNTDOWN = [3, 2, 1, "go"];
const ABSENT_END_SEC = 3;
const PREVIEW_MODES = [
  ["calibrating", "CALIB"],
  ["complete", "DONE"],
  ["timer", "TIMER"],
  ["session", "HUD"],
];

export default function WebcamSessionPage({
  streamKey,
  error,
  layoutPreview = false,
  source = "webcam",
  timeLimit = null,
  onHome,
  onEnd,
}) {
  const [status, setStatus] = useState({});
  const [mode, setMode] = useState("calibrating");
  const [counterState, setCounterState] = useState(0);
  const [sessionStartedAt, setSessionStartedAt] = useState(null);
  const [now, setNow] = useState(() => Date.now());
  const [awardLog, setAwardLog] = useState([]);
  const [lastAward, setLastAward] = useState(null);
  const readyStarted = useRef(false);
  const goSent = useRef(false);
  const wasRunning = useRef(false);
  const ended = useRef(false);
  const prevAwardSeq = useRef(0);
  const awardLogRef = useRef([]);
  const activeStarted = useRef(false);
  const sawPersonRef = useRef(false);
  const absentSinceRef = useRef(null);
  const onEndRef = useRef(onEnd);
  onEndRef.current = onEnd;

  function finish() {
    if (ended.current) {
      return;
    }
    ended.current = true;
    onEndRef.current?.(awardLogRef.current);
  }

  useEffect(() => {
    if (layoutPreview) {
      return undefined;
    }
    const id = setInterval(async () => {
      try {
        const res = await fetch("/api/status");
        setStatus(await res.json());
      } catch {
        /* backend may not be running yet */
      }
    }, 250);
    return () => clearInterval(id);
  }, [layoutPreview]);

  useEffect(() => {
    if (layoutPreview) {
      return undefined;
    }
    const phase = status.phase;
    if (phase === "active") {
      setMode("session");
      setSessionStartedAt((t) => t ?? Date.now());
      return undefined;
    }
    if (phase === "ready" && !readyStarted.current) {
      readyStarted.current = true;
      setMode("timer");
    }
    if (
      phase === "calibrating" ||
      phase === "idle" ||
      phase == null
    ) {
      if (!readyStarted.current) {
        setMode("calibrating");
      }
    }
    return undefined;
  }, [layoutPreview, status.phase]);

  useEffect(() => {
    if (layoutPreview) {
      return;
    }
    if (status.phase === "active" && !activeStarted.current) {
      activeStarted.current = true;
      prevAwardSeq.current = status.award_seq ?? 0;
      sawPersonRef.current = false;
      absentSinceRef.current = null;
      setAwardLog([]);
      setLastAward(null);
      awardLogRef.current = [];
    }
  }, [layoutPreview, status.phase, status.award_seq]);

  useEffect(() => {
    if (layoutPreview) {
      return;
    }
    const seq = status.award_seq ?? 0;
    if (seq <= prevAwardSeq.current) {
      return;
    }
    const award = {
      mult: status.last_multiplier ?? 1,
      label: status.last_award_label ?? "",
      points: status.last_award_points ?? 0,
    };
    setLastAward(award);
    playSfx("multiplier");
    setAwardLog((log) => {
      const next = [...log, award];
      awardLogRef.current = next;
      return next;
    });
    prevAwardSeq.current = seq;
  }, [
    layoutPreview,
    status.award_seq,
    status.last_multiplier,
    status.last_award_label,
    status.last_award_points,
  ]);

  useEffect(() => {
    if (layoutPreview) {
      return;
    }
    if (status.running === true) {
      wasRunning.current = true;
    }
    if (wasRunning.current && status.running === false) {
      finish();
    }
  }, [layoutPreview, status.running]);

  useEffect(() => {
    if (mode !== "timer") {
      return undefined;
    }
    playSfx("counter");
    let i = 0;
    setCounterState(COUNTDOWN[0]);
    const id = setInterval(async () => {
      i += 1;
      if (i >= COUNTDOWN.length) {
        clearInterval(id);
        if (!layoutPreview && !goSent.current) {
          goSent.current = true;
          try {
            await fetch("/api/go", { method: "POST" });
          } catch {
            goSent.current = false;
          }
        }
        return;
      }
      setCounterState(COUNTDOWN[i]);
    }, 900);
    return () => clearInterval(id);
  }, [layoutPreview, mode]);

  useEffect(() => {
    if (mode !== "session") {
      return undefined;
    }
    const id = setInterval(() => setNow(Date.now()), 100);
    return () => clearInterval(id);
  }, [mode]);

  const elapsed =
    sessionStartedAt != null ? (now - sessionStartedAt) / 1000 : 0;

  useEffect(() => {
    if (layoutPreview || timeLimit == null || mode !== "session") {
      return;
    }
    if (elapsed >= timeLimit) {
      finish();
    }
  }, [layoutPreview, timeLimit, mode, elapsed]);

  useEffect(() => {
    if (layoutPreview || mode !== "session") {
      return;
    }
    if (status.detected) {
      sawPersonRef.current = true;
      absentSinceRef.current = null;
      return;
    }
    if (!sawPersonRef.current) {
      return;
    }
    if (absentSinceRef.current == null) {
      absentSinceRef.current = now;
      return;
    }
    if ((now - absentSinceRef.current) / 1000 >= ABSENT_END_SEC) {
      finish();
    }
  }, [layoutPreview, mode, status.detected, now]);

  const previewAward =
    layoutPreview && mode === "session"
      ? { mult: 1.85, label: "deep" }
      : lastAward;

  function pickPreviewMode(next) {
    setMode(next);
    if (next === "session") {
      const t = Date.now();
      setSessionStartedAt(t);
      setNow(t);
    }
  }

  return (
    <div className="landing-page">
      <div className="landing-sizer">
        <div className="landing-stage">
          <div className="landing-bg" aria-hidden="true">
            <div className="landing-bg-photo">
              <img src="/img/0c709cc31e3edd0d195da718fbed8f3d.jpg" alt="" />
            </div>
            <div className="landing-bg-bar" />
          </div>
          {layoutPreview ? (
            <div className="layout-preview-bar">
              {PREVIEW_MODES.map(([id, label]) => (
                <button
                  key={id}
                  className={
                    mode === id
                      ? "layout-preview-btn layout-preview-btn--on"
                      : "layout-preview-btn"
                  }
                  type="button"
                  onClick={() => pickPreviewMode(id)}
                >
                  {label}
                </button>
              ))}
              {onEnd ? (
                <button
                  className="layout-preview-btn"
                  type="button"
                  onClick={onEnd}
                >
                  END
                </button>
              ) : null}
              {onHome ? (
                <button
                  className="layout-preview-btn"
                  type="button"
                  onClick={onHome}
                >
                  HOME
                </button>
              ) : null}
            </div>
          ) : (
            <div className="layout-preview-bar">
              {onEnd ? (
                <button
                  className="layout-preview-btn"
                  type="button"
                  onClick={finish}
                >
                  END
                </button>
              ) : null}
              {onHome ? (
                <button
                  className="layout-preview-btn"
                  type="button"
                  onClick={onHome}
                >
                  HOME
                </button>
              ) : null}
            </div>
          )}
          <div className="rules-page-card">
            <SessionStage
              mode={mode}
              counterState={counterState}
              reps={layoutPreview ? 12 : (status.reps ?? 0)}
              time={elapsed}
              score={layoutPreview ? 1840 : (status.score ?? 0)}
              lastAward={previewAward}
              streamSrc={streamKey ? `/api/stream?t=${streamKey}` : null}
              source={source}
            />
          </div>
          <div className="rules-page-wordmark">
            <div className="landing-wordmark-wrap">
              <p className="landing-wordmark landing-wordmark--on-dark">
                ADIFORM
              </p>
            </div>
            <p className="landing-year landing-wordmark--on-dark">2099</p>
          </div>
          {error ? <p className="error rules-page-error">{error}</p> : null}
        </div>
      </div>
    </div>
  );
}
