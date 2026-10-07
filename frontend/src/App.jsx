import { useEffect, useRef, useState } from "react";
import { assetUrl } from "./assetUrl.js";
import LandingPageButtons from "./components/LandingPageButtons.jsx";
import WebcamRulePage from "./pages/WebcamRulePage.jsx";
import WebcamSessionPage from "./pages/WebcamSessionPage.jsx";
import WebcamSessionEndedPage from "./pages/WebcamSessionEndedPage.jsx";
import {
  playMusic,
  playSfx,
  randomSessionTheme,
  SOUNDS,
  stopMusic,
  unmuteMusic,
} from "./sounds.js";

// GitHub Pages sets this so people can click through the screens with no backend.
const LAYOUT_PREVIEW = import.meta.env.VITE_LAYOUT_PREVIEW === "true";

export default function App() {
  const [view, setView] = useState("select");
  const [error, setError] = useState("");
  const [pickerState, setPickerState] = useState("start");
  const [timeLimit, setTimeLimit] = useState(null);
  const [starting, setStarting] = useState(false);
  const [streamKey, setStreamKey] = useState(0);
  const [summary, setSummary] = useState({});
  const [bestByMode, setBestByMode] = useState({});
  const [sourceKind, setSourceKind] = useState("webcam");
  const uploadRetryRef = useRef(null);
  const uploadRulesRef = useRef(null);

  useEffect(() => {
    if (view === "select" || view === "webcam-rules") {
      playMusic(SOUNDS.background, { muted: true });
    } else if (view === "webcam-session") {
      playMusic(randomSessionTheme());
    } else if (view === "webcam-ended") {
      stopMusic();
    }
  }, [view]);

  useEffect(() => {
    function onPointerDown() {
      unmuteMusic();
    }
    function onMouseOver(e) {
      const btn = e.target.closest("button.lp-btn, button.layout-preview-btn");
      if (!btn || btn.disabled) {
        return;
      }
      if (btn.contains(e.relatedTarget)) {
        return;
      }
      playSfx("buttonHover");
    }
    function onClick(e) {
      const btn = e.target.closest("button.lp-btn, button.layout-preview-btn");
      if (!btn || btn.disabled) {
        return;
      }
      playSfx("buttonHover");
    }
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("mouseover", onMouseOver);
    document.addEventListener("click", onClick);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("mouseover", onMouseOver);
      document.removeEventListener("click", onClick);
    };
  }, []);

  function resetHome() {
    setError("");
    setPickerState("start");
    setTimeLimit(null);
    setSummary({});
    setView("select");
  }

  async function goHome() {
    if (!LAYOUT_PREVIEW) {
      try {
        await fetch("/api/stop", { method: "POST" });
      } catch {
        /* backend may not be running */
      }
    }
    resetHome();
  }

  function goRetry() {
    if (sourceKind === "video") {
      uploadRetryRef.current?.click();
      return;
    }
    setError("");
    setSummary({});
    setPickerState("mode");
    setView("select");
  }

  function goMode() {
    setError("");
    setSummary({});
    setPickerState(sourceKind === "video" ? "source" : "mode");
    setView("select");
  }

  function goBack() {
    setError("");
    if (pickerState === "timed") {
      setPickerState("mode");
      return;
    }
    if (pickerState === "mode") {
      setPickerState("source");
      return;
    }
    if (pickerState === "source") {
      setPickerState("start");
    }
  }

  async function startWebcam() {
    setError("");
    setStarting(true);
    const body = new FormData();
    body.append("camera_id", "0");
    try {
      const res = await fetch("/api/start/webcam", { method: "POST", body });
      if (!res.ok) {
        setError("Could not start webcam.");
        return false;
      }
      setSourceKind("webcam");
      setStreamKey(Date.now());
      return true;
    } finally {
      setStarting(false);
    }
  }

  async function startVideo(file) {
    if (!file) {
      return;
    }
    setError("");
    setStarting(true);
    const body = new FormData();
    body.append("file", file);
    try {
      const res = await fetch("/api/start/video", { method: "POST", body });
      if (!res.ok) {
        setError("Could not start video.");
        return;
      }
      setSourceKind("video");
      setStreamKey(Date.now());
      setView("webcam-session");
    } finally {
      setStarting(false);
    }
  }

  if (view === "webcam-rules") {
    return (
      <>
        <input
          ref={uploadRulesRef}
          className="hidden-file"
          type="file"
          accept="video/*"
          onChange={(e) => {
            startVideo(e.target.files?.[0]);
            e.target.value = "";
          }}
        />
        <WebcamRulePage
          starting={starting}
          error={error}
          onBegin={async () => {
            if (sourceKind === "video") {
              uploadRulesRef.current?.click();
              return;
            }
            if (LAYOUT_PREVIEW) {
              setSourceKind("webcam");
              setView("webcam-session");
              return;
            }
            const ok = await startWebcam();
            if (ok) {
              setView("webcam-session");
            }
          }}
          onHome={goHome}
          onBack={() => {
            setError("");
            setView("select");
          }}
        />
      </>
    );
  }

  if (view === "webcam-session") {
    return (
      <WebcamSessionPage
        streamKey={streamKey}
        error={error}
        layoutPreview={LAYOUT_PREVIEW}
        source={sourceKind}
        timeLimit={timeLimit}
        onHome={goHome}
        onEnd={async (awards) => {
          if (!LAYOUT_PREVIEW) {
            try {
              const res = await fetch("/api/stop", { method: "POST" });
              if (res.ok) {
                const result = await res.json();
                const awardsList = awards ?? [];
                if (sourceKind === "webcam") {
                  const mode =
                    timeLimit == null ? "endless" : String(timeLimit);
                  const score = Number(result.score) || 0;
                  const prevBest = bestByMode[mode];
                  const isNewBest = prevBest != null && score > prevBest;
                  if (score > 0 && (prevBest == null || score > prevBest)) {
                    setBestByMode((prev) => ({ ...prev, [mode]: score }));
                  }
                  setSummary({
                    ...result,
                    awards: awardsList,
                    prevBest: prevBest ?? null,
                    isNewBest,
                  });
                } else {
                  setSummary({ ...result, awards: awardsList });
                }
              }
            } catch {
              /* backend may not be running */
            }
          }
          setView("webcam-ended");
        }}
      />
    );
  }

  if (view === "webcam-ended") {
    return (
      <>
        <input
          ref={uploadRetryRef}
          className="hidden-file"
          type="file"
          accept="video/*"
          onChange={(e) => {
            startVideo(e.target.files?.[0]);
            e.target.value = "";
          }}
        />
        <WebcamSessionEndedPage
          layoutPreview={LAYOUT_PREVIEW}
          summary={summary}
          onHome={goHome}
          onMode={goMode}
          onRetry={goRetry}
        />
      </>
    );
  }

  return (
    <div className="landing-page">
      <div className="landing-sizer">
        <div className="landing-stage">
          <div className="landing-bg" aria-hidden="true">
            <div className="landing-bg-photo">
              <img src={assetUrl("img/0c709cc31e3edd0d195da718fbed8f3d.jpg")} alt="" />
            </div>
            <div className="landing-bg-bar" />
          </div>
          {pickerState !== "start" ? (
            <div className="layout-preview-bar">
              <button
                className="layout-preview-btn"
                type="button"
                onClick={goBack}
              >
                BACK
              </button>
            </div>
          ) : null}
          <div className="landing-card">
            <div className="landing-title">
              <div className="landing-wordmark-wrap">
                <p className="landing-wordmark">ADIFORM</p>
              </div>
              <p className="landing-year">2099</p>
            </div>
            <LandingPageButtons
              state={pickerState}
              disabled={starting}
              onStart={() => {
                setPickerState("source");
              }}
              onWebcam={() => {
                setError("");
                setSourceKind("webcam");
                setPickerState("mode");
              }}
              onTimed={() => setPickerState("timed")}
              onEndless={() => {
                setTimeLimit(null);
                setView("webcam-rules");
              }}
              onTimedDuration={(seconds) => {
                setTimeLimit(seconds);
                setView("webcam-rules");
              }}
              onUpload={() => {
                setError("");
                setSourceKind("video");
                setView("webcam-rules");
              }}
            />
            {error ? <p className="error landing-error">{error}</p> : null}
          </div>
        </div>
      </div>
    </div>
  );
}
