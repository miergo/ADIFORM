import { useState } from "react";
import LandingPageButtons from "./components/LandingPageButtons.jsx";
import WebcamRulePage from "./pages/WebcamRulePage.jsx";
import WebcamSessionPage from "./pages/WebcamSessionPage.jsx";
import WebcamSessionEndedPage from "./pages/WebcamSessionEndedPage.jsx";

// Set true to browse screens without starting the backend / camera.
const LAYOUT_PREVIEW = false;

export default function App() {
  const [view, setView] = useState("select");
  const [error, setError] = useState("");
  const [sourcePick, setSourcePick] = useState(false);
  const [starting, setStarting] = useState(false);
  const [streamKey, setStreamKey] = useState(0);
  const [summary, setSummary] = useState({});
  const [sourceKind, setSourceKind] = useState("webcam");

  function resetHome() {
    setError("");
    setSourcePick(false);
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
      <WebcamRulePage
        starting={starting}
        error={error}
        onBegin={async () => {
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
      />
    );
  }

  if (view === "webcam-session") {
    return (
      <WebcamSessionPage
        streamKey={streamKey}
        error={error}
        layoutPreview={LAYOUT_PREVIEW}
        source={sourceKind}
        onHome={goHome}
        onEnd={async () => {
          if (!LAYOUT_PREVIEW) {
            try {
              const res = await fetch("/api/stop", { method: "POST" });
              if (res.ok) {
                setSummary(await res.json());
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
      <WebcamSessionEndedPage
        layoutPreview={LAYOUT_PREVIEW}
        summary={summary}
        onHome={goHome}
      />
    );
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
          <div className="landing-card">
            <div className="landing-title">
              <div className="landing-wordmark-wrap">
                <p className="landing-wordmark">ADIFORM</p>
              </div>
              <p className="landing-year">2099</p>
            </div>
            <LandingPageButtons
              state={sourcePick ? "source" : "start"}
              disabled={starting}
              onStart={() => setSourcePick(true)}
              onWebcam={() => {
                setError("");
                setSourceKind("webcam");
                setView("webcam-rules");
              }}
              onUploadFile={startVideo}
            />
            {error ? <p className="error landing-error">{error}</p> : null}
          </div>
        </div>
      </div>
    </div>
  );
}
