import { useState } from "react";
import WebCamSessionEnded from "../components/WebCamSessionEnded.jsx";

export default function WebcamSessionEndedPage({
  layoutPreview = false,
  summary = {},
  onHome,
  onView,
}) {
  const [playKey, setPlayKey] = useState(0);

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
              <button
                className="layout-preview-btn"
                type="button"
                onClick={() => setPlayKey((k) => k + 1)}
              >
                PLAY
              </button>
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
          ) : null}
          <div className="ended-page-slot">
            <WebCamSessionEnded
              key={playKey}
              onHome={onHome}
              onView={onView}
              y0Hits={layoutPreview ? 8 : (summary.y0_hits ?? 0)}
              y1Hits={layoutPreview ? 7 : (summary.y1_hits ?? 0)}
              good={layoutPreview ? 12 : (summary.posture_good ?? 0)}
              bad={layoutPreview ? 3 : (summary.posture_bad ?? 0)}
              total={layoutPreview ? 1840 : (summary.score ?? 0)}
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
        </div>
      </div>
    </div>
  );
}
