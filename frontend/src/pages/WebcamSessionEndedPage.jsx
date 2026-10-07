import { useEffect, useState } from "react";
import { assetUrl } from "../assetUrl.js";
import WebcamSessionEnded from "../components/WebcamSessionEnded.jsx";
import { playSfx, stopSfx } from "../sounds.js";

export default function WebcamSessionEndedPage({
  layoutPreview = false,
  summary = {},
  onHome,
  onMode,
  onRetry,
}) {
  const [playKey, setPlayKey] = useState(0);

  useEffect(() => {
    const sound = playSfx("score");
    return () => stopSfx(sound);
  }, [playKey]);

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
          <div className="layout-preview-bar">
            {layoutPreview ? (
              <button
                className="layout-preview-btn"
                type="button"
                onClick={() => setPlayKey((k) => k + 1)}
              >
                PLAY
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
          <div className="ended-page-slot">
            <WebcamSessionEnded
              key={playKey}
              onMode={onMode}
              onRetry={onRetry}
              topBarHits={layoutPreview ? 8 : (summary.top_bar_hits ?? 0)}
              lowBarHits={layoutPreview ? 7 : (summary.low_bar_hits ?? 0)}
              postureGood={layoutPreview ? 12 : (summary.posture_good ?? 0)}
              postureBad={layoutPreview ? 3 : (summary.posture_bad ?? 0)}
              score={layoutPreview ? 1840 : (summary.score ?? 0)}
              prevBest={layoutPreview ? 2100 : (summary.prevBest ?? null)}
              isNewBest={layoutPreview ? false : Boolean(summary.isNewBest)}
              awards={
                layoutPreview
                  ? [
                      { mult: 2, label: "full depth", points: 200 },
                      { mult: 1.85, label: "deep", points: 185 },
                      { mult: 1, label: "hit", points: 100 },
                      { mult: 1, label: "miss", points: 25 },
                    ]
                  : (summary.awards ?? [])
              }
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
