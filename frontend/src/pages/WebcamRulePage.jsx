import { useState } from "react";
import RulesComponent, { RULE_COUNT } from "../components/RulesComponent.jsx";

export default function WebcamRulePage({ starting, error, onBegin, onHome, onBack }) {
  const [step, setStep] = useState(1);

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
          {onHome ? (
            <div className="layout-preview-bar">
              <button
                className="layout-preview-btn"
                type="button"
                onClick={onHome}
              >
                HOME
              </button>
            </div>
          ) : null}
          <button
            className="layout-preview-btn rules-skip-btn"
            type="button"
            onClick={onBegin}
            disabled={starting}
          >
            SKIP
          </button>
          <div className="rules-page-card">
            <RulesComponent
              step={step}
              disabled={starting}
              onBack={onBack}
              onPrev={() => setStep((s) => Math.max(1, s - 1))}
              onNext={() => setStep((s) => Math.min(RULE_COUNT, s + 1))}
              onBegin={onBegin}
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
