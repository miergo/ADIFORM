const RULES = [
  "THE CAMERA SHOULD BE PLACED FACING BELOW WAIST LEVEL,AND MAKE SURE YOUR FULLBODY LENGTH FITS THE FULL FRAME.",
  "ONCE IN FRAME,THE CALIBRATION WILL BEGIN AND  YOU NEED TO STAY IN THE UPWARD PUSHUP POSITION FOR 3 SECONDS.",
  "A SOUND COUNTER WILL PLAY, AND THEN YOU CAN BEGIN YOUR PUSHUP MOTION",
  "HAVE FUN",
];

export default function RulesComponent({
  step,
  disabled = false,
  onHome,
  onPrev,
  onNext,
  onBegin,
}) {
  const isLast = step === RULES.length;

  return (
    <div className="rules-card">
      <h2 className="rules-heading">Rules</h2>
      <ol className="rules-text" start={step}>
        <li>{RULES[step - 1]}</li>
      </ol>
      <div className="rules-nav">
        {step === 1 ? (
          <button
            className="lp-btn lp-btn--nav"
            type="button"
            onClick={onHome}
            disabled={disabled}
          >
            HOME
          </button>
        ) : (
          <button
            className="lp-btn lp-btn--nav"
            type="button"
            onClick={onPrev}
            disabled={disabled}
          >
            {"<< PREV"}
          </button>
        )}
        <button
          className="lp-btn lp-btn--nav"
          type="button"
          onClick={isLast ? onBegin : onNext}
          disabled={disabled}
        >
          {isLast ? (
            <>
              <span className="rules-cta-idle">{"BEGIN >>"}</span>
              <span className="rules-cta-hover">{"BEGIN >>"}</span>
            </>
          ) : (
            "NEXT >>"
          )}
        </button>
      </div>
    </div>
  );
}

export const RULE_COUNT = RULES.length;
