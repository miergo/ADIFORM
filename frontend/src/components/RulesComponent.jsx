const RULES = [
  "SET THE CAMERA AT YOUR SIDE, LOW ENOUGH TO SEE YOUR WHOLE BODY.",
  "HOLD A HIGH PLANK UNTIL CALIBRATION FINISHES.",
  "WAIT FOR THE COUNTDOWN. THEN GO.",
  "KEEP YOUR ELBOWS CLOSE TO YOUR BODY. DONT FLARE THEM OUT.",
  "FULL PUSH-UPS ONLY. LEGS STRAIGHT, BEND YOUR ARMS AT THE BOTTOM, LOCK THEM OUT AT THE TOP.",
  "HAVE FUN",
];

export default function RulesComponent({
  step,
  disabled = false,
  onBack,
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
            onClick={onBack}
            disabled={disabled}
          >
            BACK
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
