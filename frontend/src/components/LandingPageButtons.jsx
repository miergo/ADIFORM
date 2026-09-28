const TIMED_DURATIONS = [
  { label: "30s", seconds: 30 },
  { label: "60s", seconds: 60 },
  { label: "120s", seconds: 120 },
];

export default function LandingPageButtons({
  state = "start",
  disabled = false,
  onStart,
  onWebcam,
  onUpload,
  onTimed,
  onEndless,
  onTimedDuration,
}) {
  if (state === "timed") {
    return (
      <div className="lp-buttons lp-buttons--timed">
        {TIMED_DURATIONS.map(({ label, seconds }) => (
          <button
            key={seconds}
            className="lp-btn lp-btn--choice"
            type="button"
            onClick={() => onTimedDuration?.(seconds)}
            disabled={disabled}
          >
            {label}
          </button>
        ))}
      </div>
    );
  }

  if (state === "mode") {
    return (
      <div className="lp-buttons lp-buttons--source">
        <button
          className="lp-btn lp-btn--choice"
          type="button"
          onClick={onTimed}
          disabled={disabled}
        >
          TIMED
        </button>
        <button
          className="lp-btn lp-btn--choice"
          type="button"
          onClick={onEndless}
          disabled={disabled}
        >
          ENDLESS
        </button>
      </div>
    );
  }

  if (state === "source") {
    return (
      <div className="lp-buttons lp-buttons--source">
        <button
          className="lp-btn lp-btn--choice"
          type="button"
          onClick={onWebcam}
          disabled={disabled}
        >
          WEBCAM
        </button>
        <button
          className="lp-btn lp-btn--choice"
          type="button"
          onClick={onUpload}
          disabled={disabled}
        >
          UPLOAD
        </button>
      </div>
    );
  }

  return (
    <div className="lp-buttons lp-buttons--start">
      <button
        className="lp-btn lp-btn--start"
        type="button"
        onClick={onStart}
        disabled={disabled}
      >
        START
      </button>
    </div>
  );
}
