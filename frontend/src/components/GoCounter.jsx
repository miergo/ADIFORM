const LABELS = {
  1: "1",
  2: "2",
  3: "3",
  4: "GO",
};

export default function GoCounter({ state = 0 }) {
  const label = LABELS[state];

  return (
    <div
      className={`go-counter go-counter--${state}`}
      role="status"
      aria-label={label ? `Countdown ${label}` : "Countdown idle"}
    >
      <div className="go-counter-outer" />
      {state >= 1 && state <= 4 ? <div className="go-counter-inner" /> : null}
      {label ? <p className="go-counter-text">{label}</p> : null}
    </div>
  );
}
