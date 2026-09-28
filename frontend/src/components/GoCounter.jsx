const LABELS = {
  1: "1",
  2: "2",
  3: "3",
  go: "GO",
};

export default function GoCounter({ state = 0 }) {
  const label = LABELS[state];
  const stepClass = state === 0 ? "0" : String(state);

  return (
    <div
      className={`go-counter go-counter--${stepClass}`}
      role="status"
      aria-label={label ? `Countdown ${label}` : "Countdown idle"}
    >
      <div className="go-counter-outer" />
      {state === 1 || state === 2 || state === 3 || state === "go" ? (
        <div className="go-counter-inner" />
      ) : null}
      {label ? <p className="go-counter-text">{label}</p> : null}
    </div>
  );
}
