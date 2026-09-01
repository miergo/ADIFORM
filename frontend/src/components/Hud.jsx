function formatTime(seconds) {
  const n = Number(seconds);
  if (!Number.isFinite(n) || n < 0) {
    return "0.00s";
  }
  return `${n.toFixed(2)}s`;
}

function pad4(value) {
  const n = Number(value) || 0;
  return String(Math.max(0, Math.trunc(n))).padStart(4, "0");
}

export default function Hud({ reps = 0, time = 0, score = 0 }) {
  return (
    <div className="hud">
      <div className="hud-stat hud-stat--reps">
        <span>REPS</span>
        <span>{reps}</span>
      </div>
      <div className="hud-stat hud-stat--time">
        <span>TIME</span>
        <span>{formatTime(time)}</span>
      </div>
      <div className="hud-stat hud-stat--score">
        <span>SCORE</span>
        <span>{pad4(score)}</span>
      </div>
    </div>
  );
}
