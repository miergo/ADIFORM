function pad4(value) {
  const n = Number(value) || 0;
  const sign = n < 0 ? "-" : "";
  return `${sign}${String(Math.abs(n)).padStart(4, "0")}`;
}

export default function ScoreCard({
  open = false,
  y0Hits = 0,
  y1Hits = 0,
  good = 0,
  bad = 0,
  total = 0,
}) {
  if (!open) {
    return <div className="score-card score-card--closed" />;
  }

  return (
    <div className="score-card score-card--open">
      <div className="score-sheet">
        <div className="score-sheet-title">
          <p>Summary</p>
        </div>
        <div className="score-sheet-body">
          <div className="score-block">
            <p className="score-block-label">REPS</p>
            <div className="score-row">
              <span>Y0 HITS</span>
              <span>{pad4(y0Hits)}</span>
            </div>
            <div className="score-row">
              <span>Y1 HITS</span>
              <span>{pad4(y1Hits)}</span>
            </div>
          </div>
          <div className="score-block">
            <p className="score-block-label">POSTURE</p>
            <div className="score-row">
              <span>GOOD</span>
              <span>{pad4(good)}</span>
            </div>
            <div className="score-row">
              <span>BAD</span>
              <span>{pad4(bad)}</span>
            </div>
          </div>
          <div className="score-block">
            <div className="score-row">
              <span>TOTAL</span>
              <span>{pad4(total)}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
