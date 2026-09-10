function pad4(value) {
  const n = Number(value) || 0;
  const sign = n < 0 ? "-" : "";
  return `${sign}${String(Math.abs(n)).padStart(4, "0")}`;
}

function countAwards(awards, label) {
  return awards.filter((a) => a.label === label).length;
}

export default function ScoreCard({
  open = false,
  y0Hits = 0,
  y1Hits = 0,
  good = 0,
  bad = 0,
  total = 0,
  awards = [],
  prevBest = null,
  isNewBest = false,
}) {
  if (!open) {
    return <div className="score-card score-card--closed" />;
  }

  const full = countAwards(awards, "full depth");
  const deep = countAwards(awards, "deep");
  const base = countAwards(awards, "hit");
  const miss = countAwards(awards, "miss");
  const showBest = prevBest != null && !isNewBest;

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
              <span>UP</span>
              <span>{pad4(y0Hits)}</span>
            </div>
            <div className="score-row">
              <span>DOWN</span>
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
            {showBest ? (
              <div className="score-row">
                <span>BEST</span>
                <span>{pad4(prevBest)}</span>
              </div>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}
