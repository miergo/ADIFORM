import ScoreCard from "./ScoreCard.jsx";

export default function WebCamSessionEnded({
  onMode,
  onRetry,
  y0Hits = 0,
  y1Hits = 0,
  good = 0,
  bad = 0,
  total = 0,
  awards = [],
  prevBest = null,
  isNewBest = false,
}) {
  return (
    <div className="ended-card ended-card--anim">
      <p className="ended-heading">
        {isNewBest ? "NEW BEST" : "SESSION ENDED"}
      </p>
      <div className="ended-bar" />
      <ScoreCard
        open
        y0Hits={y0Hits}
        y1Hits={y1Hits}
        good={good}
        bad={bad}
        total={total}
        awards={awards}
        prevBest={prevBest}
        isNewBest={isNewBest}
      />
      <div className="ended-actions">
        <button
          className="lp-btn lp-btn--start"
          type="button"
          onClick={onMode}
        >
          MODE
        </button>
        <button
          className="lp-btn lp-btn--start"
          type="button"
          onClick={onRetry}
        >
          RETRY
        </button>
      </div>
    </div>
  );
}
