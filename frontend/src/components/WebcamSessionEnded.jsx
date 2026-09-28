import ScoreCard from "./ScoreCard.jsx";

export default function WebcamSessionEnded({
  onMode,
  onRetry,
  topBarHits = 0,
  lowBarHits = 0,
  postureGood = 0,
  postureBad = 0,
  score = 0,
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
        topBarHits={topBarHits}
        lowBarHits={lowBarHits}
        postureGood={postureGood}
        postureBad={postureBad}
        score={score}
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
