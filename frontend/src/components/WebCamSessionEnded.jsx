import ScoreCard from "./ScoreCard.jsx";

export default function WebCamSessionEnded({
  onHome,
  onView,
  y0Hits = 0,
  y1Hits = 0,
  good = 0,
  bad = 0,
  total = 0,
}) {
  return (
    <div className="ended-card ended-card--anim">
      <p className="ended-heading">SESSION ENDED</p>
      <div className="ended-bar" />
      <ScoreCard
        open
        y0Hits={y0Hits}
        y1Hits={y1Hits}
        good={good}
        bad={bad}
        total={total}
      />
      <div className="ended-actions">
        <button
          className="lp-btn lp-btn--start"
          type="button"
          onClick={onHome}
        >
          HOME
        </button>
        <button
          className="lp-btn lp-btn--start"
          type="button"
          onClick={onView}
        >
          VIEW
        </button>
      </div>
    </div>
  );
}
