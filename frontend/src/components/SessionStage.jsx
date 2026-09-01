import GoCounter from "./GoCounter.jsx";
import Hud from "./Hud.jsx";

export default function SessionStage({
  mode,
  counterState = 0,
  reps = 0,
  time = 0,
  score = 0,
  streamSrc,
  source = "webcam",
}) {
  const showFeed =
    Boolean(streamSrc) &&
    (mode === "session" ||
      mode === "calibrating" ||
      mode === "complete" ||
      mode === "timer");
  const playing = mode === "session";
  const stageClass = [
    "session-stage",
    showFeed ? "session-stage--feed" : "",
    playing ? "session-stage--playing" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={stageClass}>
      {showFeed ? (
        <div className="session-feed">
          <img src={streamSrc} alt="Workout video" />
        </div>
      ) : null}
      {mode === "calibrating" || mode === "complete" ? (
        <h2 className="session-heading">CALIBRATION</h2>
      ) : null}
      {mode === "complete" ? (
        <p className="session-complete-msg">
          Calibration done, the system will now begin the session
        </p>
      ) : null}
      {mode === "timer" ? (
        <div className="session-counter">
          <GoCounter state={counterState} />
        </div>
      ) : null}
      {playing ? <Hud reps={reps} time={time} score={score} /> : null}
    </div>
  );
}
