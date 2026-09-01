import { useRef } from "react";

export default function LandingPageButtons({
  state = "start",
  disabled = false,
  onStart,
  onWebcam,
  onUploadFile,
}) {
  const fileRef = useRef(null);

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
          onClick={() => fileRef.current?.click()}
          disabled={disabled}
        >
          UPLOAD
        </button>
        <input
          ref={fileRef}
          className="hidden-file"
          type="file"
          accept="video/*"
          onChange={(e) => {
            onUploadFile?.(e.target.files?.[0]);
            e.target.value = "";
          }}
        />
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
