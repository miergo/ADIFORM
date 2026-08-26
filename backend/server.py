"""FastAPI entry point: HTTP routes around a single WorkoutSession."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import cv2

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from session import WorkoutSession

app = FastAPI(title="Pose workout API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

session = WorkoutSession()
UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Response models (self-documenting at /docs)
# ---------------------------------------------------------------------------


class SeriesPoint(BaseModel):
    """One downsampled history sample for charts."""

    t: float
    reps: int = 0
    elbow: Optional[float] = None
    speed: Optional[float] = None
    posture_angle: Optional[float] = None
    posture_status: Optional[str] = None


class MetricsSnapshot(BaseModel):
    """Shared metrics shape returned by /api/status and /api/summary."""

    reps: int = 0
    stage: str = "-"
    side: Optional[str] = None
    status: str = "idle"
    elbow: Optional[float] = None
    eye_height: Optional[float] = None
    speed: Optional[float] = None
    touch_y0: Optional[bool] = None
    touch_y1: Optional[bool] = None
    posture_angle: Optional[float] = None
    posture_status: Optional[str] = None
    phase: Optional[str] = None
    calib_progress: float = 0.0
    running: Optional[bool] = None
    error: Optional[str] = None
    source: Optional[str] = None
    has_recording: Optional[bool] = None
    series: list[SeriesPoint] = Field(default_factory=list)
    # summary-only fields
    good_frames: Optional[int] = None
    issue_frames: Optional[int] = None
    total_samples: Optional[int] = None


class OkResponse(BaseModel):
    ok: bool = True
    source: Optional[str] = None
    name: Optional[str] = None
    phase: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/api/status", response_model=MetricsSnapshot)
def status() -> dict[str, Any]:
    return session.snapshot()


@app.get("/api/summary", response_model=MetricsSnapshot)
def summary() -> dict[str, Any]:
    return session.summary()


@app.get("/api/history")
def history() -> dict[str, list[dict[str, Any]]]:
    return {"series": session.history_series(max_points=300)}


@app.post("/api/start/webcam", response_model=OkResponse)
def start_webcam(camera_id: int = Form(0)) -> dict[str, Any]:
    session.start(camera_id, "webcam")
    return {"ok": True, "source": "webcam"}


@app.post("/api/start/video", response_model=OkResponse)
async def start_video(file: UploadFile = File(...)) -> dict[str, Any]:
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    dest = UPLOAD_DIR / f"session{suffix}"
    dest.write_bytes(await file.read())
    session.start(str(dest), "video")
    return {"ok": True, "source": "video", "name": file.filename}


@app.post("/api/stop", response_model=MetricsSnapshot)
def stop() -> dict[str, Any]:
    session.stop()
    return session.summary()


@app.post("/api/go")
def go():
    """Arm the game after hold-at-top calibration (phase ready → active)."""
    ok, err = session.arm()
    if not ok:
        return JSONResponse({"ok": False, "error": err}, status_code=400)
    return {"ok": True, "phase": "active"}


@app.get("/api/recording")
def recording():
    path = session.playback_path()
    if not path:
        return JSONResponse({"error": "no recording yet"}, status_code=404)
    return FileResponse(path, media_type="video/mp4", filename="workout.mp4")


@app.get("/api/recording/download")
def download_recording():
    path = session.playback_path()
    if not path:
        return JSONResponse({"error": "no recording yet"}, status_code=404)
    name = (
        "webcam-workout.mp4"
        if session.source_kind == "webcam"
        else "uploaded-workout.mp4"
    )
    return FileResponse(path, media_type="video/mp4", filename=name)


@app.get("/api/replay/stream")
def replay_stream():
    path = session.playback_path()
    if not path:
        return JSONResponse({"error": "no recording yet"}, status_code=404)

    def frames():
        video_capture = cv2.VideoCapture(str(path))
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        delay = 1.0 / fps if fps and fps > 1 else 1.0 / 30.0
        while True:
            ok, frame = video_capture.read()
            if not ok:
                break
            ok, buffer = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            )
            if ok:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )
            time.sleep(delay)
        video_capture.release()

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/stream")
def stream() -> StreamingResponse:
    def frames():
        while True:
            jpeg = session.latest_jpeg
            if jpeg is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            time.sleep(0.03)

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


frontend_dist = ROOT.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="ui")
