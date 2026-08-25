from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
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


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/status")
def status():
    return session.snapshot()


@app.get("/api/summary")
def summary():
    return session.summary()


@app.get("/api/history")
def history():
    return {"series": session.history_series(max_points=300)}


@app.post("/api/start/webcam")
def start_webcam(camera_id: int = Form(0)):
    session.start(camera_id, "webcam")
    return {"ok": True, "source": "webcam"}


@app.post("/api/start/video")
async def start_video(file: UploadFile = File(...)):
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    dest = UPLOAD_DIR / f"session{suffix}"
    dest.write_bytes(await file.read())
    session.start(str(dest), "video")
    return {"ok": True, "source": "video", "name": file.filename}


@app.post("/api/stop")
def stop():
    session.stop()
    return session.summary()


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
    name = "webcam-workout.mp4" if session.source_kind == "webcam" else "uploaded-workout.mp4"
    return FileResponse(path, media_type="video/mp4", filename=name)


@app.get("/api/replay/stream")
def replay_stream():
    path = session.playback_path()
    if not path:
        return JSONResponse({"error": "no recording yet"}, status_code=404)

    def frames():
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = 1.0 / fps if fps and fps > 1 else 1.0 / 30.0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
                )
            time.sleep(delay)
        cap.release()

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/stream")
def stream():
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
