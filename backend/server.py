"""HTTP API for the workout session."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from session import WorkoutSession

UPLOADS = Path("uploads")
session = WorkoutSession()
app = FastAPI()


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/status")
def status():
    return session.get_status()


@app.post("/api/start/webcam")
def start_webcam(camera_id: int = Form(0)):
    try:
        session.start_webcam(camera_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True}


@app.post("/api/start/video")
async def start_video(file: UploadFile = File(...)):
    UPLOADS.mkdir(parents=True, exist_ok=True)
    name = Path(file.filename or "upload.mp4").name
    dest = UPLOADS / name
    dest.write_bytes(await file.read())
    try:
        session.start_video(dest)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True}


@app.post("/api/go")
def go():
    session.go()
    return session.get_status()


@app.post("/api/stop")
def stop():
    return session.stop()


@app.get("/api/stream")
def stream():
    boundary = b"frame"

    def frames():
        while session.running:
            jpeg = session.get_frame()
            if jpeg:
                yield (
                    b"--" + boundary + b"\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            else:
                time.sleep(0.03)

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
