"""Live workout session: capture, YOLO, Pushup tracker, clean MJPEG frames."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np

from pose import Pushup, draw_overlay

MODEL_PATH = Path("model/yolo26n-pose.pt")
JPEG_QUALITY = 80

IDLE_STATUS = {
    "running": False,
    "phase": "idle",
    "reps": 0,
    "stage": "up",
    "score": 0,
    "detected": False,
    "top_bar": None,
    "low_bar": None,
    "shoulder_y": None,
    "elbow_angle": None,
    "hip_angle": None,
    "hip_status": None,
    "award_seq": 0,
    "last_multiplier": 1.0,
    "last_award_label": "",
    "last_award_points": 0,
}


def _detections(
    result,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    boxes = None
    keypoints = None
    ids = None
    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        if result.boxes.id is not None:
            ids = result.boxes.id.cpu().numpy().astype(np.int64)
    if result.keypoints is not None and result.keypoints.data is not None:
        keypoints = result.keypoints.data.cpu().numpy().astype(np.float64)
    return boxes, keypoints, ids


def _status_from(metrics: dict, phase: str, running: bool) -> dict:
    shoulder = metrics.get("shoulder")
    return {
        "running": running,
        "phase": phase,
        "reps": metrics.get("reps", 0),
        "stage": metrics.get("stage", "up"),
        "score": metrics.get("score", 0),
        "detected": bool(metrics.get("detected")),
        "top_bar": metrics.get("top_bar"),
        "low_bar": metrics.get("low_bar"),
        "shoulder_y": None if shoulder is None else float(shoulder[1]),
        "elbow_angle": metrics.get("elbow_angle"),
        "hip_angle": metrics.get("hip_angle"),
        "hip_status": metrics.get("hip_status"),
        "award_seq": metrics.get("award_seq", 0),
        "last_multiplier": metrics.get("last_multiplier", 1.0),
        "last_award_label": metrics.get("last_award_label", ""),
        "last_award_points": metrics.get("last_award_points", 0),
    }


class WorkoutSession:
    """One in-memory session. Webcam and video share the same phases."""

    def __init__(self, model=None) -> None:
        self._model = model
        self._device = "cpu"
        self._cap = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._status = dict(IDLE_STATUS)
        self.tracker = Pushup()
        self.phase = "idle"
        self.running = False
        self._source = "idle"

    def start_webcam(self, camera_id: int = 0) -> None:
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}")
        self._start(cap, "webcam")

    def start_video(self, path: Path | str) -> None:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video {path}")
        self._start(cap, "video")

    def stop(self) -> dict:
        summary = {
            "reps": self.tracker.reps,
            "score": self.tracker.score,
            "top_bar_hits": self.tracker.top_bar_hits,
            "low_bar_hits": self.tracker.low_bar_hits,
            "posture_good": self.tracker.posture_good,
            "posture_bad": self.tracker.posture_bad,
            "top_bar": self.tracker.top_bar,
            "low_bar": self.tracker.low_bar,
        }
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.running = False
        self.phase = "idle"
        self._source = "idle"
        with self._lock:
            self._status = dict(IDLE_STATUS)
            self._jpeg = None
        return summary

    def go(self) -> None:
        if self.phase != "ready":
            return
        self.tracker.reset_reps()
        self.phase = "active"
        with self._lock:
            self._status = {
                **self._status,
                "phase": "active",
                "reps": 0,
                "stage": "up",
                "score": 0,
                "award_seq": 0,
                "last_multiplier": 1.0,
                "last_award_label": "",
                "last_award_points": 0,
            }

    def get_status(self) -> dict:
        with self._lock:
            return dict(self._status)

    def get_frame(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def apply_metrics(self, metrics: dict, frame: np.ndarray | None = None) -> None:
        """Advance phase from tracker output. Used by the worker and tests."""
        if self.phase == "calibrating" and metrics.get("top_bar") is not None:
            self.phase = "ready"
        status = _status_from(metrics, self.phase, self.running)
        jpeg = None
        if frame is not None:
            ok, buf = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
            if ok:
                jpeg = buf.tobytes()
        with self._lock:
            self._status = status
            if jpeg is not None:
                self._jpeg = jpeg

    def _start(self, cap, source: str) -> None:
        if self.running:
            self.stop()
        self._source = source
        self._ensure_model()
        # Clear Ultralytics track persist so a new session gets fresh ids.
        if hasattr(self._model, "predictor"):
            self._model.predictor = None
        self._cap = cap
        self.tracker = Pushup()
        self.phase = "calibrating"
        self.running = True
        self._stop.clear()
        with self._lock:
            self._status = _status_from(
                self.tracker.update(None, None), self.phase, True
            )
            self._jpeg = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _ensure_model(self) -> None:
        if self._model is None:
            import torch
            from ultralytics import YOLO

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._device = device
            self._model = YOLO(str(MODEL_PATH))
            self._model.to(device)

    def _should_advance(self) -> bool:
        """Return False while uploaded video is paused for the countdown."""
        return not (self._source == "video" and self.phase == "ready")

    def _video_fps(self) -> float:
        fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0) if self._cap else 0.0
        return fps if fps >= 1.0 else 30.0

    def _end_loop(self) -> None:
        self.running = False
        with self._lock:
            self._status = {**self._status, "running": False, "phase": self.phase}
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _loop(self) -> None:
        pace = self._source == "video"
        fps = self._video_fps() if pace else 0.0
        frame_i = 0
        t0: float | None = None

        while not self._stop.is_set():
            if not self._should_advance():
                t0 = None
                time.sleep(0.03)
                continue

            ok, frame = self._cap.read()
            if not ok:
                self._end_loop()
                return
            if pace:
                frame_i += 1
            result = self._model.track(
                frame, persist=True, verbose=False, device=self._device
            )[0]
            boxes, keypoints, ids = _detections(result)
            metrics = self.tracker.update(boxes, keypoints, ids)
            draw_overlay(frame, metrics)
            self.apply_metrics(metrics, frame)
            if pace:
                now = time.monotonic()
                if t0 is None:
                    t0 = now - (frame_i - 1) / fps
                delay = t0 + frame_i / fps - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            else:
                time.sleep(0.001)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
