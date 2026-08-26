"""One workout at a time: webcam or uploaded video, analyzed with YOLO pose."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2
from numpy.typing import NDArray
from ultralytics import YOLO

from pose import (
    IDLE_METRICS,
    Metrics,
    TrackerState,
    analyze_frame,
    init_tracker,
    metrics_from_tracker,
    update_calibrate_ankle,
)

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
UPLOAD_DIR = BACKEND_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


def model_path() -> str:
    """Prefer a local YOLO weights file; fall back to the package name."""
    for path in (
        PROJECT_DIR / "src" / "yolo26n-pose.pt",
        PROJECT_DIR / "yolo26n-pose.pt",
    ):
        if path.exists():
            return str(path)
    return "yolo26n-pose.pt"


def open_camera(source: Any, kind: str) -> cv2.VideoCapture:
    """Open webcam (kind='webcam') or a video file path."""
    if kind == "webcam":
        video_capture = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    else:
        video_capture = cv2.VideoCapture(str(source))
    return video_capture


def video_info(video_capture: cv2.VideoCapture) -> tuple[float, int, int]:
    """Return (fps, width, height) with safe defaults."""
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1:
        fps = 30.0
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    return fps, width, height


def open_recorder(
    path: Path,
    fps: float,
    size: tuple[int, int],
) -> Optional[cv2.VideoWriter]:
    """Try a few codecs until one actually opens."""
    for codec in ("avc1", "H264", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    return None


def frame_to_jpeg(frame: NDArray[Any]) -> Optional[bytes]:
    """Encode a BGR frame as JPEG bytes for the live MJPEG stream."""
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return None
    return buffer.tobytes()


def file_ready(path: Optional[str]) -> bool:
    """True when path exists and is a non-empty file."""
    if not path:
        return False
    file_path = Path(path)
    return file_path.exists() and file_path.stat().st_size > 0


def downsample_series(
    history: list[dict[str, Any]],
    max_points: int = 300,
) -> list[dict[str, Any]]:
    """Downsample history for charts: {t, reps, elbow, speed} relative to start."""
    if not history:
        return []

    num_samples = len(history)
    if num_samples <= max_points:
        indices = list(range(num_samples))
    else:
        # Evenly spaced indices that always include first and last
        indices = sorted(
            {
                int(round(i * (num_samples - 1) / (max_points - 1)))
                for i in range(max_points)
            }
        )

    start_time = history[0].get("t") or 0.0
    series: list[dict[str, Any]] = []
    for index in indices:
        item = history[index]
        timestamp = item.get("t")
        series.append(
            {
                "t": round(
                    (timestamp - start_time) if timestamp is not None else 0.0, 2
                ),
                "reps": item.get("reps", 0),
                "elbow": item.get("elbow"),
                "speed": item.get("speed"),
                "posture_angle": item.get("posture_angle"),
                "posture_status": item.get("posture_status"),
            }
        )
    return series


class WorkoutSession:
    """
    One workout at a time.

    Threading model:
      - The analysis loop runs on a daemon thread (_run_loop).
      - self.lock guards: tracker, metrics, history, latest_jpeg, and related
        fields that HTTP handlers read while the loop writes.
      - self.running is a stop flag (bool); the loop checks it each frame.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.writer: Optional[cv2.VideoWriter] = None
        self.model: Optional[YOLO] = None
        self.tracker: Optional[TrackerState] = None  # live pose tracker

        self.running = False
        self.error: Optional[str] = None
        self.source_kind: Optional[str] = None
        self.original_path: Optional[str] = None
        self.record_path: Optional[str] = None
        self.latest_jpeg: Optional[bytes] = None

        self.metrics: Metrics = dict(IDLE_METRICS)
        self.history: list[dict[str, Any]] = []

    def start(self, source: Any, kind: str) -> None:
        """Stop any previous workout, then start analyzing source on a new thread."""
        self.stop()

        self.running = True
        self.error = None
        self.source_kind = kind
        self.original_path = str(source) if kind == "video" else None
        self.record_path = None
        self.latest_jpeg = None
        self.tracker = None
        self.metrics = {**IDLE_METRICS, "status": "starting"}
        self.history = []

        self.thread = threading.Thread(
            target=self._run_loop,
            args=(source, kind),
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> None:
        """Signal the loop to stop and wait briefly for it to finish."""
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            if threading.current_thread() is not self.thread:
                self.thread.join(timeout=8)
        self.thread = None
        with self.lock:
            self.tracker = None
        self._close_video()

    def arm(self) -> tuple[bool, Optional[str]]:
        """Move phase ready → active so rep counting starts. Called by POST /api/go."""
        with self.lock:
            tracker = self.tracker
            if tracker is None:
                return False, "no active session"
            if tracker.get("phase") != "ready":
                return False, f"phase is {tracker.get('phase')!r}, need 'ready'"
            tracker["phase"] = "active"
            self.metrics = {
                **self.metrics,
                "phase": "active",
                "status": "go",
            }
            return True, None

    def snapshot(self) -> dict[str, Any]:
        """Current metrics + running flag + chart series (for GET /api/status)."""
        with self.lock:
            return {
                **self.metrics,
                "running": self.running,
                "error": self.error,
                "source": self.source_kind,
                "has_recording": self.playback_path() is not None,
                "series": downsample_series(self.history, max_points=300),
            }

    def summary(self) -> dict[str, Any]:
        """End-of-workout counts of good vs issue frames (for GET /api/summary).

        Uses posture_status from each history sample ("good" / "bad"), not the
        overloaded status string (which mixes phase labels like "ready" / "finished").
        Frames with no posture reading (None / missing) are ignored.
        """
        with self.lock:
            good_frames = 0
            issue_frames = 0
            for item in self.history:
                posture = item.get("posture_status")
                if posture == "good":
                    good_frames += 1
                elif posture == "bad":
                    issue_frames += 1
            return {
                **self.metrics,
                "good_frames": good_frames,
                "issue_frames": issue_frames,
                "total_samples": len(self.history),
                "source": self.source_kind,
                "has_recording": self.playback_path() is not None,
                "series": downsample_series(self.history, max_points=300),
            }

    def history_series(self, max_points: int = 300) -> list[dict[str, Any]]:
        """Downsampled chart points only."""
        with self.lock:
            return downsample_series(self.history, max_points=max_points)

    def playback_path(self) -> Optional[str]:
        """Prefer the processed recording once the writer has finished."""
        if file_ready(self.record_path) and self.writer is None:
            return self.record_path
        if file_ready(self.original_path):
            return self.original_path
        return None

    def _close_video(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _publish(self, metrics: Metrics, running: bool = True) -> None:
        """Update shared metrics/history under the lock."""
        with self.lock:
            self.metrics = dict(metrics)
            self.running = running
            self.history.append({**metrics, "t": time.time()})
            if len(self.history) > 2000:
                self.history = self.history[-1500:]

    def _save_preview(self, frame: NDArray[Any]) -> None:
        jpeg = frame_to_jpeg(frame)
        if jpeg is None:
            return
        with self.lock:
            self.latest_jpeg = jpeg

    def _run_loop(self, source: Any, kind: str) -> None:
        try:
            self._analyze_video(source, kind)
        except Exception as exc:
            self.error = str(exc)
            self._publish(
                {**self.metrics, "status": f"error: {exc}"},
                running=False,
            )
        finally:
            self._close_video()
            self.running = False

    def _analyze_video(self, source: Any, kind: str) -> None:
        if self.model is None:
            self.model = YOLO(model_path())

        video_capture = open_camera(source, kind)
        if not video_capture.isOpened():
            self.error = f"Could not open {source}"
            self._publish(
                {**IDLE_METRICS, "status": "could not open source"},
                running=False,
            )
            return

        self.cap = video_capture
        fps, width, height = video_info(video_capture)

        filename = "webcam_temp.mp4" if kind == "webcam" else "upload_replay.mp4"
        record_path = UPLOAD_DIR / filename
        writer = open_recorder(record_path, fps, (width, height))
        self.writer = writer
        self.record_path = str(record_path) if writer is not None else None

        tracker = init_tracker(
            fps,
            calib_mode="auto" if kind == "video" else "hold",
        )
        with self.lock:
            self.tracker = tracker
        next_frame_at = time.perf_counter()
        frame_gap = 1.0 / fps

        while self.running:
            ok, frame = video_capture.read()
            if not ok:
                self._publish(
                    metrics_from_tracker(tracker, "finished"),
                    running=False,
                )
                break

            metrics = self._analyze_frame(frame, tracker)
            self._publish(metrics, running=True)
            self._save_preview(frame)
            self._write_frame(writer, frame, width, height)

            if kind != "webcam":
                next_frame_at += frame_gap
                wait = next_frame_at - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)

    def _analyze_frame(
        self,
        frame: NDArray[Any],
        tracker: TrackerState,
    ) -> Metrics:
        """Run YOLO, then pose analysis. Updates tracker in place."""
        result = self.model.track(frame, persist=True, verbose=False)[0]
        has_person = result.keypoints is not None and len(result.keypoints.data) > 0
        if not has_person:
            tracker["zi"] = None
            tracker["prev_h"] = None
            # Hard-reset hold if we were still calibrating
            if tracker.get("phase") == "calibrating":
                update_calibrate_ankle(tracker, None, None, None)
            return metrics_from_tracker(tracker, "no person")

        keypoints = result.keypoints.data.cpu().numpy()[0].astype(float)
        return analyze_frame(frame, keypoints, tracker)

    def _write_frame(
        self,
        writer: Optional[cv2.VideoWriter],
        frame: NDArray[Any],
        width: int,
        height: int,
    ) -> None:
        if writer is None:
            return
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)
