"""One workout at a time: webcam or uploaded video, analyzed with YOLO pose."""

import threading
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from pose import IDLE_METRICS, analyze_frame, init_tracker

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
UPLOAD_DIR = BACKEND_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

SKIP_STATUSES = (
    "idle",
    "no person",
    "low conf",
    "detecting side...",
    "calibrating ankle...",
)


def model_path():
    for path in (
        PROJECT_DIR / "src" / "yolo26n-pose.pt",
        PROJECT_DIR / "yolo26n-pose.pt",
    ):
        if path.exists():
            return str(path)
    return "yolo26n-pose.pt"


def open_camera(source, kind):
    if kind == "webcam":
        cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    else:
        cap = cv2.VideoCapture(str(source))
    return cap


def video_info(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    return fps, width, height


def open_recorder(path, fps, size):
    """Try a few codecs until one actually opens."""
    for codec in ("avc1", "H264", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    return None


def frame_to_jpeg(frame):
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return None
    return buf.tobytes()


def file_ready(path):
    if not path:
        return False
    p = Path(path)
    return p.exists() and p.stat().st_size > 0


def downsample_series(history, max_points=300):
    """Downsample history for charts: {t, reps, elbow, speed} relative to start."""
    if not history:
        return []

    n = len(history)
    if n <= max_points:
        idxs = range(n)
    else:
        step = n / max_points
        idxs = sorted({min(n - 1, int(i * step)) for i in range(max_points)})

    t0 = history[0].get("t") or 0.0
    series = []
    for i in idxs:
        item = history[i]
        t = item.get("t")
        series.append(
            {
                "t": round((t - t0) if t is not None else 0.0, 2),
                "reps": item.get("reps", 0),
                "elbow": item.get("elbow"),
                "speed": item.get("speed"),
                "posture_angle": item.get("posture_angle"),
                "posture_status": item.get("posture_status"),
            }
        )
    return series


class WorkoutSession:
    def __init__(self):
        self.lock = threading.Lock()
        self.thread = None
        self.cap = None
        self.writer = None
        self.model = None

        self.running = False
        self.error = None
        self.source_kind = None
        self.original_path = None
        self.record_path = None
        self.latest_jpeg = None

        self.metrics = dict(IDLE_METRICS)
        self.history = []

    def start(self, source, kind):
        self.stop()

        self.running = True
        self.error = None
        self.source_kind = kind
        self.original_path = str(source) if kind == "video" else None
        self.record_path = None
        self.latest_jpeg = None
        self.metrics = {**IDLE_METRICS, "status": "starting"}
        self.history = []

        self.thread = threading.Thread(
            target=self._run_loop,
            args=(source, kind),
            daemon=True,
        )
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            if threading.current_thread() is not self.thread:
                self.thread.join(timeout=8)
        self.thread = None
        self._close_video()

    def snapshot(self):
        with self.lock:
            return {
                **self.metrics,
                "running": self.running,
                "error": self.error,
                "source": self.source_kind,
                "has_recording": self.playback_path() is not None,
                "series": downsample_series(self.history, max_points=300),
            }

    def summary(self):
        with self.lock:
            good = 0
            issues = 0
            for item in self.history:
                text = str(item.get("status", ""))
                if text.startswith("GOOD"):
                    good += 1
                elif text and text not in SKIP_STATUSES:
                    issues += 1
            return {
                **self.metrics,
                "good_frames": good,
                "issue_frames": issues,
                "total_samples": len(self.history),
                "source": self.source_kind,
                "has_recording": self.playback_path() is not None,
                "series": downsample_series(self.history, max_points=300),
            }

    def history_series(self, max_points=300):
        with self.lock:
            return downsample_series(self.history, max_points=max_points)

    def playback_path(self):
        # Prefer the processed recording once the writer has finished.
        if file_ready(self.record_path) and self.writer is None:
            return self.record_path
        if file_ready(self.original_path):
            return self.original_path
        return None

    def _close_video(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _publish(self, metrics, running=True):
        with self.lock:
            self.metrics = dict(metrics)
            self.running = running
            self.history.append({**metrics, "t": time.time()})
            if len(self.history) > 2000:
                self.history = self.history[-1500:]

    def _save_preview(self, frame):
        jpeg = frame_to_jpeg(frame)
        if jpeg is None:
            return
        with self.lock:
            self.latest_jpeg = jpeg

    def _run_loop(self, source, kind):
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

    def _analyze_video(self, source, kind):
        if self.model is None:
            self.model = YOLO(model_path())

        cap = open_camera(source, kind)
        if not cap.isOpened():
            self.error = f"Could not open {source}"
            self._publish(
                {**IDLE_METRICS, "status": "could not open source"},
                running=False,
            )
            return

        self.cap = cap
        fps, width, height = video_info(cap)

        filename = "webcam_temp.mp4" if kind == "webcam" else "upload_replay.mp4"
        record_path = UPLOAD_DIR / filename
        writer = open_recorder(record_path, fps, (width, height))
        self.writer = writer
        self.record_path = str(record_path) if writer is not None else None

        tracker = init_tracker(fps)
        next_frame_at = time.perf_counter()
        frame_gap = 1.0 / fps

        while self.running:
            ok, frame = cap.read()
            if not ok:
                self._publish(
                    metrics_from_tracker_finished(tracker),
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

    def _analyze_frame(self, frame, tracker):
        """Run YOLO, then pose analysis. Updates tracker in place."""
        result = self.model.track(frame, persist=True, verbose=False)[0]
        has_person = result.keypoints is not None and len(result.keypoints.data) > 0
        if not has_person:
            tracker["zi"] = None
            tracker["prev_h"] = None
            return {
                "reps": tracker["reps"],
                "stage": tracker["stage"],
                "side": tracker["side"],
                "status": "no person",
                "elbow": None,
                "eye_height": tracker.get("eye_height"),
                "speed": tracker.get("speed"),
                "posture_angle": None,
                "posture_status": None,
            }

        kpts = result.keypoints.data.cpu().numpy()[0].astype(float)
        return analyze_frame(frame, kpts, tracker)

    def _write_frame(self, writer, frame, width, height):
        if writer is None:
            return
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)


def metrics_from_tracker_finished(tracker):
    return {
        "reps": tracker["reps"],
        "stage": tracker["stage"],
        "side": tracker["side"],
        "status": "finished",
        "elbow": None,
        "eye_height": tracker.get("eye_height"),
        "speed": tracker.get("speed"),
        "posture_angle": tracker.get("posture_angle"),
        "posture_status": tracker.get("posture_status"),
    }
