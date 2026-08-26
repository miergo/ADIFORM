"""
Push-up pose analysis for the workout API.

WHAT HAPPENS EACH FRAME (see analyze_frame at the bottom)
  1. Detect which side faces the camera (left vs right)
  2. Smooth keypoints (Butterworth) to reduce jitter
  3. Measure elbow angle + eye height above the ankle
  4. Count reps: eye near y=0 (down), then elbow unlocks (up)
  5. Draw overlays on the frame
  6. Return a metrics dict for the API / UI

HOW TO ADD A NEW FEATURE (when you do not have AI)
  1. Write a small helper function in the right section below
  2. If it needs memory across frames, add a key in init_tracker()
  3. Call the helper from analyze_frame() in the numbered steps
  4. Put any value you want on the UI into metrics_from_tracker()

Session / video I/O lives in session.py — you usually only edit this file.
"""

from __future__ import annotations

import time
from typing import Literal, Optional, TypedDict

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfilt, sosfilt_zi

# =============================================================================
# KEYPOINT INDEX CHEATSHEET (YOLO / COCO pose)
# Each keypoint is [x, y, confidence]. Image Y grows downward.
# =============================================================================

NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
# 3 = left ear, 4 = right ear (unused)
# Arms: shoulder, elbow, wrist
LEFT_ARM = [5, 7, 9]
RIGHT_ARM = [6, 8, 10]
# Body line for posture: shoulder, hip, ankle
LEFT_LINE = (5, 11, 15)
RIGHT_LINE = (6, 12, 16)

SideName = Literal["left", "right"]
PhaseName = Literal["detecting", "calibrating", "ready", "active"]
# Metrics may also report "idle" before a session starts
MetricsPhase = Literal["idle", "detecting", "calibrating", "ready", "active"]
StageName = Literal["-", "down", "up"]
PostureStatus = Literal["good", "bad"]
CalibMode = Literal["hold", "auto"]  # hold = webcam 2s; auto = video frame count

# Keypoint array: shape (17, 3) with columns [x, y, confidence]
Keypoints = NDArray[np.float64]
# XY only: shape (num_keypoints, 2)
KeypointXY = NDArray[np.float64]
# Single point [x, y] or [x, y, confidence]
Point = NDArray[np.float64]


# =============================================================================
# TUNABLE CONSTANTS — change these to adjust behavior
# =============================================================================

# --- Side detection ---
SIDE_VOTES_NEEDED = 20  # frames of arm-visibility votes before locking a side
SIDE_MARGIN = 0.10  # min confidence gap between left/right arms

# --- Keypoint smoothing (Butterworth low-pass) ---
BUTTER_ORDER = 2
BUTTER_CUTOFF_HZ = 3.0  # lower = smoother but more lag; try 2–6

# --- Rep counting ---
# Down = eye near the ankle horizontal line (y=0)
# Up   = after a down, elbow angle unlocks past UP_ELBOW
UP_ELBOW = 145.0  # degrees; "arms locked out" at the top
DOWN_TOUCH_PX = 25.0  # eye within this many pixels of ankle height = down
MIN_Y1_PX = 40.0  # minimum length for the y1 guide line

# --- Hold-at-top calibration (y0 + y1) ---
CALIB_ELBOW_MIN = 150.0  # degrees; fully extended for calib hold
HOLD_CALIB_POSE_TIME = 2.0  # seconds to hold before locking guides (webcam)
VIDEO_AUTO_CALIB_FRAMES = 7  # consecutive top frames before locking (video)

# --- Posture overlay ---
CONF_MIN = 0.4  # ignore joints below this confidence
STRAIGHT_TOL = 15.0  # degrees off 180 still counts as "straight" body line
AXIS_LINE_LEN = 120  # fallback vertical length before y1 exists

# --- Speed (m/s) from eye-height change ---
TORSO_LENGTH_M = 0.45  # assumed shoulder–hip length in meters
SCALE_EMA = 0.15  # how fast meters_per_px adapts (0–1)


# =============================================================================
# TRACKER + METRICS TYPES (documented state shapes)
# =============================================================================


class TrackerState(TypedDict):
    """
    Per-session memory mutated every frame.

    Keys you will touch most often:
      stage, reps     — push-up FSM
      y1              — upper guide (set once at end of hold-at-top calib)
      ankle_origin    — frozen y=0 point
      eye_height      — shown in the UI (normalized by y1 when available)
      sos, zi         — Butterworth filter state (reset zi when person lost)
    """

    # Side lock
    left_votes: int
    right_votes: int
    side: Optional[SideName]
    arm: Optional[list[int]]  # [shoulder, elbow, wrist] indices
    chain: Optional[tuple[int, int, int]]  # (shoulder, hip, ankle) indices
    # Smoothing (sos = second-order sections coeffs; zi = filter memory)
    sos: NDArray[np.float64]
    zi: Optional[NDArray[np.float64]]
    fps: float
    # Reps
    stage: StageName
    reps: int
    # Calibration / game phases
    ankle_origin: Optional[NDArray[np.float64]]  # frozen y=0
    y1: Optional[float]  # frozen at end of hold-at-top calib
    phase: PhaseName  # detecting | calibrating | ready | active
    calib_mode: CalibMode  # hold = webcam timed hold; auto = video frame count
    calib_progress: float
    calib_hold_start: Optional[float]
    calib_ankle_samples: list[NDArray[np.float64]]
    calib_h_samples: list[float]
    # Overlays / UI
    eye_height: Optional[float]
    speed: Optional[float]  # m/s
    prev_h: Optional[float]  # last eye height px (for speed)
    m_per_px: Optional[float]  # meters per pixel from torso scale
    posture_angle: Optional[float]
    posture_status: Optional[PostureStatus]
    touch_y0: Optional[bool]
    touch_y1: Optional[bool]


class Metrics(TypedDict):
    """Values the API sends to the frontend. Same key set for every status."""

    reps: int
    stage: StageName
    side: Optional[SideName]
    status: str
    elbow: Optional[float]
    eye_height: Optional[float]
    speed: Optional[float]
    touch_y0: Optional[bool]
    touch_y1: Optional[bool]
    posture_angle: Optional[float]
    posture_status: Optional[PostureStatus]
    phase: Optional[MetricsPhase]
    calib_progress: float


# Default metrics when nothing is running yet (used by session.py)
IDLE_METRICS: Metrics = {
    "reps": 0,
    "stage": "-",
    "side": None,
    "status": "idle",
    "elbow": None,
    "eye_height": None,
    "speed": None,
    "touch_y0": None,
    "touch_y1": None,
    "posture_angle": None,
    "posture_status": None,
    "phase": "idle",
    "calib_progress": 0.0,
}


# =============================================================================
# MATH HELPERS
# =============================================================================


def to_pixel(point: Point) -> tuple[int, int]:
    """Convert a point to integer pixel coords for OpenCV drawing."""
    return (int(point[0]), int(point[1]))


def joint_angle(point_a: Point, point_b: Point, point_c: Point) -> Optional[float]:
    """Angle at point_b formed by points a-b-c, in degrees. None if degenerate."""
    vector_ba = point_a[:2] - point_b[:2]
    vector_bc = point_c[:2] - point_b[:2]
    length_ba = np.linalg.norm(vector_ba)
    length_bc = np.linalg.norm(vector_bc)
    if length_ba < 1e-6 or length_bc < 1e-6:
        return None
    cosine = np.clip(np.dot(vector_ba, vector_bc) / (length_ba * length_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


# =============================================================================
# SIDE DETECTION (which arm / ankle faces the camera)
# =============================================================================


def guess_side(keypoints: Keypoints) -> Optional[SideName]:
    """Guess left vs right from wrist/elbow confidence. None if too close."""
    left_confidence = (keypoints[7][2] + keypoints[9][2]) / 2
    right_confidence = (keypoints[8][2] + keypoints[10][2]) / 2
    if abs(right_confidence - left_confidence) < SIDE_MARGIN:
        return None
    return "right" if right_confidence > left_confidence else "left"


def pick_side_from_votes(
    keypoints: Keypoints,
    left_votes: int,
    right_votes: int,
) -> tuple[
    int, int, Optional[SideName], Optional[list[int]], Optional[tuple[int, int, int]]
]:
    """
    Keep voting until SIDE_VOTES_NEEDED, then lock side + arm + body chain.
    Returns: left_votes, right_votes, side, arm, chain
    """
    guess = guess_side(keypoints)
    if guess == "left":
        left_votes += 1
    elif guess == "right":
        right_votes += 1

    total_votes = left_votes + right_votes
    if total_votes < SIDE_VOTES_NEEDED:
        return left_votes, right_votes, None, None, None

    side: SideName = "right" if right_votes > left_votes else "left"
    if side == "right":
        return left_votes, right_votes, side, RIGHT_ARM, RIGHT_LINE
    return left_votes, right_votes, side, LEFT_ARM, LEFT_LINE


# =============================================================================
# SMOOTHING (Butterworth) — reduces keypoint jitter
# =============================================================================


def make_smoother(fps: Optional[float] = None) -> tuple[NDArray[np.float64], None]:
    """Build filter coeffs. Returns (sos, zi) with zi=None until first frame."""
    sample_rate = float(fps) if fps and fps > 1 else 30.0
    cutoff = min(BUTTER_CUTOFF_HZ, 0.45 * sample_rate)  # must stay below Nyquist
    sos = butter(BUTTER_ORDER, cutoff, btype="low", fs=sample_rate, output="sos")
    return sos, None


def smooth_xy(
    xy: KeypointXY,
    sos: NDArray[np.float64],
    zi: Optional[NDArray[np.float64]],
) -> tuple[KeypointXY, NDArray[np.float64]]:
    """
    Filter one frame of keypoints. xy shape: (num_keypoints, 2).
    Returns (smoothed_xy, updated_zi). Pass zi=None to re-seed after gaps.

    Uses one vectorized sosfilt call over all keypoints/axes (same math as
    filtering each keypoint axis one sample at a time).
    """
    xy = np.asarray(xy, dtype=np.float64)
    num_keypoints, num_axes = xy.shape

    if zi is None:
        # Seed filter memory so the first output equals the first input (no jump)
        base_zi = sosfilt_zi(sos)
        zi = np.zeros((num_keypoints, num_axes) + base_zi.shape, dtype=np.float64)
        for keypoint_index in range(num_keypoints):
            for axis_index in range(num_axes):
                zi[keypoint_index, axis_index] = (
                    base_zi * xy[keypoint_index, axis_index]
                )
        return xy.copy(), zi

    # One sosfilt call for all keypoint axes.
    # Input shape (num_series, 1); SciPy expects zi shape (n_sections, num_series, 2).
    # Stored zi layout is (num_keypoints, num_axes, n_sections, 2).
    num_sections = sos.shape[0]
    num_series = num_keypoints * num_axes
    flat_xy = xy.reshape(num_series, 1)
    zi_for_filter = np.moveaxis(
        zi.reshape(num_series, num_sections, 2), 0, 1
    )  # (n_sections, num_series, 2)
    filtered, zi_out = sosfilt(sos, flat_xy, axis=-1, zi=zi_for_filter)
    smoothed = filtered.reshape(num_keypoints, num_axes)
    updated_zi = np.moveaxis(zi_out, 0, 1).reshape(
        num_keypoints, num_axes, num_sections, 2
    )
    return smoothed, updated_zi


# =============================================================================
# FACE POINT + HEIGHT ABOVE ANKLE
# =============================================================================


def eye_point(keypoints: Keypoints) -> Optional[Point]:
    """Best face point: average of both eyes, else one eye, else nose."""
    left_eye, right_eye, nose = (
        keypoints[LEFT_EYE],
        keypoints[RIGHT_EYE],
        keypoints[NOSE],
    )
    left_ok = left_eye[2] >= CONF_MIN
    right_ok = right_eye[2] >= CONF_MIN
    if left_ok and right_ok:
        return (left_eye[:2] + right_eye[:2]) / 2.0
    if left_ok:
        return left_eye[:2].copy()
    if right_ok:
        return right_eye[:2].copy()
    if nose[2] >= CONF_MIN:
        return nose[:2].copy()
    return None


def eye_height_above_ankle(eye: Point, ankle: Point) -> float:
    """
    How far the eye sits above the ankle, in pixels.
    Larger = higher off the floor (image Y grows downward).
    """
    return float(ankle[1] - eye[1])


# =============================================================================
# CALIBRATION + REP COUNTING
#
# Rules:
#   - Webcam (calib_mode="hold"): hold top pose for HOLD_CALIB_POSE_TIME → ready
#   - Video (calib_mode="auto"): N consecutive extended-elbow frames → active
#   - Freeze ankle_origin (y0) + y1 from samples
#   - stage "down" when eye near y=0; after down, elbow >= UP_ELBOW → +1 rep
#   - Rep counting only when phase == "active"
# =============================================================================


def _lock_calibration_guides(tracker: TrackerState, next_phase: PhaseName) -> None:
    """Freeze ankle_origin + y1 from collected samples and advance phase."""
    tracker["ankle_origin"] = np.median(tracker["calib_ankle_samples"], axis=0)
    tracker["y1"] = max(float(np.median(tracker["calib_h_samples"])), MIN_Y1_PX)
    tracker["phase"] = next_phase
    tracker["calib_progress"] = 1.0


def update_calibrate_ankle(
    tracker: TrackerState,
    ankle: Optional[Point],
    eye_height_px: Optional[float],
    elbow: Optional[float],
) -> None:
    """
    Calibration. Locks ankle_origin + y1.

    hold → phase ready (webcam waits for POST /api/go).
    auto → phase active (video starts counting immediately).
    """
    if tracker["side"] is not None and tracker["phase"] == "detecting":
        tracker["phase"] = "calibrating"
        tracker["calib_progress"] = 0.0
        return

    if tracker["phase"] != "calibrating":
        return

    if (
        ankle is None
        or eye_height_px is None
        or elbow is None
        or elbow < CALIB_ELBOW_MIN
    ):
        tracker["calib_ankle_samples"] = []
        tracker["calib_h_samples"] = []
        tracker["calib_hold_start"] = None
        tracker["calib_progress"] = 0.0
        return

    tracker["calib_ankle_samples"].append(
        np.asarray(ankle[:2], dtype=np.float64).copy()
    )
    tracker["calib_h_samples"].append(float(eye_height_px))

    if tracker.get("calib_mode") == "auto":
        # Video: count consecutive valid top frames (no wall-clock hold)
        sample_count = len(tracker["calib_h_samples"])
        tracker["calib_progress"] = min(1.0, sample_count / VIDEO_AUTO_CALIB_FRAMES)
        if sample_count >= VIDEO_AUTO_CALIB_FRAMES:
            _lock_calibration_guides(tracker, "active")
        return

    # Webcam: timed hold at top
    now = time.time()
    if tracker["calib_hold_start"] is None:
        tracker["calib_hold_start"] = now

    elapsed = now - tracker["calib_hold_start"]
    tracker["calib_progress"] = min(1.0, elapsed / HOLD_CALIB_POSE_TIME)

    if elapsed >= HOLD_CALIB_POSE_TIME:
        _lock_calibration_guides(tracker, "ready")


def update_rep_count(
    eye_height_px: Optional[float],
    elbow: Optional[float],
    tracker: TrackerState,
) -> None:
    """Update stage / reps from eye height and elbow angle."""
    if eye_height_px is None:
        return

    # Eye (or nose) close to ankle line → bottom of the push-up
    if eye_height_px <= DOWN_TOUCH_PX:
        tracker["stage"] = "down"
        return

    # Coming up from a down: arms lock out → count one rep
    if tracker["stage"] == "down" and elbow is not None and elbow >= UP_ELBOW:
        tracker["reps"] += 1
        tracker["stage"] = "up"


# =============================================================================
# DRAWING OVERLAYS (mutates the frame in place)
# =============================================================================


def draw_ankle_axes(
    frame: NDArray[np.uint8],
    ankle: Point,
    eye: Optional[Point],
    y1: Optional[float],
    touch_y0: bool = False,
    touch_y1: bool = False,
) -> None:
    """Full-width y=0 / y1 guides from the (frozen) ankle toward the head."""
    height, width = frame.shape[:2]
    ankle_x, ankle_y = float(ankle[0]), float(ankle[1])
    red = (0, 0, 255)
    black = (20, 20, 20)
    green = (0, 255, 0)

    if eye is not None and abs(float(eye[0]) - ankle_x) > 1.0:
        toward_head = 1 if float(eye[0]) > ankle_x else -1
    else:
        toward_head = 1

    top_y = ankle_y - (y1 if y1 is not None else AXIS_LINE_LEN)
    cv2.arrowedLine(
        frame,
        to_pixel((ankle_x, ankle_y)),
        to_pixel((ankle_x, top_y)),
        black,
        3,
        tipLength=0.12,
    )

    def draw_level(y: float, label: str, color: tuple[int, int, int]) -> None:
        y_i = int(round(y))
        y_i = max(0, min(height - 1, y_i))
        cv2.line(frame, (0, y_i), (width - 1, y_i), color, 2)
        if toward_head > 0:
            cv2.arrowedLine(
                frame, (int(ankle_x), y_i), (width - 1, y_i), color, 3, tipLength=0.015
            )
            label_x = max(8, int(ankle_x) - 55)
        else:
            cv2.arrowedLine(
                frame, (int(ankle_x), y_i), (0, y_i), color, 3, tipLength=0.015
            )
            label_x = min(width - 40, int(ankle_x) + 12)
        cv2.putText(
            frame,
            label,
            (label_x, y_i - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            red,
            2,
        )

    y0_color = green if touch_y0 else red
    draw_level(ankle_y, "y=0", y0_color)

    y1_color = green if touch_y1 else red
    if y1 is not None:
        draw_level(ankle_y - y1, "y1", y1_color)

    # Position vector from y=0 (ankle) to the eye
    if eye is not None:
        cyan = (255, 255, 0)
        cv2.arrowedLine(frame, to_pixel(ankle), to_pixel(eye), cyan, 3, tipLength=0.08)

    cv2.circle(frame, to_pixel(ankle), 8, (0, 255, 0), -1)
    if eye is not None:
        cv2.circle(frame, to_pixel(eye), 8, (0, 255, 0), -1)


def draw_posture(
    frame: NDArray[np.uint8],
    keypoints: Keypoints,
    chain: tuple[int, int, int],
    tracker: TrackerState,
) -> str:
    """Shoulder–hip–ankle alignment. Sets tracker posture fields; returns status label."""
    shoulder_index, hip_index, ankle_index = chain
    if (
        min(
            keypoints[shoulder_index][2],
            keypoints[hip_index][2],
            keypoints[ankle_index][2],
        )
        < CONF_MIN
    ):
        tracker["posture_angle"] = None
        tracker["posture_status"] = None
        return "low conf"

    shoulder = keypoints[shoulder_index][:2]
    hip = keypoints[hip_index][:2]
    ank = keypoints[ankle_index][:2]
    angle = joint_angle(shoulder, hip, ank)
    if angle is None:
        tracker["posture_angle"] = None
        tracker["posture_status"] = None
        return "low conf"

    good = angle >= 180.0 - STRAIGHT_TOL
    color = (0, 255, 0) if good else (0, 0, 255)

    tracker["posture_angle"] = round(float(angle), 1)
    tracker["posture_status"] = "good" if good else "bad"

    shoulder_to_ankle = ank - shoulder
    projection_t = np.clip(
        np.dot(hip - shoulder, shoulder_to_ankle)
        / max(np.dot(shoulder_to_ankle, shoulder_to_ankle), 1e-6),
        0.0,
        1.0,
    )
    on_line = shoulder + projection_t * shoulder_to_ankle

    if good:
        label = f"GOOD Posture: {angle:.0f}°"
    elif hip[1] > on_line[1]:
        label = f"Hips sagging: {angle:.0f}°"
    else:
        label = f"Hips Too High: {angle:.0f}°"

    cv2.line(frame, to_pixel(shoulder), to_pixel(ank), color, 4)
    cv2.line(frame, to_pixel(shoulder), to_pixel(hip), (255, 255, 0), 2)
    cv2.line(frame, to_pixel(hip), to_pixel(ank), (255, 255, 0), 2)
    cv2.line(frame, to_pixel(hip), to_pixel(on_line), (0, 255, 255), 2)
    for point in (shoulder, hip, ank):
        cv2.circle(frame, to_pixel(point), 8, color, -1)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return label


def draw_arm(
    frame: NDArray[np.uint8],
    keypoints: Keypoints,
    arm: list[int],
    elbow: Optional[float],
) -> None:
    """Magenta shoulder–elbow–wrist stick figure + elbow degrees."""
    shoulder = keypoints[arm[0]][:2]
    elbow_pt = keypoints[arm[1]][:2]
    wrist = keypoints[arm[2]][:2]
    cv2.line(frame, to_pixel(shoulder), to_pixel(elbow_pt), (255, 0, 255), 3)
    cv2.line(frame, to_pixel(elbow_pt), to_pixel(wrist), (255, 0, 255), 3)
    for point in (shoulder, elbow_pt, wrist):
        cv2.circle(frame, to_pixel(point), 6, (255, 0, 255), -1)
    if elbow is not None:
        cv2.putText(
            frame,
            f"Elbow {elbow:.0f}deg",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2,
        )


def update_meters_per_px(
    keypoints: Keypoints,
    chain: tuple[int, int, int],
    tracker: TrackerState,
) -> Optional[float]:
    """EMA-smooth m/px from shoulder–hip vs TORSO_LENGTH_M. Returns m_per_px or None."""
    shoulder_index, hip_index, _ankle_index = chain
    if min(keypoints[shoulder_index][2], keypoints[hip_index][2]) < CONF_MIN:
        return tracker.get("m_per_px")

    shoulder = keypoints[shoulder_index][:2]
    hip = keypoints[hip_index][:2]
    distance_px = float(np.linalg.norm(shoulder - hip))
    if distance_px < 1.0:
        return tracker.get("m_per_px")

    sample = TORSO_LENGTH_M / distance_px
    previous = tracker.get("m_per_px")
    if previous is None:
        tracker["m_per_px"] = sample
    else:
        tracker["m_per_px"] = (1.0 - SCALE_EMA) * previous + SCALE_EMA * sample
    return tracker["m_per_px"]


def update_eye_speed(
    eye_height_px: Optional[float],
    tracker: TrackerState,
) -> Optional[float]:
    """
    Vertical speed in m/s from eye height above ankle.
    speed = (eye_height_px - prev_h) * m_per_px * fps. Positive = rising toward y1.
    """
    if eye_height_px is None:
        tracker["prev_h"] = None
        return None

    previous_height = tracker.get("prev_h")
    tracker["prev_h"] = float(eye_height_px)
    if previous_height is None:
        return None

    meters_per_px = tracker.get("m_per_px")
    fps = float(tracker.get("fps") or 30.0)
    if meters_per_px is None or meters_per_px <= 0:
        return None

    speed_mps = (float(eye_height_px) - float(previous_height)) * meters_per_px * fps
    speed = round(speed_mps, 2)
    tracker["speed"] = speed
    return speed


def draw_eye_speed(
    frame: NDArray[np.uint8],
    eye: Optional[Point],
    speed: Optional[float],
) -> None:
    """Short velocity arrow at the eye + m/s label."""
    if eye is None or speed is None:
        return

    # Scale arrow for display (m/s is small); keep direction of motion
    tip = np.asarray(eye, dtype=np.float64).copy()
    tip[1] -= speed * 40.0
    color = (0, 255, 0) if speed >= 0 else (0, 165, 255)
    cv2.arrowedLine(frame, to_pixel(eye), to_pixel(tip), color, 3, tipLength=0.35)
    cv2.putText(
        frame,
        f"Speed {speed:.2f} m/s",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )


# =============================================================================
# TRACKER STATE + METRICS (session calls init_tracker / analyze_frame)
# =============================================================================


def init_tracker(
    fps: Optional[float] = None,
    calib_mode: CalibMode = "hold",
) -> TrackerState:
    """
    Per-session memory. Add new keys here when a helper needs history.

    calib_mode:
      hold — webcam: timed 2s hold → ready (needs POST /api/go for reps)
      auto — video: N consecutive top frames → active (reps start immediately)

    Keys you will touch most often:
      stage, reps     — push-up FSM
      y1              — upper guide (set once on first unlock)
      ankle_origin    — frozen y=0 point
      eye_height      — shown in the UI
      sos, zi         — Butterworth filter state (reset zi when person lost)
    """
    sos, zi = make_smoother(fps)
    sample_rate = float(fps) if fps and fps > 1 else 30.0
    return {
        # Side lock
        "left_votes": 0,
        "right_votes": 0,
        "side": None,
        "arm": None,  # [shoulder, elbow, wrist] indices
        "chain": None,  # (shoulder, hip, ankle) indices
        # Smoothing
        "sos": sos,
        "zi": zi,
        "fps": sample_rate,
        # Reps
        "stage": "-",  # "-", "down", or "up"
        "reps": 0,
        # Calibration / game phases
        "ankle_origin": None,  # frozen y=0
        "y1": None,  # frozen at end of hold-at-top calib
        "phase": "detecting",  # detecting | calibrating | ready | active
        "calib_mode": calib_mode,
        "calib_progress": 0.0,
        "calib_hold_start": None,
        "calib_ankle_samples": [],
        "calib_h_samples": [],
        # Overlays / UI
        "eye_height": None,
        "speed": None,  # m/s
        "prev_h": None,  # last eye height px (for speed)
        "m_per_px": None,  # meters per pixel from torso scale
        "posture_angle": None,
        "posture_status": None,
        "touch_y0": None,
        "touch_y1": None,
    }


def metrics_from_tracker(
    tracker: TrackerState,
    status: str,
    elbow: Optional[float] = None,
) -> Metrics:
    """Pack values the API sends to the frontend. Add new UI fields here."""
    return {
        "reps": tracker["reps"],
        "stage": tracker["stage"],
        "side": tracker["side"],
        "status": status,
        "elbow": None if elbow is None else round(float(elbow), 1),
        "eye_height": tracker.get("eye_height"),
        "speed": tracker.get("speed"),
        "touch_y0": tracker.get("touch_y0") or None,
        "touch_y1": tracker.get("touch_y1") or None,
        "posture_angle": tracker.get("posture_angle"),
        "posture_status": tracker.get("posture_status"),
        "phase": tracker.get("phase"),
        "calib_progress": round(float(tracker.get("calib_progress") or 0.0), 2),
    }


def analyze_frame(
    frame: NDArray[np.uint8],
    keypoints: Keypoints,
    tracker: TrackerState,
) -> Metrics:
    """
    Full per-frame pipeline. Mutates frame + tracker; returns metrics dict.

    keypoints: (17, 3) array of [x, y, confidence] from YOLO.
    """
    # --- 1) Lock which side faces the camera ---
    if tracker["side"] is None:
        left_votes, right_votes, side, arm, chain = pick_side_from_votes(
            keypoints, tracker["left_votes"], tracker["right_votes"]
        )
        tracker["left_votes"] = left_votes
        tracker["right_votes"] = right_votes
        tracker["side"] = side
        tracker["arm"] = arm
        tracker["chain"] = chain

    # --- 2) Smooth keypoints ---
    smoothed_xy, tracker["zi"] = smooth_xy(
        keypoints[:, :2], tracker["sos"], tracker["zi"]
    )
    keypoints[:, :2] = smoothed_xy

    if tracker["arm"] is None:
        return metrics_from_tracker(tracker, "detecting side...")

    chain = tracker["chain"]
    arm = tracker["arm"]
    assert chain is not None and arm is not None
    ankle_index = chain[2]

    # --- 3) Elbow angle (used for "up" / unlock) ---
    elbow = joint_angle(keypoints[arm[0]], keypoints[arm[1]], keypoints[arm[2]])

    # --- 4) Eye height, hold-at-top calib (y0+y1), rep count ---
    eye = eye_point(keypoints)
    if eye is not None and keypoints[ankle_index][2] >= CONF_MIN:
        ankle_live = keypoints[ankle_index][:2]
        ankle_ref = (
            tracker["ankle_origin"]
            if tracker["ankle_origin"] is not None
            else ankle_live
        )
        eye_height_px = eye_height_above_ankle(eye, ankle_ref)

        # During calib, height is measured against live ankle until origin freezes
        update_calibrate_ankle(tracker, ankle_live, eye_height_px, elbow)

        # Refresh ref/height if calib just locked ankle_origin this frame
        if tracker["ankle_origin"] is not None:
            ankle_ref = tracker["ankle_origin"]
            eye_height_px = eye_height_above_ankle(eye, ankle_ref)

        # Rep counting only after POST /api/go moves phase to "active"
        if tracker["phase"] == "active":
            update_rep_count(eye_height_px, elbow, tracker)

        if eye_height_px is not None:
            tracker["touch_y0"] = eye_height_px <= DOWN_TOUCH_PX
        else:
            tracker["touch_y0"] = False

        y1 = tracker["y1"]

        if y1 is not None and y1 > 1e-6:
            tracker["eye_height"] = round(eye_height_px / y1, 2)
            tracker["touch_y1"] = eye_height_px >= y1
        else:
            tracker["eye_height"] = round(eye_height_px, 1)
            tracker["touch_y1"] = False

        # --- 5a) Scale, speed (m/s), y0→eye vector + velocity arrow ---
        update_meters_per_px(keypoints, chain, tracker)
        speed = update_eye_speed(eye_height_px, tracker)
        draw_ankle_axes(
            frame,
            ankle_ref,
            eye,
            y1,
            touch_y0=bool(tracker["touch_y0"]),
            touch_y1=bool(tracker["touch_y1"]),
        )
        draw_eye_speed(frame, eye, speed)
    else:
        # Lost person / ankle → hard-reset hold if still calibrating
        update_calibrate_ankle(tracker, None, None, elbow)
        tracker["eye_height"] = None
        update_eye_speed(None, tracker)

    # --- 5b) Draw arm + posture ---
    draw_arm(frame, keypoints, arm, elbow)
    status = draw_posture(frame, keypoints, chain, tracker)
    if tracker["phase"] == "calibrating":
        if tracker.get("calib_mode") == "auto":
            status = f"calibrating video... {int(tracker['calib_progress'] * 100)}%"
        else:
            status = f"calibrating... {int(tracker['calib_progress'] * 100)}%"
    elif tracker["phase"] == "ready":
        status = "ready"
    elif tracker["phase"] == "detecting":
        status = "detecting side..."

    # --- 6) Metrics for the API ---
    return metrics_from_tracker(tracker, status, elbow)
