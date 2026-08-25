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

import cv2
import numpy as np
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
# Up   = after a down, elbow angle unlocks past UP_ELBOW (sets y1 once)
UP_ELBOW = 145.0  # degrees; "arms locked out" at the top
DOWN_TOUCH_PX = 25.0  # eye within this many pixels of ankle height = down
MIN_Y1_PX = 40.0  # minimum length for the y1 guide line

# --- Ankle y=0 calibration ---
ANKLE_CALIB_FRAMES = 10  # collect this many frames, then freeze ankle
ANKLE_CALIB_LOCK_FRAMES = 15  # only average the last N (skip settling)

# --- Posture overlay ---
CONF_MIN = 0.4  # ignore joints below this confidence
STRAIGHT_TOL = 15.0  # degrees off 180 still counts as "straight" body line
AXIS_LINE_LEN = 120  # fallback vertical length before y1 exists

# --- Speed (m/s) from eye-height change ---
TORSO_LENGTH_M = 0.45  # assumed shoulder–hip length in meters
SCALE_EMA = 0.15  # how fast m_per_px adapts (0–1)

# Default metrics when nothing is running yet (used by session.py)
IDLE_METRICS = {
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
}


# =============================================================================
# MATH HELPERS
# =============================================================================


def pt(p):
    """Convert a point to integer pixel coords for OpenCV drawing."""
    return (int(p[0]), int(p[1]))


def joint_angle(a, b, c):
    """Angle at point b formed by points a-b-c, in degrees. None if degenerate."""
    ba = a[:2] - b[:2]
    bc = c[:2] - b[:2]
    na = np.linalg.norm(ba)
    nc = np.linalg.norm(bc)
    if na < 1e-6 or nc < 1e-6:
        return None
    cos = np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


# =============================================================================
# SIDE DETECTION (which arm / ankle faces the camera)
# =============================================================================


def user_pick_side(user_answer):
    if user_answer == "left":
        return "left"
    elif user_answer == "right":
        return "right"
    else:
        return "Error: Invalid user answer"


def guess_side(kpts):
    """Guess left vs right from wrist/elbow confidence. None if too close."""
    left = (kpts[7][2] + kpts[9][2]) / 2
    right = (kpts[8][2] + kpts[10][2]) / 2
    if abs(right - left) < SIDE_MARGIN:
        return None
    return "right" if right > left else "left"


def pick_side_from_votes(kpts, left_votes, right_votes):
    """
    Keep voting until SIDE_VOTES_NEEDED, then lock side + arm + body chain.
    Returns: left_votes, right_votes, side, arm, chain
    """
    guess = guess_side(kpts)
    if guess == "left":
        left_votes += 1
    elif guess == "right":
        right_votes += 1

    total = left_votes + right_votes
    if total < SIDE_VOTES_NEEDED:
        return left_votes, right_votes, None, None, None

    side = "right" if right_votes > left_votes else "left"
    if side == "right":
        return left_votes, right_votes, side, RIGHT_ARM, RIGHT_LINE
    return left_votes, right_votes, side, LEFT_ARM, LEFT_LINE


# =============================================================================
# SMOOTHING (Butterworth) — reduces keypoint jitter
# =============================================================================


def make_smoother(fps=None):
    """Build filter coeffs. Returns (sos, zi) with zi=None until first frame."""
    fs = float(fps) if fps and fps > 1 else 30.0
    cutoff = min(BUTTER_CUTOFF_HZ, 0.45 * fs)  # must stay below Nyquist
    sos = butter(BUTTER_ORDER, cutoff, btype="low", fs=fs, output="sos")
    return sos, None


def smooth_xy(xy, sos, zi):
    """
    Filter one frame of keypoints. xy shape: (num_keypoints, 2).
    Returns (smoothed_xy, updated_zi). Pass zi=None to re-seed after gaps.
    """
    xy = np.asarray(xy, dtype=np.float64)
    n_kpts, n_axes = xy.shape
    out = np.empty_like(xy)

    if zi is None:
        base = sosfilt_zi(sos)
        zi = np.zeros((n_kpts, n_axes) + base.shape, dtype=np.float64)
        for i in range(n_kpts):
            for j in range(n_axes):
                zi[i, j] = base * xy[i, j]
                out[i, j] = xy[i, j]
        return out, zi

    for i in range(n_kpts):
        for j in range(n_axes):
            filtered, zi[i, j] = sosfilt(sos, [xy[i, j]], zi=zi[i, j])
            out[i, j] = filtered[0]
    return out, zi


# =============================================================================
# FACE POINT + HEIGHT ABOVE ANKLE
# =============================================================================


def eye_point(kpts):
    """Best face point: average of both eyes, else one eye, else nose."""
    left, right, nose = kpts[LEFT_EYE], kpts[RIGHT_EYE], kpts[NOSE]
    left_ok = left[2] >= CONF_MIN
    right_ok = right[2] >= CONF_MIN
    if left_ok and right_ok:
        return (left[:2] + right[:2]) / 2.0
    if left_ok:
        return left[:2].copy()
    if right_ok:
        return right[:2].copy()
    if nose[2] >= CONF_MIN:
        return nose[:2].copy()
    return None


def eye_height_above_ankle(eye, ankle):
    """
    How far the eye sits above the ankle, in pixels.
    Larger = higher off the floor (image Y grows downward).
    """
    return float(ankle[1] - eye[1])


# =============================================================================
# ANKLE CALIBRATION + REP COUNTING
#
# Rules:
#   - Freeze ankle once → stable red y=0 line
#   - stage "down" when eye near that line
#   - after down, elbow >= UP_ELBOW → +1 rep, stage "up"
#   - y1 (upper red line) locks on the FIRST unlock only
# =============================================================================


def calibrate_ankle(tracker, ankle):
    """Freeze ankle_origin for a stable y=0. Does not set y1."""
    if tracker["ankle_origin"] is not None or ankle is None:
        return tracker["ankle_origin"]

    tracker["ankle_samples"].append(np.asarray(ankle[:2], dtype=np.float64).copy())
    if len(tracker["ankle_samples"]) < ANKLE_CALIB_FRAMES:
        return None

    ankle_tail = tracker["ankle_samples"][-ANKLE_CALIB_LOCK_FRAMES:]
    tracker["ankle_origin"] = np.median(ankle_tail, axis=0)
    return tracker["ankle_origin"]


def update_rep_count(h, elbow, tracker):
    """Update stage / reps / y1 from eye height h and elbow angle."""
    if h is None:
        return

    # Eye (or nose) close to ankle line → bottom of the push-up
    if h <= DOWN_TOUCH_PX:
        tracker["stage"] = "down"
        return

    # Coming up from a down: arms lock out → count one rep
    if tracker["stage"] == "down" and elbow is not None and elbow >= UP_ELBOW:
        if tracker["y1"] is None:
            tracker["y1"] = max(float(h), MIN_Y1_PX)
        tracker["reps"] += 1
        tracker["stage"] = "up"


# =============================================================================
# DRAWING OVERLAYS (mutates the frame in place)
# =============================================================================


def draw_ankle_axes(
    frame,
    ankle,
    eye,
    y1,
    touch_y0=False,
    touch_y1=False,
):
    """Full-width y=0 / y1 guides from the (frozen) ankle toward the head."""
    height, width = frame.shape[:2]
    ax, ay = float(ankle[0]), float(ankle[1])
    red = (0, 0, 255)
    black = (20, 20, 20)
    green = (0, 255, 0)

    if eye is not None and abs(float(eye[0]) - ax) > 1.0:
        toward_head = 1 if float(eye[0]) > ax else -1
    else:
        toward_head = 1

    top_y = ay - (y1 if y1 is not None else AXIS_LINE_LEN)
    cv2.arrowedLine(frame, pt((ax, ay)), pt((ax, top_y)), black, 3, tipLength=0.12)

    def draw_level(y, label, color):
        y_i = int(round(y))
        y_i = max(0, min(height - 1, y_i))
        cv2.line(frame, (0, y_i), (width - 1, y_i), color, 2)
        if toward_head > 0:
            cv2.arrowedLine(
                frame, (int(ax), y_i), (width - 1, y_i), color, 3, tipLength=0.015
            )
            label_x = max(8, int(ax) - 55)
        else:
            cv2.arrowedLine(frame, (int(ax), y_i), (0, y_i), color, 3, tipLength=0.015)
            label_x = min(width - 40, int(ax) + 12)
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
    draw_level(ay, "y=0", y0_color)

    y1_color = green if touch_y1 else red
    if y1 is not None:
        draw_level(ay - y1, "y1", y1_color)

    # Position vector from y=0 (ankle) to the eye
    if eye is not None:
        cyan = (255, 255, 0)
        cv2.arrowedLine(frame, pt(ankle), pt(eye), cyan, 3, tipLength=0.08)

    cv2.circle(frame, pt(ankle), 8, (0, 255, 0), -1)
    if eye is not None:
        cv2.circle(frame, pt(eye), 8, (0, 255, 0), -1)


def draw_posture(frame, kpts, chain, tracker):
    """Shoulder–hip–ankle alignment. Sets tracker posture fields; returns status label."""
    i_sh, i_hip, i_ank = chain
    if min(kpts[i_sh][2], kpts[i_hip][2], kpts[i_ank][2]) < CONF_MIN:
        tracker["posture_angle"] = None
        tracker["posture_status"] = None
        return "low conf"

    sh, hip, ank = kpts[i_sh][:2], kpts[i_hip][:2], kpts[i_ank][:2]
    angle = joint_angle(sh, hip, ank)
    if angle is None:
        tracker["posture_angle"] = None
        tracker["posture_status"] = None
        return "low conf"

    good = angle >= 180.0 - STRAIGHT_TOL
    color = (0, 255, 0) if good else (0, 0, 255)

    tracker["posture_angle"] = round(float(angle), 1)
    tracker["posture_status"] = "good" if good else "bad"

    ac = ank - sh
    t = np.clip(np.dot(hip - sh, ac) / max(np.dot(ac, ac), 1e-6), 0.0, 1.0)
    on_line = sh + t * ac

    if good:
        label = f"GOOD Posture: {angle:.0f}°"
    elif hip[1] > on_line[1]:
        label = f"Hips sagging: {angle:.0f}°"
    else:
        label = f"Hips Too High: {angle:.0f}°"

    cv2.line(frame, pt(sh), pt(ank), color, 4)
    cv2.line(frame, pt(sh), pt(hip), (255, 255, 0), 2)
    cv2.line(frame, pt(hip), pt(ank), (255, 255, 0), 2)
    cv2.line(frame, pt(hip), pt(on_line), (0, 255, 255), 2)
    for p in (sh, hip, ank):
        cv2.circle(frame, pt(p), 8, color, -1)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return label


def draw_arm(frame, kpts, arm, elbow):
    """Magenta shoulder–elbow–wrist stick figure + elbow degrees."""
    sh, el, wr = kpts[arm[0]][:2], kpts[arm[1]][:2], kpts[arm[2]][:2]
    cv2.line(frame, pt(sh), pt(el), (255, 0, 255), 3)
    cv2.line(frame, pt(el), pt(wr), (255, 0, 255), 3)
    for p in (sh, el, wr):
        cv2.circle(frame, pt(p), 6, (255, 0, 255), -1)
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


def update_meters_per_px(kpts, chain, tracker):
    """EMA-smooth m/px from shoulder–hip vs TORSO_LENGTH_M. Returns m_per_px or None."""
    i_sh, i_hip, _i_ank = chain
    if min(kpts[i_sh][2], kpts[i_hip][2]) < CONF_MIN:
        return tracker.get("m_per_px")

    sh = kpts[i_sh][:2]
    hip = kpts[i_hip][:2]
    dist_px = float(np.linalg.norm(sh - hip))
    if dist_px < 1.0:
        return tracker.get("m_per_px")

    sample = TORSO_LENGTH_M / dist_px
    prev = tracker.get("m_per_px")
    if prev is None:
        tracker["m_per_px"] = sample
    else:
        tracker["m_per_px"] = (1.0 - SCALE_EMA) * prev + SCALE_EMA * sample
    return tracker["m_per_px"]


def update_eye_speed(h, tracker):
    """
    Vertical speed in m/s from eye height above ankle.
    speed = (h - prev_h) * m_per_px * fps. Positive = rising toward y1.
    """
    if h is None:
        tracker["prev_h"] = None
        return None

    prev_h = tracker.get("prev_h")
    tracker["prev_h"] = float(h)
    if prev_h is None:
        return None

    m_per_px = tracker.get("m_per_px")
    fps = float(tracker.get("fps") or 30.0)
    if m_per_px is None or m_per_px <= 0:
        return None

    speed_mps = (float(h) - float(prev_h)) * m_per_px * fps
    speed = round(speed_mps, 2)
    tracker["speed"] = speed
    return speed


def draw_eye_speed(frame, eye, ankle, speed):
    """Short velocity arrow at the eye + m/s label."""
    if eye is None or speed is None:
        return

    _ = ankle
    # Scale arrow for display (m/s is small); keep direction of motion
    tip = np.asarray(eye, dtype=np.float64).copy()
    tip[1] -= speed * 40.0
    color = (0, 255, 0) if speed >= 0 else (0, 165, 255)
    cv2.arrowedLine(frame, pt(eye), pt(tip), color, 3, tipLength=0.35)
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


def init_tracker(fps=None):
    """
    Per-session memory. Add new keys here when a helper needs history.

    Keys you will touch most often:
      stage, reps     — push-up FSM
      y1              — upper guide (set once on first unlock)
      ankle_origin    — frozen y=0 point
      eye_height      — shown in the UI
      sos, zi         — Butterworth filter state (reset zi when person lost)
    """
    sos, zi = make_smoother(fps)
    fs = float(fps) if fps and fps > 1 else 30.0
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
        "fps": fs,
        # Reps
        "stage": "-",  # "-", "down", or "up"
        "reps": 0,
        "y1": None,  # locked after first successful up
        # Ankle calibration
        "ankle_origin": None,
        "ankle_samples": [],
        # Overlays / UI
        "eye_height": None,
        "speed": None,  # m/s
        "prev_h": None,  # last eye height px (for speed)
        "m_per_px": None,  # meters per pixel from torso scale
        "posture_angle": None,
        "posture_status": None,
    }


def metrics_from_tracker(tracker, status, elbow=None):
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
    }


def analyze_frame(frame, kpts, tracker):
    """
    Full per-frame pipeline. Mutates frame + tracker; returns metrics dict.

    kpts: (17, 3) array of [x, y, confidence] from YOLO.
    """
    # --- 1) Lock which side faces the camera ---
    if tracker["side"] is None:
        left, right, side, arm, chain = pick_side_from_votes(
            kpts, tracker["left_votes"], tracker["right_votes"]
        )
        tracker["left_votes"] = left
        tracker["right_votes"] = right
        tracker["side"] = side
        tracker["arm"] = arm
        tracker["chain"] = chain

    # --- 2) Smooth keypoints ---
    xy, tracker["zi"] = smooth_xy(kpts[:, :2], tracker["sos"], tracker["zi"])
    kpts[:, :2] = xy

    if tracker["arm"] is None:
        return metrics_from_tracker(tracker, "detecting side...")

    chain = tracker["chain"]
    arm = tracker["arm"]
    i_ank = chain[2]

    # --- 3) Elbow angle (used for "up" / unlock) ---
    elbow = joint_angle(kpts[arm[0]], kpts[arm[1]], kpts[arm[2]])

    # --- 4) Eye height, ankle y=0, rep count ---
    eye = eye_point(kpts)
    if eye is not None and kpts[i_ank][2] >= CONF_MIN:
        ankle_live = kpts[i_ank][:2]
        if tracker["ankle_origin"] is None:
            calibrate_ankle(tracker, ankle_live)

        ankle_ref = (
            tracker["ankle_origin"]
            if tracker["ankle_origin"] is not None
            else ankle_live
        )
        h = eye_height_above_ankle(eye, ankle_ref)

        if tracker["ankle_origin"] is not None:
            update_rep_count(h, elbow, tracker)

        if h is not None:
            tracker["touch_y0"] = h <= DOWN_TOUCH_PX

        else:  # This is a fallback to ensure the touch_y0 is a boolean
            tracker["touch_y0"] = False

        y1 = tracker["y1"]

        if y1 is not None and y1 > 1e-6:
            tracker["eye_height"] = round(h / y1, 2)
            tracker["touch_y1"] = h >= y1

        else:
            tracker["eye_height"] = round(h, 1)
            tracker["touch_y1"] = False
        # --- 5a) Scale, speed (m/s), y0→eye vector + velocity arrow ---
        update_meters_per_px(kpts, chain, tracker)
        speed = update_eye_speed(h, tracker)
        draw_ankle_axes(
            frame,
            ankle_ref,
            eye,
            y1,
            touch_y0=tracker["touch_y0"],
            touch_y1=tracker["touch_y1"],
        )
        draw_eye_speed(frame, eye, ankle_ref, speed)
    else:
        tracker["eye_height"] = None
        update_eye_speed(None, tracker)

    # --- 5b) Draw arm + posture ---
    draw_arm(frame, kpts, arm, elbow)
    status = draw_posture(frame, kpts, chain, tracker)
    if (
        tracker["ankle_origin"] is None
        and eye is not None
        and kpts[i_ank][2] >= CONF_MIN
    ):
        status = "calibrating ankle..."

    # --- 6) Metrics for the API ---
    return metrics_from_tracker(tracker, status, elbow)
