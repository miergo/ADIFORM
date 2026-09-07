"""Unit tests for minimal pose.py (numpy only, no YOLO)."""

import numpy as np

from pose import (
    DOWN_BAR_TOL,
    HipPosture,
    Pushup,
    UP_BAR_TOL,
    closest_person,
    hit_depth_multiplier,
    is_down_pose,
    is_ready_pose,
    is_up_pose,
    joint_angle,
    pick_side,
    stroke_points,
)

# Spike / confidence thresholds used by jitter assertions.
# pose.py does not export these yet; keep literals aligned with the plan.
try:
    from pose import CONF_MIN, JUMP_PX
except ImportError:  # pragma: no cover - current minimal pose.py
    JUMP_PX = 50.0
    CONF_MIN = 0.3

_BOXES = np.array([[0, 0, 200, 400]], dtype=np.float64)
_JITTER_JOINTS = (5, 11, 13, 15)  # shoulder, hip, knee, ankle


def _blank_keypoints() -> np.ndarray:
    """One person, 17 joints, all zero confidence."""
    return np.zeros((1, 17, 3), dtype=np.float64)


def _set_joint(kpts: np.ndarray, idx: int, x: float, y: float, conf: float = 1.0) -> None:
    kpts[0, idx] = [x, y, conf]


def _left_up_pose(kpts: np.ndarray) -> None:
    """Straight left arm and leg, shoulder above hip (up plank)."""
    _set_joint(kpts, 5, 100, 100)   # shoulder
    _set_joint(kpts, 7, 150, 100)   # elbow (horizontal arm -> ~180 deg)
    _set_joint(kpts, 9, 200, 100)   # wrist
    _set_joint(kpts, 11, 100, 200)  # hip (below shoulder)
    _set_joint(kpts, 13, 100, 250)  # knee (between hip and ankle)
    _set_joint(kpts, 15, 100, 300)  # ankle


def _left_kneeling_pose(kpts: np.ndarray) -> None:
    """Torso up but knee bent (kneeling)."""
    _set_joint(kpts, 5, 100, 100)   # shoulder
    _set_joint(kpts, 7, 150, 100)   # elbow
    _set_joint(kpts, 9, 200, 100)   # wrist
    _set_joint(kpts, 11, 100, 200)  # hip
    _set_joint(kpts, 13, 100, 280)  # knee forward/down (bent leg)
    _set_joint(kpts, 15, 130, 300)  # ankle behind knee


def _left_down_pose(kpts: np.ndarray) -> None:
    """Shoulder at locked low bar (down position)."""
    _set_joint(kpts, 5, 100, 300)   # shoulder at low_bar
    _set_joint(kpts, 7, 100, 250)   # elbow
    _set_joint(kpts, 9, 130, 250)   # wrist
    _set_joint(kpts, 11, 100, 320)  # hip
    _set_joint(kpts, 13, 100, 335)  # knee
    _set_joint(kpts, 15, 100, 350)  # ankle moved (should not affect down detect)


def _left_down_sagging_pose(kpts: np.ndarray) -> None:
    """Down position with sagging hips."""
    _set_joint(kpts, 5, 100, 300)
    _set_joint(kpts, 7, 100, 250)
    _set_joint(kpts, 9, 130, 250)
    _set_joint(kpts, 11, 100, 380)  # hip well below shoulder-ankle line
    _set_joint(kpts, 13, 100, 365)
    _set_joint(kpts, 15, 100, 350)


def _left_partial_down_pose(kpts: np.ndarray, shoulder_y: float) -> None:
    """Between top and low bars with aligned body."""
    _set_joint(kpts, 5, 100, shoulder_y)
    _set_joint(kpts, 7, 150, shoulder_y)
    _set_joint(kpts, 9, 200, shoulder_y)
    _set_joint(kpts, 11, 100, shoulder_y + 100)
    _set_joint(kpts, 13, 100, shoulder_y + 150)
    _set_joint(kpts, 15, 100, 300)


def _jitter(
    kpts: np.ndarray,
    rng: np.random.Generator,
    sigma_px: float,
    joint_idxs,
    *,
    conf: float | None = None,
) -> None:
    """Add Gaussian noise to xy of listed joints; keep conf unless overridden."""
    noise = rng.normal(0.0, sigma_px, size=(len(joint_idxs), 2))
    for n, idx in enumerate(joint_idxs):
        kpts[0, idx, 0] += noise[n, 0]
        kpts[0, idx, 1] += noise[n, 1]
        if conf is not None:
            kpts[0, idx, 2] = conf


def _sitting_shoulder_bounce(kpts: np.ndarray, shoulder_y: float) -> None:
    """
    Sitting stick figure with bent hip/knee; only shoulder_y varies for bounce.

    After calib from _left_up_pose (top_bar=100, low_bar=300), shoulder_y near
    those bars can cross locked up/down bands while hips stay up (not a plank).
    """
    _set_joint(kpts, 5, 100, shoulder_y)   # shoulder (bounces)
    _set_joint(kpts, 7, 130, shoulder_y + 15)
    _set_joint(kpts, 9, 150, shoulder_y + 30)
    _set_joint(kpts, 11, 180, 160)         # hip forward / raised
    _set_joint(kpts, 13, 240, 180)         # knee bent ~90 deg
    _set_joint(kpts, 15, 200, 300)         # ankle near floor / low bar


def _calibrate(tracker: Pushup, boxes: np.ndarray | None = None) -> dict:
    """Lock bars from a clean left up plank."""
    if boxes is None:
        boxes = _BOXES
    kpts = _blank_keypoints()
    _left_up_pose(kpts)
    return tracker.update(boxes, kpts)


def _hold(
    tracker: Pushup,
    fill_pose,
    n: int = 1,
    boxes: np.ndarray | None = None,
) -> dict:
    """Apply fill_pose to blank keypoints and update tracker n times."""
    if boxes is None:
        boxes = _BOXES
    out: dict = {}
    for _ in range(n):
        kpts = _blank_keypoints()
        fill_pose(kpts)
        out = tracker.update(boxes, kpts)
    return out


class TestClosestPerson:
    def test_picks_largest_box(self):
        boxes = np.array(
            [
                [0, 0, 50, 50],    # area 2500
                [0, 0, 100, 100],  # area 10000
            ],
            dtype=np.float64,
        )
        assert closest_person(boxes) == 1

    def test_no_boxes(self):
        assert closest_person(None) is None
        assert closest_person(np.zeros((0, 4))) is None


class TestPickSide:
    def test_picks_higher_confidence_side(self):
        kpts = np.zeros((17, 3), dtype=np.float64)
        for idx in (5, 7, 9, 11, 13, 15):
            kpts[idx] = [0, 0, 0.2]
        for idx in (6, 8, 10, 12, 14, 16):
            kpts[idx] = [0, 0, 0.9]
        assert pick_side(kpts) == "right"

    def test_defaults_left_on_tie(self):
        kpts = np.zeros((17, 3), dtype=np.float64)
        kpts[:, 2] = 0.5
        assert pick_side(kpts) == "left"


class TestJointAngle:
    def test_straight_arm_is_near_180(self):
        shoulder = np.array([0, 0, 1.0])
        elbow = np.array([1, 0, 1.0])
        wrist = np.array([2, 0, 1.0])
        assert joint_angle(shoulder, elbow, wrist) == 180.0

    def test_right_angle(self):
        shoulder = np.array([0, 0, 1.0])
        elbow = np.array([1, 0, 1.0])
        wrist = np.array([1, 1, 1.0])
        assert joint_angle(shoulder, elbow, wrist) == 90.0


class TestPoses:
    def test_ready_pose(self):
        assert is_ready_pose(100, 200, 300, 175.0, 179.0, 179.0) is True
        assert is_ready_pose(100, 200, 300, 175.0, 180.0, 180.0) is True
        assert is_ready_pose(200, 100, 300, 175.0, 179.0, 179.0) is False
        assert is_ready_pose(100, 200, 90, 175.0, 179.0, 179.0) is False  # ankle above shoulder
        assert is_ready_pose(100, 200, 300, 140.0, 179.0, 179.0) is False  # elbow too bent
        assert is_ready_pose(100, 200, 300, 175.0, 164.0, 179.0) is False  # body not straight (≤180-BODY_ALIGN_TOL)
        assert is_ready_pose(100, 200, 300, 175.0, 179.0, 154.0) is False  # knee bent (≤180-KNEE_ANGLE_TOL)
        assert is_ready_pose(100, 200, 300, 175.0, None, 179.0) is False
        assert is_ready_pose(100, 200, 300, 175.0, 179.0, None) is False

    def test_up_pose_at_top_bar(self):
        top, low = 100.0, 300.0  # range 200, tol 30
        assert is_up_pose(100, top, low) is True
        assert is_up_pose(120, top, low) is True
        assert is_up_pose(200, top, low) is False

    def test_down_pose_at_low_bar(self):
        top, low = 100.0, 300.0  # range 200, tol 30
        assert is_down_pose(300, top, low) is True
        assert is_down_pose(280, top, low) is True
        assert is_down_pose(100, top, low) is False


class TestHipPosture:
    def _pts(self, shoulder, hip, ankle):
        return (
            np.array([*shoulder, 1.0]),
            np.array([*hip, 1.0]),
            np.array([*ankle, 1.0]),
        )

    def test_aligned_collinear(self):
        sh, hp, ank = self._pts((0, 0), (100, 0), (200, 0))
        out = HipPosture().analyze(sh, hp, ank)
        assert out.hip_status == "aligned"
        assert out.hip_angle == 180.0

    def test_sagging_below_base(self):
        sh, hp, ank = self._pts((0, 0), (100, 50), (200, 0))
        out = HipPosture().analyze(sh, hp, ank)
        assert out.hip_status == "sagging"
        assert out.hip_angle < 180.0

    def test_hips_up_above_base(self):
        sh, hp, ank = self._pts((0, 0), (100, -50), (200, 0))
        out = HipPosture().analyze(sh, hp, ank)
        assert out.hip_status == "hips_up"
        assert out.hip_angle < 180.0


class TestHitDepthMultiplier:
    def test_edge_of_band_is_1x(self):
        mult, label = hit_depth_multiplier(1.0 - DOWN_BAR_TOL, DOWN_BAR_TOL)
        assert mult == 1.0
        assert label == "hit"

    def test_full_depth_is_2x(self):
        mult, label = hit_depth_multiplier(1.0, DOWN_BAR_TOL)
        assert mult == 2.0
        assert label == "full depth"

    def test_mid_band_between_1_and_2(self):
        edge = 1.0 - UP_BAR_TOL
        mid = edge + UP_BAR_TOL / 2
        mult, _ = hit_depth_multiplier(mid, UP_BAR_TOL)
        assert 1.0 < mult < 2.0


class TestStrokePoints:
    def test_hit_good(self):
        assert stroke_points(True, True) == 100

    def test_hit_bad(self):
        assert stroke_points(True, False) == 50

    def test_miss_good(self):
        assert stroke_points(False, True) == 25

    def test_miss_bad(self):
        assert stroke_points(False, False) == 10


class TestPushup:
    def test_no_person(self):
        tracker = Pushup()
        out = tracker.update(None, None)
        assert out["detected"] is False
        assert out["reps"] == 0

    def test_locks_bars_on_first_up_pose(self):
        tracker = Pushup()
        kpts = _blank_keypoints()
        _left_up_pose(kpts)
        boxes = np.array([[0, 0, 200, 400]], dtype=np.float64)

        out = tracker.update(boxes, kpts)
        assert out["detected"] is True
        assert out["top_bar"] == 100.0
        assert out["low_bar"] == 300.0
        assert out["hip_angle"] is not None
        assert out["hip_status"] in ("aligned", "hips_up", "sagging")
        assert out["knee_angle"] is not None

    def test_kneeling_does_not_lock_bars(self):
        tracker = Pushup()
        kpts = _blank_keypoints()
        _left_kneeling_pose(kpts)
        boxes = np.array([[0, 0, 200, 400]], dtype=np.float64)

        out = tracker.update(boxes, kpts)
        assert out["detected"] is True
        assert out["top_bar"] is None
        assert out["low_bar"] is None

    def test_one_full_rep(self):
        tracker = Pushup()
        boxes = np.array([[0, 0, 200, 400]], dtype=np.float64)

        # Lock bars in up pose
        kpts_up = _blank_keypoints()
        _left_up_pose(kpts_up)
        out = tracker.update(boxes, kpts_up)
        assert tracker.reps == 0
        assert tracker.stage == "up"
        assert tracker.score == 0
        assert out["score"] == 0

        # Move to down pose
        kpts_down = _blank_keypoints()
        _left_down_pose(kpts_down)
        tracker.update(boxes, kpts_down)
        assert tracker.stage == "down"
        assert tracker.reps == 0

        # Back to up pose -> count rep
        kpts_up2 = _blank_keypoints()
        _left_up_pose(kpts_up2)
        tracker.update(boxes, kpts_up2)
        assert tracker.stage == "up"
        assert tracker.reps == 1
        assert tracker.score == 400
        assert tracker.y0_hits == 1
        assert tracker.y1_hits == 1
        assert tracker.award_seq == 2
        assert tracker.last_multiplier == 2.0
        assert tracker.last_award_label == "full depth"

    def test_down_hit_aligned_scores_200(self):
        tracker = Pushup()
        boxes = np.array([[0, 0, 200, 400]], dtype=np.float64)
        kpts_up = _blank_keypoints()
        _left_up_pose(kpts_up)
        tracker.update(boxes, kpts_up)

        kpts_down = _blank_keypoints()
        _left_down_pose(kpts_down)
        tracker.update(boxes, kpts_down)

        assert tracker.score == 200
        assert tracker.y1_hits == 1
        assert tracker.y0_hits == 0
        assert tracker.reps == 0
        assert tracker.award_seq == 1
        assert tracker.last_multiplier == 2.0

    def test_down_hit_sagging_scores_100(self):
        tracker = Pushup()
        boxes = np.array([[0, 0, 200, 400]], dtype=np.float64)
        kpts_up = _blank_keypoints()
        _left_up_pose(kpts_up)
        tracker.update(boxes, kpts_up)

        kpts_down = _blank_keypoints()
        _left_down_sagging_pose(kpts_down)
        tracker.update(boxes, kpts_down)

        assert tracker.score == 100
        assert tracker.y1_hits == 1
        assert tracker.posture_bad == 1
        assert tracker.last_multiplier == 2.0

    def test_miss_down_stroke_scores_25(self):
        tracker = Pushup()
        boxes = np.array([[0, 0, 200, 400]], dtype=np.float64)
        kpts_up = _blank_keypoints()
        _left_up_pose(kpts_up)
        tracker.update(boxes, kpts_up)

        kpts_partial = _blank_keypoints()
        _left_partial_down_pose(kpts_partial, 190.0)  # 45% travel
        tracker.update(boxes, kpts_partial)

        kpts_up2 = _blank_keypoints()
        _left_up_pose(kpts_up2)
        tracker.update(boxes, kpts_up2)

        assert tracker.score == 25
        assert tracker.reps == 0
        assert tracker.y1_hits == 0
        assert tracker.posture_good == 1
        assert tracker.last_multiplier == 1.0
        assert tracker.last_award_label == "miss"

    def test_small_wobble_scores_nothing(self):
        tracker = Pushup()
        boxes = np.array([[0, 0, 200, 400]], dtype=np.float64)
        kpts_up = _blank_keypoints()
        _left_up_pose(kpts_up)
        tracker.update(boxes, kpts_up)

        kpts_partial = _blank_keypoints()
        _left_partial_down_pose(kpts_partial, 170.0)  # 35% travel
        tracker.update(boxes, kpts_partial)

        kpts_up2 = _blank_keypoints()
        _left_up_pose(kpts_up2)
        tracker.update(boxes, kpts_up2)

        assert tracker.score == 0
        assert tracker.award_seq == 0

    def test_reset_reps_keeps_bars(self):
        tracker = Pushup()
        boxes = np.array([[0, 0, 200, 400]], dtype=np.float64)
        kpts_up = _blank_keypoints()
        _left_up_pose(kpts_up)
        tracker.update(boxes, kpts_up)
        kpts_down = _blank_keypoints()
        _left_down_pose(kpts_down)
        tracker.update(boxes, kpts_down)
        kpts_up2 = _blank_keypoints()
        _left_up_pose(kpts_up2)
        tracker.update(boxes, kpts_up2)
        assert tracker.reps == 1

        tracker.reset_reps()
        assert tracker.reps == 0
        assert tracker.stage == "up"
        assert tracker.top_bar == 100.0
        assert tracker.low_bar == 300.0
        assert tracker.score == 0
        assert tracker.y0_hits == 0
        assert tracker.y1_hits == 0
        assert tracker.posture_good == 0
        assert tracker.posture_bad == 0
        assert tracker.award_seq == 0
        assert tracker.last_multiplier == 1.0

    def test_ignores_smaller_box(self):
        tracker = Pushup()
        boxes = np.array(
            [
                [0, 0, 50, 50],
                [0, 0, 200, 400],
            ],
            dtype=np.float64,
        )
        kpts = np.zeros((2, 17, 3), dtype=np.float64)
        _left_up_pose(kpts[1:2])  # up pose on the large-box person only

        out = tracker.update(boxes, kpts)
        assert out["detected"] is True
        assert out["top_bar"] == 100.0


class TestJitterScoring:
    """Seeded YOLO-jitter / cheat probes against CURRENT scoring (no pose.py changes)."""

    def test_noisy_full_rep_still_scores(self):
        """σ~3–5 px on shoulder/hip/knee/ankle; good rep + awards still land."""
        tracker = Pushup()
        boxes = _BOXES
        rng = np.random.default_rng(42)
        sigma = 4.0

        out = _calibrate(tracker, boxes)
        assert out["top_bar"] == 100.0
        assert out["low_bar"] == 300.0

        # up → down with noise on each frame
        for _ in range(3):
            kpts = _blank_keypoints()
            _left_down_pose(kpts)
            _jitter(kpts, rng, sigma, _JITTER_JOINTS)
            tracker.update(boxes, kpts)
        assert tracker.stage == "down"
        assert tracker.score > 0

        # down → up with noise
        for _ in range(3):
            kpts = _blank_keypoints()
            _left_up_pose(kpts)
            _jitter(kpts, rng, sigma, _JITTER_JOINTS)
            tracker.update(boxes, kpts)

        assert tracker.reps == 1
        assert tracker.score > 0
        assert tracker.y0_hits >= 1
        assert tracker.y1_hits >= 1

    def test_sit_and_bounce_scores(self):
        """After plank calib, sitting shoulder bounce still increases score/reps (cheat)."""
        tracker = Pushup()
        boxes = _BOXES
        rng = np.random.default_rng(7)

        out = _calibrate(tracker, boxes)
        assert out["top_bar"] == 100.0
        assert out["low_bar"] == 300.0
        score0, reps0 = tracker.score, tracker.reps

        # Clean bounce: near-top → near-low → near-top
        for shoulder_y in (110.0, 290.0, 110.0):
            kpts = _blank_keypoints()
            _sitting_shoulder_bounce(kpts, shoulder_y)
            tracker.update(boxes, kpts)

        assert tracker.score > score0 or tracker.reps > reps0
        score_clean, reps_clean = tracker.score, tracker.reps

        # Mild noise bounce (another rep while still sitting)
        for shoulder_y in (110.0, 290.0, 110.0):
            kpts = _blank_keypoints()
            _sitting_shoulder_bounce(kpts, shoulder_y)
            # Mild noise on lower body; keep shoulder Y exact for band hits
            _jitter(kpts, rng, 2.0, (11, 13, 15))
            tracker.update(boxes, kpts)

        assert tracker.score > score_clean or tracker.reps > reps_clean
        assert tracker.reps >= 1
        assert tracker.score > 0

    def test_one_frame_knee_spike_still_completes(self):
        """Mid-rep one-frame knee offset > JUMP_PX; good rep still completes."""
        tracker = Pushup()
        boxes = _BOXES

        _calibrate(tracker, boxes)

        # Mid-way down
        kpts = _blank_keypoints()
        _left_partial_down_pose(kpts, 200.0)
        tracker.update(boxes, kpts)

        # One-frame knee spike larger than JUMP_PX
        kpts = _blank_keypoints()
        _left_partial_down_pose(kpts, 220.0)
        kpts[0, 13, 0] += JUMP_PX + 15.0
        tracker.update(boxes, kpts)

        # Finish the good rep
        _hold(tracker, _left_down_pose, n=1, boxes=boxes)
        assert tracker.stage == "down"
        _hold(tracker, _left_up_pose, n=1, boxes=boxes)

        assert tracker.reps == 1
        assert tracker.score > 0

    def test_low_conf_knee_freeze_sit_bounce_still_scores(self):
        """
        After plank calib, sit-bounce with knee conf < CONF_MIN still scores today.

        Documents fragility of a future knee-only gate that freezes last high-conf
        (plank) knee when conf drops below CONF_MIN — smoothed knee would stay
        plank-like while shoulder bounce still awards points.
        """
        tracker = Pushup()
        boxes = _BOXES

        _calibrate(tracker, boxes)
        score0 = tracker.score

        for shoulder_y in (110.0, 290.0, 110.0):
            kpts = _blank_keypoints()
            _sitting_shoulder_bounce(kpts, shoulder_y)
            kpts[0, 13, 2] = CONF_MIN - 0.05  # below gate threshold
            tracker.update(boxes, kpts)

        assert tracker.score > score0
        assert tracker.reps >= 1
