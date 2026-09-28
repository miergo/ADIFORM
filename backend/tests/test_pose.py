"""Unit tests for minimal pose.py (numpy only, no YOLO)."""

from unittest.mock import patch

import numpy as np

from pose import (
    CALIB_HOLD_SEC,
    CONF_MIN,
    DOWN_BAR_TOL,
    HipPosture,
    JUMP_PX,
    Pushup,
    UP_BAR_TOL,
    largest_box_index,
    hit_depth_multiplier,
    is_down_pose,
    is_ready_pose,
    is_up_pose,
    joint_angle,
    pick_side,
    stroke_points,
)

_JITTER_JOINTS = (5, 11, 13, 15)  # shoulder, hip, knee, ankle


def _blank_keypoints() -> np.ndarray:
    """One person, 17 joints, all zero confidence."""
    return np.zeros((1, 17, 3), dtype=np.float64)


def _hold(
    tracker: Pushup,
    boxes: np.ndarray,
    kpts: np.ndarray,
    ids: np.ndarray | None = None,
    n: int = 8,
) -> dict:
    """Repeat update so EMA can settle on a sudden pose jump."""
    out = None
    for _ in range(n):
        out = tracker.update(boxes, kpts, ids)
    return out


def _calibrate(
    tracker: Pushup,
    boxes: np.ndarray,
    kpts: np.ndarray,
    ids: np.ndarray | None = None,
) -> dict:
    """Advance mocked time through the 3s ready hold so bars lock."""
    t = [0.0]
    with patch("pose.time.monotonic", side_effect=lambda: t[0]):
        out = tracker.update(boxes, kpts, ids)
        assert out["top_bar"] is None
        t[0] = CALIB_HOLD_SEC
        return _hold(tracker, boxes, kpts, ids)


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
    """Shoulder at locked low bar (down position) with bent elbow and long straight leg."""
    _set_joint(kpts, 5, 100, 300)   # shoulder at low_bar
    _set_joint(kpts, 7, 100, 250)   # elbow
    _set_joint(kpts, 9, 130, 250)   # wrist
    _set_joint(kpts, 11, 100, 360)  # hip
    _set_joint(kpts, 13, 100, 420)  # knee (long segments survive YOLO jitter)
    _set_joint(kpts, 15, 100, 480)  # ankle


def _left_down_sagging_pose(kpts: np.ndarray) -> None:
    """Down position with sagging hips."""
    _set_joint(kpts, 5, 100, 300)
    _set_joint(kpts, 7, 100, 250)
    _set_joint(kpts, 9, 130, 250)
    _set_joint(kpts, 11, 100, 560)  # hip past ankle on the line -> sagging
    _set_joint(kpts, 13, 100, 520)  # knee between hip and ankle
    _set_joint(kpts, 15, 100, 480)


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
    _set_joint(kpts, 5, 100, shoulder_y)  # shoulder (bounces)
    _set_joint(kpts, 7, 130, shoulder_y + 15)
    _set_joint(kpts, 9, 150, shoulder_y + 30)
    _set_joint(kpts, 11, 180, 160)  # hip forward / raised
    _set_joint(kpts, 13, 240, 180)  # knee bent ~90 deg
    _set_joint(kpts, 15, 200, 300)  # ankle near floor / low bar


def _left_stiff_arm_drop(kpts: np.ndarray) -> None:
    """Shoulder at low bar with locked arm and straight leg (flop, not a press)."""
    _set_joint(kpts, 5, 100, 300)   # shoulder at low_bar
    _set_joint(kpts, 7, 150, 300)   # elbow locked (horizontal arm)
    _set_joint(kpts, 9, 200, 300)   # wrist
    _set_joint(kpts, 11, 100, 360)  # hip
    _set_joint(kpts, 13, 100, 420)  # knee
    _set_joint(kpts, 15, 100, 480)  # ankle


class TestLargestBoxIndex:
    def test_picks_largest_box(self):
        boxes = np.array(
            [
                [0, 0, 50, 50],    # area 2500
                [0, 0, 100, 100],  # area 10000
            ],
            dtype=np.float64,
        )
        assert largest_box_index(boxes) == 1

    def test_no_boxes(self):
        assert largest_box_index(None) is None
        assert largest_box_index(np.zeros((0, 4))) is None


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
        assert is_ready_pose(100, 200, 300, 175.0, 154.0, 179.0) is False  # body not straight
        assert is_ready_pose(100, 200, 300, 175.0, 179.0, 149.0) is False  # knee bent
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

    def test_one_ready_frame_does_not_lock_bars(self):
        tracker = Pushup()
        kpts = _blank_keypoints()
        _left_up_pose(kpts)
        boxes = np.array([[0, 0, 200, 400]], dtype=np.float64)

        out = tracker.update(boxes, kpts)
        assert out["detected"] is True
        assert out["top_bar"] is None
        assert out["low_bar"] is None
        assert out["hip_angle"] is not None
        assert out["hip_status"] in ("aligned", "hips_up", "sagging")
        assert out["knee_angle"] is not None

    def test_locks_bars_after_hold(self):
        tracker = Pushup()
        kpts = _blank_keypoints()
        _left_up_pose(kpts)
        boxes = np.array([[0, 0, 200, 400]], dtype=np.float64)

        out = _calibrate(tracker, boxes, kpts)
        assert out["detected"] is True
        assert out["top_bar"] == 100.0
        assert out["low_bar"] == 300.0

    def test_broken_hold_resets_and_does_not_lock(self):
        tracker = Pushup()
        boxes = np.array([[0, 0, 200, 400]], dtype=np.float64)
        kpts_up = _blank_keypoints()
        _left_up_pose(kpts_up)
        kpts_kneel = _blank_keypoints()
        _left_kneeling_pose(kpts_kneel)

        t = [0.0]
        with patch("pose.time.monotonic", side_effect=lambda: t[0]):
            tracker.update(boxes, kpts_up)
            t[0] = 1.0
            # Hold kneel long enough for EMA to leave ready pose and reset calib.
            _hold(tracker, boxes, kpts_kneel)
            t[0] = 4.0
            out = _hold(tracker, boxes, kpts_up)
            assert out["top_bar"] is None
            t[0] = 4.0 + CALIB_HOLD_SEC
            out = _hold(tracker, boxes, kpts_up)
            assert out["top_bar"] == 100.0
            assert out["low_bar"] == 300.0

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
        out = _calibrate(tracker, boxes, kpts_up)
        assert tracker.reps == 0
        assert tracker.stage == "up"
        assert tracker.score == 0
        assert out["score"] == 0

        # Move to down pose
        kpts_down = _blank_keypoints()
        _left_down_pose(kpts_down)
        _hold(tracker, boxes, kpts_down)
        assert tracker.stage == "down"
        assert tracker.reps == 0

        # Back to up pose -> count rep
        kpts_up2 = _blank_keypoints()
        _left_up_pose(kpts_up2)
        _hold(tracker, boxes, kpts_up2)
        assert tracker.stage == "up"
        assert tracker.reps == 1
        assert tracker.score == 400
        assert tracker.top_bar_hits == 1
        assert tracker.low_bar_hits == 1
        assert tracker.award_seq == 2
        assert tracker.last_multiplier == 2.0
        assert tracker.last_award_label == "full depth"

    def test_down_hit_aligned_scores_200(self):
        tracker = Pushup()
        boxes = np.array([[0, 0, 200, 400]], dtype=np.float64)
        kpts_up = _blank_keypoints()
        _left_up_pose(kpts_up)
        _calibrate(tracker, boxes, kpts_up)

        kpts_down = _blank_keypoints()
        _left_down_pose(kpts_down)
        _hold(tracker, boxes, kpts_down)

        assert tracker.score == 200
        assert tracker.low_bar_hits == 1
        assert tracker.top_bar_hits == 0
        assert tracker.reps == 0
        assert tracker.award_seq == 1
        assert tracker.last_multiplier == 2.0

    def test_down_hit_sagging_scores_100(self):
        tracker = Pushup()
        boxes = np.array([[0, 0, 200, 400]], dtype=np.float64)
        kpts_up = _blank_keypoints()
        _left_up_pose(kpts_up)
        _calibrate(tracker, boxes, kpts_up)

        kpts_down = _blank_keypoints()
        _left_down_sagging_pose(kpts_down)
        _hold(tracker, boxes, kpts_down)

        assert tracker.score == 100
        assert tracker.low_bar_hits == 1
        assert tracker.posture_bad == 1
        assert tracker.last_multiplier == 2.0

    def test_miss_down_stroke_scores_25(self):
        tracker = Pushup()
        boxes = np.array([[0, 0, 200, 400]], dtype=np.float64)
        kpts_up = _blank_keypoints()
        _left_up_pose(kpts_up)
        _calibrate(tracker, boxes, kpts_up)

        kpts_partial = _blank_keypoints()
        _left_partial_down_pose(kpts_partial, 200.0)  # 50% travel
        _hold(tracker, boxes, kpts_partial)

        kpts_up2 = _blank_keypoints()
        _left_up_pose(kpts_up2)
        _hold(tracker, boxes, kpts_up2)

        assert tracker.score == 25
        assert tracker.reps == 0
        assert tracker.low_bar_hits == 0
        assert tracker.posture_good == 1
        assert tracker.last_multiplier == 1.0
        assert tracker.last_award_label == "miss"

    def test_small_wobble_scores_nothing(self):
        tracker = Pushup()
        boxes = np.array([[0, 0, 200, 400]], dtype=np.float64)
        kpts_up = _blank_keypoints()
        _left_up_pose(kpts_up)
        _calibrate(tracker, boxes, kpts_up)

        kpts_partial = _blank_keypoints()
        _left_partial_down_pose(kpts_partial, 170.0)  # 35% travel
        _hold(tracker, boxes, kpts_partial)

        kpts_up2 = _blank_keypoints()
        _left_up_pose(kpts_up2)
        _hold(tracker, boxes, kpts_up2)

        assert tracker.score == 0
        assert tracker.award_seq == 0

    def test_reset_reps_keeps_bars(self):
        tracker = Pushup()
        boxes = np.array([[0, 0, 200, 400]], dtype=np.float64)
        kpts_up = _blank_keypoints()
        _left_up_pose(kpts_up)
        _calibrate(tracker, boxes, kpts_up)
        kpts_down = _blank_keypoints()
        _left_down_pose(kpts_down)
        _hold(tracker, boxes, kpts_down)
        kpts_up2 = _blank_keypoints()
        _left_up_pose(kpts_up2)
        _hold(tracker, boxes, kpts_up2)
        assert tracker.reps == 1

        tracker.reset_reps()
        assert tracker.reps == 0
        assert tracker.stage == "up"
        assert tracker.top_bar == 100.0
        assert tracker.low_bar == 300.0
        assert tracker.score == 0
        assert tracker.top_bar_hits == 0
        assert tracker.low_bar_hits == 0
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

        out = _calibrate(tracker, boxes, kpts)
        assert out["detected"] is True
        assert out["top_bar"] == 100.0

    def test_locks_track_id_and_ignores_new_larger_person(self):
        tracker = Pushup()
        boxes = np.array(
            [
                [0, 0, 50, 50],
                [0, 0, 200, 400],
            ],
            dtype=np.float64,
        )
        kpts = np.zeros((2, 17, 3), dtype=np.float64)
        _left_up_pose(kpts[1:2])
        ids = np.array([3, 7], dtype=np.int64)

        out = _calibrate(tracker, boxes, kpts, ids)
        assert tracker._locked_id == 7
        assert out["top_bar"] == 100.0

        # Bystander becomes largest; locked id 7 still present with same pose.
        boxes2 = np.array(
            [
                [0, 0, 300, 500],
                [0, 0, 100, 200],
            ],
            dtype=np.float64,
        )
        kpts2 = np.zeros((2, 17, 3), dtype=np.float64)
        _set_joint(kpts2[0:1], 5, 100, 250)  # would change bars if selected
        _left_up_pose(kpts2[1:2])
        ids2 = np.array([99, 7], dtype=np.int64)

        out2 = tracker.update(boxes2, kpts2, ids2)
        assert out2["detected"] is True
        assert tracker._locked_id == 7
        assert out2["top_bar"] == 100.0
        assert float(out2["shoulder"][1]) == 100.0

    def test_reacquires_when_track_id_renumbered(self):
        tracker = Pushup()
        boxes = np.array([[0, 0, 200, 400]], dtype=np.float64)
        kpts = _blank_keypoints()
        _left_up_pose(kpts)
        ids = np.array([7], dtype=np.int64)

        tracker.update(boxes, kpts, ids)
        assert tracker._locked_id == 7

        # Same body, new ByteTrack id, overlapping box.
        boxes2 = np.array([[10, 10, 210, 410]], dtype=np.float64)
        kpts2 = _blank_keypoints()
        _left_up_pose(kpts2)
        ids2 = np.array([99], dtype=np.int64)

        out = tracker.update(boxes2, kpts2, ids2)
        assert out["detected"] is True
        assert tracker._locked_id == 99

    def test_rejects_distant_bystander_when_id_missing(self):
        tracker = Pushup()
        boxes = np.array([[0, 0, 200, 400]], dtype=np.float64)
        kpts = _blank_keypoints()
        _left_up_pose(kpts)
        ids = np.array([7], dtype=np.int64)

        tracker.update(boxes, kpts, ids)
        assert tracker._locked_id == 7

        # Far larger box — IoU ~0, must not steal lock.
        boxes2 = np.array([[600, 0, 900, 500]], dtype=np.float64)
        kpts2 = _blank_keypoints()
        _left_up_pose(kpts2)
        ids2 = np.array([99], dtype=np.int64)

        out = tracker.update(boxes2, kpts2, ids2)
        assert out["detected"] is False
        assert tracker._locked_id == 7

    def test_side_stays_locked(self):
        tracker = Pushup()
        boxes = np.array([[0, 0, 200, 400]], dtype=np.float64)
        kpts = np.zeros((1, 17, 3), dtype=np.float64)
        _left_up_pose(kpts)
        for idx in (5, 7, 9, 11, 13, 15):
            kpts[0, idx, 2] = 0.9
        for idx in (6, 8, 10, 12, 14, 16):
            kpts[0, idx] = [200, 200, 0.1]

        tracker.update(boxes, kpts)
        assert tracker._side == "left"

        # Opposite side now much more confident; side must not flip.
        kpts2 = np.zeros((1, 17, 3), dtype=np.float64)
        _left_up_pose(kpts2)
        for idx in (5, 7, 9, 11, 13, 15):
            kpts2[0, idx, 2] = 0.1
        for idx in (6, 8, 10, 12, 14, 16):
            kpts2[0, idx] = [200, 150, 0.99]

        out = tracker.update(boxes, kpts2)
        assert tracker._side == "left"
        assert out["detected"] is True
        # Left shoulder stays at x=100 from _left_up_pose, not right's 200.
        assert float(out["shoulder"][0]) == 100.0


class TestJitterScoring:
    """YOLO-jitter probes and valid-rep cheat rejection."""

    def test_noisy_full_rep_still_scores(self):
        """σ~4 px on shoulder/hip/knee/ankle; good rep + awards still land."""
        tracker = Pushup()
        boxes = np.array([[0, 0, 200, 400]], dtype=np.float64)
        rng = np.random.default_rng(42)
        sigma = 4.0

        kpts_up = _blank_keypoints()
        _left_up_pose(kpts_up)
        out = _calibrate(tracker, boxes, kpts_up)
        assert out["top_bar"] == 100.0
        assert out["low_bar"] == 300.0

        kpts_down = _blank_keypoints()
        _left_down_pose(kpts_down)
        _jitter(kpts_down, rng, sigma, _JITTER_JOINTS)
        _hold(tracker, boxes, kpts_down)
        assert tracker.stage == "down"
        assert tracker.score > 0

        kpts_up2 = _blank_keypoints()
        _left_up_pose(kpts_up2)
        _jitter(kpts_up2, rng, sigma, _JITTER_JOINTS)
        _hold(tracker, boxes, kpts_up2)

        assert tracker.reps == 1
        assert tracker.score > 0
        assert tracker.top_bar_hits >= 1
        assert tracker.low_bar_hits >= 1

    def test_sit_and_bounce_scores_nothing(self):
        """After plank calib, sitting shoulder bounce must not award score/reps."""
        tracker = Pushup()
        boxes = np.array([[0, 0, 200, 400]], dtype=np.float64)
        rng = np.random.default_rng(7)

        kpts_up = _blank_keypoints()
        _left_up_pose(kpts_up)
        out = _calibrate(tracker, boxes, kpts_up)
        assert out["top_bar"] == 100.0
        assert out["low_bar"] == 300.0
        score0, reps0 = tracker.score, tracker.reps

        for shoulder_y in (110.0, 290.0, 110.0):
            kpts = _blank_keypoints()
            _sitting_shoulder_bounce(kpts, shoulder_y)
            _hold(tracker, boxes, kpts)

        assert tracker.score == score0
        assert tracker.reps == reps0
        assert tracker.stage == "up"

        for shoulder_y in (110.0, 290.0, 110.0):
            kpts = _blank_keypoints()
            _sitting_shoulder_bounce(kpts, shoulder_y)
            _jitter(kpts, rng, 2.0, (11, 13, 15))
            _hold(tracker, boxes, kpts)

        assert tracker.score == score0
        assert tracker.reps == reps0
        assert tracker.stage == "up"

    def test_one_frame_knee_spike_still_completes(self):
        """Mid-rep one-frame knee offset > JUMP_PX; good rep still completes."""
        tracker = Pushup()
        boxes = np.array([[0, 0, 200, 400]], dtype=np.float64)

        kpts_up = _blank_keypoints()
        _left_up_pose(kpts_up)
        _calibrate(tracker, boxes, kpts_up)

        kpts_mid = _blank_keypoints()
        _left_partial_down_pose(kpts_mid, 200.0)
        _hold(tracker, boxes, kpts_mid)

        kpts_spike = _blank_keypoints()
        _left_partial_down_pose(kpts_spike, 220.0)
        kpts_spike[0, 13, 0] += JUMP_PX + 15.0
        tracker.update(boxes, kpts_spike)

        kpts_down = _blank_keypoints()
        _left_down_pose(kpts_down)
        _hold(tracker, boxes, kpts_down)
        assert tracker.stage == "down"

        kpts_up2 = _blank_keypoints()
        _left_up_pose(kpts_up2)
        _hold(tracker, boxes, kpts_up2)

        assert tracker.reps == 1
        assert tracker.score > 0

    def test_low_conf_knee_freeze_sit_bounce_scores_nothing(self):
        """Sit-bounce with knee conf < CONF_MIN must not score (raw conf gate)."""
        tracker = Pushup()
        boxes = np.array([[0, 0, 200, 400]], dtype=np.float64)

        kpts_up = _blank_keypoints()
        _left_up_pose(kpts_up)
        _calibrate(tracker, boxes, kpts_up)
        score0 = tracker.score
        reps0 = tracker.reps

        for shoulder_y in (110.0, 290.0, 110.0):
            kpts = _blank_keypoints()
            _sitting_shoulder_bounce(kpts, shoulder_y)
            kpts[0, 13, 2] = CONF_MIN - 0.05
            _hold(tracker, boxes, kpts)

        assert tracker.score == score0
        assert tracker.reps == reps0
        assert tracker.stage == "up"

    def test_stiff_arm_drop_scores_nothing(self):
        """Shoulder at low bar with locked elbows must not award; return is not a miss."""
        tracker = Pushup()
        boxes = np.array([[0, 0, 200, 400]], dtype=np.float64)

        kpts_up = _blank_keypoints()
        _left_up_pose(kpts_up)
        _calibrate(tracker, boxes, kpts_up)

        kpts_drop = _blank_keypoints()
        _left_stiff_arm_drop(kpts_drop)
        _hold(tracker, boxes, kpts_drop)
        assert tracker.stage == "up"
        assert tracker.score == 0
        assert tracker.reps == 0

        kpts_up2 = _blank_keypoints()
        _left_up_pose(kpts_up2)
        _hold(tracker, boxes, kpts_up2)
        assert tracker.stage == "up"
        assert tracker.score == 0
        assert tracker.reps == 0
        assert tracker.award_seq == 0

