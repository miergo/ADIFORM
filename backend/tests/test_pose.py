"""Edge tests for pose analysis helpers (no YOLO / webcam required)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from scipy.signal import sosfilt, sosfilt_zi

from pose import (
    CALIB_ELBOW_MIN,
    CONF_MIN,
    DOWN_TOUCH_PX,
    HOLD_CALIB_POSE_TIME,
    LEFT_EYE,
    MIN_Y1_PX,
    NOSE,
    RIGHT_EYE,
    SIDE_MARGIN,
    SIDE_VOTES_NEEDED,
    UP_ELBOW,
    VIDEO_AUTO_CALIB_FRAMES,
    analyze_frame,
    eye_point,
    guess_side,
    init_tracker,
    joint_angle,
    make_smoother,
    metrics_from_tracker,
    pick_side_from_votes,
    smooth_xy,
    update_calibrate_ankle,
    update_eye_speed,
    update_meters_per_px,
    update_rep_count,
)


# ---------------------------------------------------------------------------
# Helpers to build synthetic COCO-17 keypoints
# ---------------------------------------------------------------------------


def blank_keypoints(confidence: float = 0.9) -> np.ndarray:
    """(17, 3) keypoints with given confidence; xy filled by callers."""
    keypoints = np.zeros((17, 3), dtype=np.float64)
    keypoints[:, 2] = confidence
    return keypoints


def old_smooth_xy(xy, sos, zi):
    """Reference per-point loop (pre-vectorization) for regression checks."""
    xy = np.asarray(xy, dtype=np.float64)
    num_keypoints, num_axes = xy.shape
    out = np.empty_like(xy)
    if zi is None:
        base = sosfilt_zi(sos)
        zi = np.zeros((num_keypoints, num_axes) + base.shape, dtype=np.float64)
        for i in range(num_keypoints):
            for j in range(num_axes):
                zi[i, j] = base * xy[i, j]
                out[i, j] = xy[i, j]
        return out, zi
    for i in range(num_keypoints):
        for j in range(num_axes):
            filtered, zi[i, j] = sosfilt(sos, [xy[i, j]], zi=zi[i, j])
            out[i, j] = filtered[0]
    return out, zi


# =============================================================================
# joint_angle
# =============================================================================


class TestJointAngle:
    def test_degenerate_coincident_points_returns_none(self):
        point = np.array([1.0, 2.0])
        assert joint_angle(point, point, point) is None

    def test_collinear_points_are_180(self):
        point_a = np.array([0.0, 0.0])
        point_b = np.array([1.0, 0.0])
        point_c = np.array([2.0, 0.0])
        angle = joint_angle(point_a, point_b, point_c)
        assert angle is not None
        assert angle == pytest.approx(180.0, abs=1e-6)

    def test_right_angle_is_90(self):
        point_a = np.array([0.0, 1.0])
        point_b = np.array([0.0, 0.0])
        point_c = np.array([1.0, 0.0])
        angle = joint_angle(point_a, point_b, point_c)
        assert angle is not None
        assert angle == pytest.approx(90.0, abs=1e-6)


# =============================================================================
# eye_point
# =============================================================================


class TestEyePoint:
    def test_both_eyes_returns_average(self):
        keypoints = blank_keypoints()
        keypoints[LEFT_EYE] = [10.0, 20.0, 0.9]
        keypoints[RIGHT_EYE] = [30.0, 40.0, 0.9]
        eye = eye_point(keypoints)
        assert eye is not None
        np.testing.assert_allclose(eye, [20.0, 30.0])

    def test_one_eye_only(self):
        keypoints = blank_keypoints(confidence=0.0)
        keypoints[LEFT_EYE] = [5.0, 6.0, 0.9]
        eye = eye_point(keypoints)
        assert eye is not None
        np.testing.assert_allclose(eye, [5.0, 6.0])

    def test_nose_fallback(self):
        keypoints = blank_keypoints(confidence=0.0)
        keypoints[NOSE] = [7.0, 8.0, 0.9]
        eye = eye_point(keypoints)
        assert eye is not None
        np.testing.assert_allclose(eye, [7.0, 8.0])

    def test_nothing_confident_returns_none(self):
        keypoints = blank_keypoints(confidence=CONF_MIN - 0.01)
        assert eye_point(keypoints) is None


# =============================================================================
# Side detection
# =============================================================================


class TestSideDetection:
    def test_guess_side_tie_within_margin_returns_none(self):
        keypoints = blank_keypoints()
        # Elbow/wrist confidences nearly equal
        keypoints[7][2] = 0.5
        keypoints[9][2] = 0.5
        keypoints[8][2] = 0.5 + SIDE_MARGIN / 2
        keypoints[10][2] = 0.5 + SIDE_MARGIN / 2
        assert guess_side(keypoints) is None

    def test_guess_side_prefers_higher_confidence(self):
        keypoints = blank_keypoints()
        keypoints[7][2] = keypoints[9][2] = 0.3
        keypoints[8][2] = keypoints[10][2] = 0.9
        assert guess_side(keypoints) == "right"

    def test_pick_side_locks_only_after_enough_votes(self):
        keypoints = blank_keypoints()
        keypoints[7][2] = keypoints[9][2] = 0.9
        keypoints[8][2] = keypoints[10][2] = 0.2

        left_votes, right_votes, side, arm, chain = 0, 0, None, None, None
        for _ in range(SIDE_VOTES_NEEDED - 1):
            left_votes, right_votes, side, arm, chain = pick_side_from_votes(
                keypoints, left_votes, right_votes
            )
        assert side is None
        assert arm is None

        left_votes, right_votes, side, arm, chain = pick_side_from_votes(
            keypoints, left_votes, right_votes
        )
        assert side == "left"
        assert arm is not None
        assert chain is not None


# =============================================================================
# smooth_xy
# =============================================================================


class TestSmoothXy:
    def test_reseed_first_output_equals_input(self):
        sos, _ = make_smoother(30)
        xy = np.random.default_rng(1).normal(size=(17, 2))
        smoothed, zi = smooth_xy(xy, sos, None)
        np.testing.assert_allclose(smoothed, xy)
        assert zi is not None

    def test_output_finite_under_jitter(self):
        sos, zi = make_smoother(30)
        rng = np.random.default_rng(2)
        xy = rng.normal(size=(17, 2))
        smoothed, zi = smooth_xy(xy, sos, None)
        for _ in range(20):
            xy = xy + rng.normal(scale=2.0, size=(17, 2))
            smoothed, zi = smooth_xy(xy, sos, zi)
        assert np.isfinite(smoothed).all()

    def test_vectorized_matches_old_loop(self):
        sos, _ = make_smoother(30)
        rng = np.random.default_rng(3)
        xy0 = rng.normal(size=(17, 2))
        xy1 = xy0 + rng.normal(scale=0.5, size=(17, 2))

        new0, new_zi = smooth_xy(xy0, sos, None)
        new1, new_zi2 = smooth_xy(xy1, sos, new_zi)

        old0, old_zi = old_smooth_xy(xy0, sos, None)
        old1, old_zi2 = old_smooth_xy(xy1, sos, old_zi)

        np.testing.assert_allclose(new0, old0)
        np.testing.assert_allclose(new1, old1)
        np.testing.assert_allclose(new_zi2, old_zi2)


# =============================================================================
# update_rep_count FSM
# =============================================================================


class TestUpdateRepCount:
    def test_no_rep_without_prior_down(self):
        tracker = init_tracker(30)
        tracker["stage"] = "-"
        update_rep_count(DOWN_TOUCH_PX + 50, UP_ELBOW + 10, tracker)
        assert tracker["reps"] == 0
        assert tracker["stage"] == "-"

    def test_down_with_elbow_none_still_registers(self):
        tracker = init_tracker(30)
        update_rep_count(DOWN_TOUCH_PX - 1, None, tracker)
        assert tracker["stage"] == "down"
        assert tracker["reps"] == 0

    def test_down_then_up_counts_one_rep(self):
        tracker = init_tracker(30)
        update_rep_count(0.0, 90.0, tracker)
        assert tracker["stage"] == "down"
        update_rep_count(100.0, UP_ELBOW, tracker)
        assert tracker["reps"] == 1
        assert tracker["stage"] == "up"

    def test_repeated_downs_count_once_until_up(self):
        tracker = init_tracker(30)
        update_rep_count(0.0, 80.0, tracker)
        update_rep_count(0.0, 80.0, tracker)
        update_rep_count(100.0, UP_ELBOW, tracker)
        assert tracker["reps"] == 1
        update_rep_count(0.0, 80.0, tracker)
        update_rep_count(100.0, UP_ELBOW, tracker)
        assert tracker["reps"] == 2

    def test_none_height_is_noop(self):
        tracker = init_tracker(30)
        tracker["stage"] = "down"
        tracker["reps"] = 3
        update_rep_count(None, UP_ELBOW, tracker)
        assert tracker["reps"] == 3
        assert tracker["stage"] == "down"


# =============================================================================
# update_calibrate_ankle
# =============================================================================


class TestUpdateCalibrateAnkle:
    def test_hold_resets_when_ankle_lost(self):
        tracker = init_tracker(30)
        tracker["side"] = "left"
        tracker["phase"] = "calibrating"
        tracker["calib_hold_start"] = 100.0
        tracker["calib_ankle_samples"] = [np.array([1.0, 2.0])]
        tracker["calib_h_samples"] = [50.0]
        tracker["calib_progress"] = 0.5

        update_calibrate_ankle(tracker, None, 50.0, CALIB_ELBOW_MIN)
        assert tracker["calib_hold_start"] is None
        assert tracker["calib_ankle_samples"] == []
        assert tracker["calib_progress"] == 0.0

    def test_hold_resets_when_elbow_below_min(self):
        tracker = init_tracker(30)
        tracker["side"] = "left"
        tracker["phase"] = "calibrating"
        ankle = np.array([100.0, 400.0])
        update_calibrate_ankle(tracker, ankle, 80.0, CALIB_ELBOW_MIN - 1)
        assert tracker["calib_hold_start"] is None
        assert tracker["calib_progress"] == 0.0

    def test_completes_after_hold_time_and_floors_y1(self):
        tracker = init_tracker(30)
        tracker["side"] = "right"
        tracker["phase"] = "calibrating"
        ankle = np.array([100.0, 400.0])
        # Tiny height so y1 must floor at MIN_Y1_PX
        eye_height_px = 10.0

        with patch("pose.time.time", return_value=1000.0):
            update_calibrate_ankle(tracker, ankle, eye_height_px, CALIB_ELBOW_MIN)
            assert tracker["phase"] == "calibrating"
            assert tracker["calib_hold_start"] == 1000.0

        with patch(
            "pose.time.time",
            return_value=1000.0 + HOLD_CALIB_POSE_TIME,
        ):
            update_calibrate_ankle(tracker, ankle, eye_height_px, CALIB_ELBOW_MIN)

        assert tracker["phase"] == "ready"
        assert tracker["calib_progress"] == 1.0
        assert tracker["y1"] == MIN_Y1_PX
        assert tracker["ankle_origin"] is not None

    def test_detecting_moves_to_calibrating_when_side_locked(self):
        tracker = init_tracker(30)
        tracker["side"] = "left"
        tracker["phase"] = "detecting"
        update_calibrate_ankle(tracker, np.array([1.0, 2.0]), 50.0, 160.0)
        assert tracker["phase"] == "calibrating"

    def test_auto_mode_locks_after_n_frames_to_active(self):
        tracker = init_tracker(30, calib_mode="auto")
        tracker["side"] = "right"
        tracker["phase"] = "calibrating"
        ankle = np.array([100.0, 400.0])
        eye_height_px = 10.0  # floors y1 at MIN_Y1_PX

        for frame_index in range(VIDEO_AUTO_CALIB_FRAMES - 1):
            update_calibrate_ankle(tracker, ankle, eye_height_px, CALIB_ELBOW_MIN)
            assert tracker["phase"] == "calibrating"
            assert tracker["calib_progress"] == pytest.approx(
                (frame_index + 1) / VIDEO_AUTO_CALIB_FRAMES
            )

        update_calibrate_ankle(tracker, ankle, eye_height_px, CALIB_ELBOW_MIN)
        assert tracker["phase"] == "active"
        assert tracker["calib_progress"] == 1.0
        assert tracker["y1"] == MIN_Y1_PX
        assert tracker["ankle_origin"] is not None

    def test_auto_mode_bent_elbow_resets_consecutive_counter(self):
        tracker = init_tracker(30, calib_mode="auto")
        tracker["side"] = "right"
        tracker["phase"] = "calibrating"
        ankle = np.array([100.0, 400.0])

        update_calibrate_ankle(tracker, ankle, 80.0, CALIB_ELBOW_MIN)
        update_calibrate_ankle(tracker, ankle, 80.0, CALIB_ELBOW_MIN)
        assert len(tracker["calib_h_samples"]) == 2

        update_calibrate_ankle(tracker, ankle, 80.0, CALIB_ELBOW_MIN - 1)
        assert tracker["calib_h_samples"] == []
        assert tracker["calib_progress"] == 0.0
        assert tracker["phase"] == "calibrating"


# =============================================================================
# Speed / scale
# =============================================================================


class TestSpeedAndScale:
    def test_none_height_resets_prev_h(self):
        tracker = init_tracker(30)
        tracker["prev_h"] = 50.0
        assert update_eye_speed(None, tracker) is None
        assert tracker["prev_h"] is None

    def test_no_speed_without_m_per_px(self):
        tracker = init_tracker(30)
        tracker["m_per_px"] = None
        assert update_eye_speed(10.0, tracker) is None  # seeds prev_h
        assert update_eye_speed(20.0, tracker) is None  # still no scale

    def test_low_confidence_joints_keep_previous_scale(self):
        tracker = init_tracker(30)
        tracker["m_per_px"] = 0.01
        keypoints = blank_keypoints(confidence=0.0)
        chain = (5, 11, 15)
        result = update_meters_per_px(keypoints, chain, tracker)
        assert result == 0.01
        assert tracker["m_per_px"] == 0.01


# =============================================================================
# Active-only rep counting via analyze_frame
# =============================================================================


def _pushup_keypoints(
    eye_y: float,
    elbow_angle_hint: str,
    side: str = "right",
) -> np.ndarray:
    """
    Build a side-profile push-up skeleton.

    elbow_angle_hint: 'bent' (~90°) or 'straight' (~180°)
    eye_y: image Y of the face (larger = lower on screen = closer to ankle)
    """
    keypoints = blank_keypoints(0.95)
    # Strong right-arm confidence so side votes lock right
    keypoints[7][2] = keypoints[9][2] = 0.2
    keypoints[8][2] = keypoints[10][2] = 0.95

    ankle_y = 400.0
    ankle_x = 200.0
    # Right chain: shoulder 6, hip 12, ankle 16
    keypoints[16, :2] = [ankle_x, ankle_y]
    keypoints[12, :2] = [ankle_x + 20, ankle_y - 80]
    keypoints[6, :2] = [ankle_x + 40, ankle_y - 160]

    # Face points
    keypoints[LEFT_EYE, :2] = [ankle_x + 50, eye_y]
    keypoints[RIGHT_EYE, :2] = [ankle_x + 54, eye_y]
    keypoints[NOSE, :2] = [ankle_x + 52, eye_y]

    # Right arm: shoulder 6, elbow 8, wrist 10
    shoulder = keypoints[6, :2].copy()
    if elbow_angle_hint == "straight":
        keypoints[8, :2] = shoulder + np.array([0.0, 40.0])
        keypoints[10, :2] = shoulder + np.array([0.0, 80.0])
    else:
        keypoints[8, :2] = shoulder + np.array([40.0, 40.0])
        keypoints[10, :2] = shoulder + np.array([0.0, 80.0])

    # Left arm placeholders (low conf already set for elbows/wrists)
    keypoints[5, :2] = shoulder + np.array([-10.0, 0.0])
    keypoints[11, :2] = keypoints[12, :2] + np.array([-10.0, 0.0])
    keypoints[15, :2] = keypoints[16, :2] + np.array([-10.0, 0.0])
    return keypoints


class TestActiveOnlyReps:
    def test_ready_phase_does_not_count_reps(self):
        tracker = init_tracker(30)
        tracker["side"] = "right"
        tracker["arm"] = [6, 8, 10]
        tracker["chain"] = (6, 12, 16)
        tracker["phase"] = "ready"
        tracker["ankle_origin"] = np.array([200.0, 400.0])
        tracker["y1"] = 120.0

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Reset filter each frame so geometry is exact (no Butterworth lag)
        tracker["zi"] = None
        analyze_frame(frame, _pushup_keypoints(390.0, "bent"), tracker)
        tracker["zi"] = None
        analyze_frame(frame, _pushup_keypoints(280.0, "straight"), tracker)
        assert tracker["reps"] == 0

    def test_active_phase_counts_rep(self):
        tracker = init_tracker(30)
        tracker["side"] = "right"
        tracker["arm"] = [6, 8, 10]
        tracker["chain"] = (6, 12, 16)
        tracker["phase"] = "active"
        tracker["ankle_origin"] = np.array([200.0, 400.0])
        tracker["y1"] = 120.0

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracker["zi"] = None
        analyze_frame(frame, _pushup_keypoints(390.0, "bent"), tracker)
        assert tracker["stage"] == "down"
        tracker["zi"] = None
        analyze_frame(frame, _pushup_keypoints(280.0, "straight"), tracker)
        assert tracker["reps"] == 1
        assert tracker["stage"] == "up"


# =============================================================================
# Unified metrics payload
# =============================================================================


class TestMetricsPayload:
    def test_same_keys_for_every_status(self):
        tracker = init_tracker(30)
        expected_keys = set(metrics_from_tracker(tracker, "idle").keys())
        for status in ("no person", "finished", "ready", "detecting side..."):
            assert set(metrics_from_tracker(tracker, status).keys()) == expected_keys
        assert "touch_y0" in expected_keys
        assert "touch_y1" in expected_keys
