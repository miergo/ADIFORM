"""Session phase tests (no YOLO, no camera)."""

import numpy as np

from session import WorkoutSession


def _metrics(**overrides):
    base = {
        "detected": True,
        "wrist": np.array([200.0, 100.0]),
        "elbow_point": np.array([150.0, 100.0]),
        "shoulder": np.array([100.0, 100.0]),
        "hip": np.array([100.0, 200.0]),
        "ankle": np.array([100.0, 300.0]),
        "elbow_angle": 175.0,
        "hip_angle": 180.0,
        "hip_status": "aligned",
        "top_bar": None,
        "low_bar": None,
        "stage": "up",
        "reps": 0,
        "score": 0,
        "award_seq": 0,
        "last_multiplier": 1.0,
        "last_award_label": "",
        "last_award_points": 0,
    }
    base.update(overrides)
    return base


class TestWorkoutSession:
    def test_starts_idle(self):
        s = WorkoutSession(model=object())
        assert s.get_status()["phase"] == "idle"
        assert s.get_status()["running"] is False

    def test_calibrating_to_ready_to_active(self):
        s = WorkoutSession(model=object())
        s.running = True
        s.phase = "calibrating"
        s.apply_metrics(_metrics(top_bar=None, low_bar=None))
        assert s.phase == "calibrating"

        s.apply_metrics(_metrics(top_bar=100.0, low_bar=300.0))
        assert s.phase == "ready"
        assert s.get_status()["phase"] == "ready"
        assert s.get_status()["top_bar"] == 100.0

        s.go()
        assert s.phase == "active"
        assert s.get_status()["phase"] == "active"
        assert s.get_status()["reps"] == 0
        assert s.tracker.reps == 0

    def test_go_ignored_until_ready(self):
        s = WorkoutSession(model=object())
        s.phase = "calibrating"
        s.go()
        assert s.phase == "calibrating"

    def test_stop_returns_summary(self):
        s = WorkoutSession(model=object())
        s.running = True
        s.phase = "active"
        s.tracker.reps = 4
        s.tracker.score = 350
        s.tracker.top_bar_hits = 2
        s.tracker.low_bar_hits = 3
        s.tracker.posture_good = 4
        s.tracker.posture_bad = 1
        summary = s.stop()
        assert summary["reps"] == 4
        assert summary["score"] == 350
        assert summary["top_bar_hits"] == 2
        assert summary["low_bar_hits"] == 3
        assert summary["posture_good"] == 4
        assert summary["posture_bad"] == 1
        assert s.phase == "idle"
        assert s.get_status()["running"] is False

    def test_go_zeros_score(self):
        s = WorkoutSession(model=object())
        s.running = True
        s.phase = "ready"
        s.tracker.top_bar = 100.0
        s.tracker.low_bar = 300.0
        s.tracker.score = 150
        s.tracker.award_seq = 3
        s.apply_metrics(
            _metrics(
                top_bar=100.0,
                low_bar=300.0,
                score=150,
                award_seq=3,
                last_multiplier=2.0,
                last_award_label="full depth",
                last_award_points=200,
            )
        )
        s.go()
        assert s.get_status()["score"] == 0
        assert s.tracker.score == 0
        assert s.get_status()["award_seq"] == 0
        assert s.get_status()["last_multiplier"] == 1.0

    def test_apply_metrics_passes_award_fields(self):
        s = WorkoutSession(model=object())
        s.running = True
        s.phase = "active"
        s.apply_metrics(
            _metrics(
                award_seq=2,
                last_multiplier=1.5,
                last_award_label="deep",
                last_award_points=150,
            )
        )
        status = s.get_status()
        assert status["award_seq"] == 2
        assert status["last_multiplier"] == 1.5
        assert status["last_award_label"] == "deep"
        assert status["last_award_points"] == 150

    def test_video_pauses_on_ready(self):
        s = WorkoutSession(model=object())
        s._source = "video"
        s.phase = "ready"
        assert s._should_advance() is False

    def test_webcam_does_not_pause_on_ready(self):
        s = WorkoutSession(model=object())
        s._source = "webcam"
        s.phase = "ready"
        assert s._should_advance() is True

    def test_video_resumes_after_go(self):
        s = WorkoutSession(model=object())
        s._source = "video"
        s.running = True
        s.phase = "ready"
        s.go()
        assert s.phase == "active"
        assert s._should_advance() is True

    def test_restart_while_running_keeps_video_source(self):
        class FakeCap:
            def read(self):
                return False, None

            def get(self, _prop):
                return 30.0

            def release(self):
                pass

        s = WorkoutSession(model=object())
        s.running = True
        s._source = "video"
        s._start(FakeCap(), "video")
        if s._thread is not None:
            s._thread.join(timeout=2)
        assert s._source == "video"
        s.phase = "ready"
        assert s._should_advance() is False
