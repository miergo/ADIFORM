"""Edge tests for session helpers (no threads / HTTP / YOLO required)."""

from __future__ import annotations

from pathlib import Path

from pose import metrics_from_tracker, init_tracker
from session import WorkoutSession, downsample_series, file_ready


# =============================================================================
# downsample_series
# =============================================================================


class TestDownsampleSeries:
    def test_empty(self):
        assert downsample_series([]) == []

    def test_shorter_than_max_points(self):
        history = [
            {"t": 10.0, "reps": 0, "elbow": 90.0},
            {"t": 11.0, "reps": 1, "elbow": 160.0},
        ]
        series = downsample_series(history, max_points=300)
        assert len(series) == 2
        assert series[0]["t"] == 0.0
        assert series[1]["t"] == 1.0
        assert series[1]["reps"] == 1

    def test_longer_keeps_first_and_last(self):
        history = [{"t": float(i), "reps": i} for i in range(1000)]
        series = downsample_series(history, max_points=50)
        assert len(series) <= 50
        assert series[0]["t"] == 0.0
        assert series[0]["reps"] == 0
        assert series[-1]["reps"] == 999

    def test_missing_t_values(self):
        history = [{"reps": 0}, {"t": None, "reps": 1}, {"t": 5.0, "reps": 2}]
        series = downsample_series(history, max_points=10)
        assert len(series) == 3
        assert series[0]["t"] == 0.0


# =============================================================================
# file_ready
# =============================================================================


class TestFileReady:
    def test_missing_path(self):
        assert file_ready(None) is False
        assert file_ready("") is False
        assert file_ready("does_not_exist_xyz.mp4") is False

    def test_empty_file(self, tmp_path: Path):
        empty = tmp_path / "empty.mp4"
        empty.write_bytes(b"")
        assert file_ready(str(empty)) is False

    def test_real_file(self, tmp_path: Path):
        real = tmp_path / "clip.mp4"
        real.write_bytes(b"data")
        assert file_ready(str(real)) is True


# =============================================================================
# playback_path
# =============================================================================


class TestPlaybackPath:
    def test_none_when_nothing_exists(self):
        session = WorkoutSession()
        assert session.playback_path() is None

    def test_prefers_finished_recording_over_original(self, tmp_path: Path):
        session = WorkoutSession()
        original = tmp_path / "original.mp4"
        recording = tmp_path / "recorded.mp4"
        original.write_bytes(b"orig")
        recording.write_bytes(b"rec")
        session.original_path = str(original)
        session.record_path = str(recording)
        session.writer = None  # writer finished
        assert session.playback_path() == str(recording)

    def test_falls_back_to_original_while_writer_open(self, tmp_path: Path):
        session = WorkoutSession()
        original = tmp_path / "original.mp4"
        recording = tmp_path / "recorded.mp4"
        original.write_bytes(b"orig")
        recording.write_bytes(b"rec")
        session.original_path = str(original)
        session.record_path = str(recording)
        session.writer = object()  # still writing
        assert session.playback_path() == str(original)


# =============================================================================
# summary good/issue counting
# =============================================================================


class TestSummaryCounting:
    def test_good_and_issue_frames(self):
        session = WorkoutSession()
        session.history = [
            # Real posture readings (status text may still be a phase label)
            {"status": "ready", "posture_status": "good", "t": 1.0},
            {"status": "ready", "posture_status": "bad", "t": 2.0},
            {"status": "calibrating... 50%", "posture_status": "good", "t": 3.0},
            # Phase / empty posture — must NOT inflate counters
            {"status": "ready", "posture_status": None, "t": 4.0},
            {"status": "no person", "posture_status": None, "t": 5.0},
            {"status": "finished", "posture_status": None, "t": 6.0},
            {"status": "Hips Too High: 150°", "posture_status": "bad", "t": 7.0},
        ]
        result = session.summary()
        assert result["good_frames"] == 2
        assert result["issue_frames"] == 2
        assert result["total_samples"] == 7


# =============================================================================
# Unified payload from pose (session uses the same helper)
# =============================================================================


class TestUnifiedPayload:
    def test_no_person_and_finished_include_touch_fields(self):
        tracker = init_tracker(30)
        for status in ("no person", "finished"):
            metrics = metrics_from_tracker(tracker, status)
            assert "touch_y0" in metrics
            assert "touch_y1" in metrics
            assert set(metrics.keys()) == {
                "reps",
                "stage",
                "side",
                "status",
                "elbow",
                "eye_height",
                "speed",
                "touch_y0",
                "touch_y1",
                "posture_angle",
                "posture_status",
                "phase",
                "calib_progress",
            }
