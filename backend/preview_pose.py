"""Offline preview: run pose tracker on a video and save annotated output."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from pose import (
    BODY_ALIGN_TOL,
    DOWN_BAR_TOL,
    KNEE_ANGLE_TOL,
    MISS_TRAVEL_FRAC,
    UP_BAR_TOL,
    UP_ELBOW_TOL,
    Pushup,
    draw_overlay,
    is_ready_pose,
    pt,
)

INPUT_VIDEO = Path("uploads/8435261-uhd_4096_2160_25fps.mp4")
OUTPUT_DIR = Path("uploads/test_preview4")
MODEL_PATH = Path("model/yolo26n-pose.pt")

_PREVIEW_RE = re.compile(r"^pushup_test_(\d+)_preview\.mp4$")

ELBOW_MIN = 175.0 - UP_ELBOW_TOL
BODY_MIN = 180.0 - BODY_ALIGN_TOL
KNEE_MIN = 160.0 - KNEE_ANGLE_TOL


def draw_scoring_hud(frame: np.ndarray, tracker: Pushup) -> None:
    """Preview-only: live score and last depth award on the output video."""
    last_color = (0, 255, 0) if tracker.last_multiplier > 1.0 else (0, 255, 255)

    lines = [
        f"SCORE: {tracker.score}",
        f"LAST: x{tracker.last_multiplier:.2f} {tracker.last_award_label}",
        f"REPS: {tracker.reps}",
    ]
    y = 150
    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            last_color if i == 1 else (0, 255, 255),
            2,
        )
        y += 32


def draw_debug_constants(frame: np.ndarray, out: dict) -> None:
    """Preview-only: visualize tolerance constants on top of draw_overlay."""
    h, w = frame.shape[:2]
    top_bar = out.get("top_bar")
    low_bar = out.get("low_bar")

    if top_bar is not None and low_bar is not None:
        bar_range = low_bar - top_bar
        if bar_range > 0:
            up_band = UP_BAR_TOL * bar_range
            down_band = DOWN_BAR_TOL * bar_range
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (0, max(0, int(top_bar - up_band))),
                (w, min(h, int(top_bar + up_band))),
                (0, 255, 0),
                -1,
            )
            cv2.rectangle(
                overlay,
                (0, max(0, int(low_bar - down_band))),
                (w, min(h, int(low_bar + down_band))),
                (0, 0, 255),
                -1,
            )
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

            miss_y = int(top_bar + MISS_TRAVEL_FRAC * bar_range)
            cv2.line(frame, (0, miss_y), (w, miss_y), (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"miss travel {MISS_TRAVEL_FRAC:.0%}",
                (max(0, w - 260), max(20, miss_y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1,
            )

    if not out.get("detected"):
        return

    elbow = out.get("elbow")
    hip_angle = out.get("hip_angle")
    knee_angle = out.get("knee_angle")
    shoulder = out["shoulder"]
    elbow_pt = out["elbow_point"]
    wrist = out["wrist"]
    hip = out["hip"]
    knee = out.get("knee")
    ankle = out["ankle"]

    elbow_ok = elbow is not None and elbow >= ELBOW_MIN
    knee_ok = knee_angle is not None and knee_angle >= KNEE_MIN

    arm_color = (0, 255, 0) if elbow_ok else (0, 0, 255)
    cv2.line(frame, pt(shoulder), pt(elbow_pt), arm_color, 4)
    cv2.line(frame, pt(elbow_pt), pt(wrist), arm_color, 4)

    if knee is not None:
        leg_color = (0, 255, 0) if knee_ok else (0, 0, 255)
        cv2.line(frame, pt(hip), pt(knee), leg_color, 4)
        cv2.line(frame, pt(knee), pt(ankle), leg_color, 4)

    ready = is_ready_pose(
        shoulder[1],
        hip[1],
        ankle[1],
        elbow,
        hip_angle,
        knee_angle,
    )
    ready_txt = "READY" if ready else "NOT READY"
    ready_color = (0, 255, 0) if ready else (0, 0, 255)

    cv2.putText(
        frame,
        f"elbow min {ELBOW_MIN:.0f}  body min {BODY_MIN:.0f}  knee min {KNEE_MIN:.0f}",
        (20, h - 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"up tol {UP_BAR_TOL:.0%}  down tol {DOWN_BAR_TOL:.0%}",
        (20, h - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        ready_txt,
        (20, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        ready_color,
        2,
    )


def shoulder_depth(shoulder_y: float, top_bar: float, low_bar: float) -> float | None:
    """Fraction from top bar (0) to low bar (1)."""
    bar_range = low_bar - top_bar
    if bar_range <= 0:
        return None
    return (shoulder_y - top_bar) / bar_range


@dataclass
class StrokePeak:
    frame: int
    depth: float
    hit: bool


class DownStrokeCollector:
    """Preview-only: track live depth and per-down-stroke peaks."""

    DEPTH_START = 0.05
    DEPTH_FALL_EPS = 0.02
    DEPTH_FALL_FRAMES = 2

    def __init__(self) -> None:
        self.live: list[tuple[int, float]] = []
        self.peaks: list[StrokePeak] = []
        self._stroke_active = False
        self._at_bottom_after_hit = False
        self._peak_depth = 0.0
        self._peak_frame = 0
        self._prev_depth: float | None = None
        self._depth_fall_streak = 0

    def on_frame(
        self,
        frame: int,
        depth: float | None,
        tracker: Pushup,
        prev_score: int,
        prev_y1_hits: int,
        prev_stage: str,
    ) -> None:
        if depth is None:
            return

        self.live.append((frame, depth))

        score_up = tracker.score > prev_score
        y1_up = tracker.y1_hits > prev_y1_hits

        if not self._stroke_active and not self._at_bottom_after_hit:
            if prev_stage == "up" and depth > self.DEPTH_START:
                self._stroke_active = True
                self._peak_depth = depth
                self._peak_frame = frame

        if self._stroke_active or self._at_bottom_after_hit:
            if depth > self._peak_depth:
                self._peak_depth = depth
                self._peak_frame = frame

        if score_up and not y1_up and prev_stage == "up":
            self.peaks.append(
                StrokePeak(frame=self._peak_frame, depth=self._peak_depth, hit=False)
            )
            self._reset_stroke()
            self._prev_depth = depth
            return

        if y1_up:
            self._at_bottom_after_hit = True
            self._stroke_active = False

        if self._at_bottom_after_hit and self._prev_depth is not None:
            if depth < self._peak_depth - self.DEPTH_FALL_EPS:
                self._depth_fall_streak += 1
            else:
                self._depth_fall_streak = 0
            if self._depth_fall_streak >= self.DEPTH_FALL_FRAMES:
                self.peaks.append(
                    StrokePeak(frame=self._peak_frame, depth=self._peak_depth, hit=True)
                )
                self._reset_stroke()

        self._prev_depth = depth

    def _reset_stroke(self) -> None:
        self._stroke_active = False
        self._at_bottom_after_hit = False
        self._peak_depth = 0.0
        self._peak_frame = 0
        self._depth_fall_streak = 0


def save_depth_graph(
    path: Path,
    live: list[tuple[int, float]],
    peaks: list[StrokePeak],
    fps: float,
) -> None:
    """Save depth-over-time chart with stroke peaks."""
    if not live:
        return

    times = [frame / fps for frame, _ in live]
    depths = [depth for _, depth in live]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        times, depths, color="steelblue", alpha=0.6, linewidth=1, label="live depth"
    )

    hit_times = [peak.frame / fps for peak in peaks if peak.hit]
    hit_depths = [peak.depth for peak in peaks if peak.hit]
    miss_times = [peak.frame / fps for peak in peaks if not peak.hit]
    miss_depths = [peak.depth for peak in peaks if not peak.hit]

    if hit_times:
        ax.scatter(
            hit_times,
            hit_depths,
            c="green",
            s=50,
            zorder=5,
            label="down hit peak",
        )
    if miss_times:
        ax.scatter(
            miss_times,
            miss_depths,
            c="orange",
            s=50,
            zorder=5,
            label="down miss peak",
        )

    ax.axhline(1.0, linestyle="--", color="red", alpha=0.7, label="low bar")
    ax.axhline(
        1.0 - DOWN_BAR_TOL,
        linestyle=":",
        color="gray",
        alpha=0.7,
        label=f"hit band ({1.0 - DOWN_BAR_TOL:.0%})",
    )

    ax.set_xlabel("time (s)")
    ax.set_ylabel("depth (0=top, 1=low bar)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def next_output_path(output_dir: Path) -> Path:
    """Return uploads/test_preview/pushup_testN_preview.mp4 with N = max existing + 1."""
    output_dir.mkdir(parents=True, exist_ok=True)
    max_n = 0
    for path in output_dir.glob("pushup_test_*_preview.mp4"):
        match = _PREVIEW_RE.match(path.name)
        if match:
            max_n = max(max_n, int(match.group(1)))
    return output_dir / f"pushup_test_{max_n + 1}_preview.mp4"


def main() -> None:
    if not INPUT_VIDEO.is_file():
        raise FileNotFoundError(f"Input video not found: {INPUT_VIDEO.resolve()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = next_output_path(OUTPUT_DIR)

    cap = cv2.VideoCapture(str(INPUT_VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {INPUT_VIDEO}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps < 1:
        fps = 30.0

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    model = YOLO(str(MODEL_PATH))
    tracker = Pushup()
    depth_collector = DownStrokeCollector()
    frame_count = 0
    prev_award_seq = 0
    award_log: list[str] = []
    graph_path = output_path.parent / f"{output_path.stem}_depth.png"

    print(f"Input:  {INPUT_VIDEO}")
    print(f"Output: {output_path}")
    print(f"Graph:  {graph_path}")
    print(f"Size:   {w}x{h} @ {fps:.1f} fps")
    print(
        f"Depth bonus: 1.0x at hit-band edge, 2.0x at full bar "
        f"(down tol {DOWN_BAR_TOL:.0%}, up tol {UP_BAR_TOL:.0%})"
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model.predict(frame, verbose=False)[0]
        boxes = None
        keypoints = None
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
        if result.keypoints is not None and result.keypoints.data is not None:
            keypoints = result.keypoints.data.cpu().numpy().astype(np.float64)

        prev_score = tracker.score
        prev_y1_hits = tracker.y1_hits
        prev_stage = tracker.stage

        out = tracker.update(boxes, keypoints)

        if tracker.award_seq > prev_award_seq:
            award_log.append(
                f"x{tracker.last_multiplier:.2f} ({tracker.last_award_label}) "
                f"-> +{tracker.last_award_points}"
            )
            prev_award_seq = tracker.award_seq

        if (
            tracker.top_bar is not None
            and tracker.low_bar is not None
            and out.get("detected")
            and out.get("shoulder") is not None
        ):
            depth = shoulder_depth(
                float(out["shoulder"][1]), tracker.top_bar, tracker.low_bar
            )
            depth_collector.on_frame(
                frame_count,
                depth,
                tracker,
                prev_score,
                prev_y1_hits,
                prev_stage,
            )

        draw_overlay(frame, out)
        draw_scoring_hud(frame, tracker)
        draw_debug_constants(frame, out)
        writer.write(frame)
        frame_count += 1

    cap.release()
    writer.release()

    save_depth_graph(graph_path, depth_collector.live, depth_collector.peaks, fps)

    print(f"Frames: {frame_count}")
    print(f"Final reps:        {tracker.reps}")
    print(f"Final score:       {tracker.score}")
    print(f"Y0 hits (top):     {tracker.y0_hits}")
    print(f"Y1 hits (low):     {tracker.y1_hits}")
    print(f"Posture good/bad:  {tracker.posture_good}/{tracker.posture_bad}")
    print(f"top_bar:           {tracker.top_bar}")
    print(f"low_bar:           {tracker.low_bar}")
    print(
        f"Depth peaks:       {len(depth_collector.peaks)} "
        f"({sum(1 for p in depth_collector.peaks if p.hit)} hits, "
        f"{sum(1 for p in depth_collector.peaks if not p.hit)} misses)"
    )
    if award_log:
        print("Awards:")
        for entry in award_log:
            print(f"  {entry}")
    print(f"Saved preview: {output_path.resolve()}")
    print(f"Saved graph:   {graph_path.resolve()}")


if __name__ == "__main__":
    main()
