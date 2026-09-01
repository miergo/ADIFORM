"""Minimal push-up pose tracker."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# COCO pose indices per side
LEFT = {"shoulder": 5, "elbow": 7, "wrist": 9, "hip": 11, "knee": 13, "ankle": 15}
RIGHT = {"shoulder": 6, "elbow": 8, "wrist": 10, "hip": 12, "knee": 14, "ankle": 16}

SIDES = {"left": LEFT, "right": RIGHT}

UP_ELBOW_TOL = 30.0  # tolerance from 145.0 degrees for up position
# UP_ELBOW_DEG = 145.0
UP_BAR_TOL = 0.15  # fraction of locked bar range (shoulder near top bar)
DOWN_BAR_TOL = 0.35  # fraction of locked bar range (shoulder near low bar)
BODY_ALIGN_TOL = 15.0  # degrees from 180 at hip (shoulder-hip-ankle)
KNEE_ANGLE_TOL = 25.0  # degrees from 160 at knee (hip-knee-ankle)
MISS_TRAVEL_FRAC = 0.4  # min travel along bar range before a miss can resolve


def stroke_points(hit: bool, good: bool) -> int:
    if hit:
        return 100 if good else 50
    return 25 if good else 10


def hit_depth_multiplier(peak_travel: float, bar_tol: float) -> tuple[float, str]:
    """1.0x at hit-band edge, 2.0x at full bar reach. Misses use 1.0x (caller)."""
    min_travel = 1.0 - bar_tol
    if peak_travel <= min_travel:
        return 1.0, "hit"
    t = min(1.0, (peak_travel - min_travel) / bar_tol)
    mult = 1.0 + t
    if mult >= 1.95:
        return mult, "full depth"
    if mult > 1.05:
        return mult, "deep"
    return mult, "hit"


def closest_person(boxes: np.ndarray | None) -> int | None:
    """Return index of the largest box (nearest to camera)."""
    if boxes is None or len(boxes) == 0:
        return None
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return int(np.argmax(areas))


def pick_side(kpts: np.ndarray) -> str:
    """Pick left or right side by mean confidence of the six joints."""

    def mean_conf(indices: dict[str, int]) -> float:
        return float(np.mean([kpts[i][2] for i in indices.values()]))

    return "right" if mean_conf(RIGHT) > mean_conf(LEFT) else "left"


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    """Angle ABC in degrees, or None if points are too close."""
    ba = a[:2] - b[:2]
    bc = c[:2] - b[:2]
    na = np.linalg.norm(ba)
    nc = np.linalg.norm(bc)
    if na < 1e-6 or nc < 1e-6:
        return None
    cos = np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def is_ready_pose(
    shoulder_y: float,
    hip_y: float,
    ankle_y: float,
    elbow_deg: float | None,
    body_angle: float | None,
    knee_angle: float | None,
) -> bool:
    """Initial up plank: straight arm, body line, straight leg, ankle below shoulder."""
    if elbow_deg is None or elbow_deg < 180 - UP_ELBOW_TOL:
        return False
    if shoulder_y >= hip_y:
        return False
    if ankle_y <= shoulder_y:
        return False
    if body_angle is None:
        return False
    if body_angle <= 180.0 - BODY_ALIGN_TOL:
        return False
    if knee_angle is None:
        return False
    return knee_angle >= 180.0 - KNEE_ANGLE_TOL


def is_up_pose(
    shoulder_y: float,
    top_bar: float | None,
    low_bar: float | None,
) -> bool:
    """Up rep: shoulder y near the locked top bar."""
    if top_bar is None or low_bar is None:
        return False
    bar_range = low_bar - top_bar
    if bar_range <= 0:
        return False
    return abs(shoulder_y - top_bar) <= UP_BAR_TOL * bar_range


def is_down_pose(
    shoulder_y: float,
    top_bar: float | None,
    low_bar: float | None,
) -> bool:
    """Down rep: shoulder y near the locked low bar."""
    if top_bar is None or low_bar is None:
        return False
    bar_range = low_bar - top_bar
    if bar_range <= 0:
        return False
    return abs(shoulder_y - low_bar) <= DOWN_BAR_TOL * bar_range


@dataclass
class HipResult:
    hip_angle: float | None
    hip_status: str | None  # "aligned" | "hips_up" | "sagging"


class HipPosture:
    """Shoulder-hip-ankle triangle posture analysis."""

    def __init__(
        self,
        align_angle_tol: float = 15.0,
        base_tol_frac: float = 0.05,
    ) -> None:
        self.align_angle_tol = align_angle_tol
        self.base_tol_frac = base_tol_frac

    def analyze(
        self,
        shoulder: np.ndarray,
        hip: np.ndarray,
        ankle: np.ndarray,
    ) -> HipResult:
        """
        Triangle: shoulder-ankle base, sides shoulder-hip and hip-ankle.
        Angle at hip between ankle-hip and shoulder-hip.
        """
        sh, hp, ank = shoulder[:2], hip[:2], ankle[:2]
        hip_angle = joint_angle(sh, hp, ank)
        if hip_angle is None:
            return HipResult(hip_angle=None, hip_status=None)

        base = ank - sh
        base_len_sq = max(float(np.dot(base, base)), 1e-6)
        base_len = float(np.sqrt(base_len_sq))
        t = float(np.clip(np.dot(hp - sh, base) / base_len_sq, 0.0, 1.0))
        on_line = sh + t * base

        # Image y grows downward: positive offset = hip below the base line.
        offset = float(hp[1] - on_line[1])
        base_tol = self.base_tol_frac * base_len
        angle_near_straight = hip_angle >= 180.0 - self.align_angle_tol
        near_base = abs(offset) <= base_tol

        if angle_near_straight and near_base:
            status = "aligned"
        elif hip_angle < 180.0 and offset > base_tol:
            status = "sagging"
        elif hip_angle < 180.0 and offset < -base_tol:
            status = "hips_up"
        else:
            status = None

        return HipResult(hip_angle=hip_angle, hip_status=status)


def pt(xy) -> tuple[int, int]:
    return int(xy[0]), int(xy[1])


def draw_overlay(frame: np.ndarray, out: dict) -> None:
    """Draw joints, posture triangle, bars, and HUD."""
    if out["detected"]:
        wrist = out["wrist"]
        elbow = out["elbow_point"]
        shoulder = out["shoulder"]
        hip = out["hip"]
        ankle = out["ankle"]

        color = (255, 0, 255)
        for p in (wrist, elbow):
            cv2.circle(frame, pt(p), 6, color, -1)

        cv2.line(frame, pt(shoulder), pt(elbow), color, 2)
        cv2.line(frame, pt(elbow), pt(wrist), color, 2)

        status = out.get("hip_status")
        if status == "aligned":
            tri_color = (0, 255, 0)
        elif status == "sagging":
            tri_color = (0, 0, 255)
        elif status == "hips_up":
            tri_color = (0, 165, 255)
        else:
            tri_color = (255, 255, 0)

        cv2.line(frame, pt(shoulder), pt(ankle), tri_color, 4)
        cv2.line(frame, pt(shoulder), pt(hip), tri_color, 2)
        cv2.line(frame, pt(hip), pt(ankle), tri_color, 2)
        for p in (shoulder, hip, ankle):
            cv2.circle(frame, pt(p), 8, tri_color, -1)

        knee = out.get("knee")
        if knee is not None:
            leg_color = (255, 128, 0)
            cv2.line(frame, pt(hip), pt(knee), leg_color, 2)
            cv2.line(frame, pt(knee), pt(ankle), leg_color, 2)
            cv2.circle(frame, pt(knee), 8, leg_color, -1)

    h, w = frame.shape[:2]
    if out["top_bar"] is not None:
        y = int(out["top_bar"])
        cv2.line(frame, (0, y), (w, y), (0, 255, 0), 2)
    if out["low_bar"] is not None:
        y = int(out["low_bar"])
        cv2.line(frame, (0, y), (w, y), (0, 0, 255), 2)

    elbow_txt = f"{out['elbow']:.0f}" if out["elbow"] is not None else "--"
    hip_angle_txt = (
        f"{out['hip_angle']:.0f}" if out.get("hip_angle") is not None else "--"
    )
    hip_status_txt = out.get("hip_status") or "--"
    cv2.putText(
        frame,
        f"REPS: {out['reps']}  stage: {out['stage']}  elbow: {elbow_txt}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"hip: {hip_status_txt}  angle: {hip_angle_txt}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
    )
    knee_angle_txt = (
        f"{out['knee_angle']:.0f}" if out.get("knee_angle") is not None else "--"
    )
    cv2.putText(
        frame,
        f"knee: {knee_angle_txt}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
    )


def _empty_result(reps: int = 0, stage: str = "up", score: int = 0) -> dict:
    return {
        "detected": False,
        "wrist": None,
        "elbow_point": None,
        "shoulder": None,
        "hip": None,
        "knee": None,
        "ankle": None,
        "elbow": None,
        "hip_angle": None,
        "hip_status": None,
        "knee_angle": None,
        "top_bar": None,
        "low_bar": None,
        "stage": stage,
        "reps": reps,
        "score": score,
    }


class Pushup:
    """Stateful push-up rep counter."""

    def __init__(self) -> None:
        self.top_bar: float | None = None
        self.low_bar: float | None = None
        self.stage: str = "up"
        self.reps: int = 0
        self.score: int = 0
        self.y0_hits: int = 0
        self.y1_hits: int = 0
        self.posture_good: int = 0
        self.posture_bad: int = 0
        self._peak_travel: float = 0.0
        self._left_start_bar: bool = False
        self.award_seq: int = 0
        self.last_multiplier: float = 1.0
        self.last_award_label: str = ""
        self.last_award_points: int = 0
        self.hip = HipPosture()

    def reset_reps(self) -> None:
        """Clear count after calibration. Keep locked bars."""
        self.reps = 0
        self.stage = "up"
        self.score = 0
        self.y0_hits = 0
        self.y1_hits = 0
        self.posture_good = 0
        self.posture_bad = 0
        self._peak_travel = 0.0
        self._left_start_bar = False
        self.award_seq = 0
        self.last_multiplier = 1.0
        self.last_award_label = ""
        self.last_award_points = 0

    def _award(self, hit: bool, hip_status: str | None, bar: int) -> None:
        if hit and bar == 1:
            mult, label = hit_depth_multiplier(self._peak_travel, DOWN_BAR_TOL)
        elif hit and bar == 0:
            mult, label = hit_depth_multiplier(self._peak_travel, UP_BAR_TOL)
        else:
            mult, label = 1.0, "miss"

        good = hip_status == "aligned"
        base = stroke_points(hit, good)
        pts = int(round(base * mult))
        self.score += pts
        if hit and bar == 0:
            self.y0_hits += 1
        elif hit and bar == 1:
            self.y1_hits += 1
        if good:
            self.posture_good += 1
        else:
            self.posture_bad += 1

        self.award_seq += 1
        self.last_multiplier = mult
        self.last_award_label = label
        self.last_award_points = pts

    def _award_fields(self) -> dict:
        return {
            "award_seq": self.award_seq,
            "last_multiplier": self.last_multiplier,
            "last_award_label": self.last_award_label,
            "last_award_points": self.last_award_points,
        }

    def update(self, boxes: np.ndarray | None, keypoints: np.ndarray | None) -> dict:
        """Process one frame of detections. boxes: (N,4), keypoints: (N,17,3)."""
        if boxes is None or keypoints is None:
            return self._result(detected=False)

        idx = closest_person(boxes)
        if idx is None:
            return self._result(detected=False)

        kpts = keypoints[idx]
        side = pick_side(kpts)
        indices = SIDES[side]

        shoulder = kpts[indices["shoulder"]]
        elbow_pt = kpts[indices["elbow"]]
        wrist = kpts[indices["wrist"]]
        hip = kpts[indices["hip"]]
        knee = kpts[indices["knee"]]
        ankle = kpts[indices["ankle"]]

        elbow_deg = joint_angle(shoulder, elbow_pt, wrist)
        hip_result = self.hip.analyze(shoulder, hip, ankle)
        knee_angle = joint_angle(hip, knee, ankle)

        if self.top_bar is None and is_ready_pose(
            shoulder[1],
            hip[1],
            ankle[1],
            elbow_deg,
            hip_result.hip_angle,
            knee_angle,
        ):
            self.top_bar = float(shoulder[1])
            self.low_bar = float(ankle[1])

        if self.top_bar is not None:
            bar_range = self.low_bar - self.top_bar
            if bar_range > 0:
                if self.stage == "up":
                    travel = (shoulder[1] - self.top_bar) / bar_range
                    self._peak_travel = max(self._peak_travel, travel)
                    if not is_up_pose(shoulder[1], self.top_bar, self.low_bar):
                        self._left_start_bar = True
                    if is_down_pose(shoulder[1], self.top_bar, self.low_bar):
                        self._award(True, hip_result.hip_status, bar=1)
                        self.stage = "down"
                        self._peak_travel = 0.0
                        self._left_start_bar = False
                    elif (
                        is_up_pose(shoulder[1], self.top_bar, self.low_bar)
                        and self._left_start_bar
                        and self._peak_travel >= MISS_TRAVEL_FRAC
                    ):
                        self._award(False, hip_result.hip_status, bar=1)
                        self._peak_travel = 0.0
                        self._left_start_bar = False
                elif self.stage == "down":
                    travel = (self.low_bar - shoulder[1]) / bar_range
                    self._peak_travel = max(self._peak_travel, travel)
                    if not is_down_pose(shoulder[1], self.top_bar, self.low_bar):
                        self._left_start_bar = True
                    if is_up_pose(shoulder[1], self.top_bar, self.low_bar):
                        self._award(True, hip_result.hip_status, bar=0)
                        self.stage = "up"
                        self.reps += 1
                        self._peak_travel = 0.0
                        self._left_start_bar = False
                    elif (
                        is_down_pose(shoulder[1], self.top_bar, self.low_bar)
                        and self._left_start_bar
                        and self._peak_travel >= MISS_TRAVEL_FRAC
                    ):
                        self._award(False, hip_result.hip_status, bar=0)
                        self._peak_travel = 0.0
                        self._left_start_bar = False

        return self._result(
            detected=True,
            wrist=wrist[:2],
            elbow_point=elbow_pt[:2],
            shoulder=shoulder[:2],
            hip=hip[:2],
            knee=knee[:2],
            ankle=ankle[:2],
            elbow=elbow_deg,
            hip_angle=hip_result.hip_angle,
            hip_status=hip_result.hip_status,
            knee_angle=knee_angle,
        )

    def _result(
        self,
        detected: bool,
        wrist=None,
        elbow_point=None,
        shoulder=None,
        hip=None,
        knee=None,
        ankle=None,
        elbow=None,
        hip_angle=None,
        hip_status=None,
        knee_angle=None,
    ) -> dict:
        if not detected:
            out = _empty_result(self.reps, self.stage, self.score)
            out["top_bar"] = self.top_bar
            out["low_bar"] = self.low_bar
            out.update(self._award_fields())
            return out
        return {
            "detected": True,
            "wrist": wrist,
            "elbow_point": elbow_point,
            "shoulder": shoulder,
            "hip": hip,
            "knee": knee,
            "ankle": ankle,
            "elbow": elbow,
            "hip_angle": hip_angle,
            "hip_status": hip_status,
            "knee_angle": knee_angle,
            "top_bar": self.top_bar,
            "low_bar": self.low_bar,
            "stage": self.stage,
            "reps": self.reps,
            "score": self.score,
            **self._award_fields(),
        }
