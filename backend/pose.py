"""Minimal push-up pose tracker."""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

# COCO pose indices per side
LEFT = {"shoulder": 5, "elbow": 7, "wrist": 9, "hip": 11, "knee": 13, "ankle": 15}
RIGHT = {"shoulder": 6, "elbow": 8, "wrist": 10, "hip": 12, "knee": 14, "ankle": 16}

SIDES = {"left": LEFT, "right": RIGHT}

UP_ELBOW_TOL = 35.0  # tolerance from 145.0 degrees for up position
# UP_ELBOW_DEG = 145.0
UP_BAR_TOL = 0.15  # fraction of locked bar range (shoulder near top bar)
DOWN_BAR_TOL = 0.45  # fraction of locked bar range (shoulder near low bar)
BODY_ALIGN_TOL = 25.0  # degrees from 180 at hip (shoulder-hip-ankle)
KNEE_ANGLE_TOL = 30.0  # degrees from 160 at knee (hip-knee-ankle)
MISS_TRAVEL_FRAC = 0.5  # min travel along bar range before a miss can resolve
CALIB_HOLD_SEC = 1.5  # continuous ready plank before locking bars
EMA_ALPHA = 0.4  # joint smoothing: smoothed = (1-a)*prev + a*raw
CONF_MIN = 0.3  # hold previous xy when joint confidence is below this
JUMP_PX = 50.0  # snap to raw when a joint jumps farther than this
IOU_REACQUIRE = 0.3  # min IoU vs last box to adopt a new track id
JOINT_KEYS = ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")


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


def largest_box_index(boxes: np.ndarray | None) -> int | None:
    """Return index of the largest box (nearest to camera)."""
    if boxes is None or len(boxes) == 0:
        return None
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return int(np.argmax(areas))


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two xyxy boxes."""
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def best_iou_index(boxes: np.ndarray, ref: np.ndarray) -> tuple[int, float]:
    """Return (index, iou) of the box with highest IoU vs ref."""
    best_i = 0
    best_iou = -1.0
    for i in range(len(boxes)):
        iou = box_iou(ref, boxes[i])
        if iou > best_iou:
            best_iou = iou
            best_i = i
    return best_i, best_iou


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


def legs_extended(knee_angle: float | None) -> bool:
    """Straight-enough leg for full push-ups (same threshold as ready pose)."""
    return knee_angle is not None and knee_angle >= 180.0 - KNEE_ANGLE_TOL


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
    return legs_extended(knee_angle)


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
    """Draw joints, posture triangle, and bars."""
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


def _empty_result(reps: int = 0, stage: str = "up", score: int = 0) -> dict:
    return {
        "detected": False,
        "wrist": None,
        "elbow_point": None,
        "shoulder": None,
        "hip": None,
        "knee": None,
        "ankle": None,
        "elbow_angle": None,
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
        self.top_bar_hits: int = 0
        self.low_bar_hits: int = 0
        self.posture_good: int = 0
        self.posture_bad: int = 0
        self._peak_travel: float = 0.0
        self._left_start_bar: bool = False
        self.award_seq: int = 0
        self.last_multiplier: float = 1.0
        self.last_award_label: str = ""
        self.last_award_points: int = 0
        self._calib_since: float | None = None
        self._locked_id: int | None = None
        self._locked_box: np.ndarray | None = None
        self._side: str | None = None
        self._smooth: dict[str, np.ndarray] = {}
        self.hip = HipPosture()

    def reset_reps(self) -> None:
        """Clear count after calibration. Keep locked bars."""
        self.reps = 0
        self.stage = "up"
        self.score = 0
        self.top_bar_hits = 0
        self.low_bar_hits = 0
        self.posture_good = 0
        self.posture_bad = 0
        self._peak_travel = 0.0
        self._left_start_bar = False
        self.award_seq = 0
        self.last_multiplier = 1.0
        self.last_award_label = ""
        self.last_award_points = 0

    def _select_person(
        self,
        boxes: np.ndarray,
        ids: np.ndarray | None,
    ) -> int | None:
        """Sticky person: lock track id, re-acquire via IoU if id is renumbered."""
        if len(boxes) == 0:
            return None

        use_ids = ids is not None and len(ids) == len(boxes)

        if not use_ids:
            if self._locked_box is not None:
                idx, iou = best_iou_index(boxes, self._locked_box)
                if iou < IOU_REACQUIRE:
                    idx = largest_box_index(boxes)
            else:
                idx = largest_box_index(boxes)
            if idx is None:
                return None
            self._locked_box = boxes[idx].astype(np.float64).copy()
            return idx

        if self._locked_id is None:
            idx = largest_box_index(boxes)
            if idx is None:
                return None
            self._locked_id = int(ids[idx])
            self._locked_box = boxes[idx].astype(np.float64).copy()
            return idx

        matches = np.where(ids == self._locked_id)[0]
        if len(matches) > 0:
            idx = int(matches[0])
            self._locked_box = boxes[idx].astype(np.float64).copy()
            return idx

        # Id gone: re-acquire only if a box still overlaps the last known person.
        if self._locked_box is None:
            return None
        idx, iou = best_iou_index(boxes, self._locked_box)
        if iou < IOU_REACQUIRE:
            return None
        self._locked_id = int(ids[idx])
        self._locked_box = boxes[idx].astype(np.float64).copy()
        return idx

    def _smooth_joint(self, key: str, raw: np.ndarray) -> np.ndarray:
        """EMA on xy; hold previous when confidence is low; snap on large jumps."""
        xy = raw[:2].astype(np.float64)
        conf = float(raw[2]) if raw.shape[0] > 2 else 1.0
        prev = self._smooth.get(key)
        if prev is None:
            self._smooth[key] = xy.copy()
            return self._smooth[key]
        if conf < CONF_MIN:
            return prev
        if float(np.linalg.norm(xy - prev)) >= JUMP_PX:
            self._smooth[key] = xy.copy()
            return self._smooth[key]
        smoothed = (1.0 - EMA_ALPHA) * prev + EMA_ALPHA * xy
        self._smooth[key] = smoothed
        return smoothed

    def _award(self, hit: bool, hip_status: str | None, bar: str) -> None:
        if hit and bar == "low":
            mult, label = hit_depth_multiplier(self._peak_travel, DOWN_BAR_TOL)
        elif hit and bar == "top":
            mult, label = hit_depth_multiplier(self._peak_travel, UP_BAR_TOL)
        else:
            mult, label = 1.0, "miss"

        good = hip_status == "aligned"
        base = stroke_points(hit, good)
        pts = int(round(base * mult))
        self.score += pts
        if hit and bar == "top":
            self.top_bar_hits += 1
        elif hit and bar == "low":
            self.low_bar_hits += 1
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

    def update(
        self,
        boxes: np.ndarray | None,
        keypoints: np.ndarray | None,
        ids: np.ndarray | None = None,
    ) -> dict:
        """Process one frame. boxes: (N,4), keypoints: (N,17,3), ids: (N,) optional."""
        if boxes is None or keypoints is None:
            if self.top_bar is None:
                self._calib_since = None
            return self._result(detected=False)

        idx = self._select_person(boxes, ids)
        if idx is None:
            if self.top_bar is None:
                self._calib_since = None
            return self._result(detected=False)

        kpts = keypoints[idx]
        if self._side is None:
            self._side = pick_side(kpts)
        indices = SIDES[self._side]

        raw = {key: kpts[indices[key]] for key in JOINT_KEYS}
        smoothed = {key: self._smooth_joint(key, raw[key]) for key in JOINT_KEYS}
        shoulder = smoothed["shoulder"]
        elbow_pt = smoothed["elbow"]
        wrist = smoothed["wrist"]
        hip = smoothed["hip"]
        knee = smoothed["knee"]
        ankle = smoothed["ankle"]

        elbow_deg = joint_angle(shoulder, elbow_pt, wrist)
        hip_result = self.hip.analyze(shoulder, hip, ankle)
        knee_angle = joint_angle(hip, knee, ankle)
        knee_conf = float(raw["knee"][2]) if raw["knee"].shape[0] > 2 else 1.0
        legs_ok = knee_conf >= CONF_MIN and legs_extended(knee_angle)
        elbow_locked = elbow_deg is not None and elbow_deg >= 180.0 - UP_ELBOW_TOL
        elbow_bent = elbow_deg is not None and elbow_deg < 180.0 - UP_ELBOW_TOL

        if self.top_bar is None:
            ready = is_ready_pose(
                shoulder[1],
                hip[1],
                ankle[1],
                elbow_deg,
                hip_result.hip_angle,
                knee_angle,
            )
            if ready:
                now = time.monotonic()
                if self._calib_since is None:
                    self._calib_since = now
                elif now - self._calib_since >= CALIB_HOLD_SEC:
                    self.top_bar = float(shoulder[1])
                    self.low_bar = float(ankle[1])
                    self._calib_since = None
            else:
                self._calib_since = None

        if self.top_bar is not None:
            bar_range = self.low_bar - self.top_bar
            if bar_range > 0:
                if self.stage == "up":
                    travel = (shoulder[1] - self.top_bar) / bar_range
                    self._peak_travel = max(self._peak_travel, travel)
                    if not is_up_pose(shoulder[1], self.top_bar, self.low_bar):
                        self._left_start_bar = True
                    if is_down_pose(shoulder[1], self.top_bar, self.low_bar):
                        if legs_ok and elbow_bent:
                            self._award(True, hip_result.hip_status, bar="low")
                            self.stage = "down"
                            self._peak_travel = 0.0
                            self._left_start_bar = False
                        else:
                            self._peak_travel = 0.0
                            self._left_start_bar = False
                    elif (
                        is_up_pose(shoulder[1], self.top_bar, self.low_bar)
                        and self._left_start_bar
                        and self._peak_travel >= MISS_TRAVEL_FRAC
                    ):
                        if legs_ok:
                            self._award(False, hip_result.hip_status, bar="low")
                            self._peak_travel = 0.0
                            self._left_start_bar = False
                        else:
                            self._peak_travel = 0.0
                            self._left_start_bar = False
                elif self.stage == "down":
                    travel = (self.low_bar - shoulder[1]) / bar_range
                    self._peak_travel = max(self._peak_travel, travel)
                    if not is_down_pose(shoulder[1], self.top_bar, self.low_bar):
                        self._left_start_bar = True
                    if is_up_pose(shoulder[1], self.top_bar, self.low_bar):
                        if legs_ok and elbow_locked:
                            self._award(True, hip_result.hip_status, bar="top")
                            self.stage = "up"
                            self.reps += 1
                            self._peak_travel = 0.0
                            self._left_start_bar = False
                        else:
                            self._peak_travel = 0.0
                            self._left_start_bar = False
                    elif (
                        is_down_pose(shoulder[1], self.top_bar, self.low_bar)
                        and self._left_start_bar
                        and self._peak_travel >= MISS_TRAVEL_FRAC
                    ):
                        if legs_ok:
                            self._award(False, hip_result.hip_status, bar="top")
                            self._peak_travel = 0.0
                            self._left_start_bar = False
                        else:
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
            elbow_angle=elbow_deg,
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
        elbow_angle=None,
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
            "elbow_angle": elbow_angle,
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
