from __future__ import annotations

"""
ball_dedupe_fov_utils.py
========================
Two-camera ball deduplication with Angle-Constrained Comparison.

Pipeline (per the engineering notes):
  Step 1 – Direct Comparison
      For each cam1 ball and each cam2 ball, compute field-coordinate
      distance.  If distance <= same_ball_error_m they are candidate matches.

  Step 2 – Angle Constraint (geometrical sanity filter)
      Before accepting a candidate match, verify that:
        • The bearing from camera2's position+heading to the cam1 ball
          is within camera2's yaw window (yaw_offset ± max_angle_deg).
        • The bearing from camera1's position+heading to the cam2 ball
          is within camera1's yaw window (yaw_offset ± max_angle_deg).
      If either check fails the pair is geometrically impossible and is
      skipped before distance-based pairing proceeds.

  Step 3 – Greedy nearest-neighbour match (same logic as ball_dedupe_utils)
      Each cam2 ball is used at most once; each cam1 ball picks its
      closest valid cam2 partner.
"""

import math
from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Sequence, Tuple

from pose2d_reader import Pose2d

Point2 = Tuple[float, float]
KeepMode = Literal["average", "cam1", "cam2"]


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

def _dist(a: Point2, b: Point2) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _to_points(points: Iterable[Optional[Point2]]) -> List[Point2]:
    out: List[Point2] = []
    for p in points:
        if p is None:
            continue
        out.append((float(p[0]), float(p[1])))
    return out


def _merge_point(cam1_xy: Point2, cam2_xy: Point2, keep: KeepMode) -> Point2:
    if keep == "cam1":
        return cam1_xy
    if keep == "cam2":
        return cam2_xy
    return ((cam1_xy[0] + cam2_xy[0]) / 2.0, (cam1_xy[1] + cam2_xy[1]) / 2.0)


def _normalize_angle_deg(deg: float) -> float:
    """Wrap angle to (-180, +180]."""
    deg = deg % 360.0
    if deg > 180.0:
        deg -= 360.0
    return deg


def _bearing_relative_deg(
    camera_pose2d: Pose2d,
    camera_yaw_offset_deg: float,
    ball_xy: Point2,
) -> float:
    """
    Compute the signed horizontal angle (degrees) between:
      • the camera's optical axis  (heading_rad + yaw_offset)
      • the direction from the camera to ball_xy

    Positive = ball is to the left of the optical axis
    Negative = ball is to the right
    """
    dx = ball_xy[0] - camera_pose2d.x
    dy = ball_xy[1] - camera_pose2d.y

    bearing_world_rad = math.atan2(dy, dx)
    camera_axis_rad = camera_pose2d.heading_rad + math.radians(camera_yaw_offset_deg)

    relative_rad = bearing_world_rad - camera_axis_rad
    return _normalize_angle_deg(math.degrees(relative_rad))


def _angle_feasible(
    camera_pose2d: Optional[Pose2d],
    camera_yaw_offset_deg: float,
    ball_xy: Point2,
    max_angle_deg: float,
) -> bool:
    """
    Return True if ball_xy could geometrically be seen by this camera
    (i.e. |relative bearing| <= max_angle_deg).
    If camera_pose2d is None, skip the constraint (return True).
    """
    if camera_pose2d is None:
        return True
    rel = _bearing_relative_deg(camera_pose2d, camera_yaw_offset_deg, ball_xy)
    return abs(rel) <= max_angle_deg


# ──────────────────────────────────────────────
# Public data classes
# ──────────────────────────────────────────────

@dataclass
class FovMatchPair:
    cam1_index: int
    cam2_index: int
    cam1_xy: Point2
    cam2_xy: Point2
    error_m: float
    merged_xy: Point2
    angle_filtered: bool   # True → pair was kept after angle check


@dataclass
class FovDedupeResult:
    unique_points: List[Point2]
    matched_pairs: List[FovMatchPair]
    unmatched_cam1: List[Point2]
    unmatched_cam2: List[Point2]
    angle_rejected_count: int   # how many candidate pairs the angle filter dropped


# ──────────────────────────────────────────────
# Main function
# ──────────────────────────────────────────────

def dedupe_two_cameras_fov(
    cam1_ball_xys: Sequence[Optional[Point2]],
    cam2_ball_xys: Sequence[Optional[Point2]],
    camera1_pose2d: Optional[Pose2d],
    camera2_pose2d: Optional[Pose2d],
    camera1_yaw_offset_deg: float = 35.0,
    camera2_yaw_offset_deg: float = -35.0,
    same_ball_error_m: float = 0.10,
    max_angle_deg: float = 55.0,
    keep: KeepMode = "average",
) -> FovDedupeResult:
    """
    Deduplicate balls seen by two cameras using:
      1. Direct-distance comparison (same as ball_dedupe_utils)
      2. Angle-constrained pre-filter to reject geometrically impossible matches

    Parameters
    ----------
    cam1_ball_xys, cam2_ball_xys :
        Field-coordinate ball positions from each camera.
    camera1_pose2d, camera2_pose2d :
        Camera positions/headings in field frame.
        Pass None to skip the angle constraint for that camera.
    camera1_yaw_offset_deg, camera2_yaw_offset_deg :
        Yaw offset of the camera relative to the robot heading
        (same values as CAMERA1_YAW_OFFSET_DEG in main.py).
    same_ball_error_m :
        Distance threshold: pairs closer than this are candidate matches.
    max_angle_deg :
        Half-FOV used for the angle constraint.
        A ball is considered "visible" by a camera if its bearing relative
        to that camera's optical axis is within +/-max_angle_deg.
        Recommended: use the camera's actual half-FOV or a slightly larger
        value (e.g. 55 deg) to stay conservative.
    keep :
        How to merge matched pairs: "average" | "cam1" | "cam2".
    """
    cam1_pts = _to_points(cam1_ball_xys)
    cam2_pts = _to_points(cam2_ball_xys)

    used_cam2 = [False] * len(cam2_pts)
    matched_pairs: List[FovMatchPair] = []
    unique_points: List[Point2] = []
    unmatched_cam1: List[Point2] = []
    angle_rejected_count = 0

    for i, p1 in enumerate(cam1_pts):
        best_j = -1
        best_err = float("inf")

        for j, p2 in enumerate(cam2_pts):
            if used_cam2[j]:
                continue

            # ── Step 1: Distance pre-check ──────────────────────────
            err = _dist(p1, p2)
            if err > same_ball_error_m:
                continue

            # ── Step 2: Angle constraint ─────────────────────────────
            # Can cam2 see the ball that cam1 detected at p1?
            cam2_sees_p1 = _angle_feasible(
                camera2_pose2d, camera2_yaw_offset_deg, p1, max_angle_deg
            )
            # Can cam1 see the ball that cam2 detected at p2?
            cam1_sees_p2 = _angle_feasible(
                camera1_pose2d, camera1_yaw_offset_deg, p2, max_angle_deg
            )

            if not (cam2_sees_p1 and cam1_sees_p2):
                angle_rejected_count += 1
                continue  # geometrically impossible → skip

            # ── Step 3: Keep the nearest valid partner ───────────────
            if err < best_err:
                best_err = err
                best_j = j

        if best_j >= 0:
            used_cam2[best_j] = True
            p2 = cam2_pts[best_j]
            merged = _merge_point(p1, p2, keep=keep)
            matched_pairs.append(
                FovMatchPair(
                    cam1_index=i,
                    cam2_index=best_j,
                    cam1_xy=p1,
                    cam2_xy=p2,
                    error_m=best_err,
                    merged_xy=merged,
                    angle_filtered=True,
                )
            )
            unique_points.append(merged)
        else:
            unmatched_cam1.append(p1)
            unique_points.append(p1)

    unmatched_cam2: List[Point2] = []
    for j, p2 in enumerate(cam2_pts):
        if not used_cam2[j]:
            unmatched_cam2.append(p2)
            unique_points.append(p2)

    return FovDedupeResult(
        unique_points=unique_points,
        matched_pairs=matched_pairs,
        unmatched_cam1=unmatched_cam1,
        unmatched_cam2=unmatched_cam2,
        angle_rejected_count=angle_rejected_count,
    )
