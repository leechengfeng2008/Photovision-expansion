from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Sequence, Tuple

Point2 = Tuple[float, float]
KeepMode = Literal["average", "cam1", "cam2"]


@dataclass
class MatchPair:
    cam1_index: int
    cam2_index: int
    cam1_xy: Point2
    cam2_xy: Point2
    error_m: float
    merged_xy: Point2


@dataclass
class TwoCamDedupeResult:
    unique_points: List[Point2]
    matched_pairs: List[MatchPair]
    unmatched_cam1: List[Point2]
    unmatched_cam2: List[Point2]


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


def dedupe_two_cameras(
    cam1_ball_xys: Sequence[Optional[Point2]],
    cam2_ball_xys: Sequence[Optional[Point2]],
    same_ball_error_m: float = 0.10,
    keep: KeepMode = "average",
) -> TwoCamDedupeResult:
    """

    Rules：
    - The same ball is considered the same if the distance 
    between its two sides is less than or equal to `same_ball_error_m`.
    - Each ball will be paired at most once.
    - After successful pairing, you can choose:
        1. `keep="average"` -> Average the two
        2. `keep="cam1"` -> Keep the coordinates of cam1
        3. `keep="cam2"` -> Keep the coordinates of cam2

    Return:
        - `unique_points`: The final ball coordinates after removing duplicates

        - `matched_points`: Which cam1/cam2 balls are considered the same

        - `unmatched_cam1` / `unmatched_cam2`: Balls that were not paired
    """

    cam1_pts = _to_points(cam1_ball_xys)
    cam2_pts = _to_points(cam2_ball_xys)

    used_cam2 = [False] * len(cam2_pts)
    matched_pairs: List[MatchPair] = []
    unique_points: List[Point2] = []
    unmatched_cam1: List[Point2] = []

    for i, p1 in enumerate(cam1_pts):
        best_j = -1
        best_err = float("inf")

        for j, p2 in enumerate(cam2_pts):
            if used_cam2[j]:
                continue
            err = _dist(p1, p2)
            if err <= same_ball_error_m and err < best_err:
                best_err = err
                best_j = j

        if best_j >= 0:
            used_cam2[best_j] = True
            p2 = cam2_pts[best_j]
            merged = _merge_point(p1, p2, keep=keep)
            matched_pairs.append(
                MatchPair(
                    cam1_index=i,
                    cam2_index=best_j,
                    cam1_xy=p1,
                    cam2_xy=p2,
                    error_m=best_err,
                    merged_xy=merged,
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

    return TwoCamDedupeResult(
        unique_points=unique_points,
        matched_pairs=matched_pairs,
        unmatched_cam1=unmatched_cam1,
        unmatched_cam2=unmatched_cam2,
    )

'''
`unique_points` is only used to provide the coordinates of all spheres after deduplication.
Source information for each point is not retained here 
(e.g., whether it is matched, comes only from cam1, or comes only from cam2).

The current addition order is:
1. Points merged after matching
2. Points from unmatched_cam1
3. Points from unmatched_cam2

However, the calling end should not rely on this order; 
if source information is needed in the future, 
it should use matched_points / # unmatched_cam1 / unmatched_cam2 
instead of just looking at the index of `unique_points`.

'''