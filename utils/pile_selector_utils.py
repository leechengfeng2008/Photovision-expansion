from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from pose2d_reader import Pose2d

Point2 = Tuple[float, float]


@dataclass
class PileCandidate:
    pile_id: int
    center_xy: Point2
    count: int


@dataclass
class PileScoreInfo:
    pile_id: int
    center_xy: Point2
    count: int
    distance_from_robot_m: float
    near_score: float
    count_score: float
    final_score: float


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _normalize_0to1(value: float, vmin: float, vmax: float) -> float:
    if abs(vmax - vmin) < 1e-9:
        return 1.0
    return (value - vmin) / (vmax - vmin)


def build_candidates(
    center_xys: Sequence[Point2],
    counts: Sequence[int],
    pile_ids: Optional[Sequence[int]] = None,
) -> List[PileCandidate]:
    """
    把外部算好的中心座標與球數，整理成 selector 可用的候選清單。
    """
    if len(center_xys) != len(counts):
        raise ValueError("center_xys and counts must have the same length")

    if pile_ids is None:
        pile_ids = list(range(len(center_xys)))

    if len(pile_ids) != len(center_xys):
        raise ValueError("pile_ids and center_xys must have the same length")

    out: List[PileCandidate] = []
    for pid, center, count in zip(pile_ids, center_xys, counts):
        out.append(
            PileCandidate(
                pile_id=int(pid),
                center_xy=(float(center[0]), float(center[1])),
                count=int(count),
            )
        )
    return out


def select_best_pile(
    robot_pose: Optional[Pose2d],
    pile_candidates: Sequence[PileCandidate],
    ball_priority_0to10: float = 5.0,
) -> Tuple[Optional[PileCandidate], List[PileScoreInfo]]:
    """
    根據距離與球數加權，選出最佳球堆。

    ball_priority_0to10:
        0  -> 完全偏最近
        10 -> 完全偏球多
    """
    if not pile_candidates:
        return None, []

    alpha = _clamp(float(ball_priority_0to10), 0.0, 10.0) / 10.0

    counts = [c.count for c in pile_candidates]
    count_min = min(counts)
    count_max = max(counts)

    if robot_pose is not None:
        distances = [
            math.hypot(c.center_xy[0] - robot_pose.x, c.center_xy[1] - robot_pose.y)
            for c in pile_candidates
        ]
        dist_min = min(distances)
        dist_max = max(distances)
    else:
        distances = [0.0 for _ in pile_candidates]
        dist_min = 0.0
        dist_max = 0.0

    score_infos: List[PileScoreInfo] = []

    for c, dist_m in zip(pile_candidates, distances):
        count_score = _normalize_0to1(float(c.count), float(count_min), float(count_max))

        if robot_pose is None:
            near_score = 0.0
            final_score = count_score
        else:
            dist_norm = _normalize_0to1(dist_m, dist_min, dist_max)
            near_score = 1.0 - dist_norm   # 越近越高
            final_score = (1.0 - alpha) * near_score + alpha * count_score

        score_infos.append(
            PileScoreInfo(
                pile_id=c.pile_id,
                center_xy=c.center_xy,
                count=c.count,
                distance_from_robot_m=dist_m,
                near_score=near_score,
                count_score=count_score,
                final_score=final_score,
            )
        )

    score_infos.sort(
        key=lambda s: (
            -s.final_score,
            -s.count,
            s.distance_from_robot_m,
            s.pile_id,
        )
    )

    best_id = score_infos[0].pile_id
    best_candidate = next((c for c in pile_candidates if c.pile_id == best_id), None)

    return best_candidate, score_infos