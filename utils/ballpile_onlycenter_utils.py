from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

Point2 = Tuple[float, float]


@dataclass
class PileCenterInfo:
    pile_id: int
    points: List[Point2]      # 這一堆中的球座標
    center_xy: Point2         # 這一堆的中心 (平均)
    count: int                # 這一堆的球數


def _dist(a: Point2, b: Point2) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _centroid(points: Sequence[Point2]) -> Point2:
    if not points:
        raise ValueError("points must not be empty")
    n = len(points)
    return (
        sum(p[0] for p in points) / n,
        sum(p[1] for p in points) / n,
    )


def _to_points(points: Iterable[Optional[Point2]]) -> List[Point2]:
    out: List[Point2] = []
    for p in points:
        if p is None:
            continue
        out.append((float(p[0]), float(p[1])))
    return out


def cluster_ball_piles(
    ball_xys: Sequence[Point2],
    link_distance_m: float = 0.30,  # 30 cm
) -> List[List[Point2]]:
    """
    Single-linkage clustering:
    若任兩顆球距離 <= link_distance_m，視為相連，屬於同一堆。
    所有可傳遞相連的球都會被分到同一堆。

    你可調整 link_distance_m，來控制「球與球最多相隔多少還算同一堆」。
    """
    if not ball_xys:
        return []

    n = len(ball_xys)
    visited = [False] * n
    piles: List[List[Point2]] = []

    for i in range(n):
        if visited[i]:
            continue

        stack = [i]
        visited[i] = True
        comp: List[Point2] = []

        while stack:
            u = stack.pop()
            comp.append(ball_xys[u])

            for v in range(n):
                if visited[v]:
                    continue
                if _dist(ball_xys[u], ball_xys[v]) <= link_distance_m:
                    visited[v] = True
                    stack.append(v)

        piles.append(comp)

    return piles


def plan_ballpile_centers(
    ball_xys: Iterable[Optional[Point2]],
    cluster_link_m: float = 0.30,
) -> Tuple[int, List[PileCenterInfo], List[Point2]]:
    """
    Pipeline:
    1. 清理輸入球座標，轉成 Point2 list
    2. 用 cluster_link_m 分堆
    3. 計算每一堆中心
    4. 輸出：
       - pile_count: 有幾堆球
       - plans: 每一堆的完整資訊
       - all_center_xys: 所有球堆中心座標
    """
    pts = _to_points(ball_xys)
    if not pts:
        return 0, [], []

    piles = cluster_ball_piles(pts, link_distance_m=cluster_link_m)

    plans: List[PileCenterInfo] = []
    all_center_xys: List[Point2] = []

    for pile_id, pile_points in enumerate(piles):
        center_xy = _centroid(pile_points)
        all_center_xys.append(center_xy)

        plans.append(
            PileCenterInfo(
                pile_id=pile_id,
                points=list(pile_points),
                center_xy=center_xy,
                count=len(pile_points),
            )
        )

    pile_count = len(plans)
    return pile_count, plans, all_center_xys