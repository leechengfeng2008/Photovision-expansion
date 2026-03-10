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


def _density_center(
    pile_points: Sequence[Point2],
    density_radius_m: float = 0.20,
    spread_limit_m: float = 0.30,
) -> Point2:
    """
    Version B: Fixed-Radius Density Clustering center.

    Algorithm
    ---------
    1. For every ball in the pile, count how many neighbours lie within
       ``density_radius_m`` (its local density).
    2. Collect all balls that share the maximum density count.
    3. Safeguard: if those high-density balls are themselves too spread
       apart (max pairwise distance > spread_limit_m), fall back to the
       single ball with the highest density (tie broken by first index).
    4. Return the centroid of the surviving high-density ball(s).

    This avoids being misled by sparse bridging balls at the edge of
    the cluster while still being computationally lightweight.
    """
    n = len(pile_points)
    if n == 1:
        return pile_points[0]

    # Step 1 - local density per ball
    density: List[int] = []
    for i, p in enumerate(pile_points):
        count = sum(
            1
            for j, q in enumerate(pile_points)
            if i != j and _dist(p, q) <= density_radius_m
        )
        density.append(count)

    max_density = max(density)
    high_density_pts: List[Point2] = [
        p for p, d in zip(pile_points, density) if d == max_density
    ]

    # Step 2 - safeguard: check spread among high-density balls
    if len(high_density_pts) > 1:
        max_spread = max(
            _dist(high_density_pts[a], high_density_pts[b])
            for a in range(len(high_density_pts))
            for b in range(a + 1, len(high_density_pts))
        )
        if max_spread > spread_limit_m:
            # Fall back to the single densest ball (first occurrence)
            best_idx = density.index(max_density)
            return pile_points[best_idx]

    return _centroid(high_density_pts)


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
    density_radius_m: float = 0.20,
    density_spread_limit_m: float = 0.30,
) -> Tuple[int, List[PileCenterInfo], List[Point2]]:
    """
    Pipeline:
    1. 清理輸入球座標，轉成 Point2 list
    2. 用 cluster_link_m 分堆（Version A single-linkage）
    3. 用 Version B Fixed-Radius Density Clustering 計算每一堆中心
       - density_radius_m:      鄰近計數半徑（預設 20 cm）
       - density_spread_limit_m: 高密度點群的最大允許分散距離，
                                 超過此值時退化為最密單點（預設 30 cm）
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
        center_xy = _density_center(
            pile_points,
            density_radius_m=density_radius_m,
            spread_limit_m=density_spread_limit_m,
        )
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