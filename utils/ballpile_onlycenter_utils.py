from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

Point2 = Tuple[float, float]
CenterMode = Literal["centroid", "density_vb"]


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PileCenterInfo:
    pile_id: int
    points: List[Point2]          # 這一堆中的球座標
    center_xy: Point2             # 這一堆的中心
    count: int                    # 這一堆的球數
    # ── 新增 debug 欄位（density_vb 模式才有意義；centroid 模式皆為 None/False）──
    center_mode: CenterMode = "centroid"
    used_insurance_fallback: bool = False   # 保險機制是否觸發
    densest_neighbor_count: Optional[int] = None  # 最高密度球的鄰居數
    density_peak_count: Optional[int] = None      # 最高密度球的數量


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

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


def _build_grid(points: Sequence[Point2], cell: float) -> Dict[Tuple[int, int], List[int]]:
    """
    Grid 加速：把點分進正方形格子，查詢鄰居時只需搜 3×3 共 9 個格子。
    cell 建議設成 density_radius_m，使每個鄰居一定落在相鄰格內。
    平均複雜度從 O(N²) 降到 O(N × k)，k = 平均鄰居數。
    """
    cell = max(cell, 1e-9)
    grid: Dict[Tuple[int, int], List[int]] = {}
    for idx, (x, y) in enumerate(points):
        cx = int(math.floor(x / cell))
        cy = int(math.floor(y / cell))
        grid.setdefault((cx, cy), []).append(idx)
    return grid


def _count_neighbors_grid(
    points: Sequence[Point2],
    grid: Dict[Tuple[int, int], List[int]],
    radius: float,
) -> List[int]:
    """
    利用 grid 計算每個點在 radius 內的鄰居數（含自身）。
    """
    r2 = radius * radius
    counts = [0] * len(points)
    for i, (x, y) in enumerate(points):
        cx = int(math.floor(x / radius))
        cy = int(math.floor(y / radius))
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                for j in grid.get((gx, gy), []):
                    dx = points[j][0] - x
                    dy = points[j][1] - y
                    if dx * dx + dy * dy <= r2:
                        counts[i] += 1
    return counts


def _smartest_fallback(
    candidates: Sequence[Point2],
    all_points: Sequence[Point2],
) -> Point2:
    """
    保險回退：從 candidates 中選「對 all_points 中所有點的總距離最小」的那一個。
    等效於選最靠近整體重心的密集點，比直接選 index 0 更穩定。
    """
    best_pt = candidates[0]
    best_sum = float("inf")
    for cand in candidates:
        total = sum(_dist(cand, q) for q in all_points)
        if total < best_sum:
            best_sum = total
            best_pt = cand
    return best_pt


def _density_center(
    pile_points: Sequence[Point2],
    density_radius_m: float,
    spread_limit_m: float,
) -> Tuple[Point2, bool, int, int]:
    """
    Version B: Fixed-Radius Density Clustering center（含保險機制）。

    改進點
    ------
    1. Grid 加速：用九宮格查詢，平均 O(N) 取代原本 O(N²)。
    2. density 計數含自身（與 PDF 報告一致，更直觀地反映「此點附近共有幾顆球」）。
    3. 保險回退改為選「總距最小」點，而非 index 0（更靠近真實密集核心）。

    Returns
    -------
    center_xy, used_fallback, max_neighbor_count, peak_count
    """
    n = len(pile_points)
    if n == 1:
        return pile_points[0], False, 1, 1

    # ── Step 1: Grid 加速鄰居計數 ──────────────────────────────────────────
    grid = _build_grid(pile_points, cell=density_radius_m)
    neighbor_counts = _count_neighbors_grid(pile_points, grid, radius=density_radius_m)

    max_k = max(neighbor_counts)
    high_density_pts: List[Point2] = [
        pile_points[i] for i, k in enumerate(neighbor_counts) if k == max_k
    ]

    # ── Step 2: 保險 — 高密度點群是否過度分散 ──────────────────────────────
    used_fallback = False
    if len(high_density_pts) > 1:
        max_spread = max(
            _dist(high_density_pts[a], high_density_pts[b])
            for a in range(len(high_density_pts))
            for b in range(a + 1, len(high_density_pts))
        )
        if max_spread > spread_limit_m:
            # ── 改進：選「總距最小」的點，而非第一個 ──
            best = _smartest_fallback(high_density_pts, pile_points)
            return best, True, max_k, len(high_density_pts)

    center = _centroid(high_density_pts)
    return center, used_fallback, max_k, len(high_density_pts)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def cluster_ball_piles(
    ball_xys: Sequence[Point2],
    link_distance_m: float = 0.30,
) -> List[List[Point2]]:
    """
    Version A: Single-linkage clustering（連通分量）。
    若任兩顆球距離 <= link_distance_m，視為相連，屬於同一堆。
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
    # ── 可回滾旗標 ──────────────────────────────────────────────────────────
    # "centroid"   → 直接平均，行為與舊版完全一致（回滾用）
    # "density_vb" → Version B 密度中心（預設，含保險與 grid 加速）
    center_mode: CenterMode = "density_vb",
    density_radius_m: float = 0.30,
    density_spread_limit_m: float = 0.30,
) -> Tuple[int, List[PileCenterInfo], List[Point2]]:
    """
    Pipeline:
    1. 清理輸入球座標
    2. Version A single-linkage 分堆（cluster_link_m）
    3. 依 center_mode 計算每堆中心：
       - "centroid"   : 所有點平均（快速、可回滾）
       - "density_vb" : Version B 密度中心 + 保險機制 + grid 加速
    4. 輸出 pile_count, plans, all_center_xys

    回滾方式
    --------
    只需把 center_mode="centroid" 傳入（或改 main.py 的 PILE_CENTER_MODE），
    行為與修改前完全相同，無需改任何其他程式碼。
    """
    pts = _to_points(ball_xys)
    if not pts:
        return 0, [], []

    piles = cluster_ball_piles(pts, link_distance_m=cluster_link_m)

    plans: List[PileCenterInfo] = []
    all_center_xys: List[Point2] = []

    for pile_id, pile_points in enumerate(piles):
        if center_mode == "density_vb":
            center_xy, used_fb, max_k, peak_n = _density_center(
                pile_points,
                density_radius_m=density_radius_m,
                spread_limit_m=density_spread_limit_m,
            )
            info = PileCenterInfo(
                pile_id=pile_id,
                points=list(pile_points),
                center_xy=center_xy,
                count=len(pile_points),
                center_mode="density_vb",
                used_insurance_fallback=used_fb,
                densest_neighbor_count=max_k,
                density_peak_count=peak_n,
            )
        else:  # "centroid" — 回滾路徑
            center_xy = _centroid(pile_points)
            info = PileCenterInfo(
                pile_id=pile_id,
                points=list(pile_points),
                center_xy=center_xy,
                count=len(pile_points),
                center_mode="centroid",
            )

        plans.append(info)
        all_center_xys.append(info.center_xy)

    return len(plans), plans, all_center_xys
