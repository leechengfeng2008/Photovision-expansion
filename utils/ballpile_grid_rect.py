from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

Point2 = Tuple[float, float]
Cell2  = Tuple[int, int]
CenterMode = Literal["density_weighted", "rect_center"]


# ─────────────────────────────────────────────────────────────────────────────
# Data class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RectPileInfo:
    pile_id:             int
    points:              List[Point2]   # 堆內所有球的場地座標
    cells:               List[Cell2]    # 被占用的格子索引
    count:               int            # 球數
    center_xy:           Point2         # 輸出中心（依 center_mode 決定）
    rect_min_xy:         Point2         # 外接長方形左下角
    rect_max_xy:         Point2         # 外接長方形右上角
    rect_size_xy:        Point2         # 長方形寬、高
    occupied_cell_count: int            # 這堆占了幾個格子
    cell_size_m:         float          # 格子大小
    center_mode:         CenterMode = "density_weighted"
    # ── density_weighted 模式的 debug 欄位 ───────────────────────────────────
    max_neighbor_count:  Optional[int]  = None  # 最密球的鄰居數（含自身）
    used_rect_fallback:  bool           = False  # True = 球全部等密，退回矩形中心


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_points(points: Iterable[Optional[Point2]]) -> List[Point2]:
    out: List[Point2] = []
    for p in points:
        if p is None:
            continue
        out.append((float(p[0]), float(p[1])))
    return out


def _point_to_cell(p: Point2, cell_size_m: float) -> Cell2:
    return (
        int(math.floor(p[0] / cell_size_m)),
        int(math.floor(p[1] / cell_size_m)),
    )


def _build_cell_map(
    points: Sequence[Point2],
    cell_size_m: float,
) -> Dict[Cell2, List[Point2]]:
    cell_map: Dict[Cell2, List[Point2]] = {}
    for p in points:
        cell = _point_to_cell(p, cell_size_m)
        cell_map.setdefault(cell, []).append(p)
    return cell_map


def _get_neighbors(cell: Cell2, diagonal: bool) -> List[Cell2]:
    cx, cy = cell
    if diagonal:
        return [
            (cx-1, cy-1), (cx, cy-1), (cx+1, cy-1),
            (cx-1, cy  ),             (cx+1, cy  ),
            (cx-1, cy+1), (cx, cy+1), (cx+1, cy+1),
        ]
    return [(cx-1,cy),(cx+1,cy),(cx,cy-1),(cx,cy+1)]


def _cluster_cells(
    occupied_cells: Sequence[Cell2],
    diagonal_connect: bool,
) -> List[List[Cell2]]:
    occupied_set = set(occupied_cells)
    visited: set[Cell2] = set()
    components: List[List[Cell2]] = []

    for start in occupied_cells:
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        comp: List[Cell2] = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in _get_neighbors(cur, diagonal_connect):
                if nb in occupied_set and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        components.append(comp)

    return components


def _cells_to_rect(
    cells: Sequence[Cell2],
    cell_size_m: float,
) -> Tuple[Point2, Point2]:
    min_cx = min(c[0] for c in cells)
    max_cx = max(c[0] for c in cells)
    min_cy = min(c[1] for c in cells)
    max_cy = max(c[1] for c in cells)
    rect_min = (min_cx * cell_size_m,       min_cy * cell_size_m)
    rect_max = ((max_cx + 1) * cell_size_m, (max_cy + 1) * cell_size_m)
    return rect_min, rect_max


def _rect_center(rect_min: Point2, rect_max: Point2) -> Point2:
    return (
        (rect_min[0] + rect_max[0]) / 2.0,
        (rect_min[1] + rect_max[1]) / 2.0,
    )


# ── Grid-accelerated density-weighted centroid ───────────────────────────────

def _build_grid(
    points: Sequence[Point2],
    cell: float,
) -> Dict[Cell2, List[int]]:
    """把點分進 grid，查鄰居時只需搜 3×3 格，平均 O(N) 取代 O(N²)。"""
    cell = max(cell, 1e-9)
    g: Dict[Cell2, List[int]] = {}
    for i, (x, y) in enumerate(points):
        key = (int(math.floor(x / cell)), int(math.floor(y / cell)))
        g.setdefault(key, []).append(i)
    return g


def _count_neighbors(
    points: Sequence[Point2],
    grid:   Dict[Cell2, List[int]],
    radius: float,
) -> List[int]:
    """每個點在 radius 內的鄰居數（含自身）。"""
    r2 = radius * radius
    counts = [0] * len(points)
    for i, (x, y) in enumerate(points):
        cx = int(math.floor(x / radius))
        cy = int(math.floor(y / radius))
        for gx in (cx-1, cx, cx+1):
            for gy in (cy-1, cy, cy+1):
                for j in grid.get((gx, gy), []):
                    dx = points[j][0] - x
                    dy = points[j][1] - y
                    if dx*dx + dy*dy <= r2:
                        counts[i] += 1
    return counts


def _density_weighted_centroid(
    pile_points:      Sequence[Point2],
    density_radius_m: float,
    rect_min:         Point2,
    rect_max:         Point2,
) -> Tuple[Point2, bool, int]:
    """
    在矩形框內對所有球做密度加權平均。

    每顆球的權重 = 它在 density_radius_m 內的鄰居數（含自身）。
    等效於：越密集的球群貢獻越大，邊緣孤立球影響越小。

    若所有球密度相同（例如只有 1 顆球，或分布完全均勻），
    退回矩形幾何中心（used_rect_fallback=True）。

    Returns
    -------
    center_xy, used_rect_fallback, max_neighbor_count
    """
    n = len(pile_points)

    if n == 0:
        return _rect_center(rect_min, rect_max), True, 0

    if n == 1:
        return pile_points[0], False, 1

    # Grid 加速鄰居計數
    grid   = _build_grid(pile_points, cell=density_radius_m)
    counts = _count_neighbors(pile_points, grid, radius=density_radius_m)
    max_k  = max(counts)

    # 若所有球密度完全相同，退回矩形中心
    if all(c == counts[0] for c in counts):
        return _rect_center(rect_min, rect_max), True, max_k

    # 加權平均：weight = neighbor_count
    total_w = sum(counts)
    wx = sum(counts[i] * pile_points[i][0] for i in range(n)) / total_w
    wy = sum(counts[i] * pile_points[i][1] for i in range(n)) / total_w
    return (wx, wy), False, max_k


# ─────────────────────────────────────────────────────────────────────────────
# Public: plan
# ─────────────────────────────────────────────────────────────────────────────

def plan_ballpile_rect_centers(
    ball_xys:          Iterable[Optional[Point2]],
    cell_size_m:       float      = 0.50,
    diagonal_connect:  bool       = True,
    # 回滾旗標：
    #   "density_weighted" → 矩形框內密度加權中心（建議）
    #   "rect_center"      → 外接矩形幾何中心（舊版行為，一行回滾）
    center_mode:       CenterMode = "density_weighted",
    density_radius_m:  float      = 0.50,   # 加權鄰域半徑，建議 = cell_size_m
) -> Tuple[int, List[RectPileInfo], List[Point2]]:
    """
    流程
    ----
    1. 球的場地座標切到 cell_size_m × cell_size_m 格子
    2. 相鄰有球的格子連通成同一堆（diagonal_connect 控制斜角）
    3. 每堆建立外接長方形
    4. 依 center_mode 計算中心：
       - density_weighted : 矩形框內所有球的密度加權平均（更貼近真實密集核心）
       - rect_center       : 外接矩形幾何中心（快速 / 回滾用）

    回滾方式
    --------
    把 center_mode="rect_center" 傳入即可回到純幾何中心，無需改其他程式碼。
    """
    pts = _to_points(ball_xys)
    if not pts:
        return 0, [], []

    cell_map       = _build_cell_map(pts, cell_size_m)
    occupied_cells = list(cell_map.keys())
    components     = _cluster_cells(occupied_cells, diagonal_connect)

    plans:          List[RectPileInfo] = []
    all_center_xys: List[Point2]       = []

    for pile_id, cells in enumerate(components):
        pile_points: List[Point2] = []
        for cell in cells:
            pile_points.extend(cell_map[cell])

        rect_min, rect_max = _cells_to_rect(cells, cell_size_m)
        rect_size = (
            rect_max[0] - rect_min[0],
            rect_max[1] - rect_min[1],
        )

        if center_mode == "density_weighted":
            center_xy, used_fb, max_k = _density_weighted_centroid(
                pile_points,
                density_radius_m = density_radius_m,
                rect_min         = rect_min,
                rect_max         = rect_max,
            )
            info = RectPileInfo(
                pile_id             = pile_id,
                points              = pile_points,
                cells               = list(cells),
                count               = len(pile_points),
                center_xy           = center_xy,
                rect_min_xy         = rect_min,
                rect_max_xy         = rect_max,
                rect_size_xy        = rect_size,
                occupied_cell_count = len(cells),
                cell_size_m         = cell_size_m,
                center_mode         = "density_weighted",
                max_neighbor_count  = max_k,
                used_rect_fallback  = used_fb,
            )
        else:  # "rect_center" ── 回滾路徑
            center_xy = _rect_center(rect_min, rect_max)
            info = RectPileInfo(
                pile_id             = pile_id,
                points              = pile_points,
                cells               = list(cells),
                count               = len(pile_points),
                center_xy           = center_xy,
                rect_min_xy         = rect_min,
                rect_max_xy         = rect_max,
                rect_size_xy        = rect_size,
                occupied_cell_count = len(cells),
                cell_size_m         = cell_size_m,
                center_mode         = "rect_center",
            )

        plans.append(info)
        all_center_xys.append(center_xy)

    return len(plans), plans, all_center_xys


# ─────────────────────────────────────────────────────────────────────────────
# Public: select
# ─────────────────────────────────────────────────────────────────────────────

def select_best_rect_pile(
    pile_plans:          Sequence[RectPileInfo],
    robot_pose2d,                               # SimplePose2d | None
    ball_priority_0to10: float = 8.0,
) -> Optional[RectPileInfo]:
    """
    從 RectPileInfo 列表中選出最佳堆。
    評分邏輯與原 select_best_pile 相同（不依賴 PileCandidate），直接接受 RectPileInfo。

    ball_priority_0to10:
        0  → 完全偏最近
        10 → 完全偏球多

    Tie-break：final_score → count → distance（與原版一致）
    """
    if not pile_plans:
        return None

    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    def _norm(v: float, vmin: float, vmax: float) -> float:
        if abs(vmax - vmin) < 1e-9:
            return 1.0
        return (v - vmin) / (vmax - vmin)

    alpha  = _clamp(float(ball_priority_0to10), 0.0, 10.0) / 10.0
    counts = [p.count for p in pile_plans]
    c_min, c_max = min(counts), max(counts)

    if robot_pose2d is not None:
        dists = [
            math.hypot(p.center_xy[0] - robot_pose2d.x,
                       p.center_xy[1] - robot_pose2d.y)
            for p in pile_plans
        ]
        d_min, d_max = min(dists), max(dists)
    else:
        dists  = [0.0] * len(pile_plans)
        d_min  = d_max = 0.0

    best: Optional[RectPileInfo] = None
    best_key = None

    for pile, dist_m in zip(pile_plans, dists):
        count_score = _norm(float(pile.count), float(c_min), float(c_max))

        if robot_pose2d is None:
            near_score  = 0.0
            final_score = count_score
        else:
            near_score  = 1.0 - _norm(dist_m, d_min, d_max)
            final_score = (1.0 - alpha) * near_score + alpha * count_score

        key = (-final_score, -pile.count, dist_m, pile.pile_id)
        if best_key is None or key < best_key:
            best_key = key
            best = pile

    return best


# ── 舊版 select_largest_rect_pile（只看球數/面積，無距離考量，已隱藏）─────────
# def select_largest_rect_pile(
#     pile_plans: Sequence[RectPileInfo],
# ) -> Optional[RectPileInfo]:
#     if not pile_plans:
#         return None
#     def key_fn(p: RectPileInfo):
#         rect_area = p.rect_size_xy[0] * p.rect_size_xy[1]
#         return (p.count, -rect_area)
#     return max(pile_plans, key=key_fn)
