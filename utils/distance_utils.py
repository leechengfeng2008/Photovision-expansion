# distance_utils.py
from __future__ import annotations
import math
from typing import Iterable, List, Optional


def distance_calculate(
    pitch_deg_list: Iterable[float],
    camera_height_m: float,
    camera_pitch_deg: float,
    target_height_m: float,
) -> List[Optional[float]]:

    out: List[Optional[float]] = []
    delta_height = camera_height_m - target_height_m  # 目標點相對相機高度差

    for pitch_deg in pitch_deg_list:
        total_deg = camera_pitch_deg - float(pitch_deg)
        total_rad = math.radians(total_deg)

        tanvalue = math.tan(total_rad)
        if abs(tanvalue) < 1e-6:
            out.append(None)
            continue

        # 幾何：distance = delta_h / tan(total_angle)
        distance = delta_height / tanvalue

        # 若你只接受前方距離，可把負值視為無效
        if distance <= 0:
            out.append(None)
        else:
            out.append(distance)

    return out
