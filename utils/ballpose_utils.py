# utils/ballpose_utils.py
from __future__ import annotations
import math
from typing import Optional, Tuple


def ball_xy_from_camera(
    camera_pose2d,             # [x, y, heading_deg]  (跟你 Pose2d 一樣格式)
    yaw_deg: float,            # Photon yaw (deg)
    distance_m: float,         # 距離 (m)
    camera_yaw_offset_deg: float,  # 鏡頭相對機器人法線(前方)的偏角，例如左+35 右-35
    yaw_sign: float = 1.0,     # 若你發現左右顛倒，把這個改成 -1.0
) -> Optional[Tuple[float, float]]:
    """
    回傳球的場地座標 (x, y)
    """
    if camera_pose2d is None:
        return None
    if distance_m is None:
        return None

    x_cam, y_cam, heading_deg = float(camera_pose2d[0]), float(camera_pose2d[1]), float(camera_pose2d[2])

    # 依你的規則： (鏡頭距法線35度 - yawangle)
    bearing_robot_deg = camera_yaw_offset_deg - yaw_sign * float(yaw_deg)

    # 加上機器人heading -> 場地座標角度
    bearing_field_deg = heading_deg + bearing_robot_deg
    theta = math.radians(bearing_field_deg)

    x_ball = x_cam + distance_m * math.cos(theta)
    y_ball = y_cam + distance_m * math.sin(theta)

    return (x_ball, y_ball)