from __future__ import annotations
import math
from typing import Optional, Tuple
from pose2d_reader import Pose2d


def ball_xy_from_camera(
    camera_pose2d: Pose2d,
    yaw_deg: float,
    distance_m: float,
    camera_yaw_offset_deg: float,
    yaw_sign: float = 1.0,
) -> Optional[Tuple[float, float]]:
    """
    回傳球的場地座標 (x, y)
    """
    if camera_pose2d is None:
        return None
    if distance_m is None:
        return None

    x_cam = float(camera_pose2d.x)
    y_cam = float(camera_pose2d.y)
    heading_rad = float(camera_pose2d.heading_rad)

    # 鏡頭座標系下的角度（先用度數算，再轉成弧度）
    bearing_robot_deg = camera_yaw_offset_deg - yaw_sign * float(yaw_deg)
    bearing_robot_rad = math.radians(bearing_robot_deg)

    # 場地座標角度 = 相機朝向(rad) + 相對角(rad)
    theta = heading_rad + bearing_robot_rad

    x_ball = x_cam + distance_m * math.cos(theta)
    y_ball = y_cam + distance_m * math.sin(theta)

    return (x_ball, y_ball)