# pose_utils.py
from __future__ import annotations
import math
from dataclasses import dataclass
from pose2d_reader import Pose2d


def cameraPose2d_calculate(
    robot_pose: Pose2d,
    Camera_Vertical_Displacement: float,
    Camera_Horizontal_Displacement: float,
) -> Pose2d:
    """
    robot_pose:  機器人中心 Pose2d（x, y, heading_rad）
    vertical_cm: 機器人中心到相機的「前後」位移；你已定義相機在 heading 負方，所以會用 -vertical
    horizontal_cm: 機器人中心到相機的「左右」位移；你已定義右負左正（以 heading 為前）
    回傳：相機在場地座標的 Pose2d（heading 同 robot）
    """

    forward_m = -(Camera_Vertical_Displacement )  # ✅ heading 負方
    left_m = (Camera_Horizontal_Displacement  )    # ✅ 左正右負（你已定義）

    theta = robot_pose.heading_rad

    dx = forward_m * math.cos(theta) - left_m * math.sin(theta)
    dy = forward_m * math.sin(theta) + left_m * math.cos(theta)

    return Pose2d(
        x=robot_pose.x + dx,
        y=robot_pose.y + dy,
        heading_rad=robot_pose.heading_rad,
    )