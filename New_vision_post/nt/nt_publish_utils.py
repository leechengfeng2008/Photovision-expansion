from __future__ import annotations

"""
nt_publish_utils.py
===================
NetworkTables 發布工具，供 main.py（及未來其他入口）共用。

發布格式（與舊版完全相容）：
    SmartDashboard/BestPilePose2d  →  DoubleArray [x, y, heading_deg]
    - 無最佳堆 或 無機器人 pose → 發布 []，讓接收端用 len < 3 判定無效
    - heading_deg = 機器人指向球堆中心的場地角度（不是機器人朝向）
"""

import math
from typing import Optional

import ntcore


# ─────────────────────────────────────────────────────────────────────────────
# Publisher 建立
# ─────────────────────────────────────────────────────────────────────────────

def create_best_pose2d_publisher(
    server:      str,
    table:       str = "SmartDashboard",
    key:         str = "BestPilePose2d",
    client_name: str = "best-pile-publisher",
):
    """
    建立並回傳 (inst, publisher)。

    inst 必須保留在呼叫端（不可被 GC 回收），否則連線會斷。
    用法：
        _inst, pub = create_best_pose2d_publisher(NT_SERVER)
        # 保留 _inst 至 main loop 結束
    """
    inst = ntcore.NetworkTableInstance.create()
    inst.startClient4(client_name)
    inst.setServer(server)
    pub = inst.getTable(table).getDoubleArrayTopic(key).publish()
    return inst, pub


# ─────────────────────────────────────────────────────────────────────────────
# 發布
# ─────────────────────────────────────────────────────────────────────────────

def publish_best_pile(
    pub,
    best_pile,       # RectPileInfo | PileCenterInfo | 任何有 .center_xy 的物件 | None
    robot_pose2d,    # SimplePose2d | Pose2d | 任何有 .x .y .heading_rad 的物件 | None
) -> None:
    """
    把最佳球堆位置發布到 NT。

    格式：[x, y, heading_deg]
      - x, y        : 球堆中心的場地座標（公尺）
      - heading_deg : 從機器人位置指向球堆中心的角度（場地座標系，度）

    無效情況（發布 []）：
      - best_pile 是 None
      - robot_pose2d 是 None（無法算方向角）

    與舊版的差異
    ------------
    舊版 publish_best_pose2d_to_NT() 直接寫在 main.py，且在
    robot_pose2d=None 但 best_pile 不是 None 時會 crash（AttributeError）。
    此版已修正：兩者任一為 None 都安全發布 []。
    """
    if best_pile is None or robot_pose2d is None:
        pub.set([])
        return

    x = float(best_pile.center_xy[0])
    y = float(best_pile.center_xy[1])

    dx = x - float(robot_pose2d.x)
    dy = y - float(robot_pose2d.y)

    # 球堆與機器人重合（幾乎不可能，但防護）
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        heading_deg = math.degrees(float(robot_pose2d.heading_rad))
    else:
        heading_deg = math.degrees(math.atan2(dy, dx))

    pub.set([x, y, heading_deg])


# ─────────────────────────────────────────────────────────────────────────────
# 舊版（已隱藏，保留供對照）
# ─────────────────────────────────────────────────────────────────────────────
# 舊版直接寫在 main.py，簽名如下：
#
# def publish_best_pose2d_to_NT(best_pose_pub, best_pile, robot_pose2d):
#     if best_pile is None:
#         best_pose_pub.set([])
#         return
#     x = float(best_pile.center_xy[0])
#     y = float(best_pile.center_xy[1])
#     dx = x - float(robot_pose2d.x)   # ← robot_pose2d=None 時 crash
#     dy = y - float(robot_pose2d.y)
#     if abs(dx) < 1e-9 and abs(dy) < 1e-9:
#         heading_deg = math.degrees(float(robot_pose2d.heading_rad))
#     else:
#         heading_deg = math.degrees(math.atan2(dy, dx))
#     best_pose_pub.set([x, y, heading_deg])
