from __future__ import annotations
"""
main_checkCameraPose2d.py
=========================
測試用：印出每一圈的 robot_pose2d、camera1_pose2d、camera2_pose2d，
以及每個 target 的 yaw/pitch/dist/ball_xy，方便驗證：
  1. cameraPose2d_calculate() 輸出是否合理（位置是否在機器人後方正確偏移）
  2. ball_xy_from_camera()    輸出是否合理（球的場地座標方向是否正確）
  3. 兩台鏡頭的投影是否對齊（同一顆球被兩台看見時，ball_xy 是否接近）

不做分群、不做選堆、不發布 NT，只印資料。

使用方式
--------
  python3 main_checkCameraPose2d.py
或只測單台：
  CAMERAS = ["Camera1"]  / CAMERAS = ["Camera2"]
"""

import time
import math

from photon_nt_multicam import PhotonMultiCamClient
from pose2d_reader import Pose2dReader, Pose2d
from utils.distance_utils import distance_calculate
from utils.pose_utils import cameraPose2d_calculate
from utils.ballpose_utils import ball_xy_from_camera

# ─────────────────────────────────────────────────────────────────────────────
# Network
# ─────────────────────────────────────────────────────────────────────────────
NT_SERVER        = "10.69.98.2"
CAMERAS          = ["Camera1", "Camera2"]   # 可改成只測單台
ROBOT_POSE_TOPIC = "/AdvantageKit/RealOutputs/RobotState/robotPose"

# ─────────────────────────────────────────────────────────────────────────────
# Camera / Geometry（與 main_final.py 保持一致）
# ─────────────────────────────────────────────────────────────────────────────
CAMERA_HEIGHT_M  = 0.527
CAMERA_PITCH_DEG = 21.0
TARGET_HEIGHT_M  = 0.075

Camera_Vertical_Displacement = 0.641

CAMERA1_HORIZONTAL_M   = +0.246
CAMERA1_YAW_OFFSET_DEG = +30.0
CAMERA1_YAW_SIGN       = -1.0

CAMERA2_HORIZONTAL_M   = -0.246
CAMERA2_YAW_OFFSET_DEG = -30.0    # ⚠ 待校正
CAMERA2_YAW_SIGN       = -1.0     # ⚠ 待校正

DISTANCE_MIN_M = 0.15
DISTANCE_MAX_M = 5.00

# ─────────────────────────────────────────────────────────────────────────────
# Loop 控制
# ─────────────────────────────────────────────────────────────────────────────
PRINT_EVERY_N_LOOPS = 1     # 測試模式：每圈都印
LOOP_SLEEP_SEC      = 0.20  # 比 main_final 慢，讓輸出容易看
STALE_TIMEOUT_SEC   = 0.5

# ─────────────────────────────────────────────────────────────────────────────
# 每台鏡頭的設定對照表（方便動態查詢）
# ─────────────────────────────────────────────────────────────────────────────
CAM_CONFIG = {
    "Camera1": {
        "horizontal_m":   CAMERA1_HORIZONTAL_M,
        "yaw_offset_deg": CAMERA1_YAW_OFFSET_DEG,
        "yaw_sign":       CAMERA1_YAW_SIGN,
    },
    "Camera2": {
        "horizontal_m":   CAMERA2_HORIZONTAL_M,
        "yaw_offset_deg": CAMERA2_YAW_OFFSET_DEG,
        "yaw_sign":       CAMERA2_YAW_SIGN,
    },
}


def process_camera(
    cam_name:     str,
    pv:           PhotonMultiCamClient,
    robot_pose2d: Pose2d | None,
):
    """
    回傳 (camera_pose2d, results)
    results 是 list of dict，每筆包含：
      target_number, yaw, pitch, area, distance, ball_xy, dist_filtered
    """
    cfg = CAM_CONFIG[cam_name]

    camera_pose2d = None
    if robot_pose2d is not None:
        camera_pose2d = cameraPose2d_calculate(
            robot_pose2d,
            Camera_Vertical_Displacement,
            cfg["horizontal_m"],
        )

    yaw_list   = getattr(pv, f"{cam_name}_Yaw")
    pitch_list = getattr(pv, f"{cam_name}_Pitch")
    area_list  = getattr(pv, f"{cam_name}_Area")

    dist_list = distance_calculate(
        pitch_list,
        camera_height_m  = CAMERA_HEIGHT_M,
        camera_pitch_deg = CAMERA_PITCH_DEG,
        target_height_m  = TARGET_HEIGHT_M,
    )

    results = []
    n = min(len(yaw_list), len(pitch_list), len(area_list), len(dist_list))

    for idx in range(n):
        yaw   = yaw_list[idx]
        pitch = pitch_list[idx]
        area  = area_list[idx]
        dist  = dist_list[idx]

        dist_ok = (
            dist is not None
            and DISTANCE_MIN_M <= dist <= DISTANCE_MAX_M
        )
        ball_xy = None
        if camera_pose2d is not None and dist_ok and yaw is not None:
            ball_xy = ball_xy_from_camera(
                camera_pose2d         = camera_pose2d,
                yaw_deg               = yaw,
                distance_m            = dist,
                camera_yaw_offset_deg = cfg["yaw_offset_deg"],
                yaw_sign              = cfg["yaw_sign"],
            )

        results.append({
            "target_number": idx,
            "yaw":           yaw,
            "pitch":         pitch,
            "area":          area,
            "distance":      dist,
            "dist_filtered": not dist_ok,
            "ball_xy":       ball_xy,
        })

    return camera_pose2d, results


def fmt(v, fmt_str=".4f", none_str="None"):
    return none_str if v is None else format(float(v), fmt_str)


def print_pose(label, pose):
    if pose is None:
        print(f"  {label}: None")
    else:
        print(
            f"  {label}: "
            f"x={pose.x:.4f}  "
            f"y={pose.y:.4f}  "
            f"heading={math.degrees(pose.heading_rad):.2f}°"
        )


def print_camera_section(cam_name, cam_pose2d, results, state, elapsed):
    if state.last_error:
        print(f"  [{cam_name}] decode error: {state.last_error}")
        return
    if elapsed > STALE_TIMEOUT_SEC:
        print(f"  [{cam_name}] ⚠ STALE ({elapsed:.2f}s)")
        return

    print_pose(f"{cam_name} pose", cam_pose2d)

    if not results:
        print(f"  [{cam_name}] no targets")
        return

    # Header
    print(f"  [{cam_name}] {len(results)} target(s):")
    print(f"    {'t':>3}  {'yaw':>8}  {'pitch':>8}  {'area':>7}  "
          f"{'dist':>7}  {'flag':>4}  {'ball_x':>8}  {'ball_y':>8}")
    print("    " + "-" * 66)

    for r in results:
        flag = "FILT" if r["dist_filtered"] else ("OK" if r["ball_xy"] is not None else "noPose")
        bx = fmt(r["ball_xy"][0] if r["ball_xy"] else None)
        by = fmt(r["ball_xy"][1] if r["ball_xy"] else None)
        print(
            f"    t{r['target_number']:02d}  "
            f"{fmt(r['yaw'],'.3f'):>8}  "
            f"{fmt(r['pitch'],'.3f'):>8}  "
            f"{fmt(r['area'],'.3f'):>7}  "
            f"{fmt(r['distance'],'.3f'):>7}  "
            f"{flag:>4}  "
            f"{bx:>8}  {by:>8}"
        )

    # 如果兩台鏡頭都有同一顆球，印出投影差（只在同一批 results 比較沒有意義；
    # 在 main loop 裡再跨鏡頭比較）


def main():
    pv = PhotonMultiCamClient(
        server                    = NT_SERVER,
        cameras                   = CAMERAS,
        sort_targets_by_area_desc = False,
    )
    pv.start()

    pose_reader = Pose2dReader(
        server     = NT_SERVER,
        topic_path = ROBOT_POSE_TOPIC,
    )

    print("=" * 72)
    print("main_checkCameraPose2d — 相機位置與球座標即時驗證工具")
    print(f"監視鏡頭：{CAMERAS}")
    print(f"每 {LOOP_SLEEP_SEC}s 印一次，Ctrl-C 結束")
    print("=" * 72)

    loop_count = 0

    while True:
        loop_count += 1

        robot_pose2d = pose_reader.get_pose2d()

        # 各鏡頭處理
        cam_data = {}
        for cam_name in CAMERAS:
            state   = pv.get_state(cam_name)
            elapsed = time.monotonic() - state.last_update_monotonic

            if elapsed > STALE_TIMEOUT_SEC:
                cam_data[cam_name] = {
                    "pose2d":  None,
                    "results": [],
                    "state":   state,
                    "elapsed": elapsed,
                }
            else:
                pose2d, results = process_camera(cam_name, pv, robot_pose2d)
                cam_data[cam_name] = {
                    "pose2d":  pose2d,
                    "results": results,
                    "state":   state,
                    "elapsed": elapsed,
                }

        if loop_count % PRINT_EVERY_N_LOOPS != 0:
            time.sleep(LOOP_SLEEP_SEC)
            continue

        print(f"\n{'─'*72}")
        print(f"Loop #{loop_count}")

        # Robot pose
        print_pose("robot_pose2d", robot_pose2d)

        # 相機位移驗證提示（只在 loop 1 印一次）
        if loop_count == 1 and robot_pose2d is not None:
            print()
            print("  [驗證提示]")
            print("  camera1_pose 應在 robot_pose 的後方偏左：")
            print("    ‧ x 差 ≈ -0.641·cos(θ) - 0.246·sin(θ)（後方偏左）")
            print("    ‧ y 差 ≈ -0.641·sin(θ) + 0.246·cos(θ)")
            print("  camera2_pose 應在 robot_pose 的後方偏右：")
            print("    ‧ x 差 ≈ -0.641·cos(θ) + 0.246·sin(θ)（後方偏右）")
            print("    ‧ y 差 ≈ -0.641·sin(θ) - 0.246·cos(θ)")

        # 每台鏡頭的資料
        for cam_name in CAMERAS:
            d = cam_data[cam_name]
            print()
            print_camera_section(cam_name, d["pose2d"], d["results"], d["state"], d["elapsed"])

        # 跨鏡頭比較（若兩台都有 ball_xy，印出配對距離）
        if len(CAMERAS) == 2:
            c1, c2 = CAMERAS[0], CAMERAS[1]
            xys1 = [r["ball_xy"] for r in cam_data[c1]["results"] if r["ball_xy"] is not None]
            xys2 = [r["ball_xy"] for r in cam_data[c2]["results"] if r["ball_xy"] is not None]
            if xys1 and xys2:
                print()
                print("  [跨鏡頭投影比較]")
                print(f"  {c1} 有 {len(xys1)} 個 ball_xy，{c2} 有 {len(xys2)} 個 ball_xy")
                print("  最近配對（距離 ≤ 0.20m 視為同一球，> 0.20m 可能是不同球或 yaw 偏差）：")
                for p1 in xys1:
                    nearest = min(xys2, key=lambda p2: math.hypot(p1[0]-p2[0], p1[1]-p2[1]))
                    d = math.hypot(p1[0]-nearest[0], p1[1]-nearest[1])
                    flag = "✓ 對齊" if d <= 0.20 else "⚠ 偏差"
                    print(f"    ({p1[0]:.3f},{p1[1]:.3f}) ↔ "
                          f"({nearest[0]:.3f},{nearest[1]:.3f})  "
                          f"dist={d:.3f}m  {flag}")

        time.sleep(LOOP_SLEEP_SEC)


if __name__ == "__main__":
    main()