from __future__ import annotations
import time
import math

from photon_nt_multicam import PhotonMultiCamClient
from pose2d_reader import Pose2dReader, Pose2d
from utils.distance_utils import distance_calculate
from utils.pose_utils import cameraPose2d_calculate
from utils.ballpose_utils import ball_xy_from_camera
from utils.ballpile_grid_rect import plan_ballpile_rect_centers, select_best_rect_pile
from utils.nt_publish_utils import create_best_pose2d_publisher, publish_best_pile

# ─────────────────────────────────────────────────────────────────────────────
# Network
# ─────────────────────────────────────────────────────────────────────────────
NT_SERVER        = "10.69.98.2"
CAMERAS          = ["Camera2"]
ROBOT_POSE_TOPIC = "/AdvantageKit/RealOutputs/RobotState/robotPose"

# ─────────────────────────────────────────────────────────────────────────────
# Camera / Geometry
# ⚠ Camera2 的 YAW_OFFSET 與 YAW_SIGN 為初始推估值，
#   必須用 cam2_param_check_interactive.py 校正後填入正確數值
# ─────────────────────────────────────────────────────────────────────────────
CAMERA_HEIGHT_M  = 0.527
CAMERA_PITCH_DEG = 21.0
TARGET_HEIGHT_M  = 0.075

Camera_Vertical_Displacement = 0.641

CAMERA2_HORIZONTAL_M   = -0.246        # 右側，與 Camera1 鏡像
CAMERA2_YAW_OFFSET_DEG = -30.0         # ⚠ 待校正
CAMERA2_YAW_SIGN       = -1.0          # ⚠ 待校正（通常與 Camera1 相同，但需實測確認）

# 距離合法範圍過濾
DISTANCE_MIN_M = 0.15
DISTANCE_MAX_M = 5.00

# ─────────────────────────────────────────────────────────────────────────────
# Grid Rect 分群
# ─────────────────────────────────────────────────────────────────────────────
CELL_SIZE_M      = 0.50
DIAGONAL_CONNECT = True
CENTER_MODE      = "density_weighted"   # "rect_center" 一行回滾
DENSITY_RADIUS_M = 0.50

# ─────────────────────────────────────────────────────────────────────────────
# 選堆
# ─────────────────────────────────────────────────────────────────────────────
PILE_BALL_PRIORITY_0_TO_10 = 8.0

# ─────────────────────────────────────────────────────────────────────────────
# NT 發布
# ─────────────────────────────────────────────────────────────────────────────
BEST_POSE2D_TABLE = "SmartDashboard"
BEST_POSE2D_KEY   = "BestPilePose2d"

# ─────────────────────────────────────────────────────────────────────────────
# Loop 控制
# ─────────────────────────────────────────────────────────────────────────────
PRINT_EVERY_N_LOOPS = 5
LOOP_SLEEP_SEC      = 0.05
STALE_TIMEOUT_SEC   = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Camera2 處理
# ─────────────────────────────────────────────────────────────────────────────
def process_camera2(
    pv:           PhotonMultiCamClient,
    robot_pose2d: Pose2d | None,
):
    camera_pose2d = None
    if robot_pose2d is not None:
        camera_pose2d = cameraPose2d_calculate(
            robot_pose2d,
            Camera_Vertical_Displacement,
            CAMERA2_HORIZONTAL_M,
        )

    yaw_list   = getattr(pv, "Camera2_Yaw")
    pitch_list = getattr(pv, "Camera2_Pitch")
    area_list  = getattr(pv, "Camera2_Area")

    dist_list = distance_calculate(
        pitch_list,
        camera_height_m  = CAMERA_HEIGHT_M,
        camera_pitch_deg = CAMERA_PITCH_DEG,
        target_height_m  = TARGET_HEIGHT_M,
    )

    results = []
    n = min(len(yaw_list), len(pitch_list), len(area_list), len(dist_list))

    for idx in range(n):
        yaw  = yaw_list[idx]
        dist = dist_list[idx]

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
                camera_yaw_offset_deg = CAMERA2_YAW_OFFSET_DEG,
                yaw_sign              = CAMERA2_YAW_SIGN,
            )

        results.append({
            "target_number": idx,
            "distance":      dist,
            "ball_xy":       ball_xy,
        })

    return camera_pose2d, results


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
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

    _pub_inst, best_pose_pub = create_best_pose2d_publisher(
        server = NT_SERVER,
        table  = BEST_POSE2D_TABLE,
        key    = BEST_POSE2D_KEY,
    )

    loop_count = 0

    while True:
        loop_count += 1

        robot_pose2d = pose_reader.get_pose2d()
        cam2_state   = pv.get_state("Camera2")

        elapsed = time.monotonic() - cam2_state.last_update_monotonic
        if elapsed > STALE_TIMEOUT_SEC:
            ball_xys     = []
            cam2_results = []
            cam2_pose2d  = None
        else:
            cam2_pose2d, cam2_results = process_camera2(pv, robot_pose2d)
            ball_xys = [r["ball_xy"] for r in cam2_results if r["ball_xy"] is not None]

        pile_count, pile_plans, _ = plan_ballpile_rect_centers(
            ball_xys         = ball_xys,
            cell_size_m      = CELL_SIZE_M,
            diagonal_connect = DIAGONAL_CONNECT,
            center_mode      = CENTER_MODE,
            density_radius_m = DENSITY_RADIUS_M,
        )

        best_pile = select_best_rect_pile(
            pile_plans          = pile_plans,
            robot_pose2d        = robot_pose2d,
            ball_priority_0to10 = PILE_BALL_PRIORITY_0_TO_10,
        )

        publish_best_pile(
            pub          = best_pose_pub,
            best_pile    = best_pile,
            robot_pose2d = robot_pose2d,
        )

        if loop_count % PRINT_EVERY_N_LOOPS != 0:
            time.sleep(LOOP_SLEEP_SEC)
            continue

        if cam2_state.last_error:
            print("Camera2 decode error:", cam2_state.last_error)
        if elapsed > STALE_TIMEOUT_SEC:
            print(f"\n⚠ Camera2 stale ({elapsed:.2f}s since last update)")

        if robot_pose2d is None:
            print("\n=== robotPose unavailable ===")
        else:
            print(
                f"\n=== robotPose ===  "
                f"x={robot_pose2d.x:.4f}  "
                f"y={robot_pose2d.y:.4f}  "
                f"heading={math.degrees(robot_pose2d.heading_rad):.2f}°"
            )

        print(f"\n=== Camera2 ({len(cam2_results)} targets) ===")
        for r in cam2_results:
            ds = f"{r['distance']:.3f}m" if r["distance"] is not None else "None"
            print(f"  t{r['target_number']:02d}  dist={ds}  ball_xy={r['ball_xy']}")

        print(
            f"\n=== PILES  ({pile_count} piles  "
            f"cell={CELL_SIZE_M}m  mode={CENTER_MODE}) ==="
        )
        print(f"  ball_count = {len(ball_xys)}")
        for p in pile_plans:
            fb = ""
            if p.center_mode == "density_weighted":
                fb = (
                    f"  max_nb={p.max_neighbor_count}"
                    f"  rect_fb={'YES' if p.used_rect_fallback else 'no'}"
                )
            print(
                f"  pile {p.pile_id}: "
                f"center=({p.center_xy[0]:.3f},{p.center_xy[1]:.3f})  "
                f"count={p.count}  cells={p.occupied_cell_count}"
                f"{fb}"
            )

        print("\n=== BEST PILE ===")
        if best_pile is None:
            print("  None")
        else:
            print(
                f"  pile_id={best_pile.pile_id}  "
                f"center=({best_pile.center_xy[0]:.3f},{best_pile.center_xy[1]:.3f})  "
                f"count={best_pile.count}  "
                f"rect_size={best_pile.rect_size_xy}"
            )

        time.sleep(LOOP_SLEEP_SEC)


if __name__ == "__main__":
    main()











