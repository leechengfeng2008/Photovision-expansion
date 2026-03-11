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
CAMERAS          = ["Camera1"]
ROBOT_POSE_TOPIC = "/AdvantageKit/RealOutputs/RobotState/robotPose"

# ─────────────────────────────────────────────────────────────────────────────
# Camera / Geometry
# ─────────────────────────────────────────────────────────────────────────────
CAMERA_HEIGHT_M  = 0.527
CAMERA_PITCH_DEG = 21.0
TARGET_HEIGHT_M  = 0.075

Camera_Vertical_Displacement = 0.641

CAMERA1_HORIZONTAL_M   = +0.246
CAMERA1_YAW_OFFSET_DEG = +30.0
CAMERA1_YAW_SIGN       = -1.0

# 修正 [5]：舊版無此過濾，pitch≈0 時可算出 50m+
DISTANCE_MIN_M = 0.15
DISTANCE_MAX_M = 5.00

# ─────────────────────────────────────────────────────────────────────────────
# Grid Rect 分群
# ─────────────────────────────────────────────────────────────────────────────
CELL_SIZE_M      = 0.50
DIAGONAL_CONNECT = True

# 回滾旗標：
#   "density_weighted" → 矩形框內密度加權中心（建議）
#   "rect_center"      → 純幾何中心（一行回滾）
CENTER_MODE      = "density_weighted"
DENSITY_RADIUS_M = 0.50

# ─────────────────────────────────────────────────────────────────────────────
# 選堆
# ─────────────────────────────────────────────────────────────────────────────
PILE_BALL_PRIORITY_0_TO_10 = 8.0   # 0=完全偏最近, 10=完全偏球多

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
# 修正 [2]：舊版無此保護
STALE_TIMEOUT_SEC   = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# 單鏡頭處理
# ─────────────────────────────────────────────────────────────────────────────
def process_camera1(
    pv:           PhotonMultiCamClient,
    robot_pose2d: Pose2d | None,
):
    camera_pose2d = None
    if robot_pose2d is not None:
        camera_pose2d = cameraPose2d_calculate(
            robot_pose2d,
            Camera_Vertical_Displacement,
            CAMERA1_HORIZONTAL_M,
        )

    yaw_list   = getattr(pv, "Camera1_Yaw")
    pitch_list = getattr(pv, "Camera1_Pitch")
    area_list  = getattr(pv, "Camera1_Area")

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

        # 修正 [5]：距離合法範圍過濾
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
                camera_yaw_offset_deg = CAMERA1_YAW_OFFSET_DEG,
                yaw_sign              = CAMERA1_YAW_SIGN,
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

    # 修正 [6][7]：改用 Pose2dReader，NT 瞬斷時自動回傳 _last_good，不會變 None
    pose_reader = Pose2dReader(
        server     = NT_SERVER,
        topic_path = ROBOT_POSE_TOPIC,
    )

    # 修正 [9]：使用 nt_publish_utils，不在 main 內寫 local function
    _pub_inst, best_pose_pub = create_best_pose2d_publisher(
        server = NT_SERVER,
        table  = BEST_POSE2D_TABLE,
        key    = BEST_POSE2D_KEY,
    )

    loop_count = 0

    while True:
        loop_count += 1

        robot_pose2d = pose_reader.get_pose2d()
        cam1_state   = pv.get_state("Camera1")

        # 修正 [2]：stale-frame 保護
        elapsed = time.monotonic() - cam1_state.last_update_monotonic
        if elapsed > STALE_TIMEOUT_SEC:
            ball_xys     = []
            cam1_results = []
            cam1_pose2d  = None
        else:
            cam1_pose2d, cam1_results = process_camera1(pv, robot_pose2d)
            ball_xys = [r["ball_xy"] for r in cam1_results if r["ball_xy"] is not None]

        # ── 分群 ─────────────────────────────────────────────────────────────
        pile_count, pile_plans, _ = plan_ballpile_rect_centers(
            ball_xys         = ball_xys,
            cell_size_m      = CELL_SIZE_M,
            diagonal_connect = DIAGONAL_CONNECT,
            center_mode      = CENTER_MODE,
            density_radius_m = DENSITY_RADIUS_M,
        )

        # ── 選堆 ─────────────────────────────────────────────────────────────
        best_pile = select_best_rect_pile(
            pile_plans          = pile_plans,
            robot_pose2d        = robot_pose2d,
            ball_priority_0to10 = PILE_BALL_PRIORITY_0_TO_10,
        )

        # ── 發布 ─────────────────────────────────────────────────────────────
        # 修正 [1]：舊版在 robot_pose2d=None 且 best_pile!=None 時 crash
        publish_best_pile(
            pub          = best_pose_pub,
            best_pile    = best_pile,
            robot_pose2d = robot_pose2d,
        )

        # ── Debug print ──────────────────────────────────────────────────────
        if loop_count % PRINT_EVERY_N_LOOPS != 0:
            time.sleep(LOOP_SLEEP_SEC)
            continue

        if cam1_state.last_error:
            print("Camera1 decode error:", cam1_state.last_error)

        if elapsed > STALE_TIMEOUT_SEC:
            print(f"\n⚠ Camera1 stale ({elapsed:.2f}s since last update)")

        if robot_pose2d is None:
            print("\n=== robotPose unavailable ===")
        else:
            print(
                f"\n=== robotPose ===  "
                f"x={robot_pose2d.x:.4f}  "
                f"y={robot_pose2d.y:.4f}  "
                f"heading={math.degrees(robot_pose2d.heading_rad):.2f}°"
            )

        print(f"\n=== Camera1 ({len(cam1_results)} targets) ===")
        for r in cam1_results:
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
