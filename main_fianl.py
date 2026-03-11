from __future__ import annotations
import time
import math
import ntcore

from photon_nt_multicam import PhotonMultiCamClient
from utils.distance_utils import distance_calculate
from utils.pose_utils import cameraPose2d_calculate
from utils.ballpose_utils import ball_xy_from_camera
from utils.ballpile_grid_rect import plan_ballpile_rect_centers, select_best_rect_pile
from utils.nt_publish_utils import create_best_pose2d_publisher, publish_best_pile
from wpimath.geometry import Pose2d


# ─────────────────────────────────────────────────────────────────────────────
# Network / Hardware
# ─────────────────────────────────────────────────────────────────────────────
NT_SERVER = "10.69.98.2"
CAMERAS   = ["Camera1"]

ROBOT_POSE_TOPIC = "/AdvantageKit/RealOutputs/RobotState/robotPose"

# ─────────────────────────────────────────────────────────────────────────────
# Camera / Geometry
# ─────────────────────────────────────────────────────────────────────────────
CAMERA_HEIGHT_M  = 0.527
CAMERA_PITCH_DEG = 21
TARGET_HEIGHT_M  = 0.075

Camera_Vertical_Displacement   = 0.641
Camera_Horizontal_Displacement = 0.246

CAMERA1_HORIZONTAL_M   = +Camera_Horizontal_Displacement
CAMERA1_YAW_OFFSET_DEG = +30.0
YAW_SIGN               = -1.0

# 距離合法範圍（超出視為雜訊，直接丟棄）
DISTANCE_MIN_M = 0.15
DISTANCE_MAX_M = 5.00

# ─────────────────────────────────────────────────────────────────────────────
# Grid Rect 分群參數
# ─────────────────────────────────────────────────────────────────────────────
CELL_SIZE_M        = 0.50    # intake 寬度 50cm，格子與之對齊
DIAGONAL_CONNECT   = True    # 斜角相鄰格子視為同一堆

# 回滾旗標：
#   "density_weighted" → 矩形框內密度加權中心（建議）
#   "rect_center"      → 外接矩形幾何中心（一行回滾）
CENTER_MODE        = "density_weighted"
DENSITY_RADIUS_M   = 0.50    # 加權鄰域半徑，建議與 CELL_SIZE_M 相同

# ─────────────────────────────────────────────────────────────────────────────
# 選堆參數
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
STALE_TIMEOUT_SEC   = 0.5    # 超過此秒數無新影格，視為鏡頭斷線


# ─────────────────────────────────────────────────────────────────────────────
# Robot Pose2d wrapper（與 utils 相容的最小 interface）
# ─────────────────────────────────────────────────────────────────────────────
class SimplePose2d:
    def __init__(self, x: float, y: float, heading_rad: float):
        self.x           = x
        self.y           = y
        self.heading_rad = heading_rad


# ─────────────────────────────────────────────────────────────────────────────
# Robot pose subscriber
# ─────────────────────────────────────────────────────────────────────────────
def create_robot_pose_subscriber(
    server:      str,
    topic_name:  str = ROBOT_POSE_TOPIC,
    client_name: str = "ak-robotpose-reader",
):
    inst = ntcore.NetworkTableInstance.create()
    inst.startClient4(client_name)
    inst.setServer(server)
    sub = inst.getStructTopic(topic_name, Pose2d).subscribe(Pose2d())
    return inst, sub


def get_robot_pose2d(robot_pose_sub) -> SimplePose2d | None:
    pose = robot_pose_sub.get()
    if pose is None:
        return None

    # 舊版的 try/except 兩塊 heading_rad 完全相同（沒有修復效果），已清理
    # --- 舊版 ---
    # try:    heading_rad = float(pose.rotation().radians())
    # except: heading_rad = float(pose.rotation().radians())  # ← 與 try 一樣
    try:
        x = float(pose.x)
        y = float(pose.y)
    except AttributeError:
        x = float(pose.X())
        y = float(pose.Y())

    heading_rad = float(pose.rotation().radians())
    return SimplePose2d(x=x, y=y, heading_rad=heading_rad)


# ─────────────────────────────────────────────────────────────────────────────
# 單鏡頭資料處理
# ─────────────────────────────────────────────────────────────────────────────
def process_camera1(
    pv: PhotonMultiCamClient,
    robot_pose2d: SimplePose2d | None,
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

        # 舊版：所有 dist（含 50m 等異常值）都會進 ball_xy 計算
        # 修正：距離合法範圍檢查
        # --- 舊版 ---
        # ball_xy = None
        # if camera_pose2d is not None and dist is not None and yaw is not None:
        #     ball_xy = ball_xy_from_camera(...)
        dist_valid = (
            dist is not None
            and DISTANCE_MIN_M <= dist <= DISTANCE_MAX_M
        )
        ball_xy = None
        if camera_pose2d is not None and dist_valid and yaw is not None:
            ball_xy = ball_xy_from_camera(
                camera_pose2d         = camera_pose2d,
                yaw_deg               = yaw,
                distance_m            = dist,
                camera_yaw_offset_deg = CAMERA1_YAW_OFFSET_DEG,
                yaw_sign              = YAW_SIGN,
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
        server                = NT_SERVER,
        cameras               = CAMERAS,
        sort_targets_by_area_desc = False,
    )
    pv.start()

    # NT publisher（舊版直接在 main 內建立，現在改從 utils 取）
    # --- 舊版 ---
    # _, best_pose_pub = create_best_pose2d_publisher(...)  # main.py 內的 local function
    _pub_inst, best_pose_pub = create_best_pose2d_publisher(
        server = NT_SERVER,
        table  = BEST_POSE2D_TABLE,
        key    = BEST_POSE2D_KEY,
    )

    _sub_inst, robot_pose_sub = create_robot_pose_subscriber(
        server     = NT_SERVER,
        topic_name = ROBOT_POSE_TOPIC,
    )

    loop_count = 0

    while True:
        loop_count += 1

        robot_pose2d              = get_robot_pose2d(robot_pose_sub)
        camera1_pose2d, cam1_results = process_camera1(pv, robot_pose2d)
        cam1_state                = pv.get_state("Camera1")

        # Stale-frame 保護
        # 舊版：PhotonVision 斷線時舊資料持續被計算
        # 修正：超過 STALE_TIMEOUT_SEC 沒有更新 → 視為空資料
        elapsed = time.monotonic() - cam1_state.last_update_monotonic
        if elapsed > STALE_TIMEOUT_SEC:
            ball_xys = []
        else:
            ball_xys = [
                r["ball_xy"] for r in cam1_results if r["ball_xy"] is not None
            ]

        # ── Grid 格子分群（取代舊版 plan_ballpile_centers）────────────────────
        # 舊版：
        # pile_count, pile_plans, _ = plan_ballpile_centers(
        #     ball_xys=unique_ball_xys,
        #     cluster_link_m=PILE_CLUSTER_LINK_M,
        #     center_mode=PILE_CENTER_MODE,
        #     density_radius_m=PILE_DENSITY_RADIUS_M,
        #     density_spread_limit_m=PILE_DENSITY_SPREAD_LIMIT_M,
        # )
        pile_count, pile_plans, _ = plan_ballpile_rect_centers(
            ball_xys         = ball_xys,
            cell_size_m      = CELL_SIZE_M,
            diagonal_connect = DIAGONAL_CONNECT,
            center_mode      = CENTER_MODE,
            density_radius_m = DENSITY_RADIUS_M,
        )

        # ── 選堆（不經過 build_candidates + select_best_pile）────────────────
        # 舊版：
        # pile_candidates = build_candidates(...)
        # best_pile, _ = select_best_pile(robot_pose=..., pile_candidates=..., ...)
        best_pile = select_best_rect_pile(
            pile_plans          = pile_plans,
            robot_pose2d        = robot_pose2d,
            ball_priority_0to10 = PILE_BALL_PRIORITY_0_TO_10,
        )

        # ── 發布（舊版 publish_best_pose2d_to_NT 改從 utils 取）──────────────
        publish_best_pile(
            pub          = best_pose_pub,
            best_pile    = best_pile,
            robot_pose2d = robot_pose2d,
        )

        # ── Debug print（每 N 圈才印）────────────────────────────────────────
        if loop_count % PRINT_EVERY_N_LOOPS == 0:

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
                dist_str = f"{r['distance']:.3f}m" if r["distance"] is not None else "None"
                print(
                    f"  t{r['target_number']:02d}  "
                    f"dist={dist_str}  "
                    f"ball_xy={r['ball_xy']}"
                )

            print(
                f"\n=== PILES  ({pile_count} piles"
                f"  cell={CELL_SIZE_M}m  mode={CENTER_MODE}) ==="
            )
            print(f"  ball_count = {len(ball_xys)}")
            for p in pile_plans:
                fb_str = ""
                if p.center_mode == "density_weighted":
                    fb_str = (
                        f"  max_nb={p.max_neighbor_count}"
                        f"  rect_fb={'YES' if p.used_rect_fallback else 'no'}"
                    )
                print(
                    f"  pile {p.pile_id}: "
                    f"center={p.center_xy}  "
                    f"count={p.count}  "
                    f"cells={p.occupied_cell_count}  "
                    f"rect={p.rect_min_xy}~{p.rect_max_xy}"
                    f"{fb_str}"
                )

            print("\n=== BEST PILE ===")
            if best_pile is None:
                print("  best_pile = None")
            else:
                print(f"  pile_id   = {best_pile.pile_id}")
                print(f"  center_xy = {best_pile.center_xy}")
                print(f"  count     = {best_pile.count}")
                print(f"  rect_size = {best_pile.rect_size_xy}")

        time.sleep(LOOP_SLEEP_SEC)


if __name__ == "__main__":
    main()
