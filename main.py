from __future__ import annotations
import time
import math
import ntcore

from photon_nt_multicam import PhotonMultiCamClient
from utils.distance_utils import distance_calculate
from utils.pose_utils import cameraPose2d_calculate
from pose2d_reader import Pose2dReader
from utils.ballpose_utils import ball_xy_from_camera
from utils.ball_dedupe_twocam_utils import dedupe_two_cameras
from utils.ballpile_onlycenter_utils import plan_ballpile_centers
from utils.pile_selector_utils import build_candidates, select_best_pile


NT_SERVER = "10.25.6.2"
CAMERAS = ["Camera1", "Camera2"]
POSE2D_KEY = "Pose2d"
HEADING_UNITS = "deg"
POSE2D_TABLE = "SmartDashboard"

CAMERA_HEIGHT_M = 0.55
CAMERA_PITCH_DEG = 30
TARGET_HEIGHT_M = 0.075

Camera_Vertical_Displacement = 0.641
Camera_Horizontal_Displacement = 0.246
CAMERA1_HORIZONTAL_M = +Camera_Horizontal_Displacement
CAMERA2_HORIZONTAL_M = -Camera_Horizontal_Displacement

CAMERA1_YAW_OFFSET_DEG = +35.0
CAMERA2_YAW_OFFSET_DEG = -35.0

YAW_SIGN = 1.0

PILE_BALL_PRIORITY_0_TO_10 = 5.0   # 0=完全偏最近, 10=完全偏球多
BEST_POSE2D_TABLE = "SmartDashboard"
BEST_POSE2D_KEY = "BestPilePose2d"

def fmt_num(v):
    if v is None:
        return "None"
    return f"{v:+7.3f}"


def fmt_area(v):
    if v is None:
        return "None"
    return f"{v:7.3f}"


def process_one_camera(cam_name: str, pv: PhotonMultiCamClient, robot_pose2d):
    if cam_name == "Camera1":
        camera_horizontal_m = CAMERA1_HORIZONTAL_M
        camera_yaw_offset_deg = CAMERA1_YAW_OFFSET_DEG
    elif cam_name == "Camera2":
        camera_horizontal_m = CAMERA2_HORIZONTAL_M
        camera_yaw_offset_deg = CAMERA2_YAW_OFFSET_DEG
    else:
        camera_horizontal_m = 0.0
        camera_yaw_offset_deg = 0.0

    camera_pose2d = None
    if robot_pose2d is not None:
        camera_pose2d = cameraPose2d_calculate(
            robot_pose2d,
            Camera_Vertical_Displacement,
            camera_horizontal_m,
        )

    yaw_list = getattr(pv, f"{cam_name}_Yaw")
    pitch_list = getattr(pv, f"{cam_name}_Pitch")
    area_list = getattr(pv, f"{cam_name}_Area")

    dist_list = distance_calculate(
        pitch_list,
        camera_height_m=CAMERA_HEIGHT_M,
        camera_pitch_deg=CAMERA_PITCH_DEG,
        target_height_m=TARGET_HEIGHT_M,
    )

    results = []
    n = min(len(yaw_list), len(pitch_list), len(area_list), len(dist_list))

    for target_number in range(n):
        yaw = yaw_list[target_number]
        pitch = pitch_list[target_number]
        area = area_list[target_number]
        dist = dist_list[target_number]

        ball_xy = None
        if camera_pose2d is not None and dist is not None and yaw is not None:
            ball_xy = ball_xy_from_camera(
                camera_pose2d=camera_pose2d,
                yaw_deg=yaw,
                distance_m=dist,
                camera_yaw_offset_deg=camera_yaw_offset_deg,
                yaw_sign=YAW_SIGN,
            )

        results.append({
            "target_number": target_number,
            "yaw": yaw,
            "pitch": pitch,
            "area": area,
            "distance": dist,
            "ball_xy": ball_xy,
        })

    return camera_pose2d, results

def create_best_pose2d_publisher(
    server: str,
    table: str = BEST_POSE2D_TABLE,
    key: str = BEST_POSE2D_KEY,
    client_name: str = "best-pile-publisher",
):
    """
    建立一個專門用來發布最佳球堆 Pose2d 的 NT client 與 publisher。

    發布格式:
        [x, y, heading_deg]

    若沒有最佳球堆，publish_best_pose2d_to_NT() 會發布 []，
    讓接收端可用 len < 3 判定為無效。
    """
    inst = ntcore.NetworkTableInstance.create()
    inst.startClient4(client_name)
    inst.setServer(server)

    table_obj = inst.getTable(table)
    pub = table_obj.getDoubleArrayTopic(key).publish()
    return inst, pub


def publish_best_pose2d_to_NT(best_pose_pub, best_pile, robot_pose2d):

    if best_pile is None:
        best_pose_pub.set([])
        return

    x = float(best_pile.center_xy[0])
    y = float(best_pile.center_xy[1])

    if robot_pose2d is None:
        heading_deg = 0.0
    else:
        dx = x - float(robot_pose2d.x)
        dy = y - float(robot_pose2d.y)

        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            heading_deg = math.degrees(float(robot_pose2d.heading_rad))
        else:
            heading_deg = math.degrees(math.atan2(dy, dx))

    best_pose_pub.set([x, y, heading_deg])


def main():
    pose_reader = Pose2dReader(
        server=NT_SERVER,
        table=POSE2D_TABLE,
        key=POSE2D_KEY,
        heading_units=HEADING_UNITS,
    )

    pv = PhotonMultiCamClient(
        server=NT_SERVER,
        cameras=CAMERAS,
        sort_targets_by_area_desc=False,
    )
    pv.start()

    best_pose_nt_inst, best_pose_pub = create_best_pose2d_publisher(
    server=NT_SERVER,
    table=BEST_POSE2D_TABLE,
    key=BEST_POSE2D_KEY,
    )

    while True:
        robot_pose2d = pose_reader.get_pose2d()

        camera1_pose2d, cam1_results = process_one_camera("Camera1", pv, robot_pose2d)
        camera2_pose2d, cam2_results = process_one_camera("Camera2", pv, robot_pose2d)

        cam1_state = pv.get_state("Camera1")
        cam2_state = pv.get_state("Camera2")

        if cam1_state.last_error:
            print("Camera1 decode error:", cam1_state.last_error)
        if cam2_state.last_error:
            print("Camera2 decode error:", cam2_state.last_error)

        print("\n=== Camera1 ===")
        for r in cam1_results:
            print(
                f"t{r['target_number']:02d}  "
                f"yaw={fmt_num(r['yaw'])}  "
                f"pitch={fmt_num(r['pitch'])}  "
                f"area={fmt_area(r['area'])}  "
                f"dist={r['distance']}  "
                f"ball_xy={r['ball_xy']}"
            )

        print("\n=== Camera2 ===")
        for r in cam2_results:
            print(
                f"t{r['target_number']:02d}  "
                f"yaw={fmt_num(r['yaw'])}  "
                f"pitch={fmt_num(r['pitch'])}  "
                f"area={fmt_area(r['area'])}  "
                f"dist={r['distance']}  "
                f"ball_xy={r['ball_xy']}"
            )

        # --------------------------------------------------
        # 1) 整理兩顆鏡頭各自的球座標
        # --------------------------------------------------
        cam1_ball_xys = [r["ball_xy"] for r in cam1_results if r["ball_xy"] is not None]
        cam2_ball_xys = [r["ball_xy"] for r in cam2_results if r["ball_xy"] is not None]

        # --------------------------------------------------
        # 2) 雙鏡頭刪重
        # same_ball_error_m 單位是公尺
        # 0.10 = 10 cm 內視為同一顆球
        # --------------------------------------------------
        dedupe_result = dedupe_two_cameras(
            cam1_ball_xys=cam1_ball_xys,
            cam2_ball_xys=cam2_ball_xys,
            same_ball_error_m=0.10,
            keep="average",   # 可改成 "cam1" 或 "cam2"
        )

        unique_ball_xys = dedupe_result.unique_points

        print("\n=== DEDUPE RESULT ===")
        print("cam1_ball_count =", len(cam1_ball_xys))
        print("cam2_ball_count =", len(cam2_ball_xys))
        print("unique_ball_count =", len(unique_ball_xys))
        print("unique_ball_xys =", unique_ball_xys)

        # --------------------------------------------------
        # 3) 球堆分堆 / 中心輸出
        # cluster_link_m 單位是公尺
        # 0.30 = 30 cm 內視為同一堆可連結
        # --------------------------------------------------
        pile_count, pile_plans, all_center_xys = plan_ballpile_centers(
            ball_xys=unique_ball_xys,
            cluster_link_m=0.30,
        )

        # --------------------------------------------------
        # 4) 球堆選擇
        # 0 = 完全偏最近
        # 10 = 完全偏球多
        # --------------------------------------------------
        pile_candidates = build_candidates(
            center_xys=[p.center_xy for p in pile_plans],
            counts=[p.count for p in pile_plans],
            pile_ids=[p.pile_id for p in pile_plans],
        )

        best_pile, pile_score_infos = select_best_pile(
            robot_pose=robot_pose2d,
            pile_candidates=pile_candidates,
            ball_priority_0to10=PILE_BALL_PRIORITY_0_TO_10,
        )

        publish_best_pose2d_to_NT(
        best_pose_pub=best_pose_pub,
        best_pile=best_pile,
        robot_pose2d=robot_pose2d,
        )

        print("\n=== BALL PILES ===")
        print("pile_count =", pile_count)

        for p in pile_plans:
            print(
                f"pile {p.pile_id}: "
                f"center={p.center_xy}, "
                f"count={p.count}"
            )

        print("all_center_xys =", all_center_xys)

        print("\n=== PILE SCORES ===")
        print("PILE_BALL_PRIORITY_0_TO_10 =", PILE_BALL_PRIORITY_0_TO_10)

        for s in pile_score_infos:
            print(
                f"pile {s.pile_id}: "
                f"center={s.center_xy}, "
                f"count={s.count}, "
                f"dist={s.distance_from_robot_m:.3f} m, "
                f"near_score={s.near_score:.3f}, "
                f"count_score={s.count_score:.3f}, "
                f"final_score={s.final_score:.3f}"
            )

        print("\n=== BEST PILE ===")
        if best_pile is None:
            print("best_pile = None")
        else:
            print("best_pile_id =", best_pile.pile_id)
            print("best_center_xy =", best_pile.center_xy)
            print("best_count =", best_pile.count)

        print("PILE_BALL_PRIORITY_0_TO_10 =", PILE_BALL_PRIORITY_0_TO_10)


        time.sleep(0.015)


if __name__ == "__main__":
    main()