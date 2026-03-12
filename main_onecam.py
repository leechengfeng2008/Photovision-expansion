from __future__ import annotations
import time
import math
import ntcore

from photon_nt_multicam import PhotonMultiCamClient
from utils.distance_utils import distance_calculate
from utils.pose_utils import cameraPose2d_calculate
from utils.ballpose_utils import ball_xy_from_camera
from utils.ballpile_onlycenter_utils import plan_ballpile_centers
from utils.pile_selector_utils import build_candidates, select_best_pile
from wpimath.geometry import Pose2d


NT_SERVER = "10.69.98.2"
CAMERAS = ["Camera1"]

# AdvantageKit robot pose topic
ROBOT_POSE_TOPIC = "/AdvantageKit/RealOutputs/RobotState/robotPose"

# =============================
# Camera / Geometry Parameters
# =============================
CAMERA_HEIGHT_M = 0.527
CAMERA_PITCH_DEG = 25
TARGET_HEIGHT_M = 0.075

Camera_Vertical_Displacement = 0.641
Camera_Horizontal_Displacement = 0.246

CAMERA1_HORIZONTAL_M = +Camera_Horizontal_Displacement
CAMERA1_YAW_OFFSET_DEG = +35.0
YAW_SIGN = -1.0

PILE_BALL_PRIORITY_0_TO_10 = 8.0

# =============================
# Pile Center Parameters
# =============================
PILE_CENTER_MODE = "density_vb"
PILE_DENSITY_RADIUS_M = 0.40
PILE_DENSITY_SPREAD_LIMIT_M = 1
PILE_CLUSTER_LINK_M = 0.30

BEST_POSE2D_TABLE = "SmartDashboard"
BEST_POSE2D_KEY = "BestPilePose2d"

PRINT_EVERY_N_LOOPS = 5
LOOP_SLEEP_SEC = 0.05


class SimplePose2d:
    def __init__(self, x: float, y: float, heading_rad: float):
        self.x = x
        self.y = y
        self.heading_rad = heading_rad


def create_best_pose2d_publisher(
    server: str,
    table: str = BEST_POSE2D_TABLE,
    key: str = BEST_POSE2D_KEY,
    client_name: str = "best-pile-publisher-onecam",
):
    inst = ntcore.NetworkTableInstance.create()
    inst.startClient4(client_name)
    inst.setServer(server)

    table_obj = inst.getTable(table)
    pub = table_obj.getDoubleArrayTopic(key).publish()
    return inst, pub


def create_robot_pose_subscriber(
    server: str,
    topic_name: str = ROBOT_POSE_TOPIC,
    client_name: str = "ak-robotpose-reader",
):
    inst = ntcore.NetworkTableInstance.create()
    inst.startClient4(client_name)
    inst.setServer(server)

    sub = inst.getStructTopic(topic_name, Pose2d).subscribe(Pose2d())
    return inst, sub


def get_robot_pose2d(robot_pose_sub):
    pose = robot_pose_sub.get()
    if pose is None:
        return None

    try:
        x = float(pose.x)
        y = float(pose.y)
    except Exception:
        x = float(pose.X())
        y = float(pose.Y())

    try:
        heading_rad = float(pose.rotation().radians())
    except Exception:
        heading_rad = float(pose.rotation().radians())

    return SimplePose2d(x=x, y=y, heading_rad=heading_rad)


def publish_best_pose2d_to_NT(best_pose_pub, best_pile, robot_pose2d):
    if best_pile is None:
        best_pose_pub.set([])
        return

    x = float(best_pile.center_xy[0])
    y = float(best_pile.center_xy[1])

    dx = x - float(robot_pose2d.x)
    dy = y - float(robot_pose2d.y)

    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        heading_deg = math.degrees(float(robot_pose2d.heading_rad))
    else:
        heading_deg = math.degrees(math.atan2(dy, dx))

    best_pose_pub.set([x, y, heading_deg])


def process_camera1(pv: PhotonMultiCamClient, robot_pose2d):
    camera_pose2d = None
    if robot_pose2d is not None:
        camera_pose2d = cameraPose2d_calculate(
            robot_pose2d,
            Camera_Vertical_Displacement,
            CAMERA1_HORIZONTAL_M,
        )

    yaw_list = getattr(pv, "Camera1_Yaw")
    pitch_list = getattr(pv, "Camera1_Pitch")
    area_list = getattr(pv, "Camera1_Area")

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
        dist = dist_list[target_number]

        ball_xy = None
        if camera_pose2d is not None and dist is not None and yaw is not None:
            ball_xy = ball_xy_from_camera(
                camera_pose2d=camera_pose2d,
                yaw_deg=yaw,
                distance_m=dist,
                camera_yaw_offset_deg=CAMERA1_YAW_OFFSET_DEG,
                yaw_sign=YAW_SIGN,
            )

        results.append({
            "target_number": target_number,
            "distance": dist,
            "ball_xy": ball_xy,
        })

    return camera_pose2d, results


def main():
    pv = PhotonMultiCamClient(
        server=NT_SERVER,
        cameras=CAMERAS,
        sort_targets_by_area_desc=False,
    )
    pv.start()

    _, best_pose_pub = create_best_pose2d_publisher(
        server=NT_SERVER,
        table=BEST_POSE2D_TABLE,
        key=BEST_POSE2D_KEY,
    )

    _, robot_pose_sub = create_robot_pose_subscriber(
        server=NT_SERVER,
        topic_name=ROBOT_POSE_TOPIC,
    )

    loop_count = 0

    while True:
        loop_count += 1

        robot_pose2d = get_robot_pose2d(robot_pose_sub)
        camera1_pose2d, cam1_results = process_camera1(pv, robot_pose2d)

        cam1_state = pv.get_state("Camera1")
        unique_ball_xys = [r["ball_xy"] for r in cam1_results if r["ball_xy"] is not None]

        pile_count, pile_plans, all_center_xys = plan_ballpile_centers(
            ball_xys=unique_ball_xys,
            cluster_link_m=PILE_CLUSTER_LINK_M,
            center_mode=PILE_CENTER_MODE,
            density_radius_m=PILE_DENSITY_RADIUS_M,
            density_spread_limit_m=PILE_DENSITY_SPREAD_LIMIT_M,
        )

        pile_candidates = build_candidates(
            center_xys=[p.center_xy for p in pile_plans],
            counts=[p.count for p in pile_plans],
            pile_ids=[p.pile_id for p in pile_plans],
        )

        best_pile, _ = select_best_pile(
            robot_pose=robot_pose2d,
            pile_candidates=pile_candidates,
            ball_priority_0to10=PILE_BALL_PRIORITY_0_TO_10,
        )

        if robot_pose2d is not None:
            publish_best_pose2d_to_NT(
                best_pose_pub=best_pose_pub,
                best_pile=best_pile,
                robot_pose2d=robot_pose2d,
            )
        else:
            best_pose_pub.set([])

        if loop_count % PRINT_EVERY_N_LOOPS == 0:
            if cam1_state.last_error:
                print("Camera1 decode error:", cam1_state.last_error)

            if robot_pose2d is None:
                print("\n=== robotPose unavailable ===")
            else:
                print(
                    f"\n=== robotPose === x={robot_pose2d.x:.4f}, "
                    f"y={robot_pose2d.y:.4f}, "
                    f"heading_deg={math.degrees(robot_pose2d.heading_rad):.2f}"
                )

            print("\n=== Camera1 Dist Only ===")
            for r in cam1_results:
                print(
                    f"t{r['target_number']:02d}  "
                    f"dist={r['distance']}  "
                    f"ball_xy={r['ball_xy']}"
                )

            print("\n=== ONE CAM RESULT ===")
            print("unique_ball_count =", len(unique_ball_xys))
            print("unique_ball_xys =", unique_ball_xys)

            print("\n=== BEST PILE ===")
            if best_pile is None:
                print("best_pile = None")
            else:
                print("best_pile_id =", best_pile.pile_id)
                print("best_center_xy =", best_pile.center_xy)
                print("best_count =", best_pile.count)

        time.sleep(LOOP_SLEEP_SEC)


if __name__ == "__main__":
    main()