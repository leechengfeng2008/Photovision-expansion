from __future__ import annotations
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
CAMERAS          = ["Camera1"]
ROBOT_POSE_TOPIC = "/AdvantageKit/RealOutputs/RobotState/robotPose"

# ─────────────────────────────────────────────────────────────────────────────
# Camera / Geometry
# ─────────────────────────────────────────────────────────────────────────────
CAMERA_HEIGHT_M  = 0.527
CAMERA_PITCH_DEG = 25.0
TARGET_HEIGHT_M  = 0.075

Camera_Vertical_Displacement = -0.641

CAMERA1_HORIZONTAL_M   = -0.246
CAMERA1_YAW_OFFSET_DEG = -30.0
CAMERA1_YAW_SIGN       = +1.0

DISTANCE_MIN_M = 0.15
DISTANCE_MAX_M = 8.00

# ─────────────────────────────────────────────────────────────────────────────
# Loop
# ─────────────────────────────────────────────────────────────────────────────
PRINT_EVERY_N_LOOPS = 5
LOOP_SLEEP_SEC      = 0.05
STALE_TIMEOUT_SEC   = 0.5


def xy_to_pose2d(ball_xy: tuple[float, float] | None) -> Pose2d | None:
    if ball_xy is None:
        return None
    return Pose2d(
        x=float(ball_xy[0]),
        y=float(ball_xy[1]),
        heading_rad=0.0,
    )


def process_camera1(
    pv: PhotonMultiCamClient,
    robot_pose2d: Pose2d | None,
):
    camera_pose2d = None
    if robot_pose2d is not None:
        camera_pose2d = cameraPose2d_calculate(
            robot_pose=robot_pose2d,
            Camera_Vertical_Displacement=Camera_Vertical_Displacement,
            Camera_Horizontal_Displacement=CAMERA1_HORIZONTAL_M,
        )

    yaw_list   = getattr(pv, "Camera1_Yaw")
    pitch_list = getattr(pv, "Camera1_Pitch")
    area_list  = getattr(pv, "Camera1_Area")

    dist_list = distance_calculate(
        pitch_list,
        camera_height_m=CAMERA_HEIGHT_M,
        camera_pitch_deg=CAMERA_PITCH_DEG,
        target_height_m=TARGET_HEIGHT_M,
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
        ball_pose2d = None

        if camera_pose2d is not None and dist_ok and yaw is not None:
            ball_xy = ball_xy_from_camera(
                camera_pose2d=camera_pose2d,
                yaw_deg=yaw,
                distance_m=dist,
                camera_yaw_offset_deg=CAMERA1_YAW_OFFSET_DEG,
                yaw_sign=CAMERA1_YAW_SIGN,
            )
            ball_pose2d = xy_to_pose2d(ball_xy)

        results.append({
            "target_number": idx,
            "yaw": yaw,
            "pitch": pitch,
            "area": area,
            "distance": dist,
            "distance_ok": dist_ok,
            "ball_xy": ball_xy,
            "ball_pose2d": ball_pose2d,
        })

    return camera_pose2d, results


def main():
    pv = PhotonMultiCamClient(
        server=NT_SERVER,
        cameras=CAMERAS,
        sort_targets_by_area_desc=False,
    )
    pv.start()

    pose_reader = Pose2dReader(
        server=NT_SERVER,
        topic_path=ROBOT_POSE_TOPIC,
    )

    loop_count = 0

    while True:
        loop_count += 1

        robot_pose2d = pose_reader.get_pose2d()
        cam1_state = pv.get_state("Camera1")

        elapsed = time.monotonic() - cam1_state.last_update_monotonic
        if elapsed > STALE_TIMEOUT_SEC:
            cam1_pose2d = None
            cam1_results = []
        else:
            cam1_pose2d, cam1_results = process_camera1(pv, robot_pose2d)

        if loop_count % PRINT_EVERY_N_LOOPS != 0:
            time.sleep(LOOP_SLEEP_SEC)
            continue

        if cam1_state.last_error:
            print("Camera1 decode error:", cam1_state.last_error)

        if elapsed > STALE_TIMEOUT_SEC:
            print(f"\n⚠ Camera1 stale ({elapsed:.2f}s since last update)")
            time.sleep(LOOP_SLEEP_SEC)
            continue

        if robot_pose2d is None:
            print("\n=== robotPose unavailable ===")
        else:
            print(
                f"\n=== robotPose ===  "
                f"x={robot_pose2d.x:.4f}  "
                f"y={robot_pose2d.y:.4f}  "
                f"heading={math.degrees(robot_pose2d.heading_rad):.2f}°"
            )

        if cam1_pose2d is None:
            print("\n=== Camera1 pose unavailable ===")
        else:
            print(
                f"\n=== Camera1 pose ===  "
                f"x={cam1_pose2d.x:.4f}  "
                f"y={cam1_pose2d.y:.4f}  "
                f"heading={math.degrees(cam1_pose2d.heading_rad):.2f}°"
            )

        print(f"\n=== Camera1 targets ({len(cam1_results)}) ===")
        print("  t       yaw     pitch     area     dist   ok     ball_xy                 ball_pose2d")
        print("  --------------------------------------------------------------------------------------")

        for r in cam1_results:
            yaw_s   = f"{r['yaw']:.3f}" if r["yaw"] is not None else "None"
            pitch_s = f"{r['pitch']:.3f}" if r["pitch"] is not None else "None"
            area_s  = f"{r['area']:.3f}" if r["area"] is not None else "None"
            dist_s  = f"{r['distance']:.3f}" if r["distance"] is not None else "None"

            ball_xy = r["ball_xy"]
            if ball_xy is None:
                ball_xy_s = "None"
            else:
                ball_xy_s = f"({ball_xy[0]:.4f}, {ball_xy[1]:.4f})"

            bp = r["ball_pose2d"]
            if bp is None:
                ball_pose_s = "None"
            else:
                ball_pose_s = f"Pose2d(x={bp.x:.4f}, y={bp.y:.4f}, heading=0.00°)"

            print(
                f"  t{r['target_number']:02d}  "
                f"{yaw_s:>8}  "
                f"{pitch_s:>8}  "
                f"{area_s:>7}  "
                f"{dist_s:>7}  "
                f"{'OK' if r['distance_ok'] else 'FILT':>4}   "
                f"{ball_xy_s:<22}  "
                f"{ball_pose_s}"
            )

        time.sleep(LOOP_SLEEP_SEC)


if __name__ == "__main__":
    main()