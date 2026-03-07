# main.py
from __future__ import annotations
import time

from photon_nt_multicam import PhotonMultiCamClient
from utils.distance_utils import distance_calculate
from utils.pose_utils import cameraPose2d_calculate
from pose2d_reader import Pose2dReader
from utils.ballpose_utils import ball_xy_from_camera


NT_SERVER = "10.25.6.2"
CAMERAS = ["Camera1", "Camera2"]
POSE2D_KEY = "Pose2d"             # key
HEADING_UNITS = "deg"
POSE2D_TABLE = "SmartDashboard"

CAMERA_HEIGHT_M = 0.55       # 相機高度（m）
CAMERA_PITCH_DEG = 30       # 相機安裝仰角（deg）
TARGET_HEIGHT_M = 0.075      # 目標點高度（m），例如球心高度 7.5cm

Camera_Vertical_Displacement = 0.641 #meter
Camera_Horizontal_Displacement = 0.246 #meter
CAMERA1_HORIZONTAL_M = +Camera_Horizontal_Displacement  # 1->Left
CAMERA2_HORIZONTAL_M = -Camera_Horizontal_Displacement  # 2->Right

CAMERA1_YAW_OFFSET_DEG = +35.0
CAMERA2_YAW_OFFSET_DEG = -35.0

YAW_SIGN = 1.0

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


def main():

    pose_reader = Pose2dReader(
        server=NT_SERVER,
        table=POSE2D_TABLE,       # "SmartDashboard"
        key=POSE2D_KEY,           # "Pose2d"
        heading_units=HEADING_UNITS,  # "deg"
    )
        
    pv = PhotonMultiCamClient(
        server=NT_SERVER,
        cameras=CAMERAS,
        sort_targets_by_area_desc=False,
    )
    pv.start()

    while True:
        robot_pose2d = pose_reader.get_pose2d()

        camera1_pose2d, cam1_results = process_one_camera("Camera1", pv, robot_pose2d)
        camera2_pose2d, cam2_results = process_one_camera("Camera2", pv, robot_pose2d)

        print("\n=== Camera1 ===")
        for r in cam1_results:
            print(
                f"t{r['target_number']:02d}  yaw={r['yaw']:+7.3f}  pitch={r['pitch']:+7.3f}  "
                f"area={r['area']:7.3f}  dist={r['distance']} ball_xy={r['ball_xy']}"
            )

        print("\n=== Camera2 ===")
        for r in cam2_results:
            print(
                f"t{r['target_number']:02d}  yaw={r['yaw']:+7.3f}  pitch={r['pitch']:+7.3f}  "
                f"area={r['area']:7.3f}  dist={r['distance']} ball_xy={r['ball_xy']}"
            )
            
        time.sleep(0.015)



if __name__ == "__main__":
    main()