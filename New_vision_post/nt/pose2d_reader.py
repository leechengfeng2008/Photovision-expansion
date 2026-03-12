# pose2d_reader.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import math
import ntcore
from wpimath.geometry import Pose2d as WpiPose2d


@dataclass
class Pose2d:
    x: float
    y: float
    heading_rad: float


class Pose2dReader:
    """
    Read AdvantageKit Pose2d struct topic from NT4.
    Example topic:
        /Advantagekit/RealOutputs/RobotState/robotPose
    """

    def __init__(
        self,
        server: str,
        topic_path: str,
        client_name: str = "orangepi-pose2d-reader",
    ):
        self.server = server
        self.topic_path = topic_path

        self._inst = ntcore.NetworkTableInstance.create()
        self._inst.startClient4(client_name)
        self._inst.setServer(server)

        self._sub = self._inst.getStructTopic(topic_path, WpiPose2d).subscribe(WpiPose2d())
        self._last_good: Optional[Pose2d] = None

    def get_pose2d(self) -> Optional[Pose2d]:
        raw = self._sub.get()
        if raw is None:
            return self._last_good

        try:
            x = float(raw.x)
            y = float(raw.y)
        except Exception:
            x = float(raw.X())
            y = float(raw.Y())

        heading_rad = float(raw.rotation().radians())

        p = Pose2d(x=x, y=y, heading_rad=heading_rad)
        self._last_good = p
        return p