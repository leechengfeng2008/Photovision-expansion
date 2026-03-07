# pose2d_reader.py
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional, List
import ntcore
import math


@dataclass
class Pose2d:
    x: float
    y: float
    heading_rad: float 


class Pose2dReader:
    """
    from NT4 read a DoubleArray topic which contains [x,y,degree]
    """

    def __init__(
        self,
        server: str,
        table: str,
        key: str,
        heading_units: str = "deg", 
        client_name: str = "orangepi-pose2d-reader",
    ):
        self.server = server
        self.table = table
        self.key = key
        self.heading_units = heading_units

        self._inst = ntcore.NetworkTableInstance.create()
        self._inst.startClient4(client_name)
        self._inst.setServer(server)

        t = self._inst.getTable(table)
        # self._sub = t.getDoubleArrayTopic(key).subscribe([0.0, 0.0, 0.0]) fro no robot test
        self._sub = t.getDoubleArrayTopic(key).subscribe([])

        self._last_good: Optional[Pose2d] = None

    def get_pose2d(self) -> Optional[Pose2d]:
        arr: List[float] = self._sub.get()
        if not arr or len(arr) < 3:
            return self._last_good

        x, y, h = float(arr[0]), float(arr[1]), float(arr[2])

        if self.heading_units == "deg":

            h = math.radians(h)

        p = Pose2d(x=x, y=y, heading_rad=h)
        self._last_good = p
        return p