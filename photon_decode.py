# photon_decode.py
from __future__ import annotations
import struct
from typing import Any, Dict, List, Tuple


class Buf:
    def __init__(self, b: bytes):
        self.b = b
        self.i = 0

    def remaining(self) -> int:
        return len(self.b) - self.i

    def _need(self, n: int):
        if self.remaining() < n:
            raise ValueError(f"buffer underrun: need {n}, remaining {self.remaining()}")

    def u8(self) -> int:
        self._need(1)
        v = self.b[self.i]
        self.i += 1
        return v

    def i32(self) -> int:
        self._need(4)
        v = struct.unpack_from("<i", self.b, self.i)[0]
        self.i += 4
        return v

    def i64(self) -> int:
        self._need(8)
        v = struct.unpack_from("<q", self.b, self.i)[0]
        self.i += 8
        return v

    def f32(self) -> float:
        self._need(4)
        v = struct.unpack_from("<f", self.b, self.i)[0]
        self.i += 4
        return v

    def f64(self) -> float:
        self._need(8)
        v = struct.unpack_from("<d", self.b, self.i)[0]
        self.i += 8
        return v


def _skip_transform3d(buf: Buf):
    # Transform3d: Translation3d(3 doubles) + Rotation3d(4 doubles) => 7 doubles => 56 bytes
    buf._need(56)
    buf.i += 56


def _read_corner(buf: Buf) -> Tuple[float, float]:
    # TargetCorner: float64 x, float64 y
    return (buf.f64(), buf.f64())


def _read_corner_list(buf: Buf) -> List[Tuple[float, float]]:
    # SmallVector length: uint8
    n = buf.u8()
    return [_read_corner(buf) for _ in range(n)]


def _read_metadata(buf: Buf) -> Dict[str, int]:
    # PhotonPipelineMetadata: 4x int64
    return {
        "sequenceID": buf.i64(),
        "captureTimestampMicros": buf.i64(),
        "publishTimestampMicros": buf.i64(),
        "timeSinceLastPong": buf.i64(),
    }


def _read_target(buf: Buf) -> Dict[str, Any]:
    # PhotonTrackedTarget (你目前要的 yaw/pitch/area/conf... 都在這裡)
    yaw = buf.f64()
    pitch = buf.f64()
    area = buf.f64()
    skew = buf.f64()
    fid = buf.i32()
    oid = buf.i32()
    conf = buf.f32()

    _skip_transform3d(buf)  # bestCameraToTarget
    _skip_transform3d(buf)  # altCameraToTarget

    amb = buf.f64()
    _min_rect = _read_corner_list(buf)
    _detected = _read_corner_list(buf)

    return {
        "yaw": yaw,
        "pitch": pitch,
        "area": area,
        "skew": skew,
        "fiducialId": fid,
        "objDetectId": oid,
        "objDetectConf": conf,
        "poseAmbiguity": amb,
    }


def decode_pipeline_result(raw: bytes):
    """
    Returns:
      md: dict
      targets: list[dict]
      multitag_present: int (0/1)
      leftover_bytes: int
    """
    buf = Buf(raw)

    md = _read_metadata(buf)

    n_targets = buf.u8()
    targets = [_read_target(buf) for _ in range(n_targets)]

    multitag_present = 0
    if buf.remaining() > 0:
        multitag_present = buf.u8()
        # multitag payload 不解析，先保留

    return md, targets, multitag_present, buf.remaining()