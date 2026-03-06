# photon_nt_multicam.py
from __future__ import annotations
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import ntcore

from photon_decode import decode_pipeline_result


@dataclass
class CameraState:
    seq: Optional[int] = None
    targets: List[Dict[str, Any]] = field(default_factory=list)
    multitag_present: int = 0
    leftover: int = 0
    raw_len: int = 0
    last_error: Optional[str] = None
    last_update_monotonic: float = 0.0


class PhotonMultiCamClient:
    """
    - 連 NT server
    - 訂閱 /photonvision/<CameraName>/rawBytes
    - 每顆鏡頭各自保留「最新解包結果」
    - 支援用屬性讀取：Camera1_Yaw / Camera2_Pitch / Camera1_Conf ... 等
    """

    def __init__(
        self,
        server: str,
        cameras: List[str],
        client_name: str = "orangepi-multicam",
        table_name: str = "photonvision",
        poll_dt: float = 0.02,
        sort_targets_by_area_desc: bool = True,
    ):
        self.server = server
        self.cameras = cameras
        self.client_name = client_name
        self.table_name = table_name
        self.poll_dt = poll_dt
        self.sort_targets_by_area_desc = sort_targets_by_area_desc

        self._inst = ntcore.NetworkTableInstance.getDefault()
        self._subs: Dict[str, Any] = {}
        self._states: Dict[str, CameraState] = {c: CameraState() for c in cameras}
        self._locks: Dict[str, threading.Lock] = {c: threading.Lock() for c in cameras}
        self._threads: List[threading.Thread] = []
        self._stop_evt = threading.Event()

    def start(self):
        self._inst.startClient4(self.client_name)
        self._inst.setServer(self.server)

        root = self._inst.getTable(self.table_name)
        for cam in self.cameras:
            subtable = root.getSubTable(cam)
            sub = subtable.getRawTopic("rawBytes").subscribe("raw", b"")
            self._subs[cam] = sub

            th = threading.Thread(target=self._cam_loop, args=(cam,), daemon=True)
            th.start()
            self._threads.append(th)

    def stop(self):
        self._stop_evt.set()
        # daemon threads會自動結束；需要更嚴謹可自行 join

    def _cam_loop(self, cam: str):
        sub = self._subs[cam]
        last_seq = None

        while not self._stop_evt.is_set():
            raw = sub.get()
            if not raw:
                time.sleep(self.poll_dt)
                continue

            try:
                md, targets, mt_present, leftover = decode_pipeline_result(raw)
                seq = md.get("sequenceID", None)

                # 避免同一包反覆刷
                if seq is not None and seq == last_seq:
                    time.sleep(self.poll_dt)
                    continue
                last_seq = seq

                if self.sort_targets_by_area_desc and targets:
                    targets = sorted(targets, key=lambda t: float(t.get("area", 0.0)), reverse=True)

                with self._locks[cam]:
                    st = self._states[cam]
                    st.seq = seq
                    st.targets = targets
                    st.multitag_present = mt_present
                    st.leftover = leftover
                    st.raw_len = len(raw)
                    st.last_error = None
                    st.last_update_monotonic = time.monotonic()

            except Exception as e:
                with self._locks[cam]:
                    self._states[cam].last_error = f"{type(e).__name__}: {e}"

            time.sleep(self.poll_dt)

    # ---------
    # 對外 API
    # ---------
    def get_state(self, cam: str) -> CameraState:
        with self._locks[cam]:
            # 回傳複製（避免外部改到內部）
            st = self._states[cam]
            return CameraState(
                seq=st.seq,
                targets=list(st.targets),
                multitag_present=st.multitag_present,
                leftover=st.leftover,
                raw_len=st.raw_len,
                last_error=st.last_error,
                last_update_monotonic=st.last_update_monotonic,
            )

    def _get_field_list(self, cam: str, key: str) -> List[Any]:
        with self._locks[cam]:
            return [t.get(key) for t in self._states[cam].targets]

    def _get_best_field(self, cam: str, key: str):
        with self._locks[cam]:
            if not self._states[cam].targets:
                return None
            return self._states[cam].targets[0].get(key)

    def __getattr__(self, name: str):
        """
        支援這些格式（你要的 Camera1_Yaw 就是這類）：
          - <CameraName>_Yaw      => list[float]
          - <CameraName>_Pitch
          - <CameraName>_Area
          - <CameraName>_Conf
          - <CameraName>_ObjId
          - <CameraName>_Fid
          - <CameraName>_Skew
          - <CameraName>_Amb
          - <CameraName>_Seq
          - <CameraName>_BestYaw  => float|None（最大 area 的目標）
          - <CameraName>_Targets  => list[dict]（完整 targets）
        """
        if "_" not in name:
            raise AttributeError(name)

        cam, field = name.split("_", 1)
        if cam not in self._states:
            raise AttributeError(name)

        field_map = {
            "Yaw": "yaw",
            "Pitch": "pitch",
            "Area": "area",
            "Skew": "skew",
            "Conf": "objDetectConf",
            "ObjId": "objDetectId",
            "Fid": "fiducialId",
            "Amb": "poseAmbiguity",
        }

        if field == "Seq":
            with self._locks[cam]:
                return self._states[cam].seq

        if field == "Targets":
            with self._locks[cam]:
                return list(self._states[cam].targets)

        if field.startswith("Best"):
            base = field[len("Best") :]
            if base not in field_map:
                raise AttributeError(name)
            return self._get_best_field(cam, field_map[base])

        if field in field_map:
            return self._get_field_list(cam, field_map[field])

        raise AttributeError(name)