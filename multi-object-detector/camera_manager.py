"""
camera_manager.py
----------------------------------------------------------------------
Har bir kamera (RTSP oqim / video fayl / vebkamera) uchun ALOHIDA background
thread ochadi:
  - kadrlarni uzluksiz o'qiydi,
  - GLOBAL ravishda YOQILGAN modellarga (fire / fall / danger_zone) beradi
    (qarang: models_manager.is_enabled),
  - natijani (annotatsiyalangan kadr) va hodisalar logini xotirada saqlaydi.

Kameralar soni DINAMIK: add_camera() / remove_camera() orqali istalgan
vaqtda qo'shiladi yoki o'chiriladi - serverni qayta ishga tushirish shart
emas.
"""

from __future__ import annotations

import os
import threading
import time
import uuid

# RTSP oqimlarini serverda barqaror va TCP protokoli orqali ochish uchun:
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

import config
import models_manager
import fall_detector
import danger_zone_detector


@dataclass
class CameraEvent:
    ts: float
    model: str            # "fire" | "fall" | "danger_zone"
    label: str             # foydalanuvchiga ko'rsatiladigan tayyor matn
    extra: dict = field(default_factory=dict)


class CameraWorker:
    def __init__(self, cam_id: str, name: str, source: str):
        self.id = cam_id
        self.name = name
        self.source = source

        self._lock = threading.Lock()
        self._raw_frame: Optional[np.ndarray] = None
        self._annotated_frame: Optional[np.ndarray] = None
        self._connected = False
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_error: Optional[str] = None

        self.events: deque = deque(maxlen=300)
        self.frame_idx = 0
        self.fps_estimate = 25.0
        self._last_frame_time: Optional[float] = None

        # --- har kamera uchun ALOHIDA (holat saqlaydigan) "fall" instanslari ---
        # (bo'lishilsa track ID'lar kameralar orasida aralashib ketadi)
        self._fall_model = None
        self._fall_pose = None
        self._fall_track_states: dict[int, fall_detector.TrackState] = {}

        # --- xavfli hudud (poligon har kamerada boshqacha bo'lishi mumkin) ---
        self._zone_polygon: Optional[np.ndarray] = None
        self._zone_state: Optional[danger_zone_detector.DangerZoneState] = None
        self._zone_lock = threading.Lock()

    # -------------------------------------------------------------- state --
    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def last_error(self) -> Optional[str]:
        with self._lock:
            return self._last_error

    def get_raw_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._raw_frame is None else self._raw_frame.copy()

    def get_annotated_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._annotated_frame is None else self._annotated_frame.copy()

    def get_recent_events(self, limit: int = 50) -> list[dict]:
        with self._lock:
            items = list(self.events)[-limit:]
        return [{"ts": e.ts, "model": e.model, "label": e.label, **e.extra} for e in reversed(items)]

    # ------------------------------------------------------- xavfli hudud --
    def set_zone(self, polygon_px: list[list[int]]) -> None:
        """Poligon PIKSEL koordinatalarida beriladi: [[x, y], [x, y], ...]"""
        arr = np.array(polygon_px, dtype=np.int32)
        new_state = models_manager.new_danger_zone_state(arr)
        with self._zone_lock:
            self._zone_polygon = arr
            self._zone_state = new_state

    def clear_zone(self) -> None:
        with self._zone_lock:
            self._zone_polygon = None
            self._zone_state = None

    def get_zone(self) -> Optional[list[list[int]]]:
        with self._zone_lock:
            if self._zone_polygon is None:
                return None
            return self._zone_polygon.tolist()

    # ------------------------------------------------------------ control --
    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"cam-{self.id}")
        self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)
        if self._fall_pose is not None:
            try:
                self._fall_pose.close()
            except Exception:
                pass

    # -------------------------------------------------------------- loop  --
    def _open_capture(self) -> cv2.VideoCapture:
        src = self.source
        # Vebkamera raqami sifatida berilgan bo'lsa ("0", "1", ...)
        try:
            return cv2.VideoCapture(int(src))
        except (TypeError, ValueError):
            pass
        # Aks holda RTSP URL yoki video fayl yo'li
        return cv2.VideoCapture(src, cv2.CAP_FFMPEG)

    def _run(self) -> None:
        cap = None
        while True:
            with self._lock:
                if not self._running:
                    break

            if cap is None or not cap.isOpened():
                cap = self._open_capture()
                if not cap.isOpened():
                    with self._lock:
                        self._connected = False
                        self._last_error = "Manbaga ulanib bo'lmadi"
                    time.sleep(config.CAMERA_RECONNECT_DELAY_SEC)
                    continue

            ok, frame = cap.read()
            if not ok or frame is None:
                with self._lock:
                    self._connected = False
                    self._last_error = "Kadr o'qilmadi (oqim uzilgan bo'lishi mumkin)"
                cap.release()
                cap = None
                time.sleep(config.CAMERA_RECONNECT_DELAY_SEC)
                continue

            self._update_fps_estimate()

            with self._lock:
                self._connected = True
                self._last_error = None
                self._raw_frame = frame
            self.frame_idx += 1

            try:
                annotated = self._analyze(frame.copy())
            except Exception as e:
                annotated = frame
                with self._lock:
                    self._last_error = f"Tahlil xatoligi: {e}"

            with self._lock:
                self._annotated_frame = annotated

        if cap is not None:
            cap.release()
        with self._lock:
            self._connected = False

    def _update_fps_estimate(self) -> None:
        now = time.time()
        if self._last_frame_time is not None:
            dt = now - self._last_frame_time
            if 0 < dt < 2.0:
                inst_fps = 1.0 / dt
                self.fps_estimate = 0.9 * self.fps_estimate + 0.1 * inst_fps
        self._last_frame_time = now

    # ---------------------------------------------------------- analysis  --
    def _log_event(self, model: str, label: str, **extra) -> None:
        with self._lock:
            self.events.append(CameraEvent(ts=time.time(), model=model, label=label, extra=extra))

    def _analyze(self, frame: np.ndarray) -> np.ndarray:
        out = frame

        if models_manager.is_enabled("fire"):
            out, fire_events, _has_fire, _has_smoke = models_manager.analyze_fire(out)
            for ev in fire_events:
                self._log_event(
                    "fire",
                    f"{ev['type']} aniqlandi ({ev['confidence'] * 100:.0f}%)",
                    box=ev["box"], confidence=ev["confidence"], type=ev["type"],
                )

        if models_manager.is_enabled("fall"):
            if self._fall_model is None:
                self._fall_model = fall_detector.create_tracking_model(config.PERSON_MODEL_PATH)
                self._fall_pose = fall_detector.create_pose_instance()

            out, fall_events, _any_fall = fall_detector.process_fall_frame(
                out,
                self.frame_idx,
                max(1, int(round(self.fps_estimate))),
                self._fall_model,
                self._fall_pose,
                self._fall_track_states,
            )
            for ev in fall_events:
                self._log_event("fall", f"Yiqilish aniqlandi (ID {ev['track_id']})", track_id=ev["track_id"])

        if models_manager.is_enabled("danger_zone"):
            with self._zone_lock:
                zone_state = self._zone_state
            if zone_state is not None:
                out, people_in_zone, breach = models_manager.analyze_danger_zone(zone_state, out)
                if breach:
                    self._log_event(
                        "danger_zone",
                        f"Xavfli hududda {people_in_zone} kishi aniqlandi",
                        people_in_zone=people_in_zone,
                    )

        return out


# ---------------------------------------------------------------------------
# GLOBAL KAMERALAR REESTRI (dinamik ro'yxat)
# ---------------------------------------------------------------------------
_cameras: dict[str, CameraWorker] = {}
_registry_lock = threading.Lock()


def add_camera(name: str, source: str) -> CameraWorker:
    cam_id = uuid.uuid4().hex[:8]
    worker = CameraWorker(cam_id, name, source)
    with _registry_lock:
        _cameras[cam_id] = worker
    worker.start()
    return worker


def remove_camera(cam_id: str) -> bool:
    with _registry_lock:
        worker = _cameras.pop(cam_id, None)
    if worker is None:
        return False
    worker.stop()
    return True


def get_camera(cam_id: str) -> Optional[CameraWorker]:
    with _registry_lock:
        return _cameras.get(cam_id)


def list_cameras() -> list[CameraWorker]:
    with _registry_lock:
        return list(_cameras.values())


def stop_all() -> None:
    with _registry_lock:
        workers = list(_cameras.values())
    for w in workers:
        w.stop()