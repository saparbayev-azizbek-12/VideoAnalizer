from __future__ import annotations
import os
import cv2
import time
import json
import uuid
import threading
import numpy as np
from typing import Optional
from collections import deque
from dataclasses import dataclass, field

import config
from analysis import models_manager
from analysis.detectors import fall_detector, danger_zone_detector

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

@dataclass
class CameraEvent:
    ts: float
    model: str
    label: str
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
        self._fall_model = None
        self._fall_pose = None
        self._fall_track_states: dict[int, fall_detector.TrackState] = {}
        self._zone_polygon: Optional[np.ndarray] = None
        self._zone_state: Optional[danger_zone_detector.DangerZoneState] = None
        self._zone_lock = threading.Lock()
        self._last_dataset_save: dict[str, float] = {}

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

    def set_zone(self, polygon_px: list[list[int]]) -> None:
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

    def _open_capture(self) -> cv2.VideoCapture:
        src = self.source
        try:
            return cv2.VideoCapture(int(src))
        except (TypeError, ValueError):
            pass
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

    def _log_event(self, model: str, label: str, **extra) -> None:
        with self._lock:
            self.events.append(CameraEvent(ts=time.time(), model=model, label=label, extra=extra))

    def _save_dataset_snapshot(self, model: str, frame_to_save: np.ndarray, extra_info: dict) -> Optional[str]:
        now = time.time()
        last_time = self._last_dataset_save.get(model, 0.0)
        if now - last_time < config.SNAPSHOT_COOLDOWN_SEC:
            return None
        self._last_dataset_save[model] = now

        try:
            model_dir = os.path.join(config.DATASET_DIR, model)
            os.makedirs(model_dir, exist_ok=True)
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            ms = int((now % 1) * 1000)
            filename = f"{self.id}_{model}_{timestamp_str}_{ms:03d}.jpg"
            file_path = os.path.join(model_dir, filename)

            cv2.imwrite(file_path, frame_to_save)

            meta_path = os.path.join(model_dir, f"{self.id}_{model}_{timestamp_str}_{ms:03d}.json")
            meta_data = {
                "camera_id": self.id,
                "camera_name": self.name,
                "model": model,
                "timestamp": now,
                "timestamp_str": timestamp_str,
                "image_filename": filename,
                "extra": extra_info,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, ensure_ascii=False, indent=2)

            return filename
        except Exception as e:
            print(f"[Dataset] Snapshot saqlashda xatolik: {e}")
            return None

    def _analyze(self, frame: np.ndarray) -> np.ndarray:
        out = frame
        if models_manager.is_enabled("fire"):
            out, fire_events, _has_fire, _has_smoke = models_manager.analyze_fire(out)
            for ev in fire_events:
                snap = self._save_dataset_snapshot("fire", out, ev)
                extra = dict(box=ev["box"], confidence=ev["confidence"], type=ev["type"])
                if snap:
                    extra["image"] = snap
                self._log_event("fire", f"{ev['type']} aniqlandi ({ev['confidence'] * 100:.0f}%)", **extra)

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
                snap = self._save_dataset_snapshot("fall", out, ev)
                extra = dict(track_id=ev["track_id"])
                if snap:
                    extra["image"] = snap
                self._log_event("fall", f"Yiqilish aniqlandi (ID {ev['track_id']})", **extra)

        if models_manager.is_enabled("danger_zone"):
            with self._zone_lock:
                zone_state = self._zone_state
            if zone_state is not None:
                out, people_in_zone, breach = models_manager.analyze_danger_zone(zone_state, out)
                if breach:
                    snap = self._save_dataset_snapshot("danger_zone", out, {"people_in_zone": people_in_zone})
                    extra = dict(people_in_zone=people_in_zone)
                    if snap:
                        extra["image"] = snap
                    self._log_event("danger_zone", f"Xavfli hududda {people_in_zone} kishi aniqlandi", **extra)

        return out

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