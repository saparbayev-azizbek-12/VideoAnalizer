from __future__ import annotations
import os
import cv2
import time
import json
import uuid
import threading
import numpy as np
import supervision as sv
from typing import Optional
from collections import deque
from dataclasses import dataclass, field

from multi_object_detector.core import config
from multi_object_detector.core.system_logger import sys_logger
from multi_object_detector.core.stream_logger import stream_logger
from multi_object_detector.detectors import manager as models_manager, frame_filter
from multi_object_detector.detectors import fall_detector, danger_zone_detector, fire_detector

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"


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
        self._capture_thread: Optional[threading.Thread] = None
        self._analysis_thread: Optional[threading.Thread] = None
        self._new_frame_event = threading.Event()
        self._last_error: Optional[str] = None
        self._stream_filter = frame_filter.StreamCorruptionFilter(camera_id=cam_id)
        self.events: deque = deque(maxlen=300)
        self.frame_idx = 0
        self.captured_frame_count = 0
        self.fps_estimate = 25.0
        self._last_frame_time: Optional[float] = None
        self._fall_tracker = sv.ByteTrack()
        self._fall_pose: Optional[fall_detector.RTMPoseEstimator] = None
        self._fall_track_states: dict[int, fall_detector.TrackState] = {}
        self._fire_validator = fire_detector.TemporalValidator(window=6, min_hits=3)
        self._zone_polygon: Optional[np.ndarray] = None
        self._zone_polygon_norm: Optional[list[list[float]]] = None
        self._zone_resolution: Optional[tuple[int, int]] = None
        self._zone_state: Optional[danger_zone_detector.DangerZoneState] = None
        self._zone_lock = threading.Lock()
        self._last_dataset_save: dict[str, float] = {}
        src_str = str(self.source).strip()
        self._is_file_source = (
            os.path.exists(src_str)
            or (not src_str.isdigit() and not src_str.lower().startswith(("rtsp://", "rtsps://", "http://", "https://")))
        )

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
            if self._annotated_frame is not None:
                return self._annotated_frame.copy()
            if self._raw_frame is not None:
                frame = self._raw_frame.copy()
                with self._zone_lock:
                    poly = self._zone_polygon
                if poly is not None and len(poly) >= 3:
                    cv2.polylines(frame, [poly], isClosed=True, color=(0, 0, 220), thickness=2, lineType=cv2.LINE_AA)
                return frame
            return None

    def get_recent_events(self, limit: int = 50) -> list[dict]:
        with self._lock:
            items = list(self.events)[-limit:]
        return [{"ts": e.ts, "model": e.model, "label": e.label, **e.extra} for e in reversed(items)]

    def set_zone(self, polygon: list[list[float | int]]) -> None:
        if not polygon or len(polygon) < 3:
            return
        with self._lock:
            cur_frame = self._raw_frame
        h, w = (cur_frame.shape[:2]) if cur_frame is not None else (1080, 1920)

        is_norm = all(0.0 <= float(p[0]) <= 1.0 and 0.0 <= float(p[1]) <= 1.0 for p in polygon)
        if is_norm:
            norm_poly = [[float(p[0]), float(p[1])] for p in polygon]
            pixel_poly = np.array([[int(p[0] * w), int(p[1] * h)] for p in norm_poly], dtype=np.int32)
        else:
            pixel_poly = np.array(polygon, dtype=np.int32)
            norm_poly = [[float(p[0]) / max(1, w), float(p[1]) / max(1, h)] for p in pixel_poly]

        new_state = models_manager.new_danger_zone_state(pixel_poly)
        with self._zone_lock:
            self._zone_polygon = pixel_poly
            self._zone_polygon_norm = norm_poly
            self._zone_resolution = (w, h)
            self._zone_state = new_state

    def clear_zone(self) -> None:
        with self._zone_lock:
            self._zone_polygon = None
            self._zone_polygon_norm = None
            self._zone_resolution = None
            self._zone_state = None

    def get_zone(self) -> Optional[list[list[float]]]:
        with self._zone_lock:
            if self._zone_polygon_norm is not None:
                return self._zone_polygon_norm
            if self._zone_polygon is not None:
                return self._zone_polygon.tolist()
            return None

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._new_frame_event.clear()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True, name=f"cap-{self.id}")
        self._analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True, name=f"ana-{self.id}")
        self._capture_thread.start()
        self._analysis_thread.start()

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._new_frame_event.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2)
        if self._analysis_thread is not None:
            self._analysis_thread.join(timeout=2)
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
        if self._is_file_source:
            return cv2.VideoCapture(src)
        return cv2.VideoCapture(src, cv2.CAP_FFMPEG)

    def _capture_loop(self) -> None:
        cap = None
        consecutive_read_fails = 0
        while self._running:
            if cap is None or not cap.isOpened():
                consecutive_read_fails = 0
                cap = self._open_capture()
                if not cap.isOpened():
                    diag_err = stream_logger.log_connect_fail(self.id, self.name, self.source)
                    with self._lock:
                        self._connected = False
                        self._last_error = diag_err
                    time.sleep(config.CAMERA_RECONNECT_DELAY_SEC)
                    continue
                else:
                    stream_logger.log_reconnect_success(self.id, self.name, self.source)

            ok, frame = cap.read()
            if not ok or frame is None:
                if self._is_file_source and cap is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.02)
                    continue

                drop_err = stream_logger.log_stream_drop(self.id, self.name, self.source)
                with self._lock:
                    self._connected = False
                    self._last_error = drop_err
                if cap is not None:
                    cap.release()
                cap = None
                time.sleep(config.CAMERA_RECONNECT_DELAY_SEC)
                continue

            self._update_fps_estimate()
            self.captured_frame_count += 1

            with self._lock:
                self._connected = True
                self._last_error = None
                self._raw_frame = frame
                if self.captured_frame_count % max(1, config.ANALYSIS_EVERY_N_FRAMES) == 0:
                    self._new_frame_event.set()

            if self._is_file_source:
                time.sleep(1.0 / max(1.0, self.fps_estimate))

        if cap is not None:
            cap.release()
        with self._lock:
            self._connected = False

    def _analysis_loop(self) -> None:
        while self._running:
            signaled = self._new_frame_event.wait(timeout=0.1)
            self._new_frame_event.clear()

            if not self._running:
                break

            with self._lock:
                frame = self._raw_frame.copy() if self._raw_frame is not None else None
                connected = self._connected

            if frame is None or not connected:
                time.sleep(0.05)
                continue

            self.frame_idx += 1
            is_valid, corrupt_reason = self._stream_filter.check_frame(frame)
            if not is_valid:
                stream_logger.log_corruption(self.id, self.name, self.source, corrupt_reason or "Kadr buzilgan")
                with self._lock:
                    self._annotated_frame = frame
                continue

            try:
                annotated = self._analyze(frame)
            except Exception as e:
                import traceback
                traceback.print_exc()
                stream_logger.log_error(self.id, self.name, self.source, f"Tahlil xatoligi: {e}")
                annotated = frame
                with self._lock:
                    self._last_error = f"Tahlil xatoligi: {e}"

            with self._lock:
                self._annotated_frame = annotated

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
        except Exception:
            return None

    def _analyze(self, frame: np.ndarray) -> np.ndarray:
        if not frame_filter.is_frame_valid(frame):
            return frame
        out = frame.copy()

        if models_manager.is_enabled("fire"):
            try:
                out, fire_events, has_fire, has_smoke = models_manager.analyze_fire(
                    out, validator=self._fire_validator
                )
                for ev in fire_events:
                    snap = self._save_dataset_snapshot("fire", out, ev)
                    extra = dict(box=ev["box"], confidence=ev["confidence"], type=ev["type"])
                    if snap:
                        extra["image"] = snap
                    self._log_event("fire", f"{ev['type']} aniqlandi ({ev['confidence'] * 100:.0f}%)", **extra)
            except Exception as e:
                sys_logger.error("CameraWorker", f"[{self.name}] FIRE xatoligi: {e}", exc=e)

        if models_manager.is_enabled("fall"):
            try:
                if self._fall_pose is None:
                    self._fall_pose = fall_detector.create_pose_instance()

                out, fall_events, any_fall = models_manager.analyze_fall(
                    frame=out,
                    frame_idx=self.frame_idx,
                    fps=max(1, int(round(self.fps_estimate))),
                    pose=self._fall_pose,
                    track_states=self._fall_track_states,
                    tracker=self._fall_tracker,
                )
                for ev in fall_events:
                    snap = self._save_dataset_snapshot("fall", out, ev)
                    extra = dict(track_id=ev["track_id"])
                    if snap:
                        extra["image"] = snap
                    self._log_event("fall", f"Yiqilish aniqlandi (ID {ev['track_id']})", **extra)
            except Exception as e:
                sys_logger.error("CameraWorker", f"[{self.name}] FALL xatoligi: {e}", exc=e)

        if models_manager.is_enabled("danger_zone"):
            try:
                with self._zone_lock:
                    zone_state = self._zone_state
                    zone_norm = self._zone_polygon_norm
                    zone_res = self._zone_resolution

                if zone_norm is not None:
                    h, w = frame.shape[:2]
                    if zone_res != (w, h) or zone_state is None:
                        pixel_poly = np.array([[int(p[0] * w), int(p[1] * h)] for p in zone_norm], dtype=np.int32)
                        zone_state = models_manager.new_danger_zone_state(pixel_poly)
                        with self._zone_lock:
                            self._zone_polygon = pixel_poly
                            self._zone_state = zone_state
                            self._zone_resolution = (w, h)

                if zone_state is not None:
                    out, people_in_zone, breach = models_manager.analyze_danger_zone(zone_state, out)
                    if breach:
                        snap = self._save_dataset_snapshot("danger_zone", out, {"people_in_zone": people_in_zone})
                        extra = dict(people_in_zone=people_in_zone)
                        if snap:
                            extra["image"] = snap
                        self._log_event("danger_zone", f"Xavfli hududda {people_in_zone} kishi aniqlandi", **extra)
            except Exception as e:
                sys_logger.error("CameraWorker", f"[{self.name}] DANGER_ZONE xatoligi: {e}", exc=e)

        if models_manager.is_enabled("ppe"):
            try:
                out, ppe_violations, has_ppe_violation = models_manager.analyze_ppe(out)
                if has_ppe_violation:
                    snap = self._save_dataset_snapshot("ppe", out, {"violation_count": len(ppe_violations)})
                    for i, viol in enumerate(ppe_violations):
                        extra = dict(
                            box=viol["box"],
                            confidence=viol["confidence"],
                            missing=viol["missing"],
                            type=viol["type"],
                        )
                        if snap and i == 0:
                            extra["image"] = snap
                        missing_str = ", ".join(
                            "Kask" if m == "Safety Helmet" else "Xavfsizlik kiyimi"
                            for m in viol["missing"]
                        )
                        self._log_event("ppe", f"PPE yo'q: {missing_str}", **extra)
            except Exception as e:
                sys_logger.error("CameraWorker", f"[{self.name}] PPE xatoligi: {e}", exc=e)

        with self._lock:
            self._annotated_frame = out

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
