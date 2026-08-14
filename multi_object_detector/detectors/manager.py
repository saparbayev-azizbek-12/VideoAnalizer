from __future__ import annotations
import threading
import numpy as np
from ultralytics import YOLO
from typing import Optional, Any

from multi_object_detector.core import config
from multi_object_detector.core.system_logger import sys_logger
from multi_object_detector.detectors import fire_detector, danger_zone_detector, ppe_detector, fall_detector

_state_lock = threading.Lock()
_enabled: dict[str, bool] = dict(config.MODEL_DEFAULT_ENABLED)


def set_enabled(model_id: str, enabled: bool) -> None:
    if model_id not in config.MODEL_IDS:
        raise ValueError(f"Noma'lum model: {model_id}")
    with _state_lock:
        _enabled[model_id] = bool(enabled)
    sys_logger.info("ModelsManager", f"Model '{model_id}' enabled status set to: {enabled}")


def is_enabled(model_id: str) -> bool:
    with _state_lock:
        return _enabled.get(model_id, False)


def get_all_enabled() -> dict:
    with _state_lock:
        return dict(_enabled)


_fire_model = None
_fire_model_lock = threading.Lock()
_fire_infer_lock = threading.Lock()

_zone_person_model = None
_zone_model_lock = threading.Lock()
_zone_infer_lock = threading.Lock()

_ppe_model = None
_ppe_model_lock = threading.Lock()
_ppe_infer_lock = threading.Lock()

_fall_model = None
_fall_model_lock = threading.Lock()
_fall_infer_lock = threading.Lock()


def get_fire_model():
    global _fire_model
    with _fire_model_lock:
        if _fire_model is None:
            _fire_model = fire_detector.get_model()
    return _fire_model


def get_zone_person_model():
    global _zone_person_model
    with _zone_model_lock:
        if _zone_person_model is None:
            _zone_person_model = danger_zone_detector.get_model()
    return _zone_person_model


def get_ppe_model() -> YOLO:
    global _ppe_model
    with _ppe_model_lock:
        if _ppe_model is None:
            _ppe_model = ppe_detector.get_model()
    return _ppe_model


def get_fall_model() -> Any:
    global _fall_model
    with _fall_model_lock:
        if _fall_model is None:
            _fall_model = fall_detector.get_model()
    return _fall_model


def preload_all() -> None:
    get_fire_model()
    get_zone_person_model()
    get_ppe_model()
    get_fall_model()


def analyze_fire(
    frame: np.ndarray,
    validator: Optional[fire_detector.TemporalValidator] = None,
) -> tuple[np.ndarray, list[dict], bool, bool]:
    model = get_fire_model()
    with _fire_infer_lock:
        return fire_detector.detect_fire_frame(
            frame,
            model=model,
            conf_threshold=config.VIT_FIRE_CONF_THRESHOLD,
            fire_conf_threshold=config.VIT_FIRE_CONF_THRESHOLD,
            min_color_ratio=fire_detector._MIN_FIRE_PIXEL_RATIO,
            validator=validator,
        )


def new_danger_zone_state(polygon: np.ndarray) -> danger_zone_detector.DangerZoneState:
    return danger_zone_detector.DangerZoneState(polygon=polygon, model=get_zone_person_model())


def analyze_danger_zone(
    state: danger_zone_detector.DangerZoneState,
    frame: np.ndarray,
    draw_boxes: bool = True,
) -> tuple[np.ndarray, int, bool]:
    with _zone_infer_lock:
        return state.analyze(
            frame,
            conf=config.DANGER_ZONE_CONF_THRESHOLD,
            draw_boxes=draw_boxes,
        )


def analyze_ppe(frame: np.ndarray, draw_person_boxes: bool = True) -> tuple[np.ndarray, list[dict], bool]:
    model = get_ppe_model()
    with _ppe_infer_lock:
        return ppe_detector.analyze_ppe_frame(
            frame,
            model=model,
            conf=config.PPE_CONF_THRESHOLD,
            draw_person_boxes=draw_person_boxes,
        )


def analyze_fall(
    frame: np.ndarray,
    frame_idx: int,
    fps: int,
    pose: fall_detector.RTMPoseEstimator,
    track_states: dict[int, fall_detector.TrackState],
    tracker: Optional[Any] = None,
) -> tuple[np.ndarray, list[dict], bool]:
    model = get_fall_model()
    with _fall_infer_lock:
        return fall_detector.process_fall_frame(
            frame=frame,
            frame_idx=frame_idx,
            fps=fps,
            model=model,
            pose=pose,
            track_states=track_states,
            tracker=tracker,
        )
