from __future__ import annotations
import threading
import numpy as np

import config
import fire_detector
import danger_zone_detector

_state_lock = threading.Lock()
_enabled: dict[str, bool] = dict(config.MODEL_DEFAULT_ENABLED)

def set_enabled(model_id: str, enabled: bool) -> None:
    if model_id not in config.MODEL_IDS:
        raise ValueError(f"Noma'lum model: {model_id}")
    with _state_lock:
        _enabled[model_id] = bool(enabled)

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
            from ultralytics import YOLO
            _zone_person_model = YOLO(config.PERSON_MODEL_PATH)
    return _zone_person_model

def preload_all() -> None:
    get_fire_model()
    get_zone_person_model()

def analyze_fire(frame: np.ndarray) -> tuple[np.ndarray, list[dict], bool, bool]:
    model = get_fire_model()
    with _fire_infer_lock:
        return fire_detector.detect_fire_frame(
            frame,
            model=model,
            conf_threshold=config.FIRE_CONF_THRESHOLD,
            fire_conf_threshold=config.FIRE_ONLY_CONF_THRESHOLD,
        )

def new_danger_zone_state(polygon: np.ndarray) -> danger_zone_detector.DangerZoneState:
    return danger_zone_detector.DangerZoneState(polygon=polygon, model=get_zone_person_model())

def analyze_danger_zone(
    state: danger_zone_detector.DangerZoneState,
    frame: np.ndarray,
    draw_boxes: bool | None = None,
) -> tuple[np.ndarray, int, bool]:
    if draw_boxes is None:
        draw_boxes = not is_enabled("fall")
    with _zone_infer_lock:
        return state.analyze(
            frame,
            conf=config.DANGER_ZONE_CONF_THRESHOLD,
            draw_boxes=draw_boxes,
        )