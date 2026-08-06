"""
models_manager.py
----------------------------------------------------------------------
Barcha aniqlash modellarining YAGONA markazi:
  - fire         -> fire_detector.py         (best.onnx / best.pt)
  - fall         -> fall_detector_v2.py      (YOLO tracker + MediaPipe Pose)
  - danger_zone  -> danger_zone_detector.py  (YOLO person + poligon)

Vazifalari:
  1) Har bir model GLOBAL ravishda yoqilgan/o'chirilganligini saqlaydi
     (GUI dagi checkbox shu holatni o'zgartiradi; camera_manager.py shu
     holatga qarab tegishli modelni chaqiradi yoki chaqirmaydi).
  2) Og'ir modellarni faqat BIR MARTA yuklaydi (xotira tejash uchun) va
     ularni barcha kameralar orasida (xavfsiz tarzda) bo'lishadi.

DIQQAT: "fall" modeli bu yerda YUKLANMAYDI - u tracking holati saqlagani
uchun har bir kamera o'zining ALOHIDA instansini yaratadi
(qarang: fall_detector_v2.create_tracking_model, camera_manager.py).
"""

from __future__ import annotations

import threading

import numpy as np

import config
import fire_detector
import danger_zone_detector

# ---------------------------------------------------------------------------
# 1) MODELLARNI YOQISH / O'CHIRISH HOLATI (global, barcha kameralarga tegishli)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 2) OG'IR (STATELESS) MODELLARNI BIR MARTA YUKLASH - lazy-loading
# ---------------------------------------------------------------------------
_fire_model = None
_fire_model_lock = threading.Lock()
_fire_infer_lock = threading.Lock()   # GPU'ga bir vaqtning o'zida bitta so'rov boradi

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
    """
    Server ishga tushganda (FastAPI startup) chaqiriladi - shunda birinchi
    kamera ulanganda modelni yuklash uchun kutish bo'lmaydi.
    """
    get_fire_model()
    get_zone_person_model()


# ---------------------------------------------------------------------------
# 3) BIR KADRNI TAHLIL QILISH (thread-xavfsiz wrapper'lar)
# ---------------------------------------------------------------------------
def analyze_fire(frame: np.ndarray) -> tuple[np.ndarray, list[dict], bool, bool]:
    """fire_detector.detect_fire_frame() ni GPU-lock bilan chaqiradi."""
    model = get_fire_model()
    with _fire_infer_lock:
        return fire_detector.detect_fire_frame(
            frame,
            model=model,
            conf_threshold=config.FIRE_CONF_THRESHOLD,
            fire_conf_threshold=config.FIRE_ONLY_CONF_THRESHOLD,
        )


def new_danger_zone_state(polygon: np.ndarray) -> danger_zone_detector.DangerZoneState:
    """
    Har bir kamera uchun (poligoni boshqacha bo'lgani sababli) alohida
    DangerZoneState yaratadi, lekin OG'IR YOLO modeli barcha kameralar
    orasida ULASHILADI (bitta marta yuklanadi).
    """
    return danger_zone_detector.DangerZoneState(polygon=polygon, model=get_zone_person_model())


def analyze_danger_zone(
    state: danger_zone_detector.DangerZoneState, frame: np.ndarray
) -> tuple[np.ndarray, int, bool]:
    with _zone_infer_lock:
        return state.analyze(frame, conf=config.DANGER_ZONE_CONF_THRESHOLD)