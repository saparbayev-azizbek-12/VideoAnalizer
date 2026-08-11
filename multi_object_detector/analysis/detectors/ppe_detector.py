"""
PPE (Personal Protective Equipment) detektor.

sfchd_yolov8s.pt modeli yordamida xavfsizlik jihozlarini (kask, kiyim) aniqlaydi.
Kadrda kask yoki xavfsizlik kiyimi kiyilmagan odam aniqlansa alarm beradi.

Sinf nomlari (model.names):
    Person, Safety Helmet, Safety Clothing, Other Clothing,
    Head, Blurred Clothing, Blurred Head
"""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from collections import deque
from ultralytics import YOLO
from multi_object_detector import config

# ── Model sinf nomlari ─────────────────────────────────────────────────────────
CLASS_PERSON           = "Person"
CLASS_SAFETY_HELMET    = "Safety Helmet"
CLASS_SAFETY_CLOTHING  = "Safety Clothing"
CLASS_OTHER_CLOTHING   = "Other Clothing"
CLASS_HEAD             = "Head"
CLASS_BLURRED_CLOTHING = "Blurred Clothing"
CLASS_BLURRED_HEAD     = "Blurred Head"

# Alarm beradigan sinflar (bunlar yo'q bo'lsa xavfli)
REQUIRED_PPE_CLASSES = {CLASS_SAFETY_HELMET, CLASS_SAFETY_CLOTHING}

# "Odam bor" deb hisoblanadigan sinflar
PERSON_LIKE_CLASSES = {CLASS_PERSON, CLASS_HEAD, CLASS_BLURRED_HEAD}

# ── Ranglar ────────────────────────────────────────────────────────────────────
COLOR_OK      = (0, 200, 60)     # Yashil: PPE mavjud
COLOR_VIOLATE = (0, 40, 220)     # Qizil: PPE yo'q
COLOR_BANNER  = (0, 0, 180)      # Qizil: banner fon

# ── Singleton model ────────────────────────────────────────────────────────────
_ppe_model: Optional[YOLO] = None


def get_model() -> YOLO:
    global _ppe_model
    if _ppe_model is None:
        _ppe_model = YOLO(config.PPE_MODEL_PATH)
    return _ppe_model


# ── Natija tuzilmasi ──────────────────────────────────────────────────────────
@dataclass
class PPEViolation:
    """Bitta "PPE yo'q" hodisasi."""
    box: list[int]          # [x1, y1, x2, y2]
    confidence: float
    missing: list[str]      # yo'q jihozlar ro'yxati (masalan, ["Safety Helmet"])


# ── Yordamchi: IoU hisoblash ──────────────────────────────────────────────────
def _iou(boxA: list[float], boxB: list[float]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter + 1e-6)


# ── Asosiy tahlil funksiyasi ──────────────────────────────────────────────────
def analyze_ppe_frame(
    frame: np.ndarray,
    model: Optional[YOLO] = None,
    conf: float = 0.40,
    iou_match_thresh: float = 0.10,
) -> tuple[np.ndarray, list[dict], bool]:
    """
    Kadrda PPE aniqlash.

    Returns:
        annotated_frame  — annotatsiya qilingan kadr
        violations       — [{box, confidence, missing, type}] ro'yxati
        has_violation    — kamida 1 ta qoidabuzarlik bor/yo'qligi
    """
    if model is None:
        model = get_model()

    result = model(frame, conf=conf, verbose=False)[0]
    annotated = frame.copy()
    names: dict[int, str] = result.names  # {0: "Person", 1: "Safety Helmet", ...}

    if result.boxes is None or len(result.boxes) == 0:
        return annotated, [], False

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()       # (N, 4)
    confs      = result.boxes.conf.cpu().numpy()       # (N,)
    class_ids  = result.boxes.cls.cpu().numpy().astype(int)  # (N,)

    # Natijalarni sinflarga ajratish
    person_like: list[tuple[np.ndarray, float]] = []   # [(box, conf), ...]
    ppe_boxes: dict[str, list[np.ndarray]] = {
        CLASS_SAFETY_HELMET:    [],
        CLASS_SAFETY_CLOTHING:  [],
    }
    all_detections: list[tuple[np.ndarray, float, str]] = []  # (box, conf, name)

    for box, cf, cid in zip(boxes_xyxy, confs, class_ids):
        name = names.get(cid, "")
        all_detections.append((box, float(cf), name))
        if name in PERSON_LIKE_CLASSES:
            person_like.append((box, float(cf)))
        elif name in ppe_boxes:
            ppe_boxes[name].append(box)

    # ── Har bir "odam" uchun PPE tekshirish ──────────────────────────────────
    violations: list[dict] = []

    for p_box, p_conf in person_like:
        missing = []
        for req_class in REQUIRED_PPE_CLASSES:
            overlaps = any(
                _iou(p_box.tolist(), eq_box.tolist()) >= iou_match_thresh
                for eq_box in ppe_boxes[req_class]
            )
            if not overlaps:
                missing.append(req_class)

        x1, y1, x2, y2 = map(int, p_box)

        if missing:
            # PPE YO'Q → qizil
            cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_VIOLATE, 2)
            label = "NO PPE: " + ", ".join(
                "Helmet" if m == CLASS_SAFETY_HELMET else "Clothing"
                for m in missing
            )
            _draw_label(annotated, label, x1, y1, COLOR_VIOLATE)
            violations.append({
                "box": [x1, y1, x2, y2],
                "confidence": round(p_conf, 3),
                "missing": missing,
                "type": "ppe_violation",
            })
        else:
            # PPE bor → yashil
            cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_OK, 2)
            _draw_label(annotated, "PPE OK", x1, y1, COLOR_OK)

    # Boshqa aniqlangan obyektlarni ingichka chiziq bilan ko'rsatish
    for box, cf, name in all_detections:
        if name in PERSON_LIKE_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, box)
        color = COLOR_OK if name in REQUIRED_PPE_CLASSES else (180, 180, 180)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            annotated, f"{name} {cf:.2f}",
            (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
        )

    # ── Alarm banneri ─────────────────────────────────────────────────────────
    has_violation = len(violations) > 0
    if has_violation:
        width = frame.shape[1]
        banner_txt = f"DIQQAT! {len(violations)} ta PPE QOIDABUZARLIK"
        cv2.rectangle(annotated, (0, 0), (width, 46), COLOR_BANNER, -1)
        cv2.putText(
            annotated, banner_txt,
            (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
        )

    return annotated, violations, has_violation


def _draw_label(img: np.ndarray, text: str, x: int, y: int, color: tuple) -> None:
    """Matn uchun to'ldirilgan fon bilan label chizish."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.5, 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    ty = max(th + 4, y)
    cv2.rectangle(img, (x, ty - th - 4), (x + tw + 4, ty + baseline), color, -1)
    cv2.putText(img, text, (x + 2, ty - 2), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
