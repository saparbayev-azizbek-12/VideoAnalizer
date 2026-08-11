from __future__ import annotations
import cv2
import numpy as np
from typing import Optional
from ultralytics import YOLO
from dataclasses import dataclass
from multi_object_detector import config

CLASS_PERSON = "Person"
CLASS_SAFETY_HELMET = "Safety Helmet"
CLASS_SAFETY_CLOTHING = "Safety Clothing"
CLASS_OTHER_CLOTHING = "Other Clothing"
CLASS_HEAD = "Head"
CLASS_BLURRED_CLOTHING = "Blurred Clothing"
CLASS_BLURRED_HEAD = "Blurred Head"

REQUIRED_PPE_CLASSES = {CLASS_SAFETY_HELMET, CLASS_SAFETY_CLOTHING}
PERSON_LIKE_CLASSES = {CLASS_PERSON}
IGNORED_CLASSES = {CLASS_HEAD, CLASS_BLURRED_HEAD}

COLOR_OK = (0, 200, 60)
COLOR_VIOLATE = (0, 40, 220)
COLOR_BANNER = (0, 0, 180)

_ppe_model: Optional[YOLO] = None

def get_model() -> YOLO:
    global _ppe_model
    if _ppe_model is None:
        _ppe_model = YOLO(config.PPE_MODEL_PATH)
    return _ppe_model

@dataclass
class PPEViolation:
    box: list[int]
    confidence: float
    missing: list[str]

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

def analyze_ppe_frame(
    frame: np.ndarray,
    model: Optional[YOLO] = None,
    conf: float = 0.40,
    iou_match_thresh: float = 0.10,
) -> tuple[np.ndarray, list[dict], bool]:
    if model is None:
        model = get_model()

    result = model(frame, conf=conf, verbose=False)[0]
    annotated = frame.copy()
    names: dict[int, str] = result.names
    if result.boxes is None or len(result.boxes) == 0:
        return annotated, [], False

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)

    person_like: list[tuple[np.ndarray, float]] = []
    ppe_boxes: dict[str, list[np.ndarray]] = {
        CLASS_SAFETY_HELMET: [],
        CLASS_SAFETY_CLOTHING: [],
    }
    all_detections: list[tuple[np.ndarray, float, str]] = []

    for box, cf, cid in zip(boxes_xyxy, confs, class_ids):
        name = names.get(cid, "")
        if name in IGNORED_CLASSES:
            continue
        all_detections.append((box, float(cf), name))
        if name in PERSON_LIKE_CLASSES:
            person_like.append((box, float(cf)))
        elif name in ppe_boxes:
            ppe_boxes[name].append(box)

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
            cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_VIOLATE, 3)
            label = "⚠ NO PPE: " + ", ".join(
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
            cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_OK, 3)
            _draw_label(annotated, "✓ PPE OK", x1, y1, COLOR_OK)

    for box, cf, name in all_detections:
        if name in PERSON_LIKE_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, box)
        color = COLOR_OK if name in REQUIRED_PPE_CLASSES else (200, 200, 200)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        lbl = f"{name} {cf:.2f}"
        (tw, th), bl = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(th + 4, y1)
        cv2.rectangle(annotated, (x1, ty - th - 6), (x1 + tw + 6, ty + bl + 2), color, -1)
        cv2.putText(
            annotated, lbl,
            (x1 + 3, ty - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA
        )

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
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.7, 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    pad = 6
    ty = max(th + pad + 2, y)
    cv2.rectangle(img, (x, ty - th - pad), (x + tw + pad * 2, ty + baseline + 2), color, -1)
    cv2.rectangle(img, (x, ty - th - pad), (x + tw + pad * 2, ty + baseline + 2), (0, 0, 0), 1)
    cv2.putText(img, text, (x + pad, ty - 2), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
