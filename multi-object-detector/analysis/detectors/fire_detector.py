from __future__ import annotations
import sys
import cv2
import subprocess
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import Callable, Optional
from dataclasses import dataclass, field
from collections import deque

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH_PT = BASE_DIR / "../models/best.pt"

_YOLO_MODEL: Optional[YOLO] = None

CLASS_FIRE = 0
CLASS_SMOKE = 2

def get_model() -> YOLO:
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        if MODEL_PATH_PT.exists():
            _YOLO_MODEL = YOLO(str(MODEL_PATH_PT))
        else:
            raise FileNotFoundError("Na best.onnx va na best.pt fayli topilmadi!")
    return _YOLO_MODEL

@dataclass
class FireEvent:
    frame_index: int
    timestamp_sec: float
    event_type: str
    confidence: float
    box: list[int]

@dataclass
class ProcessingResult:
    output_path: str
    fps: int
    width: int
    height: int
    total_frames: int
    fire_detected: bool
    smoke_detected: bool
    events: list[FireEvent] = field(default_factory=list)

ProgressCallback = Callable[[int, int], None]

def _fire_color_ratio(frame: np.ndarray, box: list[int]) -> float:
    x1, y1, x2, y2 = box
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hue_mask = (h <= 35) | (h >= 160)
    fire_mask = hue_mask & (s >= 60) & (v >= 140)
    return float(np.count_nonzero(fire_mask)) / float(fire_mask.size)

def _is_valid_fire_box(frame: np.ndarray, box: list[int], min_color_ratio: float) -> bool:
    return _fire_color_ratio(frame, box) >= min_color_ratio

class TemporalValidator:
    def __init__(self, window: int = 6, min_hits: int = 3):
        self.window = window
        self.min_hits = min_hits
        self._fire_hist: deque[bool] = deque(maxlen=window)
        self._smoke_hist: deque[bool] = deque(maxlen=window)

    def update(self, has_fire: bool, has_smoke: bool) -> tuple[bool, bool]:
        self._fire_hist.append(has_fire)
        self._smoke_hist.append(has_smoke)
        confirmed_fire = sum(self._fire_hist) >= self.min_hits
        confirmed_smoke = sum(self._smoke_hist) >= self.min_hits
        return confirmed_fire, confirmed_smoke

    def reset(self) -> None:
        self._fire_hist.clear()
        self._smoke_hist.clear()

def _draw_detection(frame: np.ndarray, box: list[int], color: tuple[int, int, int], label: str) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, max(y1 - th - 10, 0)), (x1 + tw + 8, max(y1, th + 10)), color, -1)
    cv2.putText(frame, label, (x1 + 4, max(y1 - 5, th + 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

def _draw_banner(frame: np.ndarray, width: int, confirmed_fire: bool, confirmed_smoke: bool) -> str:
    if confirmed_fire and confirmed_smoke:
        status_text = "ALARM: FIRE & SMOKE DETECTED"
        event_type = "FIRE & SMOKE"
        bg_color = (46, 87, 228)
    elif confirmed_fire:
        status_text = "ALARM: FIRE DETECTED"
        event_type = "FIRE"
        bg_color = (46, 87, 228)
    else:
        status_text = "ALARM: SMOKE DETECTED"
        event_type = "SMOKE"
        bg_color = (65, 164, 217)
    cv2.rectangle(frame, (0, 0), (width, 42), bg_color, -1)
    cv2.putText(frame, status_text, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    return event_type

def process_video(
    input_path: str,
    output_path: str,
    progress_callback: Optional[ProgressCallback] = None,
    conf_threshold: float = 0.30,
    fire_conf_threshold: float = 0.60,
    min_color_ratio: float = 0.18,
    confirm_window: int = 6,
    confirm_min_hits: int = 3,
) -> ProcessingResult:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video ochilmadi: {input_path}")
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    fire_detected_global = False
    smoke_detected_global = False
    events: list[FireEvent] = []
    frame_idx = 0
    model = get_model()
    last_event_sec = -1.0
    validator = TemporalValidator(window=confirm_window, min_hits=confirm_min_hits)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        timestamp_sec = frame_idx / fps
        if progress_callback:
            progress_callback(frame_idx, total_frames)
        results = model(frame, verbose=False, conf=conf_threshold)
        frame_has_fire = False
        frame_has_smoke = False
        max_conf = 0.0
        primary_box = [0, 0, 0, 0]
        pending_draws: list[tuple[list[int], tuple[int, int, int], str]] = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                bbox = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                if cls_id == CLASS_FIRE:
                    if conf < fire_conf_threshold:
                        continue
                    if not _is_valid_fire_box(frame, bbox, min_color_ratio):
                        continue
                    frame_has_fire = True
                    color = (46, 87, 228)
                    label = f"FIRE {conf * 100:.0f}%"
                elif cls_id == CLASS_SMOKE:
                    frame_has_smoke = True
                    color = (65, 164, 217)
                    label = f"SMOKE {conf * 100:.0f}%"
                else:
                    continue
                if conf > max_conf:
                    max_conf = conf
                    primary_box = bbox
                pending_draws.append((bbox, color, label))

        confirmed_fire, confirmed_smoke = validator.update(frame_has_fire, frame_has_smoke)

        if confirmed_fire or confirmed_smoke:
            for bbox, color, label in pending_draws:
                _draw_detection(frame, bbox, color, label)
            event_type = _draw_banner(frame, width, confirmed_fire, confirmed_smoke)
            if confirmed_fire:
                fire_detected_global = True
            if confirmed_smoke:
                smoke_detected_global = True
            if timestamp_sec - last_event_sec >= 0.8:
                events.append(FireEvent(frame_index=frame_idx, timestamp_sec=round(timestamp_sec, 2), event_type=event_type, confidence=round(max_conf, 2), box=primary_box))
                last_event_sec = timestamp_sec

        out.write(frame)

    cap.release()
    out.release()
    return ProcessingResult(output_path=output_path, fps=fps, width=width, height=height, total_frames=frame_idx, fire_detected=fire_detected_global, smoke_detected=smoke_detected_global, events=events)

def detect_fire_frame(
    frame: np.ndarray,
    model: Optional[YOLO] = None,
    conf_threshold: float = 0.30,
    fire_conf_threshold: float = 0.60,
    min_color_ratio: float = 0.18,
    validator: Optional[TemporalValidator] = None,
    draw_banner: bool = True,
) -> tuple[np.ndarray, list[dict], bool, bool]:
    if model is None:
        model = get_model()
    results = model(frame, verbose=False, conf=conf_threshold)
    frame_has_fire = False
    frame_has_smoke = False
    detections: list[dict] = []
    pending_draws: list[tuple[list[int], tuple[int, int, int], str]] = []

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            bbox = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
            if cls_id == CLASS_FIRE:
                if conf < fire_conf_threshold:
                    continue
                if not _is_valid_fire_box(frame, bbox, min_color_ratio):
                    continue
                frame_has_fire = True
                color = (46, 87, 228)
                label = f"FIRE {conf * 100:.0f}%"
                etype = "FIRE"
            elif cls_id == CLASS_SMOKE:
                frame_has_smoke = True
                color = (65, 164, 217)
                label = f"SMOKE {conf * 100:.0f}%"
                etype = "SMOKE"
            else:
                continue
            detections.append({"type": etype, "confidence": round(conf, 2), "box": bbox})
            pending_draws.append((bbox, color, label))

    if validator is not None:
        confirmed_fire, confirmed_smoke = validator.update(frame_has_fire, frame_has_smoke)
    else:
        confirmed_fire, confirmed_smoke = frame_has_fire, frame_has_smoke

    if confirmed_fire or confirmed_smoke:
        for bbox, color, label in pending_draws:
            _draw_detection(frame, bbox, color, label)
        if draw_banner:
            _draw_banner(frame, frame.shape[1], confirmed_fire, confirmed_smoke)

    return frame, detections, confirmed_fire, confirmed_smoke

def reencode_for_web(input_path: str, output_path: str) -> None:
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_exe = "ffmpeg"
    cmd = [ffmpeg_exe, "-y", "-i", input_path, "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", output_path]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Foydalanish: python fire_detector.py <input_video> <output_video>")
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]

    def _print_progress(cur: int, total: int) -> None:
        if total:
            pct = cur / total * 100
            print(f"\rQayta ishlanmoqda: {cur}/{total} ({pct:.1f}%)", end="", flush=True)

    res = process_video(src, dst, progress_callback=_print_progress)
    print()
    print(f"Tayyor: {res.output_path}")
    print(f"Yong'in aniqlandimi: {res.fire_detected}")
    print(f"Tutun aniqlandimi: {res.smoke_detected}")
    for ev in res.events:
        print(f"  -> [{ev.event_type}] frame {ev.frame_index}, {ev.timestamp_sec:.2f}s ({ev.confidence*100:.0f}%)")