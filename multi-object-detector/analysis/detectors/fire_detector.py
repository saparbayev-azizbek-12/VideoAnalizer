from __future__ import annotations
import sys
import cv2
import subprocess
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import Callable, Optional
from dataclasses import dataclass, field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH_PT = BASE_DIR / "../models/best.pt"

_YOLO_MODEL: Optional[YOLO] = None

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

def process_video(
    input_path: str,
    output_path: str,
    progress_callback: Optional[ProgressCallback] = None,
    conf_threshold: float = 0.30,
    fire_conf_threshold: float = 0.55,
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

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                if cls_id == 0:
                    if conf < fire_conf_threshold:
                        continue
                    frame_has_fire = True
                    fire_detected_global = True
                    color = (46, 87, 228)
                    label = f"FIRE {conf * 100:.0f}%"
                elif cls_id == 2:
                    frame_has_smoke = True
                    smoke_detected_global = True
                    color = (65, 164, 217)
                    label = f"SMOKE {conf * 100:.0f}%"
                else:
                    continue
                if conf > max_conf:
                    max_conf = conf
                    primary_box = [int(x1), int(y1), int(x2), int(y2)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, max(y1 - th - 10, 0)), (x1 + tw + 8, max(y1, th + 10)), color, -1)
                cv2.putText(frame, label, (x1 + 4, max(y1 - 5, th + 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        if frame_has_fire or frame_has_smoke:
            if frame_has_fire and frame_has_smoke:
                status_text = "ALARM: FIRE & SMOKE DETECTED"
                event_type = "FIRE & SMOKE"
                bg_color = (46, 87, 228)
            elif frame_has_fire:
                status_text = "ALARM: FIRE DETECTED"
                event_type = "FIRE"
                bg_color = (46, 87, 228)
            else:
                status_text = "ALARM: SMOKE DETECTED"
                event_type = "SMOKE"
                bg_color = (65, 164, 217)

            cv2.rectangle(frame, (0, 0), (width, 42), bg_color, -1)
            cv2.putText(frame, status_text, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
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
    fire_conf_threshold: float = 0.55,
    draw_banner: bool = True,
) -> tuple[np.ndarray, list[dict], bool, bool]:
    if model is None:
        model = get_model()
    results = model(frame, verbose=False, conf=conf_threshold)
    frame_has_fire = False
    frame_has_smoke = False
    detections: list[dict] = []

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            if cls_id == 0:
                if conf < fire_conf_threshold:
                    continue
                frame_has_fire = True
                color = (46, 87, 228)
                label = f"FIRE {conf * 100:.0f}%"
                etype = "FIRE"
            elif cls_id == 2:
                frame_has_smoke = True
                color = (65, 164, 217)
                label = f"SMOKE {conf * 100:.0f}%"
                etype = "SMOKE"
            else:
                continue
            detections.append({"type": etype, "confidence": round(conf, 2), "box": [int(x1), int(y1), int(x2), int(y2)]})
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, max(y1 - th - 10, 0)), (x1 + tw + 8, max(y1, th + 10)), color, -1)
            cv2.putText(frame, label, (x1 + 4, max(y1 - 5, th + 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    if draw_banner and (frame_has_fire or frame_has_smoke):
        width = frame.shape[1]
        if frame_has_fire and frame_has_smoke:
            status_text = "ALARM: FIRE & SMOKE DETECTED"
            bg_color = (46, 87, 228)
        elif frame_has_fire:
            status_text = "ALARM: FIRE DETECTED"
            bg_color = (46, 87, 228)
        else:
            status_text = "ALARM: SMOKE DETECTED"
            bg_color = (65, 164, 217)
        cv2.rectangle(frame, (0, 0), (width, 42), bg_color, -1)
        cv2.putText(frame, status_text, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    return frame, detections, frame_has_fire, frame_has_smoke

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