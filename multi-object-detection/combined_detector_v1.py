"""
combined_detector_v1.py

Olov (fire), tutun (smoke) hamda odam yiqilishini (fall v1) bir vaqtning o'zida
bitta video kadrlarida tahlil qiluvchi va natijalarni yagona kadr ustida
birlashtiruvchi v1 moduli.

PyTorch (.pt) & TensorRT / GPU Optimallashuvi:
  - ONNX o'rniga PyTorch (.pt) va TensorRT (.engine) modellari ustuvor o'qiladi.
  - CUDA / NVIDIA DGX GPU apparat tezlatgichidan foydalanadi (device="cuda").
  - FP16 Half Precision throughput optimallashuvi (half=True).
"""

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

BASE_DIR = Path(__file__).resolve().parent

# --- GPU / CUDA Qurilmani aniqlash ---
def get_device() -> str:
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[GPU OPTIMIZATION] CUDA / PyTorch GPU faol: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
        return "cuda"
    print("[GPU OPTIMIZATION] CUDA topilmadi, CPU rejimida ishlatilmoqda.")
    return "cpu"

DEVICE = get_device()
USE_HALF = (DEVICE == "cuda")

_FIRE_MODEL: Optional[YOLO] = None
_FALL_MODEL: Optional[YOLO] = None


def get_fire_model() -> YOLO:
    """Olov/tutun modelini (.pt -> .engine -> .onnx tartibida) o'qiydi."""
    global _FIRE_MODEL
    if _FIRE_MODEL is None:
        pt_path = BASE_DIR / "best.pt"
        pt_alt = BASE_DIR.parent / "fire-detection" / "best.pt"
        engine_path = BASE_DIR / "best.engine"
        onnx_path = BASE_DIR / "best.onnx"
        onnx_alt = BASE_DIR.parent / "fire-detection" / "best.onnx"

        if pt_path.exists():
            print(f"[MODEL LOAD] Fire Model (.pt) yuklanmoqda: {pt_path}")
            _FIRE_MODEL = YOLO(str(pt_path))
        elif pt_alt.exists():
            print(f"[MODEL LOAD] Fire Model (.pt) yuklanmoqda: {pt_alt}")
            _FIRE_MODEL = YOLO(str(pt_alt))
        elif engine_path.exists():
            print(f"[MODEL LOAD] Fire Model TensorRT (.engine) yuklanmoqda: {engine_path}")
            _FIRE_MODEL = YOLO(str(engine_path))
        elif onnx_path.exists():
            print(f"[MODEL LOAD] Fire Model ONNX yuklanmoqda: {onnx_path}")
            _FIRE_MODEL = YOLO(str(onnx_path), task="detect")
        elif onnx_alt.exists():
            print(f"[MODEL LOAD] Fire Model ONNX yuklanmoqda: {onnx_alt}")
            _FIRE_MODEL = YOLO(str(onnx_alt), task="detect")
        else:
            raise FileNotFoundError("Olov/tutun modeli (best.pt / best.engine / best.onnx) topilmadi!")

        if DEVICE == "cuda":
            _FIRE_MODEL.to(DEVICE)
    return _FIRE_MODEL


def get_fall_model() -> YOLO:
    """Yiqilish modelini (.pt -> .engine -> .onnx tartibida) o'qiydi."""
    global _FALL_MODEL
    if _FALL_MODEL is None:
        pt_path = BASE_DIR / "yolov8l.pt"
        pt_alt = BASE_DIR.parent / "fall-detection" / "yolov8l.pt"
        engine_path = BASE_DIR / "yolov8l.engine"
        onnx_path = BASE_DIR / "yolov8l.onnx"
        onnx_alt = BASE_DIR.parent / "fall-detection" / "yolov8l.onnx"

        if pt_path.exists():
            print(f"[MODEL LOAD] Fall Model (.pt) yuklanmoqda: {pt_path}")
            _FALL_MODEL = YOLO(str(pt_path))
        elif pt_alt.exists():
            print(f"[MODEL LOAD] Fall Model (.pt) yuklanmoqda: {pt_alt}")
            _FALL_MODEL = YOLO(str(pt_alt))
        elif engine_path.exists():
            print(f"[MODEL LOAD] Fall Model TensorRT (.engine) yuklanmoqda: {engine_path}")
            _FALL_MODEL = YOLO(str(engine_path))
        elif onnx_path.exists():
            print(f"[MODEL LOAD] Fall Model ONNX yuklanmoqda: {onnx_path}")
            _FALL_MODEL = YOLO(str(onnx_path), task="detect")
        elif onnx_alt.exists():
            print(f"[MODEL LOAD] Fall Model ONNX yuklanmoqda: {onnx_alt}")
            _FALL_MODEL = YOLO(str(onnx_alt), task="detect")
        else:
            raise FileNotFoundError("Yiqilish modeli (yolov8l.pt / yolov8l.engine / yolov8l.onnx) topilmadi!")

        if DEVICE == "cuda":
            _FALL_MODEL.to(DEVICE)
    return _FALL_MODEL


def calculate_angle(shoulder_center: tuple[float, float], hip_center: tuple[float, float]) -> float:
    """Yelka-son chizig'i bilan vertikal o'q orasidagi burchak (gradusda)."""
    dy = shoulder_center[1] - hip_center[1]
    dx = shoulder_center[0] - hip_center[0]
    angle = math.atan2(dy, dx)
    return abs(90 - np.degrees(angle))


def classify_posture(torso_angle: float, standing_threshold: float = 10, lying_threshold: float = 60) -> str:
    if torso_angle < standing_threshold:
        return "Standing"
    elif torso_angle > lying_threshold:
        return "Lying Down"
    else:
        return "Falling"


@dataclass
class DetectionEvent:
    frame_index: int
    timestamp_sec: float
    event_type: str  # "FIRE", "SMOKE", "FALL DETECTED"
    confidence: float
    box: list[int]  # [x1, y1, x2, y2]


@dataclass
class CombinedProcessingResult:
    output_path: str
    fps: int
    width: int
    height: int
    total_frames: int
    fire_detected: bool
    smoke_detected: bool
    fall_detected: bool
    events: list[DetectionEvent] = field(default_factory=list)


ProgressCallback = Callable[[int, int], None]


def process_video(
    input_path: str,
    output_path: str,
    progress_callback: Optional[ProgressCallback] = None,
    conf_threshold: float = 0.30,
    fire_conf_threshold: float = 0.55,
) -> CombinedProcessingResult:
    """
    Videoni kadr-ma-kadr tahlil qilib:
      1) Fire & Smoke PyTorch/ONNX modeli orqali Olov va Tutunni aniqlaydi.
      2) YOLOv8 + MediaPipe Pose orqali yiqilishni aniqlaydi.
      3) Kadr ustida ramkalar va 'FALL DETECTED' yozuvini chizadi.
      4) Barcha hodisalar ro'yxatini qaytaradi.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video ochilmadi: {input_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    fire_model = get_fire_model()
    fall_model = get_fall_model()

    fire_detected_global = False
    smoke_detected_global = False
    fall_detected_global = False
    events: list[DetectionEvent] = []

    falling_count = 0
    frame_fall_detected = False
    last_event_sec = -1.0
    frame_idx = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, torch.inference_mode():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            timestamp_sec = frame_idx / fps

            if progress_callback:
                progress_callback(frame_idx, total_frames)

            # --- 1. FIRE & SMOKE DETECTION ---
            fire_results = fire_model(
                frame,
                verbose=False,
                conf=conf_threshold,
                device=DEVICE,
                half=USE_HALF,
            )
            frame_has_fire = False
            frame_has_smoke = False
            fire_max_conf = 0.0
            fire_primary_box = [0, 0, 0, 0]

            for result in fire_results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy

                    if cls_id == 0:  # Fire
                        if conf < fire_conf_threshold:
                            continue
                        frame_has_fire = True
                        fire_detected_global = True
                        color = (46, 87, 228)  # Qizil (BGR)
                        label = f"FIRE {conf * 100:.0f}%"
                    elif cls_id == 2:  # Smoke
                        frame_has_smoke = True
                        smoke_detected_global = True
                        color = (65, 164, 217)  # Amber/Sariq (BGR)
                        label = f"SMOKE {conf * 100:.0f}%"
                    else:
                        continue

                    if conf > fire_max_conf:
                        fire_max_conf = conf
                        fire_primary_box = [int(x1), int(y1), int(x2), int(y2)]

                    # Ramka chizish
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Label background va text
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, max(y1 - th - 10, 0)), (x1 + tw + 8, max(y1, th + 10)), color, -1)
                    cv2.putText(
                        frame,
                        label,
                        (x1 + 4, max(y1 - 5, th + 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

            # --- 2. FALL DETECTION ---
            person_results = fall_model(
                frame,
                verbose=False,
                device=DEVICE,
                half=USE_HALF,
            )
            for result in person_results:
                if result.boxes is None:
                    continue

                for bbox, cls in zip(result.boxes.xyxy, result.boxes.cls):
                    if int(cls) != 0:  # faqat 'person' klassi
                        continue

                    x1, y1, x2, y2 = map(int, bbox)
                    x1, y1 = max(x1, 0), max(y1, 0)
                    x2, y2 = min(x2, width), min(y2, height)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    person_bbox = frame[y1:y2, x1:x2]
                    if person_bbox.size == 0:
                        continue
                    person_bbox_rgb = cv2.cvtColor(person_bbox, cv2.COLOR_BGR2RGB)
                    pose_results = pose.process(person_bbox_rgb)

                    if pose_results.pose_landmarks:
                        landmarks = pose_results.pose_landmarks.landmark
                        h, w = person_bbox.shape[:2]

                        shoulders = [
                            (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h),
                            (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                             landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h),
                        ]
                        hips = [
                            (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
                             landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h),
                            (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h),
                        ]

                        shoulder_center = (
                            (shoulders[0][0] + shoulders[1][0]) / 2,
                            (shoulders[0][1] + shoulders[1][1]) / 2,
                        )
                        hip_center = (
                            (hips[0][0] + hips[1][0]) / 2,
                            (hips[0][1] + hips[1][1]) / 2,
                        )

                        torso_angle = calculate_angle(hip_center, shoulder_center)
                        posture = classify_posture(torso_angle)

                        mp_drawing.draw_landmarks(
                            person_bbox, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                        )
                        cv2.putText(
                            frame,
                            f"{posture} ({torso_angle:.1f} deg)",
                            (x1, max(y1 - 10, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                        if posture == "Falling":
                            falling_count += 1
                        else:
                            falling_count = 0

                        if 0.1 * fps <= falling_count <= 0.5 * fps:
                            frame_fall_detected = True
                            fall_detected_global = True
                        if posture == "Standing":
                            frame_fall_detected = False

                    frame[y1:y2, x1:x2] = person_bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 120, 0), 2)

            # Agar yiqilish sezilgan bo'lsa kadr ustida "FALL DETECTED" yozuvi
            if frame_fall_detected:
                cv2.putText(
                    frame,
                    "FALL DETECTED",
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.4,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )

            # Event yozish (har 0.8s da)
            if timestamp_sec - last_event_sec >= 0.8:
                if frame_has_fire:
                    events.append(
                        DetectionEvent(
                            frame_index=frame_idx,
                            timestamp_sec=round(timestamp_sec, 2),
                            event_type="FIRE",
                            confidence=round(fire_max_conf, 2),
                            box=fire_primary_box,
                        )
                    )
                    last_event_sec = timestamp_sec
                elif frame_has_smoke:
                    events.append(
                        DetectionEvent(
                            frame_index=frame_idx,
                            timestamp_sec=round(timestamp_sec, 2),
                            event_type="SMOKE",
                            confidence=round(fire_max_conf, 2),
                            box=fire_primary_box,
                        )
                    )
                    last_event_sec = timestamp_sec

                if frame_fall_detected:
                    events.append(
                        DetectionEvent(
                            frame_index=frame_idx,
                            timestamp_sec=round(timestamp_sec, 2),
                            event_type="FALL DETECTED",
                            confidence=0.95,
                            box=[0, 0, 0, 0],
                        )
                    )
                    last_event_sec = timestamp_sec

            out.write(frame)

    cap.release()
    out.release()

    return CombinedProcessingResult(
        output_path=output_path,
        fps=fps,
        width=width,
        height=height,
        total_frames=frame_idx,
        fire_detected=fire_detected_global,
        smoke_detected=smoke_detected_global,
        fall_detected=fall_detected_global,
        events=events,
    )


def reencode_for_web(input_path: str, output_path: str) -> None:
    """
    OpenCV yozgan mp4 video brauzerda ijro etilishi uchun H.264 ga o'tkaziladi.
    GPU mavjud bo'lsa h264_nvenc apparat tezlatgichidan foydalanadi.
    """
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_exe = "ffmpeg"

    if DEVICE == "cuda":
        cmd_gpu = [
            ffmpeg_exe,
            "-y",
            "-i",
            input_path,
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            output_path,
        ]
        res = subprocess.run(cmd_gpu, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0:
            return

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        input_path,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Foydalanish: python combined_detector_v1.py <input_video> <output_video>")
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
    print(f"Yiqilish aniqlandimi: {res.fall_detected}")
    for ev in res.events:
        print(f"  -> [{ev.event_type}] frame {ev.frame_index}, {ev.timestamp_sec:.2f}s ({ev.confidence*100:.0f}%)")
