"""
combined_detector_v2.py

Olov (fire), tutun (smoke) hamda odam yiqilishini (fall v2 - yaxshilangan)
bir vaqtning o'zida bitta video kadrlarida tahlil qiluvchi va natijalarni
yagona kadr ustida birlashtiruvchi v2 moduli.

PyTorch (.pt) & TensorRT / GPU Optimallashuvi:
  - ONNX o'rniga PyTorch (.pt) va TensorRT (.engine) modellari ustuvor o'qiladi.
  - CUDA / NVIDIA DGX GPU apparat tezlatgichidan foydalanadi (device="cuda").
  - FP16 Half Precision throughput optimallashuvi (half=True).
  - PyTorch inference_mode yordamida ortiqcha xotira va hisoblashlarni tejaydi.
  - Video eksportida NVIDIA NVENC (h264_nvenc) apparat video kodlashidan foydalanadi.

Parallel Inference (DGX Spark 128GB uchun):
  - Olov va Yiqilish modellari ThreadPoolExecutor + torch.cuda.Stream orqali
    PARALLEL ravishda bitta kadrni bir vaqtda tahlil qiladi.
  - Har model o'z CUDA stream'ida ishlaydi: GPU SM'lari bo'linib ikki xil
    inference pipeline'ni bir vaqtda bajaradi.
  - 128GB VRAM bilan ikkala model (best.pt ~6MB + yolov8l.pt ~87MB) hamda
    MediaPipe CPU pipeline birgalikda sig'adi.
"""

from __future__ import annotations

import math
import subprocess
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Any

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

# --- Sozlanadigan chegaralar (Fall Detection v2) -----------------------------
STANDING_ANGLE_DEG = 10.0
LYING_ANGLE_DEG = 60.0
ANGLE_SMOOTH_WINDOW = 5          # so'nggi N kadr bo'yicha burchak o'rtachasi
FALL_MIN_FRAMES_RATIO = 0.1      # fps ga nisbatan minimal davomiylik
FALL_MAX_FRAMES_RATIO = 0.5      # fps ga nisbatan maksimal davomiylik
VELOCITY_WINDOW_SEC = 0.3        # tezlikni shu oraliqdagi harakatdan hisoblaymiz
VELOCITY_THRESHOLD = 1.2         # bbox-balandlik/soniya (tana uzunligiga nisbatan)
ASPECT_DROP_RATIO = 0.6          # joriy H/W nisbati so'nggi maksimumning shuncha ulushidan past bo'lsa
ASPECT_HISTORY_SEC = 1.0
VISIBILITY_MIN = 0.5
TRACK_TTL_FRAMES = 15            # shuncha kadr ko'rinmasa, track state o'chiriladi

# Parallel inference thread pool (GPU inference thread-safe, GIL inference bilan)
# 2 worker: bitta fire_model uchun, bitta fall_model uchun
_EXECUTOR: Optional[ThreadPoolExecutor] = None

# CUDA Streams — fire va fall modellari uchun alohida GPU stream
_STREAM_FIRE: Optional[Any] = None
_STREAM_FALL: Optional[Any] = None

_FIRE_MODEL: Optional[YOLO] = None
_FALL_MODEL: Optional[YOLO] = None


def _get_executor() -> ThreadPoolExecutor:
    """Global ThreadPoolExecutor, ikkita modell uchun parallel inference."""
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="detector")
        print("[PARALLEL] ThreadPoolExecutor (2 workers) ishga tushirildi.")
    return _EXECUTOR


def _get_cuda_streams():
    """CUDA mavjud bo'lsa har model uchun alohida stream qaytaradi."""
    global _STREAM_FIRE, _STREAM_FALL
    if DEVICE == "cuda" and _STREAM_FIRE is None:
        _STREAM_FIRE = torch.cuda.Stream(device=DEVICE)
        _STREAM_FALL = torch.cuda.Stream(device=DEVICE)
        print("[PARALLEL] Olov va Yiqilish modellari uchun alohida CUDA Stream'lar yaratildi.")
    return _STREAM_FIRE, _STREAM_FALL


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


def classify_posture(
    torso_angle: float,
    standing_threshold: float = STANDING_ANGLE_DEG,
    lying_threshold: float = LYING_ANGLE_DEG,
) -> str:
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
    track_id: Optional[int] = None


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


@dataclass
class TrackState:
    """Har bir YOLO track_id uchun alohida holat."""
    angle_history: deque = field(default_factory=lambda: deque(maxlen=ANGLE_SMOOTH_WINDOW))
    hip_history: deque = field(default_factory=lambda: deque(maxlen=30))   # (frame_idx, hip_y_norm, bbox_h)
    aspect_history: deque = field(default_factory=lambda: deque(maxlen=30))  # (frame_idx, aspect_ratio)
    falling_count: int = 0
    fall_detected: bool = False
    last_seen_frame: int = 0


ProgressCallback = Callable[[int, int], None]


def _vertical_velocity(hip_history: deque, fps: int) -> float:
    """Son markazining bbox-balandligiga nisbatan normallashtirilgan vertikal tezligi."""
    if len(hip_history) < 2:
        return 0.0
    window_frames = max(1, int(VELOCITY_WINDOW_SEC * fps))
    cur_idx, cur_y, cur_h = hip_history[-1]
    ref = None
    for item in reversed(hip_history):
        if cur_idx - item[0] >= window_frames:
            ref = item
            break
    if ref is None:
        ref = hip_history[0]
    ref_idx, ref_y, ref_h = ref
    dt = (cur_idx - ref_idx) / fps
    if dt <= 0 or cur_h <= 0:
        return 0.0
    dy_norm = abs(cur_y - ref_y) / cur_h
    return dy_norm / dt


def _aspect_dropped(aspect_history: deque, fps: int) -> bool:
    """So'nggi ~1s ichidagi maksimal aspekt nisbatiga nisbatan keskin qisqarish bormi."""
    if len(aspect_history) < 2:
        return False
    window_frames = max(1, int(ASPECT_HISTORY_SEC * fps))
    cur_idx, cur_ratio = aspect_history[-1]
    past = [r for (i, r) in aspect_history if cur_idx - i <= window_frames]
    if not past:
        return False
    recent_max = max(past)
    if recent_max <= 0:
        return False
    return cur_ratio < recent_max * ASPECT_DROP_RATIO


def _infer_fire(
    model: YOLO,
    frame: np.ndarray,
    conf: float,
    stream: Any,
) -> list:
    """
    Olov/Tutun modeli inferensi — alohida CUDA stream da ishlaydi.
    Ultralytics YOLO thread-safe: har chaqiruvda mustaqil forward pass bajaradi.
    """
    if stream is not None:
        with torch.cuda.stream(stream):
            results = model(
                frame,
                verbose=False,
                conf=conf,
                device=DEVICE,
                half=USE_HALF,
            )
            torch.cuda.current_stream().synchronize()
    else:
        results = model(
            frame,
            verbose=False,
            conf=conf,
            device=DEVICE,
            half=USE_HALF,
        )
    return results


def _infer_fall(
    model: YOLO,
    frame: np.ndarray,
    stream: Any,
) -> list:
    """
    Yiqilish modeli (ByteTrack) inferensi — alohida CUDA stream da ishlaydi.
    persist=True: YOLO tracker internal state'ni saqlab qoladi (track_id uchun zarur).
    """
    if stream is not None:
        with torch.cuda.stream(stream):
            try:
                results = model.track(
                    frame,
                    persist=True,
                    classes=[0],
                    tracker="bytetrack.yaml",
                    verbose=False,
                    device=DEVICE,
                    half=USE_HALF,
                )
            except Exception:
                results = model(
                    frame,
                    classes=[0],
                    verbose=False,
                    device=DEVICE,
                    half=USE_HALF,
                )
            torch.cuda.current_stream().synchronize()
    else:
        try:
            results = model.track(
                frame,
                persist=True,
                classes=[0],
                tracker="bytetrack.yaml",
                verbose=False,
                device=DEVICE,
                half=USE_HALF,
            )
        except Exception:
            results = model(
                frame,
                classes=[0],
                verbose=False,
                device=DEVICE,
                half=USE_HALF,
            )
    return results


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
      2) YOLOv8 ByteTrack Tracker + MediaPipe Pose (v2) orqali odamlarni track_id
         bilan kuzatib, burchak + tezlik/aspekt signallari asosida yiqilishni aniqlaydi.
      3) [PARALLEL] Ikkala model ThreadPoolExecutor + torch.cuda.Stream orqali
         BITTA KADR uchun BIR VAQTDA parallel ishlaydi (DGX Spark 128GB optimized).
      4) Kadr ustida ramkalar va 'FALL DETECTED' yozuvini chizadi (ogohlantirish bannerisiz).
      5) Barcha hodisalar ro'yxatini qaytaradi.
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
    executor = _get_executor()
    stream_fire, stream_fall = _get_cuda_streams()

    fire_detected_global = False
    smoke_detected_global = False
    fall_detected_global = False
    events: list[DetectionEvent] = []

    track_states: dict[int, TrackState] = {}
    last_event_sec = -1.0
    frame_idx = 0

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose, torch.inference_mode():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            timestamp_sec = frame_idx / fps
            any_fall_this_frame = False

            if progress_callback:
                progress_callback(frame_idx, total_frames)

            # ================================================================
            # PARALLEL INFERENCE: Ikkala model BIR VAQTDA parallel ishlatiladi
            # executor.submit() darhol qaytadi (non-blocking), ikkala future
            # parallel GPU stream'larda bajariladi, so'ng future.result() kutiladi
            # ================================================================
            future_fire: Future = executor.submit(
                _infer_fire, fire_model, frame.copy(), conf_threshold, stream_fire
            )
            future_fall: Future = executor.submit(
                _infer_fall, fall_model, frame.copy(), stream_fall
            )

            # Ikkala natijani kutamiz (har ikki CUDA stream bajarilishini kutadi)
            fire_results = future_fire.result()
            person_results = future_fall.result()

            # ----------------------------------------------------------------
            # --- 1. FIRE & SMOKE natijalari qayta ishlash ---
            # ----------------------------------------------------------------
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

            # (fall natijalari future_fall.result() dan allaqachon keldi)

            for result in person_results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                boxes_ids = boxes.id if hasattr(boxes, "id") and boxes.id is not None else list(range(len(boxes)))

                for bbox, track_id_t in zip(boxes.xyxy, boxes_ids):
                    track_id = int(track_id_t)
                    x1, y1, x2, y2 = map(int, bbox)
                    x1, y1 = max(x1, 0), max(y1, 0)
                    x2, y2 = min(x2, width), min(y2, height)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    state = track_states.setdefault(track_id, TrackState())
                    state.last_seen_frame = frame_idx

                    person_bbox = frame[y1:y2, x1:x2]
                    if person_bbox.size == 0:
                        continue
                    person_bbox_rgb = cv2.cvtColor(person_bbox, cv2.COLOR_BGR2RGB)
                    pose_results = pose.process(person_bbox_rgb)

                    posture = None
                    if pose_results.pose_landmarks:
                        landmarks = pose_results.pose_landmarks.landmark
                        h, w = person_bbox.shape[:2]

                        needed = [
                            mp_pose.PoseLandmark.LEFT_SHOULDER,
                            mp_pose.PoseLandmark.RIGHT_SHOULDER,
                            mp_pose.PoseLandmark.LEFT_HIP,
                            mp_pose.PoseLandmark.RIGHT_HIP,
                        ]
                        if all(landmarks[lm.value].visibility >= VISIBILITY_MIN for lm in needed):
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

                            raw_angle = calculate_angle(hip_center, shoulder_center)
                            state.angle_history.append(raw_angle)
                            smoothed_angle = sum(state.angle_history) / len(state.angle_history)
                            posture = classify_posture(smoothed_angle)

                            # Tezlik va aspekt signallari uchun tarixni yangilash
                            bbox_h = y2 - y1
                            bbox_w = x2 - x1
                            hip_y_abs = y1 + hip_center[1]
                            state.hip_history.append((frame_idx, hip_y_abs, bbox_h))
                            aspect_ratio = bbox_h / max(bbox_w, 1)
                            state.aspect_history.append((frame_idx, aspect_ratio))

                            velocity = _vertical_velocity(state.hip_history, fps)
                            aspect_drop = _aspect_dropped(state.aspect_history, fps)
                            fast_signal = (velocity >= VELOCITY_THRESHOLD) or aspect_drop

                            mp_drawing.draw_landmarks(
                                person_bbox, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                            )
                            label = f"ID{track_id}: {posture} ({smoothed_angle:.1f} deg)"
                            cv2.putText(
                                frame,
                                label,
                                (x1, max(y1 - 10, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255, 255, 255),
                                2,
                                cv2.LINE_AA,
                            )

                            if posture == "Falling":
                                state.falling_count += 1
                            else:
                                state.falling_count = 0

                            min_f = FALL_MIN_FRAMES_RATIO * fps
                            max_f = FALL_MAX_FRAMES_RATIO * fps
                            angle_condition = min_f <= state.falling_count <= max_f

                            if angle_condition and fast_signal:
                                if not state.fall_detected:
                                    events.append(
                                        DetectionEvent(
                                            frame_index=frame_idx,
                                            timestamp_sec=round(timestamp_sec, 2),
                                            event_type="FALL DETECTED",
                                            confidence=0.95,
                                            box=[x1, y1, x2, y2],
                                            track_id=track_id,
                                        )
                                    )
                                state.fall_detected = True
                                fall_detected_global = True

                            if posture == "Standing":
                                state.fall_detected = False

                            if state.fall_detected:
                                any_fall_this_frame = True

                    frame[y1:y2, x1:x2] = person_bbox
                    box_color = (0, 0, 255) if (posture == "Falling" and state.fall_detected) else (255, 120, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Yo'qolgan track'larni tozalaymiz
            stale_ids = [tid for tid, st in track_states.items() if frame_idx - st.last_seen_frame > TRACK_TTL_FRAMES]
            for tid in stale_ids:
                del track_states[tid]

            # Agar bu kadrda yiqilish sezilgan bo'lsa, "FALL DETECTED" yozuvi
            if any_fall_this_frame:
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

            # Olov va tutun hodisalarini yozish (har 0.8s da)
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
        print("Foydalanish: python combined_detector_v2.py <input_video> <output_video>")
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
        track_str = f" [ID{ev.track_id}]" if ev.track_id is not None else ""
        print(f"  -> [{ev.event_type}]{track_str} frame {ev.frame_index}, {ev.timestamp_sec:.2f}s ({ev.confidence*100:.0f}%)")
