"""
fall_detector.py

Asl Jupyter notebookdagi (Elderly_Fall_Detection-YOLOv8_MediaPipe.ipynb) mantiq
bitta funksiyaga jamlangan holda: YOLOv8 orqali odamni topish, MediaPipe Pose
orqali tana nuqtalarini aniqlash, gavda burchagi asosida turish/yotish/yiqilish
holatini aniqlash va "FALL DETECTED" ogohlantirishini videoga chizish.

Bu modul FastAPI ilovasi (main.py) tomonidan chaqiriladi, lekin istasangiz
mustaqil skript sifatida ham ishlatishingiz mumkin:

    python fall_detector.py in_video.mp4 out_video.mp4
"""

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np
from ultralytics import YOLO

import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# YOLO modelini modul yuklanganda BIR MARTA o'qiymiz (har so'rovda emas),
# aks holda har video yuklanganda model qayta yuklanib, sekinlashadi.
_YOLO_MODEL: Optional[YOLO] = None


def get_model() -> YOLO:
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        _YOLO_MODEL = YOLO("yolov8l.pt")
    return _YOLO_MODEL


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
class FallEvent:
    frame_index: int
    timestamp_sec: float


@dataclass
class ProcessingResult:
    output_path: str
    fps: int
    width: int
    height: int
    total_frames: int
    fall_detected: bool
    fall_events: list[FallEvent] = field(default_factory=list)


ProgressCallback = Callable[[int, int], None]


def process_video(
    input_path: str,
    output_path: str,
    progress_callback: Optional[ProgressCallback] = None,
) -> ProcessingResult:
    """
    Videoni frame-frame o'qiydi, har frame uchun:
      1) YOLOv8 bilan 'person' klassidagi obyektlarni topadi,
      2) har bir odam uchun MediaPipe Pose bilan tana nuqtalarini chiqaradi,
      3) yelka/son markazlari asosida gavda burchagini hisoblaydi,
      4) Standing / Falling / Lying Down holatini aniqlaydi,
      5) agar ketma-ket "Falling" holati 0.1s-0.5s oralig'ida davom etsa,
         yiqilish deb belgilaydi va "FALL DETECTED" yozuvini chizadi.

    Natija: annotatsiya qilingan video + yiqilish hodisalari ro'yxati.
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

    falling_count = 0
    fall_detected = False
    fall_events: list[FallEvent] = []
    frame_idx = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            results = get_model()(frame, verbose=False)
            for result in results:
                for bbox, cls in zip(result.boxes.xyxy, result.boxes.cls):
                    if int(cls) != 0:  # faqat 'person' klassi
                        continue

                    x1, y1, x2, y2 = map(int, bbox)
                    x1, y1 = max(x1, 0), max(y1, 0)
                    x2, y2 = min(x2, width), min(y2, height)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    person_bbox = frame[y1:y2, x1:x2]
                    person_bbox_rgb = cv2.cvtColor(person_bbox, cv2.COLOR_BGR2RGB)
                    person_results = pose.process(person_bbox_rgb)

                    if person_results.pose_landmarks:
                        landmarks = person_results.pose_landmarks.landmark
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
                            person_bbox, person_results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                        )
                        cv2.putText(frame, f"{posture} ({torso_angle:.1f} deg)", (x1, max(y1 - 10, 15)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                        if posture == "Falling":
                            falling_count += 1
                        else:
                            falling_count = 0

                        if 0.1 * fps <= falling_count <= 0.5 * fps:
                            if not fall_detected:
                                fall_events.append(FallEvent(frame_index=frame_idx, timestamp_sec=frame_idx / fps))
                            fall_detected = True
                        if posture == "Standing":
                            fall_detected = False

                    frame[y1:y2, x1:x2] = person_bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            if fall_detected:
                cv2.putText(frame, "FALL DETECTED", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 0, 255), 3, cv2.LINE_AA)

            out.write(frame)

            if progress_callback is not None:
                progress_callback(frame_idx, total_frames)

    cap.release()
    out.release()

    return ProcessingResult(
        output_path=output_path,
        fps=fps,
        width=width,
        height=height,
        total_frames=frame_idx,
        fall_detected=len(fall_events) > 0,
        fall_events=fall_events,
    )


def reencode_for_web(input_path: str, output_path: str) -> None:
    """
    OpenCV VideoWriter yozgan 'mp4v' video ko'p brauzerlarda ijro etilmaydi.
    Shu sabab ffmpeg yordamida H.264 (libx264) ga qayta kodlaymiz.

    ffmpeg dasturi tizimda alohida o'rnatilmagan bo'lsa ham, 'imageio-ffmpeg'
    paketi (moviepy orqali o'rnatiladi) portativ ffmpeg binaryni ta'minlaydi.
    """
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_exe = "ffmpeg"  # tizimdagi ffmpeg'ga umid qilamiz

    cmd = [
        ffmpeg_exe, "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Foydalanish: python fall_detector.py <input_video> <output_video>")
        sys.exit(1)

    src, dst = sys.argv[1], sys.argv[2]

    def _print_progress(cur: int, total: int) -> None:
        if total:
            pct = cur / total * 100
            print(f"\rQayta ishlanmoqda: {cur}/{total} ({pct:.1f}%)", end="", flush=True)
        else:
            print(f"\rQayta ishlanmoqda: {cur}-frame", end="", flush=True)

    res = process_video(src, dst, progress_callback=_print_progress)
    print()
    print(f"Tayyor: {res.output_path}")
    print(f"Yiqilish aniqlandimi: {res.fall_detected}")
    for ev in res.fall_events:
        print(f"  -> frame {ev.frame_index}, {ev.timestamp_sec:.2f}s")
