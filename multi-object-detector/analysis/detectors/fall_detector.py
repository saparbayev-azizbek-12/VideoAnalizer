from __future__ import annotations
import os
import cv2
import sys
import math
import subprocess
import numpy as np
import mediapipe as mp
from pathlib import Path
from ultralytics import YOLO
from collections import deque
from typing import Callable, Optional
from dataclasses import dataclass, field

import config

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

_YOLO_MODEL: Optional[YOLO] = None
BASE_DIR = Path(__file__).resolve().parent.parent
LOCAL_MODEL_PATH = str(BASE_DIR / "models" / "yolov8l.pt")


def _get_person_model_path(custom_path: Optional[str] = None) -> str:
    if custom_path and os.path.exists(custom_path):
        return custom_path
    if hasattr(config, "PERSON_MODEL_PATH") and os.path.exists(config.PERSON_MODEL_PATH):
        return config.PERSON_MODEL_PATH
    if os.path.exists(LOCAL_MODEL_PATH):
        return LOCAL_MODEL_PATH
    return "yolov8l.pt"


STANDING_ANGLE_DEG = 22.0       
SITTING_ANGLE_DEG = 40.0        
LYING_ANGLE_DEG = 58.0         
ANGLE_SMOOTH_WINDOW = 7         
VELOCITY_WINDOW_SEC = 0.3
VELOCITY_THRESHOLD = 1.2
ASPECT_DROP_RATIO = 0.6
ASPECT_HISTORY_SEC = 1.0
LYING_CONFIRM_SEC = 0.6          
RECOVERY_CONFIRM_SEC = 0.8      
VISIBILITY_MIN = 0.65            
TRACK_TTL_FRAMES = 15
MIN_PERSON_HEIGHT_PX = 90         
MIN_PERSON_AREA_PX = 3500         
MIN_PERSON_HEIGHT_RATIO = 0.10    
MAX_OCCLUSION_RATIO = 0.40   
BASELINE_WINDOW_SEC = 8.0
BASELINE_MIN_SAMPLES = 10
BASELINE_DEVIATION_DEG = 28.0

ALL_POSE_LANDMARKS = list(mp_pose.PoseLandmark)


def is_valid_person_crop(bbox_w: int, bbox_h: int, frame_w: int, frame_h: int) -> bool:
    if bbox_h < MIN_PERSON_HEIGHT_PX:
        return False
    if (bbox_w * bbox_h) < MIN_PERSON_AREA_PX:
        return False
    if frame_h > 0 and (bbox_h / frame_h) < MIN_PERSON_HEIGHT_RATIO:
        return False
    return True


def compute_occlusion_ratio(landmarks) -> float:
    total = len(ALL_POSE_LANDMARKS)
    hidden = sum(1 for lm in ALL_POSE_LANDMARKS if landmarks[lm.value].visibility < VISIBILITY_MIN)
    return hidden / total


def is_pose_reliable(landmarks) -> bool:
    core_landmarks = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP,
    ]
    core_visible = all(landmarks[lm.value].visibility >= VISIBILITY_MIN for lm in core_landmarks)
    if not core_visible:
        return False
    return compute_occlusion_ratio(landmarks) <= MAX_OCCLUSION_RATIO


def estimate_sitting(landmarks, w: int, h: int) -> Optional[bool]:
    knee_lms = [mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE]
    hip_lms = [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP]
    if any(landmarks[lm.value].visibility < VISIBILITY_MIN for lm in knee_lms + hip_lms):
        return None
    hip_x = (landmarks[hip_lms[0].value].x + landmarks[hip_lms[1].value].x) / 2 * w
    hip_y = (landmarks[hip_lms[0].value].y + landmarks[hip_lms[1].value].y) / 2 * h
    knee_x = (landmarks[knee_lms[0].value].x + landmarks[knee_lms[1].value].x) / 2 * w
    knee_y = (landmarks[knee_lms[0].value].y + landmarks[knee_lms[1].value].y) / 2 * h
    dy = abs(knee_y - hip_y)
    dx = abs(knee_x - hip_x)
    thigh_angle_from_vertical = math.degrees(math.atan2(dx, dy)) if (dx or dy) else 0.0
    return thigh_angle_from_vertical > 45.0  


def get_model() -> YOLO:
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        _YOLO_MODEL = YOLO(_get_person_model_path())
    return _YOLO_MODEL


def calculate_angle(shoulder_center: tuple[float, float], hip_center: tuple[float, float]) -> float:
    dx = hip_center[0] - shoulder_center[0]
    dy = hip_center[1] - shoulder_center[1]
    angle = math.degrees(math.atan2(dx, dy)) if (dx or dy) else 0.0
    return abs(angle)


def classify_posture(torso_angle: float, sitting: Optional[bool]) -> str:
    if torso_angle > LYING_ANGLE_DEG:
        return "Lying Down"
    if sitting:
        return "Sitting" if torso_angle < SITTING_ANGLE_DEG else "Falling"
    return "Standing" if torso_angle < STANDING_ANGLE_DEG else "Falling"


@dataclass
class FallEvent:
    frame_index: int
    timestamp_sec: float
    track_id: int


@dataclass
class ProcessingResult:
    output_path: str
    fps: int
    width: int
    height: int
    total_frames: int
    fall_detected: bool
    fall_events: list[FallEvent] = field(default_factory=list)


@dataclass
class TrackState:
    angle_history: deque = field(default_factory=lambda: deque(maxlen=ANGLE_SMOOTH_WINDOW))
    hip_history: deque = field(default_factory=lambda: deque(maxlen=30))
    aspect_history: deque = field(default_factory=lambda: deque(maxlen=30))
    baseline_history: deque = field(default_factory=lambda: deque(maxlen=500))
    lying_count: int = 0
    recovery_count: int = 0
    fast_signal_ttl: int = 0
    fall_detected: bool = False
    last_seen_frame: int = 0


ProgressCallback = Callable[[int, int], None]


def _vertical_velocity(hip_history: deque, fps: int) -> float:
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


def process_fall_frame(
    frame: np.ndarray,
    frame_idx: int,
    fps: int,
    model: YOLO,
    pose,
    track_states: dict[int, TrackState],
) -> tuple[np.ndarray, list[dict], bool]:
    width, height = frame.shape[1], frame.shape[0]
    events: list[dict] = []
    any_fall_this_frame = False

    try:
        results = model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml", verbose=False)
    except Exception:
        results = model(frame, classes=[0], verbose=False)

    for result in results:
        boxes = result.boxes
        if boxes.id is None:
            continue
        for bbox, track_id_t in zip(boxes.xyxy, boxes.id):
            track_id = int(track_id_t)
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, width), min(y2, height)
            if x2 <= x1 or y2 <= y1:
                continue

            state = track_states.setdefault(track_id, TrackState())
            state.last_seen_frame = frame_idx
            bbox_w, bbox_h = x2 - x1, y2 - y1

            if not is_valid_person_crop(bbox_w, bbox_h, width, height):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
                continue

            person_bbox = frame[y1:y2, x1:x2]
            person_bbox_rgb = cv2.cvtColor(person_bbox, cv2.COLOR_BGR2RGB)
            person_results = pose.process(person_bbox_rgb)

            if person_results.pose_landmarks:
                landmarks = person_results.pose_landmarks.landmark
                if is_pose_reliable(landmarks):
                    h, w = person_bbox.shape[:2]
                    shoulders = [
                        (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h),
                        (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h),
                    ]
                    hips = [
                        (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h),
                        (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h),
                    ]
                    shoulder_center = ((shoulders[0][0] + shoulders[1][0]) / 2, (shoulders[0][1] + shoulders[1][1]) / 2)
                    hip_center = ((hips[0][0] + hips[1][0]) / 2, (hips[0][1] + hips[1][1]) / 2)

                    raw_angle = calculate_angle(shoulder_center, hip_center)
                    state.angle_history.append(raw_angle)
                    smoothed_angle = sum(state.angle_history) / len(state.angle_history)

                    sitting = estimate_sitting(landmarks, w, h)
                    posture = classify_posture(smoothed_angle, sitting)

                    hip_y_abs = y1 + hip_center[1]
                    state.hip_history.append((frame_idx, hip_y_abs, bbox_h))
                    aspect_ratio = bbox_h / max(bbox_w, 1)
                    state.aspect_history.append((frame_idx, aspect_ratio))
                    velocity = _vertical_velocity(state.hip_history, fps)
                    aspect_drop = _aspect_dropped(state.aspect_history, fps)
                    fast_signal = (velocity >= VELOCITY_THRESHOLD) or aspect_drop
                    if fast_signal:
                        state.fast_signal_ttl = int(LYING_CONFIRM_SEC * fps) + int(VELOCITY_WINDOW_SEC * fps) + 1
                    elif state.fast_signal_ttl > 0:
                        state.fast_signal_ttl -= 1

                    if posture in ("Standing", "Sitting") and not fast_signal:
                        state.baseline_history.append((frame_idx, smoothed_angle))
                    window_frames = int(BASELINE_WINDOW_SEC * fps)
                    while state.baseline_history and frame_idx - state.baseline_history[0][0] > window_frames:
                        state.baseline_history.popleft()
                    if len(state.baseline_history) >= BASELINE_MIN_SAMPLES:
                        baseline_angle = sum(a for _, a in state.baseline_history) / len(state.baseline_history)
                        deviation_ok = (smoothed_angle - baseline_angle) >= BASELINE_DEVIATION_DEG
                    else:
                        deviation_ok = True

                    mp_drawing.draw_landmarks(person_bbox, person_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    label = f"ID{track_id}: {posture} ({smoothed_angle:.1f} deg)"
                    cv2.putText(frame, label, (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                    state.lying_count = state.lying_count + 1 if posture == "Lying Down" else 0
                    lying_confirm_frames = LYING_CONFIRM_SEC * fps
                    recovery_confirm_frames = RECOVERY_CONFIRM_SEC * fps

                    reached_lying = state.lying_count >= lying_confirm_frames
                    had_fast_signal = state.fast_signal_ttl > 0

                    if reached_lying and had_fast_signal and deviation_ok:
                        if not state.fall_detected:
                            events.append({"track_id": track_id, "frame_index": frame_idx, "timestamp_sec": round(frame_idx / fps, 2)})
                        state.fall_detected = True
                        state.recovery_count = 0
                    elif posture in ("Standing", "Sitting"):
                        state.recovery_count += 1
                        if state.recovery_count >= recovery_confirm_frames:
                            state.fall_detected = False
                    else:
                        state.recovery_count = 0

                    if state.fall_detected:
                        any_fall_this_frame = True
                else:
                    cv2.putText(frame, f"ID{track_id}: occluded", (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)

            frame[y1:y2, x1:x2] = person_bbox
            box_color = (0, 0, 255) if state.fall_detected else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    stale_ids = [tid for tid, st in track_states.items() if frame_idx - st.last_seen_frame > TRACK_TTL_FRAMES]
    for tid in stale_ids:
        del track_states[tid]

    if any_fall_this_frame:
        cv2.putText(frame, "FALL DETECTED", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)

    return frame, events, any_fall_this_frame


def process_video(
    input_path: str,
    output_path: str,
    progress_callback: Optional[ProgressCallback] = None,
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
    fall_events: list[FallEvent] = []
    frame_idx = 0
    track_states: dict[int, TrackState] = {}
    model = get_model()

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            frame, frame_events, _ = process_fall_frame(frame, frame_idx, fps, model, pose, track_states)
            for ev in frame_events:
                fall_events.append(FallEvent(frame_index=ev["frame_index"], timestamp_sec=ev["timestamp_sec"], track_id=ev["track_id"]))
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


def create_tracking_model(model_path: str = "yolov8l.pt") -> YOLO:
    return YOLO(model_path)


def create_pose_instance():
    return mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)


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
        print(f"  -> ID{ev.track_id}, frame {ev.frame_index}, {ev.timestamp_sec:.2f}s")