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
from multi_object_detector import config


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

_YOLO_MODEL: Optional[YOLO] = None
BASE_DIR = Path(__file__).resolve().parent.parent
LOCAL_CALIB_PATH = str(BASE_DIR / "models" / "calib.npz")

def _get_calib_path(custom_path: Optional[str] = None) -> Optional[str]:
    if custom_path and os.path.exists(custom_path):
        return custom_path
    if hasattr(config, "CALIB_PATH") and os.path.exists(config.CALIB_PATH):
        return config.CALIB_PATH
    if os.path.exists(LOCAL_CALIB_PATH):
        return LOCAL_CALIB_PATH
    return None

_UNDISTORT_CACHE: dict[tuple[int, int], Optional[tuple[np.ndarray, np.ndarray]]] = {}

def get_undistort_maps(frame_w: int, frame_h: int, calib_path: Optional[str] = None):
    key = (frame_w, frame_h)
    if key in _UNDISTORT_CACHE:
        return _UNDISTORT_CACHE[key]

    path = _get_calib_path(calib_path)
    if path is None:
        _UNDISTORT_CACHE[key] = None
        return None

    data = np.load(path)
    K = data["camera_matrix"].astype(np.float64).copy()
    dist = data["dist_coeffs"].astype(np.float64)
    calib_w, calib_h = data["image_size"]

    scale_x = frame_w / float(calib_w)
    scale_y = frame_h / float(calib_h)
    K[0, 0] *= scale_x
    K[0, 2] *= scale_x
    K[1, 1] *= scale_y
    K[1, 2] *= scale_y

    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (frame_w, frame_h), alpha=0)
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, new_K, (frame_w, frame_h), cv2.CV_16SC2)
    _UNDISTORT_CACHE[key] = (map1, map2)
    return _UNDISTORT_CACHE[key]

def undistort_frame(frame: np.ndarray, calib_path: Optional[str] = None) -> np.ndarray:
    h, w = frame.shape[:2]
    maps = get_undistort_maps(w, h, calib_path)
    if maps is None:
        return frame
    map1, map2 = maps
    return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

STANDING_ANGLE_DEG = 22.0
LYING_ANGLE_DEG = 62.0
ANGLE_SMOOTH_WINDOW = 11

FALL_MIN_FRAMES_RATIO = 0.28
FALL_MAX_FRAMES_RATIO = 0.90

VELOCITY_WINDOW_SEC = 0.4
VELOCITY_THRESHOLD = 2.2
ASPECT_DROP_RATIO = 0.42
ASPECT_HISTORY_SEC = 1.5

VISIBILITY_MIN = 0.70
TRACK_TTL_FRAMES = 15

MIN_PERSON_HEIGHT_PX = 90
MIN_PERSON_AREA_PX = 5000
MAX_OCCLUSION_RATIO = 0.15
MIN_VISIBLE_LANDMARKS_RATIO = 1.0 - MAX_OCCLUSION_RATIO

FALL_CONFIRM_FRAMES = 4
STANDING_MIN_FRAMES = 10

def is_valid_person_crop(bbox_w: int, bbox_h: int) -> bool:
    if bbox_h < MIN_PERSON_HEIGHT_PX:
        return False
    if (bbox_w * bbox_h) < MIN_PERSON_AREA_PX:
        return False
    return True

def is_pose_reliable(landmarks) -> bool:
    key_landmarks = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE,
        mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE,
        mp_pose.PoseLandmark.RIGHT_ANKLE,
        mp_pose.PoseLandmark.LEFT_WRIST,
        mp_pose.PoseLandmark.RIGHT_WRIST,
    ]
    visible_key_count = sum(
        1 for lm in key_landmarks if landmarks[lm.value].visibility >= VISIBILITY_MIN
    )
    visible_ratio = visible_key_count / len(key_landmarks)
    needed_core = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP,
    ]
    core_visible = all(landmarks[lm.value].visibility >= VISIBILITY_MIN for lm in needed_core)
    return (visible_ratio >= MIN_VISIBLE_LANDMARKS_RATIO) and core_visible

def get_model() -> YOLO:
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        _YOLO_MODEL = YOLO(config.PERSON_MODEL_PATH)
    return _YOLO_MODEL

def calculate_angle(hip_center: tuple[float, float], shoulder_center: tuple[float, float]) -> float:
    dy = hip_center[1] - shoulder_center[1]
    dx = hip_center[0] - shoulder_center[0]
    angle = math.atan2(dy, dx)
    return abs(90 - np.degrees(angle))

def classify_posture(torso_angle: float,
                      standing_threshold: float = STANDING_ANGLE_DEG,
                      lying_threshold: float = LYING_ANGLE_DEG) -> str:
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
    falling_count: int = 0
    fall_detected: bool = False
    last_seen_frame: int = 0
    confirm_count: int = 0
    standing_frames: int = 0

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

    with mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75,
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            any_fall_this_frame = False
            frame = undistort_frame(frame)
            try:
                results = get_model().track(frame, persist=True, classes=[0], tracker="bytetrack.yaml", verbose=False)
            except Exception:
                results = get_model()(frame, classes=[0], verbose=False)
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
                    bbox_w = x2 - x1
                    bbox_h = y2 - y1
                    if not is_valid_person_crop(bbox_w, bbox_h):
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
                        continue
                    person_bbox = frame[y1:y2, x1:x2]
                    person_bbox_rgb = cv2.cvtColor(person_bbox, cv2.COLOR_BGR2RGB)
                    person_results = pose.process(person_bbox_rgb)
                    posture = None
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
                            smoothed_angle = float(np.median(state.angle_history))
                            posture = classify_posture(smoothed_angle)
                            hip_y_abs = y1 + hip_center[1]
                            state.hip_history.append((frame_idx, hip_y_abs, bbox_h))
                            aspect_ratio = bbox_h / max(bbox_w, 1)
                            state.aspect_history.append((frame_idx, aspect_ratio))
                            velocity = _vertical_velocity(state.hip_history, fps)
                            aspect_drop = _aspect_dropped(state.aspect_history, fps)
                            fast_signal = (velocity >= VELOCITY_THRESHOLD) and aspect_drop
                            mp_drawing.draw_landmarks(person_bbox, person_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                            label = f"ID{track_id}: {posture} ({smoothed_angle:.1f}°) v={velocity:.2f}"
                            cv2.putText(frame, label, (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                            if posture in ("Falling", "Lying Down"):
                                state.falling_count += 1
                            else:
                                state.falling_count = 0
                            min_f = FALL_MIN_FRAMES_RATIO * fps
                            max_f = FALL_MAX_FRAMES_RATIO * fps
                            angle_condition = min_f <= state.falling_count <= max_f
                            can_trigger = (
                                angle_condition
                                and fast_signal
                                and state.standing_frames >= STANDING_MIN_FRAMES
                            )
                            if can_trigger:
                                state.confirm_count += 1
                            else:
                                state.confirm_count = max(0, state.confirm_count - 1)
                            if state.confirm_count >= FALL_CONFIRM_FRAMES:
                                if not state.fall_detected:
                                    fall_events.append(FallEvent(frame_index=frame_idx, timestamp_sec=frame_idx / fps, track_id=track_id))
                                state.fall_detected = True
                            if posture == "Standing":
                                state.fall_detected = False
                                state.confirm_count = 0
                                state.standing_frames += 1
                            else:
                                state.standing_frames = 0
                            if state.fall_detected:
                                any_fall_this_frame = True
                    frame[y1:y2, x1:x2] = person_bbox
                    box_color = (0, 0, 255) if (posture == "Falling" and state.fall_detected) else (255, 0, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            stale_ids = [tid for tid, st in track_states.items() if frame_idx - st.last_seen_frame > TRACK_TTL_FRAMES]
            for tid in stale_ids:
                del track_states[tid]

            if any_fall_this_frame:
                cv2.putText(frame, "FALL DETECTED", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
            out.write(frame)
            if progress_callback is not None:
                progress_callback(frame_idx, total_frames)

    cap.release()
    out.release()
    return ProcessingResult(output_path=output_path, fps=fps, width=width, height=height, total_frames=frame_idx, fall_detected=len(fall_events) > 0, fall_events=fall_events)

def create_pose_instance():
    return mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75,
    )

def process_fall_frame(
    frame: np.ndarray,
    frame_idx: int,
    fps: int,
    model: YOLO,
    pose,
    track_states: dict[int, TrackState],
) -> tuple[np.ndarray, list[dict], bool]:
    frame = undistort_frame(frame)
    width, height = frame.shape[1], frame.shape[0]
    events: list[dict] = []
    any_fall_this_frame = False
    posture = None

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
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            if not is_valid_person_crop(bbox_w, bbox_h):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
                continue
            person_bbox = frame[y1:y2, x1:x2]
            person_bbox_rgb = cv2.cvtColor(person_bbox, cv2.COLOR_BGR2RGB)
            person_results = pose.process(person_bbox_rgb)
            posture = None
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
                    smoothed_angle = float(np.median(state.angle_history))
                    posture = classify_posture(smoothed_angle)
                    hip_y_abs = y1 + hip_center[1]
                    state.hip_history.append((frame_idx, hip_y_abs, bbox_h))
                    aspect_ratio = bbox_h / max(bbox_w, 1)
                    state.aspect_history.append((frame_idx, aspect_ratio))
                    velocity = _vertical_velocity(state.hip_history, fps)
                    aspect_drop = _aspect_dropped(state.aspect_history, fps)
                    fast_signal = (velocity >= VELOCITY_THRESHOLD) and aspect_drop
                    mp_drawing.draw_landmarks(person_bbox, person_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    label = f"ID{track_id}: {posture} ({smoothed_angle:.1f}°) v={velocity:.2f}"
                    cv2.putText(frame, label, (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                    if posture in ("Falling", "Lying Down"):
                        state.falling_count += 1
                    else:
                        state.falling_count = 0
                    min_f = FALL_MIN_FRAMES_RATIO * fps
                    max_f = FALL_MAX_FRAMES_RATIO * fps
                    angle_condition = min_f <= state.falling_count <= max_f
                    can_trigger = (
                        angle_condition
                        and fast_signal
                        and state.standing_frames >= STANDING_MIN_FRAMES
                    )
                    if can_trigger:
                        state.confirm_count += 1
                    else:
                        state.confirm_count = max(0, state.confirm_count - 1)
                    if state.confirm_count >= FALL_CONFIRM_FRAMES:
                        if not state.fall_detected:
                            events.append({"track_id": track_id, "frame_index": frame_idx, "timestamp_sec": round(frame_idx / fps, 2)})
                        state.fall_detected = True
                    if posture == "Standing":
                        state.fall_detected = False
                        state.confirm_count = 0
                        state.standing_frames += 1
                    else:
                        state.standing_frames = 0
                    if state.fall_detected:
                        any_fall_this_frame = True
            frame[y1:y2, x1:x2] = person_bbox
            box_color = (0, 0, 255) if (posture == "Falling" and state.fall_detected) else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    stale_ids = [tid for tid, st in track_states.items() if frame_idx - st.last_seen_frame > TRACK_TTL_FRAMES]
    for tid in stale_ids:
        del track_states[tid]

    if any_fall_this_frame:
        cv2.putText(frame, "FALL DETECTED", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)

    return frame, events, any_fall_this_frame

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