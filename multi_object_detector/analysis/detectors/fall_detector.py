from __future__ import annotations
import os
import cv2
import sys
import math
import subprocess
import numpy as np
import supervision as sv
from pathlib import Path
from ultralytics import YOLO
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional, Any

from multi_object_detector import config
from multi_object_detector.system_logger import sys_logger

KEYPOINT_NOSE = 0
KEYPOINT_LEFT_EYE = 1
KEYPOINT_RIGHT_EYE = 2
KEYPOINT_LEFT_EAR = 3
KEYPOINT_RIGHT_EAR = 4
KEYPOINT_LEFT_SHOULDER = 5
KEYPOINT_RIGHT_SHOULDER = 6
KEYPOINT_LEFT_ELBOW = 7
KEYPOINT_RIGHT_ELBOW = 8
KEYPOINT_LEFT_WRIST = 9
KEYPOINT_RIGHT_WRIST = 10
KEYPOINT_LEFT_HIP = 11
KEYPOINT_RIGHT_HIP = 12
KEYPOINT_LEFT_KNEE = 13
KEYPOINT_RIGHT_KNEE = 14
KEYPOINT_LEFT_ANKLE = 15
KEYPOINT_RIGHT_ANKLE = 16

SKELETON_CONNECTIONS = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6)
]

BASE_DIR = Path(__file__).resolve().parent.parent
LOCAL_CALIB_PATH = str(BASE_DIR / "models" / "calib.npz")
_UNDISTORT_CACHE: dict[tuple[int, int], Optional[tuple[np.ndarray, np.ndarray]]] = {}


def _get_calib_path(custom_path: Optional[str] = None) -> Optional[str]:
    if custom_path and os.path.exists(custom_path):
        return custom_path
    if hasattr(config, "CALIB_PATH") and os.path.exists(config.CALIB_PATH):
        return config.CALIB_PATH
    if os.path.exists(LOCAL_CALIB_PATH):
        return LOCAL_CALIB_PATH
    return None


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
    if max(bbox_w, bbox_h) < 35:
        return False
    if (bbox_w * bbox_h) < 600:
        return False
    return True


def is_pose_reliable(scores: np.ndarray, threshold: float = 0.25) -> bool:
    if scores is None or len(scores) < 13:
        return False
    if scores.ndim > 1:
        scores = scores[0]
    sh_vis = max(float(scores[KEYPOINT_LEFT_SHOULDER]), float(scores[KEYPOINT_RIGHT_SHOULDER]))
    hip_vis = max(float(scores[KEYPOINT_LEFT_HIP]), float(scores[KEYPOINT_RIGHT_HIP]))
    return sh_vis >= threshold and hip_vis >= threshold


_PERSON_MODEL: Optional[Any] = None


def get_model() -> Any:
    global _PERSON_MODEL
    if _PERSON_MODEL is None:
        try:
            sys_logger.info("FallDetector", "RFDETRLarge yuklanmoqda...")
            from rfdetr import RFDETRLarge
            _PERSON_MODEL = RFDETRLarge()
            sys_logger.info("FallDetector", "RFDETRLarge muvaffaqiyatli yuklandi.")
        except Exception as e:
            sys_logger.warning("FallDetector", f"RFDETRLarge yuklanmadi: {e}. YOLO modeliga o'tilmoqda.", exc=e)
            try:
                _PERSON_MODEL = YOLO(config.PERSON_MODEL_PATH)
                sys_logger.info("FallDetector", f"YOLO person modeli yuklandi: {config.PERSON_MODEL_PATH}")
            except Exception as e2:
                sys_logger.error("FallDetector", f"YOLO modelini ham yuklab bo'lmadi: {e2}", exc=e2)
    return _PERSON_MODEL


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


def check_5frame_transition(posture_history: deque) -> bool:
    if len(posture_history) < 5:
        return False
    p5 = list(posture_history)[-5:]
    if p5[-1] not in ("Falling", "Lying Down"):
        return False
    p_rank = {"Standing": 0, "Falling": 1, "Lying Down": 2}
    r = [p_rank.get(p, 0) for p in p5]
    start_min = min(r[0], r[1])
    end_max = max(r[3], r[4])
    downward_sum = sum(r[2:])
    return (start_min < end_max) and (downward_sum >= 3)


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
    posture_history: deque = field(default_factory=lambda: deque(maxlen=30))
    hip_history: deque = field(default_factory=lambda: deque(maxlen=30))
    aspect_history: deque = field(default_factory=lambda: deque(maxlen=30))
    falling_count: int = 0
    falled_count: int = 0
    is_falled: bool = False
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
    ref_idx, ref_y, _ = ref
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


KEYPOINT_COLORS = [
    (0, 255, 255),
    (255, 255, 0),
    (255, 255, 0),
    (255, 0, 255),
    (255, 0, 255),
    (0, 255, 0),
    (0, 255, 0),
    (0, 200, 255),
    (255, 200, 0),
    (0, 100, 255),
    (255, 100, 0),
    (0, 255, 128),
    (0, 255, 128),
    (0, 255, 255),
    (255, 255, 0),
    (0, 165, 255),
    (255, 165, 0),
]

LIMB_COLORS = [
    (0, 255, 255),
    (0, 255, 255),
    (255, 255, 0),
    (255, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 165, 255),
    (255, 0, 255),
    (0, 100, 255),
    (255, 0, 128),
    (255, 255, 0),
    (255, 255, 0),
    (255, 255, 0),
    (255, 255, 0),
    (255, 255, 0),
    (255, 255, 0),
    (255, 255, 0),
]


def draw_person_detection(
    image: np.ndarray,
    detections: Any,
    conf_thr: float = 0.20,
    in_zone_flags: Optional[Any] = None,
) -> np.ndarray:
    if detections is None or len(detections) == 0:
        return image

    for i in range(len(detections)):
        try:
            conf = float(detections.confidence[i]) if detections.confidence is not None else 1.0
            if conf < conf_thr:
                continue

            if in_zone_flags is not None and i < len(in_zone_flags) and in_zone_flags[i]:
                base_color = (0, 0, 255)
            else:
                base_color = (0, 200, 60)

            tid = None
            if hasattr(detections, "tracker_id") and detections.tracker_id is not None:
                try:
                    tid = int(detections.tracker_id[i])
                except Exception:
                    pass

            has_mask = False
            if hasattr(detections, "mask") and detections.mask is not None:
                try:
                    mask = detections.mask[i]
                    if mask is not None and len(mask) >= 3:
                        pts = np.array(mask, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(image, [pts], isClosed=True, color=base_color, thickness=2, lineType=cv2.LINE_AA)
                        overlay = image.copy()
                        cv2.fillPoly(overlay, [pts], color=base_color)
                        cv2.addWeighted(overlay, 0.18, image, 0.82, 0, image)
                        cx, cy = int(np.mean(mask[:, 0])), int(np.mean(mask[:, 1]))
                        has_mask = True
                except Exception:
                    pass

            if not has_mask:
                x1, y1, x2, y2 = map(int, detections.xyxy[i])
                h_img, w_img = image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)
                cv2.rectangle(image, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0, 0, 0), 2)
                cv2.rectangle(image, (x1, y1), (x2, y2), base_color, 2, cv2.LINE_AA)
                cx, cy = (x1 + x2) // 2, y1

            label_parts = []
            if tid is not None:
                label_parts.append(f"ID{tid}")
            label_parts.append(f"person {conf:.2f}")
            label = " ".join(label_parts)

            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            lx = cx - tw // 2
            ly = max(th + 4, cy - 4)
            cv2.rectangle(image, (lx - 2, ly - th - 4), (lx + tw + 2, ly + bl + 1), base_color, -1)
            cv2.putText(image, label, (lx, ly - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        except Exception:
            continue
    return image


def draw_pose_skeleton(
    image: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    kpt_thr: float = 0.20,
    draw_coords: bool = False,
) -> np.ndarray:
    if keypoints is None or scores is None or len(keypoints) == 0:
        return image

    kpts = keypoints[0] if keypoints.ndim == 3 else keypoints
    scs = scores[0] if scores.ndim == 2 else scores

    for idx, (p1_idx, p2_idx) in enumerate(SKELETON_CONNECTIONS):
        if p1_idx < len(kpts) and p2_idx < len(kpts):
            if scs[p1_idx] >= kpt_thr and scs[p2_idx] >= kpt_thr:
                pt1 = (int(round(kpts[p1_idx][0])), int(round(kpts[p1_idx][1])))
                pt2 = (int(round(kpts[p2_idx][0])), int(round(kpts[p2_idx][1])))
                color = LIMB_COLORS[idx] if idx < len(LIMB_COLORS) else (0, 255, 255)
                cv2.line(image, pt1, pt2, color, 2, cv2.LINE_AA)

    for idx, (x, y) in enumerate(kpts):
        if scs[idx] >= kpt_thr:
            px, py = int(round(x)), int(round(y))
            color = KEYPOINT_COLORS[idx] if idx < len(KEYPOINT_COLORS) else (0, 255, 0)
            cv2.circle(image, (px, py), 4, color, -1, cv2.LINE_AA)
            cv2.circle(image, (px, py), 5, (255, 255, 255), 1, cv2.LINE_AA)
            if draw_coords:
                coord_text = f"({px},{py})"
                cv2.putText(image, coord_text, (px + 4, py - 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def draw_pose_skeleton_global(
    frame: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    offset_x: int,
    offset_y: int,
    kpt_thr: float = 0.20,
    draw_coords: bool = False,
) -> np.ndarray:
    if keypoints is None or scores is None or len(keypoints) == 0:
        return frame

    kpts = keypoints[0] if keypoints.ndim == 3 else keypoints
    scs = scores[0] if scores.ndim == 2 else scores

    for idx, (p1_idx, p2_idx) in enumerate(SKELETON_CONNECTIONS):
        if p1_idx < len(kpts) and p2_idx < len(kpts):
            if scs[p1_idx] >= kpt_thr and scs[p2_idx] >= kpt_thr:
                pt1 = (int(round(kpts[p1_idx][0])) + offset_x,
                       int(round(kpts[p1_idx][1])) + offset_y)
                pt2 = (int(round(kpts[p2_idx][0])) + offset_x,
                       int(round(kpts[p2_idx][1])) + offset_y)
                color = LIMB_COLORS[idx] if idx < len(LIMB_COLORS) else (0, 255, 255)
                cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)

    for idx, (x, y) in enumerate(kpts):
        if scs[idx] >= kpt_thr:
            px = int(round(x)) + offset_x
            py = int(round(y)) + offset_y
            color = KEYPOINT_COLORS[idx] if idx < len(KEYPOINT_COLORS) else (0, 255, 0)
            cv2.circle(frame, (px, py), 4, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 5, (255, 255, 255), 1, cv2.LINE_AA)
            if draw_coords:
                coord_text = f"({px},{py})"
                cv2.putText(frame, coord_text, (px + 4, py - 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.32, (220, 220, 220), 1, cv2.LINE_AA)
    return frame


class RTMPoseEstimator:
    def __init__(
        self,
        mode: str = "balanced",
        backend: str = "onnxruntime",
        device: str = "cpu",
        onnx_model: Optional[str] = None,
    ):
        self.mode = mode
        self.backend = backend
        self.device = device
        self.onnx_model = onnx_model
        self._body = None
        self._init_model()

    def _init_model(self):
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        user_home = Path.home()
        possible_rtmlib_paths = [
            project_root / "rtmlib",
            project_root.parent / "rtmlib",
            user_home / "rtmlib",
            user_home / "AppData" / "Roaming" / "Python" / "Python312" / "site-packages",
            user_home / "AppData" / "Local" / "Programs" / "Python" / "Python312" / "Lib" / "site-packages",
            Path("C:/Python312/Lib/site-packages"),
            Path("C:/Users/ADMIN/AppData/Local/Programs/Python/Python312/Lib/site-packages"),
            Path("C:/Users/ADMIN/AppData/Roaming/Python/Python312/site-packages"),
        ]
        for p in possible_rtmlib_paths:
            if p.exists() and str(p) not in sys.path:
                sys.path.insert(0, str(p))

        try:
            sys_logger.info("RTMPose", f"RTMPose initialized (mode={self.mode}, backend={self.backend}, device={self.device})")
            from rtmlib import Body, RTMPose
            if self.onnx_model and os.path.exists(self.onnx_model):
                self._body = RTMPose(
                    onnx_model=self.onnx_model,
                    backend=self.backend,
                    device=self.device,
                )
                sys_logger.info("RTMPose", f"RTMPose loaded ONNX model: {self.onnx_model}")
            else:
                self._body = Body(
                    mode=self.mode,
                    backend=self.backend,
                    device=self.device,
                )
                sys_logger.info("RTMPose", "rtmlib Body muvaffaqiyatli yuklandi.")
        except Exception as e:
            sys_logger.warning("RTMPose", f"rtmlib Body init ogohlantirish: {e}", exc=e)
            try:
                from rtmlib import Body
                self._body = Body(mode=self.mode, backend="opencv", device="cpu")
                sys_logger.info("RTMPose", "rtmlib Body OpenCV CPU fallback yuklandi.")
            except Exception as e2:
                sys_logger.error("RTMPose", f"RTMPose fallback init ham muvaffaqiyatsiz bo'ldi: {e2}", exc=e2)

    def __call__(self, img: np.ndarray, bboxes: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        if self._body is None:
            return np.empty((0, 17, 2)), np.empty((0, 17))
        try:
            if hasattr(self._body, "pose_model"):
                if bboxes is None:
                    h, w = img.shape[:2]
                    bboxes = np.array([[0, 0, w, h]], dtype=np.float32)
                return self._body.pose_model(img, bboxes=bboxes)
            if bboxes is not None:
                return self._body(img, bboxes=bboxes)
            return self._body(img)
        except Exception:
            try:
                return self._body(img)
            except Exception:
                return np.empty((0, 17, 2)), np.empty((0, 17))


def create_pose_instance(
    mode: Optional[str] = None,
    backend: Optional[str] = None,
    device: Optional[str] = None,
    model_path: Optional[str] = None,
) -> RTMPoseEstimator:
    m = mode or getattr(config, "RTMPOSE_MODE", "balanced")
    b = backend or getattr(config, "RTMPOSE_BACKEND", "onnxruntime")
    d = device or getattr(config, "RTMPOSE_DEVICE", "cpu")
    p = model_path or getattr(config, "RTMPOSE_MODEL_PATH", None)
    return RTMPoseEstimator(mode=m, backend=b, device=d, onnx_model=p)


def process_video(
    input_path: str,
    output_path: str,
    progress_callback: Optional[ProgressCallback] = None,
) -> ProcessingResult:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video ochilmadi: {input_path}")
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    fall_events: list[FallEvent] = []
    frame_idx = 0
    track_states: dict[int, TrackState] = {}

    model = get_model()
    tracker = sv.ByteTrack()
    pose = create_pose_instance()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        frame, new_events, any_fall = process_fall_frame(
            frame=frame,
            frame_idx=frame_idx,
            fps=fps,
            model=model,
            pose=pose,
            track_states=track_states,
            tracker=tracker,
        )

        for ev in new_events:
            fall_events.append(FallEvent(
                frame_index=ev.get("frame_index", frame_idx),
                timestamp_sec=ev.get("timestamp_sec", frame_idx / max(fps, 1)),
                track_id=ev.get("track_id", 0)
            ))

        if any_fall:
            cv2.rectangle(frame, (0, 0), (width, 42), (0, 0, 220), -1)
            cv2.putText(frame, "ALARM: FALL DETECTED (YIQILISH)", (20, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(frame)
        if progress_callback is not None:
            progress_callback(frame_idx, total_frames)

    cap.release()
    out.release()
    return ProcessingResult(output_path=output_path, fps=fps, width=width, height=height, total_frames=frame_idx, fall_detected=len(fall_events) > 0, fall_events=fall_events)


def process_fall_frame(
    frame: np.ndarray,
    frame_idx: int,
    fps: int,
    model: Any,
    pose: RTMPoseEstimator,
    track_states: dict[int, TrackState],
    tracker: Optional[Any] = None,
) -> tuple[np.ndarray, list[dict], bool]:
    frame = undistort_frame(frame)
    width, height = frame.shape[1], frame.shape[0]
    events: list[dict] = []
    any_fall_this_frame = False
    posture = None

    if tracker is None:
        if not hasattr(model, "_tracker"):
            model._tracker = sv.ByteTrack()
        tracker = model._tracker

    if hasattr(model, "predict") and not isinstance(model, YOLO):
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = model.predict(frame_rgb, threshold=config.FALL_PERSON_CONF_THRESHOLD)
        except TypeError:
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = model.predict(frame_rgb)
            except Exception as _e2:
                sys_logger.error("process_fall_frame", f"RF-DETR predict() error: {_e2}", exc=_e2)
                return frame, events, False
        except Exception as _e:
            sys_logger.error("process_fall_frame", f"RF-DETR predict() error: {_e}", exc=_e)
            return frame, events, False
    else:
        try:
            res = model(frame, classes=[0], verbose=False)[0]
            detections = sv.Detections.from_ultralytics(res)
        except Exception as _e:
            sys_logger.error("process_fall_frame", f"YOLO predict error: {_e}", exc=_e)
            return frame, events, False

    if detections is None or not hasattr(detections, "class_id") or detections.class_id is None:
        sys_logger.warning("process_fall_frame", f"Frame {frame_idx}: Detections object missing class_id or None (total det count={len(detections) if detections is not None else 0})")
        return frame, events, False

    raw_det_count = len(detections)
    try:
        unique_class_ids = np.unique(detections.class_id) if hasattr(detections, "class_id") and detections.class_id is not None else []
        person_mask = (detections.class_id == 0) | (detections.class_id == 1)
        persons = detections[person_mask]
        person_det_count = len(persons)
    except Exception as _e:
        sys_logger.error("process_fall_frame", f"person_mask filtering error: {_e}", exc=_e)
        return frame, events, False

    try:
        tracked_persons = tracker.update_with_detections(persons)
        track_count = len(tracked_persons)
    except Exception as _e:
        sys_logger.error("process_fall_frame", f"ByteTrack update error: {_e}", exc=_e)
        tracked_persons = persons
        track_count = len(tracked_persons)

    if frame_idx % 30 == 0 or raw_det_count > 0:
        sys_logger.info("process_fall_frame", f"Frame {frame_idx} [{width}x{height}]: Raw Detections={raw_det_count} (class_ids={list(unique_class_ids)}), Persons={person_det_count}, Tracked={track_count}")

    if len(tracked_persons) > 0 and tracked_persons.tracker_id is not None:
        for _, (bbox, track_id_t) in enumerate(zip(tracked_persons.xyxy, tracked_persons.tracker_id)):
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

            posture = "Standing"
            smoothed_angle = 0.0

            if is_valid_person_crop(bbox_w, bbox_h):
                person_bbox = frame[y1:y2, x1:x2].copy()
                kpts, scs = pose(person_bbox)

                if len(kpts) > 0 and len(scs) > 0:
                    kpt = kpts[0]
                    sc = scs[0]

                    draw_pose_skeleton_global(frame, kpts, scs, offset_x=x1, offset_y=y1, kpt_thr=0.20, draw_coords=False)

                    if is_pose_reliable(sc):
                        l_sh = kpt[KEYPOINT_LEFT_SHOULDER]
                        r_sh = kpt[KEYPOINT_RIGHT_SHOULDER]
                        l_hip = kpt[KEYPOINT_LEFT_HIP]
                        r_hip = kpt[KEYPOINT_RIGHT_HIP]

                        l_sh_sc = sc[KEYPOINT_LEFT_SHOULDER]
                        r_sh_sc = sc[KEYPOINT_RIGHT_SHOULDER]
                        l_hip_sc = sc[KEYPOINT_LEFT_HIP]
                        r_hip_sc = sc[KEYPOINT_RIGHT_HIP]

                        if l_sh_sc >= 0.20 and r_sh_sc >= 0.20:
                            shoulder_center = ((l_sh[0] + r_sh[0]) / 2.0, (l_sh[1] + r_sh[1]) / 2.0)
                        elif l_sh_sc >= 0.20:
                            shoulder_center = (float(l_sh[0]), float(l_sh[1]))
                        else:
                            shoulder_center = (float(r_sh[0]), float(r_sh[1]))

                        if l_hip_sc >= 0.20 and r_hip_sc >= 0.20:
                            hip_center = ((l_hip[0] + r_hip[0]) / 2.0, (l_hip[1] + r_hip[1]) / 2.0)
                        elif l_hip_sc >= 0.20:
                            hip_center = (float(l_hip[0]), float(l_hip[1]))
                        else:
                            hip_center = (float(r_hip[0]), float(r_hip[0]))

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

                        sh_pt = (int(x1 + shoulder_center[0]), int(y1 + shoulder_center[1]))
                        hip_pt = (int(x1 + hip_center[0]), int(y1 + hip_center[1]))
                        cv2.line(frame, sh_pt, hip_pt, (0, 0, 255), 3, cv2.LINE_AA)
                        cv2.circle(frame, sh_pt, 5, (0, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(frame, hip_pt, 5, (255, 0, 255), -1, cv2.LINE_AA)

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
                                events.append({"track_id": track_id, "frame_index": frame_idx, "timestamp_sec": round(frame_idx / max(fps, 1), 2)})
                            state.fall_detected = True

                        if posture == "Standing":
                            state.fall_detected = False
                            state.confirm_count = 0
                            state.standing_frames += 1
                        else:
                            state.standing_frames = 0

            if state.fall_detected:
                any_fall_this_frame = True
                box_color = (0, 0, 255)
                badge_text = f"FALL! ID{track_id} ({smoothed_angle:.0f}deg)"
            elif posture == "Falling":
                box_color = (0, 140, 255)
                badge_text = f"ID{track_id}: Falling ({smoothed_angle:.0f}deg)"
            elif posture == "Lying Down":
                box_color = (0, 140, 255)
                badge_text = f"ID{track_id}: Lying ({smoothed_angle:.0f}deg)"
            else:
                box_color = (0, 200, 60)
                badge_text = f"ID{track_id}: Standing ({smoothed_angle:.0f}deg)"

            cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0, 0, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

            (tw, th), bl = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            ty = max(th + 6, y1 - 4)
            cv2.rectangle(frame, (x1, ty - th - 5), (x1 + tw + 6, ty + bl + 1), box_color, -1)
            cv2.rectangle(frame, (x1, ty - th - 5), (x1 + tw + 6, ty + bl + 1), (0, 0, 0), 1)
            cv2.putText(frame, badge_text, (x1 + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

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