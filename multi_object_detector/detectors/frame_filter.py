from __future__ import annotations
import cv2
import time
import numpy as np
from typing import Optional, Tuple, Dict, Any


def check_frame_structure(frame: np.ndarray) -> Tuple[bool, Optional[str]]:
    if frame is None:
        return False, "Kadr mavjud emas (None)"
    if not isinstance(frame, np.ndarray):
        return False, "Kadr numpy massivi emas"
    if frame.size == 0:
        return False, "Kadr o'lchami bo'sh (size == 0)"
    if frame.ndim != 3 or frame.shape[2] != 3:
        return False, f"Kadr kanallari noto'g'ri: shape={frame.shape}"
    h, w = frame.shape[:2]
    if h < 32 or w < 32:
        return False, f"Kadr o'lchami juda kichik ({w}x{h})"
    return True, None


def check_chroma_green_glitch(frame_bgr: np.ndarray) -> Tuple[bool, Optional[str]]:
    target_w, target_h = 320, 180
    small = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)

    b = small[:, :, 0].astype(np.float32)
    g = small[:, :, 1].astype(np.float32)
    r = small[:, :, 2].astype(np.float32)

    green_chroma = (g > 110) & (b < 60) & (r < 60) & (g > (b + r) * 1.3)
    if not np.any(green_chroma):
        return True, None

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    flat_green = green_chroma & (gradient_mag < 8.0)
    flat_green_count = int(np.sum(flat_green))
    total_pixels = target_w * target_h
    ratio = flat_green_count / total_pixels

    if ratio > 0.015:
        return False, f"Buzilgan kadr: H.264/RTSP yashil dekodlash artefakti (Green glitch {ratio*100:.1f}%)"

    return True, None


def check_missing_slice_corruption(frame_bgr: np.ndarray) -> Tuple[bool, Optional[str]]:
    target_w, target_h = 320, 180
    small = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    slice_h = 16
    n_slices = target_h // slice_h

    corrupted_slices = 0
    for i in range(n_slices):
        strip = gray[i * slice_h : (i + 1) * slice_h, :]
        strip_std = float(np.std(strip))
        strip_mean = float(np.mean(strip))

        if strip_std < 1.2 and (strip_mean < 5.0 or abs(strip_mean - 128.0) < 6.0 or strip_mean > 250.0):
            corrupted_slices += 1

    if corrupted_slices >= 2:
        return False, f"Buzilgan kadr: Yo'qolgan paketlar o'rni to'ldirilgan {corrupted_slices} ta bo'sh tasma (Missing slice concealment)"

    overall_std = float(np.std(gray))
    if overall_std < 1.8:
        return False, f"Buzilgan kadr: Butunlay tekis bo'sh kadr (std={overall_std:.1f})"

    return True, None


def check_frame_stream_validity(frame_bgr: np.ndarray) -> Tuple[bool, Optional[str]]:
    ok, reason = check_frame_structure(frame_bgr)
    if not ok:
        return False, reason

    ok, reason = check_chroma_green_glitch(frame_bgr)
    if not ok:
        return False, reason

    ok, reason = check_missing_slice_corruption(frame_bgr)
    if not ok:
        return False, reason

    return True, None


def is_frame_valid(frame_bgr: np.ndarray) -> bool:
    valid, _ = check_frame_stream_validity(frame_bgr)
    return valid


class StreamCorruptionFilter:
    def __init__(self, camera_id: str = "default", max_frozen_frames: int = 400):
        self.camera_id = camera_id
        self.max_frozen_frames = max_frozen_frames
        self.total_frames = 0
        self.corrupt_frames = 0
        self.consecutive_corrupt = 0
        self.consecutive_frozen = 0
        self.last_valid_time: float = time.time()
        self._prev_sample: Optional[np.ndarray] = None

    def check_frame(self, frame: np.ndarray) -> Tuple[bool, Optional[str]]:
        self.total_frames += 1

        is_valid, reason = check_frame_stream_validity(frame)
        if not is_valid:
            self.corrupt_frames += 1
            self.consecutive_corrupt += 1
            return False, reason

        if frame is not None and frame.size > 0:
            sample = cv2.resize(frame, (32, 18))
            if self._prev_sample is not None and np.array_equal(sample, self._prev_sample):
                self.consecutive_frozen += 1
                if self.consecutive_frozen > self.max_frozen_frames:
                    return False, f"Oqim muzlab qolgan ({self.consecutive_frozen} ta bir xil kadr)"
            else:
                self.consecutive_frozen = 0
                self._prev_sample = sample

        self.consecutive_corrupt = 0
        self.last_valid_time = time.time()
        return True, None

    def is_valid(self, frame: np.ndarray) -> bool:
        valid, _ = self.check_frame(frame)
        return valid

    def get_stats(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "total_frames": self.total_frames,
            "corrupt_frames": self.corrupt_frames,
            "corrupt_ratio": (self.corrupt_frames / self.total_frames) if self.total_frames > 0 else 0.0,
            "consecutive_corrupt": self.consecutive_corrupt,
            "consecutive_frozen": self.consecutive_frozen,
            "last_valid_time": self.last_valid_time,
        }

    def reset(self) -> None:
        self.total_frames = 0
        self.corrupt_frames = 0
        self.consecutive_corrupt = 0
        self.consecutive_frozen = 0
        self._prev_sample = None
