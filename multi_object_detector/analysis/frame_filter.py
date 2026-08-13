from __future__ import annotations
import cv2
import time
import numpy as np
from typing import Optional, Tuple, Dict, Any


def check_frame_structure(frame: np.ndarray) -> Tuple[bool, Optional[str]]:
    """Faqat asosiy tuzilma tekshiruvi — bo'sh yoki noto'g'ri kadrlarni rad etadi."""
    if frame is None:
        return False, "Kadr mavjud emas (None)"
    if not isinstance(frame, np.ndarray):
        return False, "Kadr numpy massivi emas"
    if frame.size == 0:
        return False, "Kadr o'lchami bo'sh (size == 0)"
    if frame.ndim != 3 or frame.shape[2] != 3:
        return False, f"Kadr kanallari noto'g'ri: shape={frame.shape}"
    h, w = frame.shape[:2]
    if h < 16 or w < 16:
        return False, f"Kadr o'lchami juda kichik ({w}x{h})"
    return True, None


def check_frame_stream_validity(frame_bgr: np.ndarray) -> Tuple[bool, Optional[str]]:
    """
    Minimal tekshiruv: faqat bo'sh yoki noto'g'ri kadrlarni rad etadi.
    RTSP/CCTV kameralardan kelgan haqiqiy kadrlarni rad etmasligi uchun
    murakkab FFmpeg concealment/tearing tekshiruvlari o'chirilgan.
    """
    return check_frame_structure(frame_bgr)


def is_frame_valid(frame_bgr: np.ndarray) -> bool:
    valid, _ = check_frame_stream_validity(frame_bgr)
    return valid


class StreamCorruptionFilter:
    def __init__(self, camera_id: str = "default", max_frozen_frames: int = 300):
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

        is_valid, reason = check_frame_structure(frame)
        if not is_valid:
            self.corrupt_frames += 1
            self.consecutive_corrupt += 1
            return False, reason

        # Muzlab qolgan oqimni aniqlash (kichik namunaviy solishtirish)
        if frame is not None and frame.size > 0:
            sample = cv2.resize(frame, (48, 27))
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
