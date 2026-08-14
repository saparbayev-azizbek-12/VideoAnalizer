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


def check_solid_color_corruption(
    frame_bgr: np.ndarray,
    block_size: int = 32,
    max_flat_block_ratio: float = 0.20,
    max_consecutive_flat_rows_ratio: float = 0.15,
) -> Tuple[bool, Optional[str]]:
    h, w = frame_bgr.shape[:2]
    target_w, target_h = 320, 180
    small = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)

    b, g, r = small[:, :, 0], small[:, :, 1], small[:, :, 2]
    green_mask = (g > 95) & (b < 45) & (r < 45)
    green_ratio = float(np.sum(green_mask)) / (target_w * target_h)
    if green_ratio > 0.08:
        return False, f"Buzilgan kadr: Yashil ekran (Green artifact) aniqlandi ({green_ratio*100:.1f}% maydon)"

    bs = 16
    n_by = target_h // bs
    n_bx = target_w // bs

    flat_blocks = 0
    total_blocks = n_by * n_bx
    row_flat_counts = np.zeros(n_by, dtype=np.int32)

    for by in range(n_by):
        y1, y2 = by * bs, (by + 1) * bs
        for bx in range(n_bx):
            x1, x2 = bx * bs, (bx + 1) * bs
            blk = small[y1:y2, x1:x2]
            std_b = np.std(blk[:, :, 0])
            std_g = np.std(blk[:, :, 1])
            std_r = np.std(blk[:, :, 2])
            if std_b < 2.5 and std_g < 2.5 and std_r < 2.5:
                flat_blocks += 1
                row_flat_counts[by] += 1

    flat_ratio = flat_blocks / max(1, total_blocks)
    if flat_ratio > max_flat_block_ratio:
        return False, f"Buzilgan kadr: Sun'iy tekis bloklar juda ko'p ({flat_ratio*100:.1f}%)"

    max_consec_flat_rows = 0
    curr_consec = 0
    for cnt in row_flat_counts:
        if cnt >= (n_bx * 0.75):
            curr_consec += 1
            max_consec_flat_rows = max(max_consec_flat_rows, curr_consec)
        else:
            curr_consec = 0

    consec_ratio = max_consec_flat_rows / max(1, n_by)
    if consec_ratio > max_consecutive_flat_rows_ratio:
        return False, f"Buzilgan kadr: {consec_ratio*100:.1f}% qatorlar bir tekis rangda muzlagan/yo'qolgan"

    return True, None


def check_decode_artifact_noise(
    frame_bgr: np.ndarray,
    band_ratio: float = 0.22,
) -> Tuple[bool, Optional[str]]:
    h, w = frame_bgr.shape[:2]
    band_h = max(16, int(h * band_ratio))

    top_band = frame_bgr[:band_h, :, :]
    bottom_band = frame_bgr[h - band_h:, :, :]

    top_hsv = cv2.cvtColor(top_band, cv2.COLOR_BGR2HSV)
    top_sat = top_hsv[:, :, 1].astype(np.float32)

    row_means = np.mean(top_sat, axis=1)
    row_diffs = np.abs(np.diff(row_means))
    max_row_jump = float(np.max(row_diffs)) if len(row_diffs) > 0 else 0.0

    sat_mean = float(np.mean(top_sat))
    sat_std = float(np.std(top_sat))

    bot_hsv = cv2.cvtColor(bottom_band, cv2.COLOR_BGR2HSV)
    bot_sat_mean = float(np.mean(bot_hsv[:, :, 1]))

    if sat_mean > 65.0 and sat_std > 50.0 and (sat_mean > bot_sat_mean * 1.8):
        return False, f"Buzilgan kadr: Yuqori qismda H.264 decode shovqini/glitch aniqlandi (sat_mean={sat_mean:.1f}, sat_std={sat_std:.1f})"

    if max_row_jump > 45.0 and sat_mean > 50.0:
        return False, f"Buzilgan kadr: Qatorlararo kuchli gorizontal uzilish (artifact tearing jump={max_row_jump:.1f})"

    return True, None


def check_frame_stream_validity(frame_bgr: np.ndarray) -> Tuple[bool, Optional[str]]:
    ok, reason = check_frame_structure(frame_bgr)
    if not ok:
        return False, reason

    ok, reason = check_solid_color_corruption(frame_bgr)
    if not ok:
        return False, reason

    ok, reason = check_decode_artifact_noise(frame_bgr)
    if not ok:
        return False, reason

    return True, None


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

        is_valid, reason = check_frame_stream_validity(frame)
        if not is_valid:
            self.corrupt_frames += 1
            self.consecutive_corrupt += 1
            return False, reason

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
