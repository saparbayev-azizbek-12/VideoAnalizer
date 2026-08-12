from __future__ import annotations
import cv2
import time
import numpy as np
from typing import Optional, Tuple, Dict, Any

DEFAULT_BLOCK_SIZE = 16
MAX_CONCEALMENT_BLOCK_RATIO = 0.04
MAX_CONCEALMENT_ROW_RATIO = 0.35
MB_TEAR_JUMP_THRESHOLD = 65.0
MB_TEAR_RATIO_THRESHOLD = 4.5


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
    if h < 16 or w < 16:
        return False, f"Kadr o'lchami juda kichik ({w}x{h})"
    return True, None


def check_ffmpeg_concealment(
    frame_bgr: np.ndarray,
    block_size: int = DEFAULT_BLOCK_SIZE,
    max_corrupt_block_ratio: float = MAX_CONCEALMENT_BLOCK_RATIO,
    max_corrupt_row_ratio: float = MAX_CONCEALMENT_ROW_RATIO,
) -> Tuple[bool, Optional[str]]:
    h, w = frame_bgr.shape[:2]
    nb_y = h // block_size
    nb_x = w // block_size
    if nb_y < 2 or nb_x < 2:
        return True, None

    cropped = frame_bgr[: nb_y * block_size, : nb_x * block_size]
    blocks = cropped.reshape(nb_y, block_size, nb_x, block_size, 3).transpose(0, 2, 1, 3, 4)

    block_std = blocks.std(axis=(2, 3, 4))
    block_mean = blocks.mean(axis=(2, 3, 4))

    is_ffmpeg_gray = (block_std < 1.0) & (np.abs(block_mean - 128.0) <= 3.0)

    total_blocks = nb_y * nb_x
    corrupt_blocks = int(np.sum(is_ffmpeg_gray))
    corrupt_ratio = corrupt_blocks / total_blocks

    if corrupt_ratio >= max_corrupt_block_ratio:
        return False, f"FFmpeg concealment (kulrang) bloklari aniqlandi ({corrupt_ratio * 100:.1f}% bloklar)"

    row_corrupt_ratio = is_ffmpeg_gray.sum(axis=1) / nb_x
    max_row_ratio = float(np.max(row_corrupt_ratio)) if len(row_corrupt_ratio) > 0 else 0.0
    if max_row_ratio >= max_corrupt_row_ratio:
        return False, f"Kadr qatorida oqim uzilishi aniqlandi (qatordagi buzilish: {max_row_ratio * 100:.1f}%)"

    return True, None


def check_slice_tearing(
    gray: np.ndarray,
    block_size: int = DEFAULT_BLOCK_SIZE,
    jump_threshold: float = MB_TEAR_JUMP_THRESHOLD,
    ratio_threshold: float = MB_TEAR_RATIO_THRESHOLD,
) -> Tuple[bool, Optional[str]]:
    h, w = gray.shape[:2]
    if h < block_size * 4 or w < block_size * 4:
        return True, None

    diff_y = np.abs(gray[1:, :].astype(np.int32) - gray[:-1, :].astype(np.int32)).mean(axis=1)
    mb_boundaries = np.arange(block_size - 1, h - 1, block_size)
    if len(mb_boundaries) == 0:
        return True, None

    for b_idx in mb_boundaries:
        b_jump = float(diff_y[b_idx])
        if b_jump < jump_threshold:
            continue

        local_start = max(0, b_idx - block_size // 2)
        local_end = min(len(diff_y), b_idx + block_size // 2 + 1)
        local_diffs = np.delete(diff_y[local_start:local_end], b_idx - local_start)

        if len(local_diffs) == 0:
            continue

        local_baseline = float(np.median(local_diffs))
        if local_baseline < 1.0:
            local_baseline = 1.0

        if (b_jump / local_baseline >= ratio_threshold) and (b_jump >= jump_threshold):
            return False, f"Makroblok uzilishi (slice tearing) aniqlandi (chegara sakrashi: {b_jump:.1f})"

    return True, None


def check_chroma_banding(
    frame_bgr: np.ndarray,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> Tuple[bool, Optional[str]]:
    h, w = frame_bgr.shape[:2]
    nb_y = h // block_size
    if nb_y < 4:
        return True, None

    ycrcb = cv2.cvtColor(frame_bgr[: nb_y * block_size], cv2.COLOR_BGR2YCrCb)

    for ch_idx, ch_name in [(1, "Cr"), (2, "Cb")]:
        ch = ycrcb[:, :, ch_idx]
        stripes = ch.reshape(nb_y, block_size, w)
        stripe_std = stripes.std(axis=(1, 2))
        stripe_mean = stripes.mean(axis=(1, 2))

        corrupt_stripe = (stripe_std < 1.0) & (np.abs(stripe_mean - 128.0) > 115.0)
        if np.any(corrupt_stripe):
            return False, f"G'ayritabiiy rang uzilishi (chroma corruption) aniqlandi ({ch_name} kanali)"

    return True, None


def check_frame_stream_validity(frame_bgr: np.ndarray) -> Tuple[bool, Optional[str]]:
    valid_struct, reason = check_frame_structure(frame_bgr)
    if not valid_struct:
        return False, reason

    valid_conceal, reason = check_ffmpeg_concealment(frame_bgr)
    if not valid_conceal:
        return False, reason

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    valid_tearing, reason = check_slice_tearing(gray)
    if not valid_tearing:
        return False, reason

    valid_chroma, reason = check_chroma_banding(frame_bgr)
    if not valid_chroma:
        return False, reason

    return True, None


def is_frame_valid(frame_bgr: np.ndarray) -> bool:
    valid, _ = check_frame_stream_validity(frame_bgr)
    return valid


class StreamCorruptionFilter:
    def __init__(self, camera_id: str = "default", max_frozen_frames: int = 150):
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

