from __future__ import annotations
import cv2
import numpy as np

MIN_BLUR_SCORE = 45.0
MIN_CONTRAST = 18.0
MIN_SATURATION = 10.0
MAX_UNIFORM_REGION_RATIO = 0.28
MAX_SOLID_HUE_RATIO = 0.45
MAX_COLOR_CHAOS_SCORE = 35.0

def _check_blur(gray: np.ndarray) -> bool:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) >= MIN_BLUR_SCORE

def _check_contrast(gray: np.ndarray) -> bool:
    return float(gray.std()) >= MIN_CONTRAST

def _check_saturation(frame_bgr: np.ndarray) -> bool:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean()) >= MIN_SATURATION

def _check_uniform_blocks(frame_bgr: np.ndarray) -> bool:
    h, w = frame_bgr.shape[:2]
    block_h, block_w = h // 4, w // 4
    if block_h < 4 or block_w < 4:
        return True
    total_blocks = 0
    uniform_blocks = 0
    for r in range(4):
        for c in range(4):
            y1, y2 = r * block_h, (r + 1) * block_h
            x1, x2 = c * block_w, (c + 1) * block_w
            block = frame_bgr[y1:y2, x1:x2]
            total_blocks += 1
            if float(block.std()) < 6.0:
                uniform_blocks += 1
    return (uniform_blocks / total_blocks) < MAX_UNIFORM_REGION_RATIO

def _check_solid_color_dominance(frame_bgr: np.ndarray) -> bool:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    hue = hsv[:, :, 0]

    saturated_mask = (saturation > 80) & (value > 40)
    total_pixels = frame_bgr.shape[0] * frame_bgr.shape[1]
    saturated_count = int(saturated_mask.sum())

    if saturated_count < total_pixels * 0.05:
        return True

    hue_vals = hue[saturated_mask]
    hist, _ = np.histogram(hue_vals, bins=18, range=(0, 180))
    dominant_bin_count = int(hist.max())
    dominant_ratio = dominant_bin_count / total_pixels

    return dominant_ratio < MAX_SOLID_HUE_RATIO

def _check_color_chaos(frame_bgr: np.ndarray) -> bool:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32)
    h_diff = np.abs(np.diff(s, axis=1))
    return float(h_diff.mean()) < MAX_COLOR_CHAOS_SCORE

def is_frame_valid(frame_bgr: np.ndarray) -> bool:
    if frame_bgr is None or frame_bgr.size == 0:
        return False
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if not _check_blur(gray):
        return False
    if not _check_contrast(gray):
        return False
    if not _check_saturation(frame_bgr):
        return False
    if not _check_uniform_blocks(frame_bgr):
        return False
    if not _check_solid_color_dominance(frame_bgr):
        return False
    if not _check_color_chaos(frame_bgr):
        return False
    return True
