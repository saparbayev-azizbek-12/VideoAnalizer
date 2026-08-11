from __future__ import annotations
import sys
import cv2
import subprocess
import numpy as np
from pathlib import Path
from collections import deque
from typing import Callable, Optional
from dataclasses import dataclass, field

BASE_DIR = Path(__file__).resolve().parent.parent.parent
VIT_MODEL_DIR = BASE_DIR / "analysis" / "models" / "vit-fire-detection"

LABEL_FIRE = "Fire"
LABEL_SMOKE = "Smoke"
LABEL_NORMAL = "Normal"


_vit_processor = None
_vit_model = None
HF_REPO = "EdBianchi/vit-fire-detection"

def _get_model_source() -> str:
    bin_file = VIT_MODEL_DIR / "pytorch_model.bin"
    if VIT_MODEL_DIR.exists() and bin_file.exists() and bin_file.stat().st_size > 10_000_000:
        return str(VIT_MODEL_DIR)
    return HF_REPO

def get_model():
    global _vit_processor, _vit_model
    if _vit_processor is None or _vit_model is None:
        from transformers import ViTForImageClassification, ViTImageProcessor
        source = _get_model_source()
        _vit_processor = ViTImageProcessor.from_pretrained(source)
        _vit_model = ViTForImageClassification.from_pretrained(source)
        _vit_model.eval()
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _vit_model = _vit_model.to(device)
        except Exception:
            pass
    return _vit_processor, _vit_model

def _classify_frame(processor, model, frame_bgr: np.ndarray) -> tuple[str, float]:
    import torch
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    from PIL import Image as PILImage
    pil_img = PILImage.fromarray(rgb)
    inputs = processor(images=pil_img, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
    id2label = model.config.id2label
    results = [(id2label[i], p) for i, p in enumerate(probs)]
    best_label, best_conf = max(results, key=lambda x: x[1])
    return best_label, best_conf

@dataclass
class FireEvent:
    frame_index: int
    timestamp_sec: float
    event_type: str
    confidence: float
    box: list[int]

@dataclass
class ProcessingResult:
    output_path: str
    fps: int
    width: int
    height: int
    total_frames: int
    fire_detected: bool
    smoke_detected: bool
    events: list[FireEvent] = field(default_factory=list)

ProgressCallback = Callable[[int, int], None]

class TemporalValidator:
    def __init__(self, window: int = 6, min_hits: int = 3):
        self.window = window
        self.min_hits = min_hits
        self._fire_hist: deque[bool] = deque(maxlen=window)
        self._smoke_hist: deque[bool] = deque(maxlen=window)

    def update(self, has_fire: bool, has_smoke: bool) -> tuple[bool, bool]:
        self._fire_hist.append(has_fire)
        self._smoke_hist.append(has_smoke)
        confirmed_fire = sum(self._fire_hist) >= self.min_hits
        confirmed_smoke = sum(self._smoke_hist) >= self.min_hits
        return confirmed_fire, confirmed_smoke

    def reset(self) -> None:
        self._fire_hist.clear()
        self._smoke_hist.clear()

def _draw_banner(frame: np.ndarray, width: int, confirmed_fire: bool, confirmed_smoke: bool) -> str:
    if confirmed_fire and confirmed_smoke:
        status_text = "ALARM: FIRE & SMOKE DETECTED"
        event_type = "FIRE & SMOKE"
        bg_color = (46, 87, 228)
    elif confirmed_fire:
        status_text = "ALARM: FIRE DETECTED"
        event_type = "FIRE"
        bg_color = (46, 87, 228)
    else:
        status_text = "ALARM: SMOKE DETECTED"
        event_type = "SMOKE"
        bg_color = (65, 164, 217)
    cv2.rectangle(frame, (0, 0), (width, 42), bg_color, -1)
    cv2.putText(frame, status_text, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    return event_type

def process_video(
    input_path: str,
    output_path: str,
    progress_callback: Optional[ProgressCallback] = None,
    fire_conf_threshold: float = 0.70,
    smoke_conf_threshold: float = 0.65,
    confirm_window: int = 6,
    confirm_min_hits: int = 3,
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
    fire_detected_global = False
    smoke_detected_global = False
    events: list[FireEvent] = []
    frame_idx = 0
    last_event_sec = -1.0
    validator = TemporalValidator(window=confirm_window, min_hits=confirm_min_hits)
    processor, model = get_model()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        timestamp_sec = frame_idx / fps
        if progress_callback:
            progress_callback(frame_idx, total_frames)

        label, conf = _classify_frame(processor, model, frame)
        frame_has_fire = label == LABEL_FIRE and conf >= fire_conf_threshold
        frame_has_smoke = label == LABEL_SMOKE and conf >= smoke_conf_threshold

        confirmed_fire, confirmed_smoke = validator.update(frame_has_fire, frame_has_smoke)

        if confirmed_fire or confirmed_smoke:
            event_type = _draw_banner(frame, width, confirmed_fire, confirmed_smoke)
            if confirmed_fire:
                fire_detected_global = True
            if confirmed_smoke:
                smoke_detected_global = True
            if timestamp_sec - last_event_sec >= 0.8:
                events.append(FireEvent(
                    frame_index=frame_idx,
                    timestamp_sec=round(timestamp_sec, 2),
                    event_type=event_type,
                    confidence=round(conf, 2),
                    box=[0, 0, 0, 0],
                ))
                last_event_sec = timestamp_sec

        out.write(frame)

    cap.release()
    out.release()
    return ProcessingResult(
        output_path=output_path, fps=fps, width=width, height=height,
        total_frames=frame_idx, fire_detected=fire_detected_global,
        smoke_detected=smoke_detected_global, events=events,
    )

def detect_fire_frame(
    frame: np.ndarray,
    model=None,
    conf_threshold: float = 0.70,
    fire_conf_threshold: float = 0.70,
    min_color_ratio: float = 0.0,
    validator: Optional[TemporalValidator] = None,
    draw_banner: bool = True,
) -> tuple[np.ndarray, list[dict], bool, bool]:
    if model is None:
        processor, vit_model = get_model()
    else:
        processor, vit_model = model

    label, conf = _classify_frame(processor, vit_model, frame)

    smoke_threshold = conf_threshold * 0.93
    frame_has_fire = label == LABEL_FIRE and conf >= fire_conf_threshold
    frame_has_smoke = label == LABEL_SMOKE and conf >= smoke_threshold

    detections: list[dict] = []
    if frame_has_fire:
        detections.append({"type": "FIRE", "confidence": round(conf, 2), "box": [0, 0, 0, 0]})
    elif frame_has_smoke:
        detections.append({"type": "SMOKE", "confidence": round(conf, 2), "box": [0, 0, 0, 0]})

    if validator is not None:
        confirmed_fire, confirmed_smoke = validator.update(frame_has_fire, frame_has_smoke)
    else:
        confirmed_fire, confirmed_smoke = frame_has_fire, frame_has_smoke

    if (confirmed_fire or confirmed_smoke) and draw_banner:
        _draw_banner(frame, frame.shape[1], confirmed_fire, confirmed_smoke)

    return frame, detections, confirmed_fire, confirmed_smoke

def reencode_for_web(input_path: str, output_path: str) -> None:
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_exe = "ffmpeg"
    cmd = [
        ffmpeg_exe, "-y", "-i", input_path,
        "-c:v", "libx264", "-preset", "veryfast",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        output_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Foydalanish: python fire_detector.py <input_video> <output_video>")
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
    for ev in res.events:
        print(f"  -> [{ev.event_type}] frame {ev.frame_index}, {ev.timestamp_sec:.2f}s ({ev.confidence*100:.0f}%)")