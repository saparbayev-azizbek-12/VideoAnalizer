from __future__ import annotations
import os
import sys
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_DIR = os.path.dirname(BASE_DIR)

for env_path in [os.path.join(PARENT_DIR, ".env"), os.path.join(BASE_DIR, ".env")]:
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path)
        break

DETECTORS_DIR = os.path.join(BASE_DIR, "detectors")
if os.path.exists(DETECTORS_DIR) and DETECTORS_DIR not in sys.path:
    sys.path.insert(0, DETECTORS_DIR)

MODELS_DIR = os.path.join(DETECTORS_DIR, "models")
VIT_FIRE_MODEL_DIR = os.path.join(MODELS_DIR, "vit-fire-detection")
VIT_FIRE_HF_REPO = "EdBianchi/vit-fire-detection"
PERSON_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8l.pt")
PERSON_DETECTOR_MODEL = os.getenv("PERSON_DETECTOR_MODEL", "rfdetr_large")
PERSON_CLASS_ID = 0
PPE_MODEL_PATH = os.path.join(MODELS_DIR, "sfchd_yolov8s.pt")

CALIB_PATH = os.path.join(BASE_DIR, "calibration", "calib.npz")

VIT_FIRE_CONF_THRESHOLD = float(os.getenv("VIT_FIRE_CONF_THRESHOLD", "0.78"))
VIT_SMOKE_CONF_THRESHOLD = float(os.getenv("VIT_SMOKE_CONF_THRESHOLD", "0.65"))
FIRE_CONF_THRESHOLD = VIT_FIRE_CONF_THRESHOLD
FIRE_ONLY_CONF_THRESHOLD = VIT_FIRE_CONF_THRESHOLD
DANGER_ZONE_CONF_THRESHOLD = float(os.getenv("DANGER_ZONE_CONF_THRESHOLD", "0.35"))
FALL_PERSON_CONF_THRESHOLD = float(os.getenv("FALL_PERSON_CONF_THRESHOLD", "0.40"))
PPE_CONF_THRESHOLD = float(os.getenv("PPE_CONF_THRESHOLD", "0.40"))

RTMPOSE_MODE = os.getenv("RTMPOSE_MODE", "balanced")
RTMPOSE_BACKEND = os.getenv("RTMPOSE_BACKEND", "onnxruntime")
RTMPOSE_DEVICE = os.getenv("RTMPOSE_DEVICE", "cpu")
RTMPOSE_MODEL_PATH = os.getenv("RTMPOSE_MODEL_PATH", "")

ANALYSIS_EVERY_N_FRAMES = int(os.getenv("ANALYSIS_EVERY_N_FRAMES", "3"))

MODEL_IDS = ("fire", "fall", "danger_zone", "ppe")

MODEL_LABELS = {
    "fire": "🔥 Yong'in / Tutun",
    "fall": "🚨 Yiqilib tushish",
    "danger_zone": "⛔ Xavfli hudud",
    "ppe": "🦺 PPE / Xavfsizlik kiyimi",
}

MODEL_DEFAULT_ENABLED = {
    "fire": True,
    "fall": True,
    "danger_zone": True,
    "ppe": True,
}

CAMERA_RECONNECT_DELAY_SEC = 2.0

APP_TITLE = "AI Video Monitoring - Ko'p Kamera / Ko'p Model Tizimi"

GUI_UPDATE_MS = 33
GRID_UPDATE_MS = 50
STATUS_POLL_MS = 2000
POPOUT_UPDATE_MS = 33

GRID_TILE_W, GRID_TILE_H = 360, 210

SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
DEFAULT_SERVER_URL = os.getenv("DEFAULT_SERVER_URL", "http://localhost:8000")
DEFAULT_CAMERA_SOURCE = os.getenv("DEFAULT_CAMERA_SOURCE", "rtsp://rtsp:Qazwsx12@10.41.120.60:554/Streaming/Channels/102")

API_TIMEOUT = 8

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
SNAPSHOT_COOLDOWN_SEC = float(os.getenv("SNAPSHOT_COOLDOWN_SEC", "2.0"))
CAMERA_LOG_PATH = os.getenv("CAMERA_LOG_PATH", os.path.join(PARENT_DIR, "camera_stream_errors.log"))
SYSTEM_LOG_PATH = os.getenv("SYSTEM_LOG_PATH", os.path.join(PARENT_DIR, "system_debug.log"))
