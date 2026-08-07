import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTORS_DIR = os.path.join(BASE_DIR, "analysis", "detectors")
if os.path.exists(DETECTORS_DIR) and DETECTORS_DIR not in sys.path:
    sys.path.insert(0, DETECTORS_DIR)

FIRE_MODEL_ONNX = os.path.join(BASE_DIR, "best.onnx")
FIRE_MODEL_PT = os.path.join(BASE_DIR, "best.pt")
PERSON_MODEL_PATH = os.path.join(BASE_DIR, "yolov8l.pt")
PERSON_CLASS_ID = 0

FIRE_CONF_THRESHOLD = 0.30
FIRE_ONLY_CONF_THRESHOLD = 0.55
DANGER_ZONE_CONF_THRESHOLD = 0.35
FALL_PERSON_CONF_THRESHOLD = 0.40

ANALYSIS_EVERY_N_FRAMES = 1

MODEL_IDS = ("fire", "fall", "danger_zone")

MODEL_LABELS = {
    "fire": "🔥 Yong'in / Tutun",
    "fall": "🚨 Yiqilib tushish",
    "danger_zone": "⛔ Xavfli hudud",
}

MODEL_DEFAULT_ENABLED = {
    "fire": True,
    "fall": True,
    "danger_zone": True,
}

CAMERA_RECONNECT_DELAY_SEC = 2.0

APP_TITLE = "AI Video Monitoring - Ko'p Kamera / Ko'p Model Tizimi"

GUI_UPDATE_MS = 120
GRID_UPDATE_MS = 350
STATUS_POLL_MS = 2000
POPOUT_UPDATE_MS = 150

GRID_TILE_W, GRID_TILE_H = 360, 210

DEFAULT_SERVER_URL = "http://10.0.89.251:8000"
API_TIMEOUT = 8

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
SNAPSHOT_COOLDOWN_SEC = 2.0