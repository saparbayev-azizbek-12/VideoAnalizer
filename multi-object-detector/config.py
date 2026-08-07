import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

def _load_env():
    for env_path in [os.path.join(PARENT_DIR, ".env"), os.path.join(BASE_DIR, ".env")]:
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
            except Exception:
                pass

_load_env()

DETECTORS_DIR = os.path.join(BASE_DIR, "analysis", "detectors")
if os.path.exists(DETECTORS_DIR) and DETECTORS_DIR not in sys.path:
    sys.path.insert(0, DETECTORS_DIR)

FIRE_MODEL_ONNX = os.path.join(BASE_DIR, "best.onnx")
FIRE_MODEL_PT = os.path.join(BASE_DIR, "best.pt")
PERSON_MODEL_PATH = os.path.join(BASE_DIR, "yolov8l.pt")
PERSON_CLASS_ID = 0

FIRE_CONF_THRESHOLD = float(os.getenv("FIRE_CONF_THRESHOLD", "0.30"))
FIRE_ONLY_CONF_THRESHOLD = float(os.getenv("FIRE_ONLY_CONF_THRESHOLD", "0.55"))
DANGER_ZONE_CONF_THRESHOLD = float(os.getenv("DANGER_ZONE_CONF_THRESHOLD", "0.35"))
FALL_PERSON_CONF_THRESHOLD = float(os.getenv("FALL_PERSON_CONF_THRESHOLD", "0.40"))

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

SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
DEFAULT_SERVER_URL = os.getenv("DEFAULT_SERVER_URL", "http://localhost:8000")
DEFAULT_CAMERA_SOURCE = os.getenv("DEFAULT_CAMERA_SOURCE", "rtsp://user:pass@192.168.1.10:554/Streaming/Channels/101")

API_TIMEOUT = 8

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
SNAPSHOT_COOLDOWN_SEC = float(os.getenv("SNAPSHOT_COOLDOWN_SEC", "2.0"))