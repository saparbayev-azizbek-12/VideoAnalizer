import os

NVR_IP = "10.41.120.60"
NVR_USER = "rtsp"
NVR_PASSWORD = "Qazwsx12"
CHANNELS = [1]

USE_SUB_STREAM = False


def channel_rtsp_url(channel: int) -> str:
    stream_suffix = "02" if USE_SUB_STREAM else "01"
    return f"rtsp://{NVR_USER}:{NVR_PASSWORD}@{NVR_IP}:554/Streaming/Channels/{channel}{stream_suffix}"


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "..", "dataset", "unlabeled")
MODEL_PATH = os.path.join(BASE_DIR, "yolo11n.pt")
PERSON_CLASS_ID = 0
CONF_THRESHOLD = 0.4
MIN_BOX_AREA_RATIO = 0.02
BLUR_THRESHOLD = 60.0
CHECK_INTERVAL_SEC = 1.0

TILE_W, TILE_H = 400, 225
GRID_COLS = 4

APP_TITLE = "NVR & Video Dataset Yig'uvchi (Client-Server)"
GUI_UPDATE_MS = 40

DEFAULT_SERVER_URL = "http://localhost:8000"
API_TIMEOUT = 10
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")



