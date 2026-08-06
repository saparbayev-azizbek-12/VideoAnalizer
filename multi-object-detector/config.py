"""
config.py
----------------------------------------------------------------------
Server (server_app.py) va desktop klient (main.py) uchun UMUMIY sozlamalar.

DIQQAT (eski config.py dan farqi):
Ilgari bu loyiha "dataset yig'ish" (bitta model bilan person crop saqlash)
uchun ishlatilgan. Endi u 3 ta modelni (fire / fall / danger_zone) bir
nechta kamerada REAL VAQTDA ishga tushiradigan monitoring tizimiga
o'zgartirildi, shu sabab NVR_IP/CHANNELS kabi statik kamera sozlamalari
olib tashlandi - kameralar endi GUI orqali DINAMIK qo'shiladi/o'chiriladi.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# YOLO MODEL FAYLLARI
# (fayllarni shu papkaga qo'ying, yoki quyidagi yo'llarni o'zingizga moslang)
# ---------------------------------------------------------------------------
FIRE_MODEL_ONNX = os.path.join(BASE_DIR, "best.onnx")      # fire_detector.py custom modeli
FIRE_MODEL_PT = os.path.join(BASE_DIR, "best.pt")
PERSON_MODEL_PATH = os.path.join(BASE_DIR, "yolov8l.pt")   # fall + danger_zone uchun "person" modeli
PERSON_CLASS_ID = 0

# ---------------------------------------------------------------------------
# HAR BIR MODEL UCHUN ANIQLASH CHEGARALARI
# ---------------------------------------------------------------------------
FIRE_CONF_THRESHOLD = 0.30
FIRE_ONLY_CONF_THRESHOLD = 0.55     # "fire" klassi uchun qo'shimcha qattiqroq chegara
DANGER_ZONE_CONF_THRESHOLD = 0.35
FALL_PERSON_CONF_THRESHOLD = 0.40

# ---------------------------------------------------------------------------
# TAHLIL CHASTOTASI / ISHLASH TEZLIGI
# ---------------------------------------------------------------------------
# 3 ta og'ir model N ta kamerada bir vaqtda ishlaganda GPU tez tiqilib qolishi
# mumkin. Zarurat bo'lsa buni oshiring - shunda har kadr emas, har N-kadr
# tahlil qilinadi (video baribir suyuq ko'rinadi, faqat aniqlash "sekinroq"
# yangilanadi). Hozircha 1 = har bir kadr tahlil qilinadi.
ANALYSIS_EVERY_N_FRAMES = 1

# ---------------------------------------------------------------------------
# MODELLAR RO'YXATI (server + GUI shu ro'yxatdan foydalanadi)
# ---------------------------------------------------------------------------
MODEL_IDS = ("fire", "fall", "danger_zone")

MODEL_LABELS = {
    "fire": "🔥 Yong'in / Tutun",
    "fall": "🚨 Yiqilib tushish",
    "danger_zone": "⛔ Xavfli hudud",
}

# Server ishga tushganda modellar yoqilganmi (keyinchalik GUI orqali
# istalgan vaqtda o'zgartiriladi)
MODEL_DEFAULT_ENABLED = {
    "fire": True,
    "fall": True,
    "danger_zone": True,
}

# ---------------------------------------------------------------------------
# KAMERA OQIMI SOZLAMALARI
# ---------------------------------------------------------------------------
CAMERA_RECONNECT_DELAY_SEC = 2.0

# ---------------------------------------------------------------------------
# GUI (DESKTOP KLIENT)
# ---------------------------------------------------------------------------
APP_TITLE = "AI Video Monitoring - Ko'p Kamera / Ko'p Model Tizimi"

GUI_UPDATE_MS = 120           # katta/yagona kamera oynasi yangilanish tezligi
GRID_UPDATE_MS = 350          # mozaika (barcha kameralar) yangilanish tezligi
STATUS_POLL_MS = 2000         # kameralar ro'yxati / model holati / hodisalar so'rovi
POPOUT_UPDATE_MS = 150        # alohida ("katta oyna") kamera oynasi yangilanish tezligi

GRID_TILE_W, GRID_TILE_H = 360, 210     # mozaikadagi bitta plitka o'lchami (server tomonda)

DEFAULT_SERVER_URL = "http://localhost:8000"
API_TIMEOUT = 8