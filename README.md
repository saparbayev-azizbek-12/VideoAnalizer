# AI Video Monitoring - Ko'p Kamera va Ko'p Model Tizimi

Sun'iy intellektga asoslangan ko'p kamerali video monitoring tizimi. Tizim real vaqt rejimida RTSP video oqimlari, vebkamera yoki video fayllardan kelayotgan kadrlarni tahlil qilib, quyidagi xavfli holatlarni aniqlaydi:

- 🔥 **Yong'in va tutun aniqlash** (Fire & Smoke Detection)
- 🚨 **Inson yiqilib tushishini aniqlash** (Fall Detection via MediaPipe Pose & YOLO)
- ⛔ **Xavfli hududga kirishni nazorat qilish** (Danger Zone Polygon Detection)
- 📦 **Hodisalar bo'yicha avtomatik Dataset yig'ish** va bir tugma bilan `.zip` yuklab olish

---

## 🛠 Arxitektura va Imkoniyatlar

Tizim **Server-Mijoz (Client-Server)** arxitekturasida ishlaydi:
1. **GPU Server (`server.py`)**: FastAPIda qurilgan. Kameralar oqimini qabul qiladi, AI modellar yordamida kadrlarni tahlil qiladi va REST API hamda video oqim snapshotlarini uzatadi.
2. **GUI Monitoring Dasturi (`main.py`)**: Tkinter grafik interfeysi. GPU serverga ulanib, real vaqt rejimida kameralar to'rini (Grid view), alohida kamera ko'rinishini va so'nggi bildirishnomalarni ko meksiyada namoyish etadi.

---

## 📋 Tizim Talablari

- **Python**: 3.12 yoki undan yuqori versiya
- **Paket Menejeri**: `uv` (tavsiya etiladi) yoki `pip`
- **Akkeleratsiya**: NVIDIA GPU va CUDA (tavsiya etiladi, CPU rejimi ham mavjud)

---

## 🚀 O'rnatish va Sozlash

### 1. Repozitoriyani klonlash va virtual muhit yaratish

`uv` paket menejeri yordamida:
```bash
# Virtual muhitni yaratish va barcha bog'liqliklarni o'rnatish
uv sync
```

Yoki standart `pip` yordamida:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r pyproject.toml
```

### 2. Atrof-muhit o'zgaruvchilarini (`.env`) sozlash

Loyiha ildiz papkasida `.env.example` faylidan nusxa olib `.env` faylini yarating:

```bash
cp .env.example .env
```

`.env` faylini ochib, server va kamera sozlamalarini kiriting:

```env
# Server Sozlamalari
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
DEFAULT_SERVER_URL=http://localhost:8000

# Standart Kamera Manbasi (RTSP URL / Video Fayl / Vebkamera indeksi)
DEFAULT_CAMERA_SOURCE=rtsp://user:password@192.168.1.100:554/Streaming/Channels/101

# AI Modellari Bo'sag'a Qiymatlari (Confidence Thresholds)
FIRE_CONF_THRESHOLD=0.30
FIRE_ONLY_CONF_THRESHOLD=0.55
DANGER_ZONE_CONF_THRESHOLD=0.35
FALL_PERSON_CONF_THRESHOLD=0.40

# Dataset Snapshot Oralig'i (soniyada)
SNAPSHOT_COOLDOWN_SEC=2.0
```

---

## 🖥 Dasturni Ishga Tushirish

### 1-Usul: GPU Serverni ishga tushirish

Analiz serverini ishga tushirish uchun:

```bash
cd multi-object-detector
uv run uvicorn server:app --host 0.0.0.0 --port 8000
```
*Server ishga tushgach, REST API va Swagger hujjatlari `http://localhost:8000/docs` manzilida ochiq bo'ladi.*

### 2-Usul: Monitoring GUI Dasturini ishga tushirish

Grafik interfeys mijozini ishga tushirish uchun:

```bash
cd multi-object-detector
uv run main.py
```

---

## 💡 Interfeysdan Foydalanish

1. **Serverga Ulanish**: GUI dasturida **"🔌 Ulanishni Tekshirish"** tugmasini bosing.
2. **Kamera Qo'shish**: **"➕ Kamera qo'shish"** tugmasini bosib, kamera nomi va RTSP manbasini kiriting.
3. **Xavfli Hudud Belgilash**: Har bir kamera kartochkasidagi **"⛶"** tugmasini bosib, rasmda kamida 3 nuqta tanlash orqali poligon (zone) chizing va saqlang.
4. **Dataset Yuklab Olish**: Modellar xavfli holatni aniqlaganda avtomatik tarzda `dataset/` papkasiga kadrlarni saqlaydi. Yuqori paneldagi **"📦 Datasetni yuklab olish"** tugmasini bosib, to'plangan datasetni `.zip` shaklida kompyuteringizga saqlab olishingiz mumkin.

---

## 📁 Loyiha Tuzilishi

```text
VideoAnalizer/
├── .env                  # Maxfiy sozlamalar va konfiguratsiya (Git'ga kiritilmaydi)
├── .env.example          # Konfiguratsiya uchun shablon fayl
├── pyproject.toml        # Loyiha bog'liqliklari va paketlar
├── README.md             # Loyiha bo'yicha qo'llanma
└── multi-object-detector/
    ├── main.py           # Tkinter GUI monitoring mijoz dasturi
    ├── server.py         # FastAPI AI tahlil serveri
    ├── config.py         # Konfiguratsiya va .env yuklagich
    ├── camera_manager.py # Kameralar oqimi va tahlil logikasi
    ├── dataset/          # Aniqlangan snapshotlar saqlanadigan papka
    └── analysis/
        ├── models_manager.py
        └── detectors/
            ├── fire_detector.py        # Yong'in / tutun detektori
            ├── fall_detector.py        # Yiqilish detektori
            └── danger_zone_detector.py # Xavfli hudud detektori
```