# AI Video Monitoring - Ko'p Kamera va Ko'p Model Tizimi

Sun'iy intellektga asoslangan ko'p kamerali video monitoring va offline video tahlil tizimi. Tizim real vaqt rejimida RTSP video oqimlari, vebkamera yoki video fayllardan kelayotgan kadrlarni tahlil qilib, quyidagi xavfli holatlarni aniqlaydi:

- 🔥 **Yong'in va tutun aniqlash** (Fire & Smoke Detection via Vision Transformer ViT)
- 🚨 **Inson yiqilib tushishini aniqlash** (Fall Detection via RTMPose & YOLO)
- ⛔ **Xavfli hududga kirishni nazorat qilish** (Danger Zone Polygon Detection via RF-DETR / YOLO)
- 🦺 **Maxsus xavfsizlik kiyimi va kaska nazorati** (PPE Compliance via YOLO)
- 📦 **Hodisalar bo'yicha avtomatik Dataset yig'ish** va bir tugma bilan `.zip` yuklab olish

---

## 🛠 Modulli Arxitektura

Loyiha to'liq modulli, har bir papkasi va fayli o'z vazifasiga mos, sodda va mantiqiy nomlangan tuzilishga ega:

```text
VideoAnalizer/
├── .env                              # Maxfiy sozlamalar va konfiguratsiya
├── .env.example                      # Konfiguratsiya uchun shablon fayl
├── pyproject.toml                    # Loyiha bog'liqliklari va paketlar
├── README.md                         # Loyiha bo'yicha to'liq qo'llanma
└── multi_object_detector/            # Asosiy paket
    ├── core/                         # Asosiy konfiguratsiya va loggerlar
    │   ├── config.py                 # Barcha sozlamalar, .env yuklash, yo'llar
    │   ├── system_logger.py          # Tizim debug va xatoliklar loggeri
    │   └── stream_logger.py          # RTSP diagnostika va oqim loggeri
    │
    ├── detectors/                    # AI modellari va kadrlar tahlili yadrosi
    │   ├── manager.py                # Modellarni yuklash va boshqarish (ModelsManager)
    │   ├── frame_filter.py           # Buzilgan (yashil, glitch, muzlash) kadrlarni filtrlash
    │   ├── fire_detector.py          # Yong'in va tutun detektori (ViT)
    │   ├── fall_detector.py          # Inson yiqilish detektori (RTMPose / YOLO)
    │   ├── danger_zone_detector.py   # Xavfli hudud poligon nazorati (RFDETR / YOLO)
    │   ├── ppe_detector.py           # Maxsus kiyim va kaska (PPE) detektori
    │   └── models/                   # Model og'irliklari (.pt va ViT fayllari)
    │
    ├── streaming/                    # Ko'p kamerali video oqimlari boshqaruvi
    │   └── camera_manager.py         # Ko'p kamerali parallel oqim va tahlil menejeri
    │
    ├── api/                          # FastAPI REST API veb-serverlari
    │   ├── live_server.py            # Jonli monitoring va GPU tahlil serveri
    │   └── video_server.py           # Offline video fayllarni tahlil qilish serveri
    │
    ├── ui/                           # Grafik foydalanuvchi interfeyslari (Tkinter GUI)
    │   ├── live_monitor_ui.py        # Jonli ko'p kamera monitoring oynasi
    │   └── video_analyzer_ui.py      # Offline video tahlilchi va pleyer interfeysi
    │
    ├── calibration/                  # Kamera optikasi va linza kalibratsiyasi
    │   ├── calib_tool.py             # CLI orqali checkerboard kalibratsiyasi
    │   ├── calib_server.py           # Interaktiv veb kalibratsiya serveri
    │   └── calib.npz                 # Kalibratsiya matritsasi
    │
    ├── dataset/                      # Avtomatik yig'ilgan snapshotlar
    │   ├── fire/
    │   ├── fall/
    │   └── danger_zone/
    │
    ├── main.py                       # Asosiy ishga tushirish fayli (Jonli GUI monitoring)
    ├── run_server.py                 # Jonli tahlil serverini ishga tushirish
    ├── run_video_server.py           # Video tahlil serverini ishga tushirish
    └── run_analyzer.py               # Video tahlil GUI dasturini ishga tushirish
```

---

## 📋 Tizim Talablari

- **Python**: 3.12 yoki undan yuqori versiya
- **Paket Menejeri**: `uv` (tavsiya etiladi) yoki `pip`
- **Akkeleratsiya**: NVIDIA GPU va CUDA (tavsiya etiladi, CPU rejimi ham mavjud)

---

## 🚀 O'rnatish va Sozlash

### 1. Bog'liqliklarni o'rnatish

`uv` yordamida:
```bash
uv sync
```

### 2. Atrof-muhit o'zgaruvchilarini (`.env`) sozlash

```bash
cp .env.example .env
```

`.env` faylidagi sozlamalar:
```env
# Server sozlamalari
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
DEFAULT_SERVER_URL=http://localhost:8000

# Standart kamera manbasi (RTSP URL / Video Fayl / Vebkamera indeksi)
DEFAULT_CAMERA_SOURCE=rtsp://user:password@192.168.1.100:554/Streaming/Channels/101

# AI Modellari bo'sag'a qiymatlari (Confidence Thresholds)
VIT_FIRE_CONF_THRESHOLD=0.78
VIT_SMOKE_CONF_THRESHOLD=0.65
DANGER_ZONE_CONF_THRESHOLD=0.35
FALL_PERSON_CONF_THRESHOLD=0.40
PPE_CONF_THRESHOLD=0.40

# Dataset Snapshot oraliq vaqti (soniyada)
SNAPSHOT_COOLDOWN_SEC=2.0
```

### 3. AI Modellarni serverga bir marta yuklab olish

Katta hajmdagi `.safetensors` va AI model og'irliklarini serverning o'zida to'g'ridan-to'g'ri yuklab olish uchun:
```bash
cd multi_object_detector
uv run download_models.py
```
*Ushbu buyruq barcha kerakli AI modellarni bir marta yuklab `detectors/models/` papkasiga saqlaydi, shundan so'ng server to'liq offline ishlaydi.*

---

## 🖥 Dasturlarni Ishga Tushirish

### 1. Jonli Ko'p Kamera Monitoring GUI (Asosiy rejim)
```bash
cd multi_object_detector
uv run main.py
```

### 2. Jonli GPU Tahlil Serverini ishga tushirish (FastAPI)
```bash
cd multi_object_detector
uv run run_server.py
uv run run_server.py --port 8004
```
*API hujjatlari: `http://localhost:8000/docs` (yoki ko'rsatilgan portda)*

### 3. Offline Video Fayllarni Tahlil Qilish GUI Dasturi
```bash
cd multi_object_detector
uv run run_analyzer.py
```

### 4. Offline Video Tahlil Serverini ishga tushirish (FastAPI)
```bash
cd multi_object_detector
uv run run_video_server.py
uv run run_video_server.py --port 8005
```
*Video API hujjatlari: `http://localhost:8001/docs` (yoki ko'rsatilgan portda)*

### 5. Kamera Linzasini Kalibratsiya Qilish (Undistortion)
```bash
cd multi_object_detector
uv run python calibration/calib_server.py
```

---

## 💡 Interfeysdan Foydalanish

1. **Serverga Ulanish**: GUI dasturida **"🔌 Ulanishni Tekshirish"** tugmasini bosing.
2. **Kamera Qo'shish**: **"➕ Kamera qo'shish"** tugmasini bosib, kamera nomi va RTSP manbasini kiriting.
3. **Xavfli Hudud Belgilash**: Har bir kamera kartochkasidagi **"⛶"** tugmasini bosib, rasmda kamida 3 nuqta tanlash orqali poligon (zone) chizing va saqlang.
4. **Kamerani Alohida Oynada Ochish**: **"🔍"** tugmasi orqali alohida popout oynada kattalashtirib kuzating.
5. **Modellarni Yoqish/O'chirish**: Yuqori paneldagi checkboxlar orqali kerakli AI modellarini real vaqtda yoqing yoki o'chiring.
6. **Dataset Yuklab Olish**: Modellar xavfli holatni aniqlaganda avtomatik tarzda `dataset/` papkasiga kadrlarni saqlaydi. Yuqori paneldagi **"📦 Datasetni yuklab olish"** tugmasini bosib, to'plangan datasetni `.zip` shaklida kompyuteringizga saqlab oling.