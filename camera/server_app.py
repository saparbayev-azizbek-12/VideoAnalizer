import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"

import cv2
import time
import torch
import shutil
import threading
import numpy as np
from ultralytics import YOLO
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import FastAPI, UploadFile, File, Form, HTTPException

import config


os.makedirs(config.OUT_DIR, exist_ok=True)
os.makedirs(config.UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="NVR AI Processing GPU Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = YOLO(config.MODEL_PATH)
model.to(device)

running = False
job_lock = threading.Lock()
job_info = {
    "running": False,
    "source_type": "idle",
    "source_name": "",
    "saved_count": 0,
    "processed_frames": 0,
    "total_frames": 0,
    "fps": 0.0
}
latest_preview_frame = None


def is_blurry(image, threshold=config.BLUR_THRESHOLD):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


def process_video_file_job(file_path, conf_thresh, min_area_ratio):
    global running, job_info, latest_preview_frame

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        with job_lock:
            running = False
            job_info["running"] = False
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_skip = max(1, int(fps * config.CHECK_INTERVAL_SEC))

    with job_lock:
        job_info["total_frames"] = total_frames
        job_info["fps"] = fps

    frame_idx = 0

    while True:
        with job_lock:
            if not running:
                break

        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_idx += 1
        with job_lock:
            job_info["processed_frames"] = frame_idx

        if frame_idx % frame_skip == 0:
            h, w = frame.shape[:2]
            results = model.predict(
                frame,
                conf=conf_thresh,
                classes=[config.PERSON_CLASS_ID],
                verbose=False
            )[0]

            preview = frame.copy()
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                area_ratio = ((x2 - x1) * (y2 - y1)) / (w * h)
                saved = False

                if area_ratio >= min_area_ratio:
                    crop = frame[max(0, y1):y2, max(0, x1):x2]
                    if crop.size > 0 and not is_blurry(crop):
                        filename = f"crop_{int(time.time() * 1000)}_{frame_idx}.jpg"
                        cv2.imwrite(os.path.join(config.OUT_DIR, filename), crop)
                        with job_lock:
                            job_info["saved_count"] += 1
                        saved = True

                color = (0, 255, 0) if saved else (0, 140, 255)
                cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)

            with job_lock:
                latest_preview_frame = preview

    cap.release()
    with job_lock:
        running = False
        job_info["running"] = False


from fastapi import Response

nvr_frames = {}
nvr_lock = threading.Lock()
nvr_readers_started = False
boxes_by_channel_store = {}


def nvr_channel_reader(ch):
    url = config.channel_rtsp_url(ch)
    cap = None
    while True:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        ok, frame = cap.read()
        if ok and frame is not None:
            with nvr_lock:
                nvr_frames[ch] = frame
            time.sleep(0.02)
        else:
            if cap is not None:
                cap.release()
            cap = None
            time.sleep(2)
    if cap is not None:
        cap.release()


def start_nvr_readers_once():
    global nvr_readers_started
    if not nvr_readers_started:
        nvr_readers_started = True
        for ch in config.CHANNELS:
            t = threading.Thread(target=nvr_channel_reader, args=(ch,), daemon=True)
            t.start()


def process_nvr_job(channels, conf_thresh, min_area_ratio):
    global running, job_info, latest_preview_frame, boxes_by_channel_store

    start_nvr_readers_once()

    while True:
        with job_lock:
            if not running:
                break

        with nvr_lock:
            current_frames = {ch: f.copy() for ch, f in nvr_frames.items()}

        for ch in channels:
            frame = current_frames.get(ch)
            if frame is not None:
                h, w = frame.shape[:2]
                results = model.predict(
                    frame,
                    conf=conf_thresh,
                    classes=[config.PERSON_CLASS_ID],
                    verbose=False
                )[0]

                ch_boxes = []
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    area_ratio = ((x2 - x1) * (y2 - y1)) / (w * h)
                    saved = False

                    if area_ratio >= min_area_ratio:
                        crop = frame[max(0, y1):y2, max(0, x1):x2]
                        if crop.size > 0 and not is_blurry(crop):
                            filename = f"ch{ch}_{int(time.time() * 1000)}.jpg"
                            cv2.imwrite(os.path.join(config.OUT_DIR, filename), crop)
                            with job_lock:
                                job_info["saved_count"] += 1
                            saved = True

                    ch_boxes.append((x1, y1, x2, y2, saved))

                with job_lock:
                    boxes_by_channel_store[ch] = ch_boxes

        time.sleep(config.CHECK_INTERVAL_SEC)

    with job_lock:
        running = False
        job_info["running"] = False


@app.get("/api/health")
async def health():
    start_nvr_readers_once()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU (CPU Mode)"
    return {
        "status": "ok",
        "device": device,
        "gpu_name": gpu_name,
        "running": running
    }


@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    dest_path = os.path.join(config.UPLOAD_DIR, file.filename)
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"ok": True, "filename": file.filename, "file_path": dest_path}


@app.post("/api/start")
async def start_job(
    source_type: str = Form(...),
    filename: str = Form(None),
    conf_threshold: float = Form(0.4),
    min_area_ratio: float = Form(0.02)
):
    global running, job_info

    with job_lock:
        if running:
            return JSONResponse({"ok": False, "error": "Serverda allaqachon jarayon ishlamoqda"})

        running = True
        job_info["running"] = True
        job_info["source_type"] = source_type
        job_info["source_name"] = filename or "NVR Channels"
        job_info["saved_count"] = 0
        job_info["processed_frames"] = 0
        job_info["total_frames"] = 0

    if source_type == "file":
        if not filename:
            raise HTTPException(status_code=400, detail="Fayl nomi kiritilmadi")
        file_path = os.path.join(config.UPLOAD_DIR, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Serverda yuklangan fayl topilmadi")

        t = threading.Thread(
            target=process_video_file_job,
            args=(file_path, conf_threshold, min_area_ratio),
            daemon=True
        )
        t.start()
    else:
        t = threading.Thread(
            target=process_nvr_job,
            args=(config.CHANNELS, conf_threshold, min_area_ratio),
            daemon=True
        )
        t.start()

    return {"ok": True, "message": "Qayta ishlash GPU serverda boshlandi"}


@app.post("/api/stop")
async def stop_job():
    global running
    with job_lock:
        running = False
        job_info["running"] = False
    return {"ok": True, "message": "Jarayon to'xtatildi"}


@app.get("/api/status")
async def status():
    with job_lock:
        return dict(job_info)


@app.get("/api/preview_image")
async def preview_image(channel: int = 0):
    start_nvr_readers_once()

    if channel > 0:
        with nvr_lock:
            raw = nvr_frames.get(channel)
        if raw is not None:
            frame = raw.copy()
            h, w = frame.shape[:2]

            with job_lock:
                is_run = running
                ch_boxes = list(boxes_by_channel_store.get(channel, [])) if is_run else []

            for (x1, y1, x2, y2, saved_flag) in ch_boxes:
                color = (0, 255, 0) if saved_flag else (0, 140, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            cv2.rectangle(frame, (0, 0), (w, 45), (30, 30, 30), -1)
            cv2.putText(
                frame, f"Kanal {channel} | To'liq Ko'rinish (Orqaga qaytish uchun bosing)",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
            )
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                return Response(content=buf.tobytes(), media_type="image/jpeg")

    cols = config.GRID_COLS
    channels = config.CHANNELS
    tiles = []

    with nvr_lock:
        current_frames = {ch: f.copy() for ch, f in nvr_frames.items()}
    with job_lock:
        is_run = running
        current_boxes = {ch: list(b) for ch, b in boxes_by_channel_store.items()} if is_run else {}

    for ch in channels:
        f = current_frames.get(ch)
        if f is not None:
            tile = cv2.resize(f, (config.TILE_W, config.TILE_H))
            if ch in current_boxes:
                scale_x = config.TILE_W / f.shape[1]
                scale_y = config.TILE_H / f.shape[0]
                for (x1, y1, x2, y2, saved_flag) in current_boxes[ch]:
                    color = (0, 255, 0) if saved_flag else (0, 140, 255)
                    cv2.rectangle(
                        tile,
                        (int(x1 * scale_x), int(y1 * scale_y)),
                        (int(x2 * scale_x), int(y2 * scale_y)),
                        color, 2
                    )
            cv2.rectangle(tile, (0, 0), (config.TILE_W, 28), (30, 30, 30), -1)
            cv2.putText(tile, f"Kanal {ch}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        else:
            tile = np.zeros((config.TILE_H, config.TILE_W, 3), dtype=np.uint8)
            cv2.putText(tile, f"Kanal {ch}: ulanmoqda...", (15, config.TILE_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
        tiles.append(tile)

    rows = []
    for r in range(0, len(tiles), cols):
        row = tiles[r:r + cols]
        while len(row) < cols:
            row.append(np.zeros((config.TILE_H, config.TILE_W, 3), dtype=np.uint8))
        rows.append(np.hstack(row))
    grid = np.vstack(rows)

    ok, buf = cv2.imencode(".jpg", grid)
    if not ok:
        raise HTTPException(status_code=500, detail="Kadr kodlanmadi")
    return Response(content=buf.tobytes(), media_type="image/jpeg")


def preview_generator():
    while True:
        with job_lock:
            frame = latest_preview_frame.copy() if latest_preview_frame is not None else None

        if frame is not None:
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.1)


@app.get("/api/preview_feed")
async def preview_feed():
    return StreamingResponse(preview_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_app:app", host="0.0.0.0", port=8000, reload=True)


