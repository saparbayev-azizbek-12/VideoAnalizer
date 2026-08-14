from __future__ import annotations
import os
import io
import cv2
import time
import torch
import zipfile
import numpy as np
from pydantic import BaseModel
from fastapi.responses import Response
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from multi_object_detector.core import config
from multi_object_detector.streaming import camera_manager
from multi_object_detector.core.system_logger import sys_logger
from multi_object_detector.detectors import manager as models_manager

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"
app = FastAPI(title="AI Video Monitoring Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"


@app.on_event("startup")
def _startup() -> None:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None (CPU mode)"
    sys_logger.info("Server", f"Server startup on device: {device} ({gpu_name})")
    sys_logger.info("Server", f"Enabled models: {models_manager.get_all_enabled()}")
    try:
        models_manager.preload_all()
        sys_logger.info("Server", "All models preloaded successfully.")
    except Exception as e:
        sys_logger.error("Server", f"Model preloading error: {e}", exc=e)


@app.on_event("shutdown")
def _shutdown() -> None:
    sys_logger.info("Server", "Server shutting down...")
    camera_manager.stop_all()


@app.get("/api/health")
async def health():
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU yo'q (CPU rejimi)"
    return {
        "status": "ok",
        "device": device,
        "gpu_name": gpu_name,
        "camera_count": len(camera_manager.list_cameras()),
        "models": models_manager.get_all_enabled(),
    }


class ModelToggleRequest(BaseModel):
    model_id: str
    enabled: bool


@app.get("/api/models")
async def list_models():
    enabled = models_manager.get_all_enabled()
    return {
        "models": [
            {"id": mid, "label": config.MODEL_LABELS[mid], "enabled": enabled[mid]}
            for mid in config.MODEL_IDS
        ]
    }


@app.post("/api/models/toggle")
async def toggle_model(req: ModelToggleRequest):
    try:
        models_manager.set_enabled(req.model_id, req.enabled)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "model_id": req.model_id, "enabled": req.enabled}


class CameraCreateRequest(BaseModel):
    name: str
    source: str


def _camera_to_dict(cam: camera_manager.CameraWorker) -> dict:
    return {
        "id": cam.id,
        "name": cam.name,
        "source": cam.source,
        "connected": cam.connected,
        "running": cam.running,
        "has_zone": cam.get_zone() is not None,
        "last_error": cam.last_error,
    }


@app.get("/api/cameras")
async def list_cameras():
    return {"cameras": [_camera_to_dict(c) for c in camera_manager.list_cameras()]}


@app.post("/api/cameras")
async def create_camera(req: CameraCreateRequest):
    if not req.name.strip() or not req.source.strip():
        raise HTTPException(status_code=400, detail="Kamera nomi va manbasi bo'sh bo'lmasligi kerak")
    cam = camera_manager.add_camera(req.name.strip(), req.source.strip())
    return {"ok": True, "camera": _camera_to_dict(cam)}


@app.delete("/api/cameras/{cam_id}")
async def delete_camera(cam_id: str):
    ok = camera_manager.remove_camera(cam_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Kamera topilmadi")
    return {"ok": True}


class ZoneRequest(BaseModel):
    polygon: list[list[int]]


@app.get("/api/cameras/{cam_id}/zone")
async def get_zone(cam_id: str):
    cam = camera_manager.get_camera(cam_id)
    if cam is None:
        raise HTTPException(status_code=404, detail="Kamera topilmadi")
    return {"polygon": cam.get_zone()}


@app.post("/api/cameras/{cam_id}/zone")
async def set_zone(cam_id: str, req: ZoneRequest):
    cam = camera_manager.get_camera(cam_id)
    if cam is None:
        raise HTTPException(status_code=404, detail="Kamera topilmadi")
    if len(req.polygon) < 3:
        raise HTTPException(status_code=400, detail="Poligon kamida 3 nuqtadan iborat bo'lishi kerak")
    cam.set_zone(req.polygon)
    return {"ok": True}


@app.delete("/api/cameras/{cam_id}/zone")
async def delete_zone(cam_id: str):
    cam = camera_manager.get_camera(cam_id)
    if cam is None:
        raise HTTPException(status_code=404, detail="Kamera topilmadi")
    cam.clear_zone()
    return {"ok": True}


@app.get("/api/cameras/{cam_id}/events")
async def camera_events(cam_id: str, limit: int = 50):
    cam = camera_manager.get_camera(cam_id)
    if cam is None:
        raise HTTPException(status_code=404, detail="Kamera topilmadi")
    return {"events": cam.get_recent_events(limit)}


@app.get("/api/events")
async def all_events(limit: int = 100):
    combined = []
    for cam in camera_manager.list_cameras():
        for ev in cam.get_recent_events(limit):
            combined.append({"camera_id": cam.id, "camera_name": cam.name, **ev})
    combined.sort(key=lambda e: e["ts"], reverse=True)
    return {"events": combined[:limit]}


@app.get("/api/dataset/info")
async def dataset_info():
    dataset_dir = config.DATASET_DIR
    if not os.path.exists(dataset_dir):
        return {"total_images": 0, "models": {}, "total_size_mb": 0.0}

    total_images = 0
    total_bytes = 0
    models_stats = {}

    for root, _, files in os.walk(dataset_dir):
        rel_dir = os.path.relpath(root, dataset_dir)
        img_files = [f for f in files if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        if img_files:
            model_name = rel_dir if rel_dir != "." else "general"
            models_stats[model_name] = len(img_files)
            total_images += len(img_files)
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                total_bytes += os.path.getsize(fp)

    return {
        "total_images": total_images,
        "models": models_stats,
        "total_size_mb": round(total_bytes / (1024 * 1024), 2),
    }


@app.get("/api/dataset/download")
async def download_dataset():
    dataset_dir = config.DATASET_DIR
    if not os.path.exists(dataset_dir) or not os.listdir(dataset_dir):
        raise HTTPException(status_code=400, detail="Datasetda hali rasmlar mavjud emas")

    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(dataset_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, os.path.dirname(dataset_dir))
                zf.write(abs_path, arcname=rel_path)

    mem_zip.seek(0)
    filename = f"dataset_{int(time.time())}.zip"
    return Response(
        content=mem_zip.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


def _placeholder_tile(w: int, h: int, text: str) -> np.ndarray:
    tile = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(tile, text, (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2, cv2.LINE_AA)
    return tile


def _encode_jpeg(frame: np.ndarray, quality: int = 75) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise HTTPException(status_code=500, detail="Kadr kodlanmadi")
    return buf.tobytes()


@app.get("/api/cameras/{cam_id}/preview")
async def camera_preview(cam_id: str, raw: int = 0, max_w: int = 960, max_h: int = 540):
    cam = camera_manager.get_camera(cam_id)
    if cam is None:
        raise HTTPException(status_code=404, detail="Kamera topilmadi")

    frame = cam.get_raw_frame() if raw else cam.get_annotated_frame()
    if frame is None:
        frame = _placeholder_tile(640, 360, f"{cam.name}: ulanmoqda...")
    else:
        h, w = frame.shape[:2]
        if max_w > 0 and max_h > 0 and (w > max_w or h > max_h):
            scale = min(max_w / w, max_h / h)
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    return Response(content=_encode_jpeg(frame, quality=72), media_type="image/jpeg")


@app.get("/api/cameras_grid_preview")
async def cameras_grid_preview():
    cams = camera_manager.list_cameras()
    tw, th = config.GRID_TILE_W, config.GRID_TILE_H

    if not cams:
        grid = _placeholder_tile(tw, th, "Kamera qo'shilmagan")
        return Response(content=_encode_jpeg(grid), media_type="image/jpeg")

    cols = int(np.ceil(np.sqrt(len(cams))))
    rows = int(np.ceil(len(cams) / cols))

    tiles = []
    for cam in cams:
        frame = cam.get_annotated_frame()
        if frame is None:
            status = cam.last_error or "ulanmoqda..."
            tile = _placeholder_tile(tw, th, f"{cam.name}: {status}")
        else:
            tile = cv2.resize(frame, (tw, th))

        cv2.rectangle(tile, (0, 0), (tw, 26), (30, 30, 30), -1)
        dot_color = (0, 230, 0) if cam.connected else (0, 0, 255)
        cv2.circle(tile, (14, 13), 5, dot_color, -1)
        cv2.putText(tile, cam.name, (26, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)

    while len(tiles) < rows * cols:
        tiles.append(np.zeros((th, tw, 3), dtype=np.uint8))

    row_imgs = [np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)]
    grid = np.vstack(row_imgs)

    return Response(content=_encode_jpeg(grid), media_type="image/jpeg")


@app.get("/api/logs/stream")
async def get_stream_logs(limit: int = 100):
    log_path = config.CAMERA_LOG_PATH
    if not os.path.exists(log_path):
        return {"logs": [], "total": 0}
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        recent = lines[-limit:]
        recent.reverse()
        return {"logs": recent, "total": len(lines), "log_file": log_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Log faylini o'qishda xatolik: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("multi_object_detector.api.live_server:app", host=config.SERVER_HOST, port=config.SERVER_PORT, reload=False)
