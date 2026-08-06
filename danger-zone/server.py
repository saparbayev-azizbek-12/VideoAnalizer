"""
server.py
---------
Superkompyuterda (yoki serverda) ishlaydigan FastAPI backend ilovasi.
YOLO model va Supervision poligon xavfli hudud tahlilini bajaradi.

Ishlatilishi:
    uv run python server.py --host 0.0.0.0 --port 8000
"""

import argparse
import asyncio
import csv
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import supervision as sv
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO

PERSON_CLASS_ID = 0  # COCO dataset: "person"

BASE_DIR = Path(__file__).parent.resolve()
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Danger Zone Detection API",
    description="Superkompyuter xavfli hudud monitoring API xizmati",
    version="1.0.0",
)

# CORS sozlamalari (Mahalliy PC brauzeridan ulanish uchun)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Versiya bo'yicha YOLO modellarini saqlash kesh
LOADED_MODELS: Dict[str, YOLO] = {}

# Vazifalar va ularning holati
TASKS: Dict[str, dict] = {}


def get_model(model_name: str) -> YOLO:
    if model_name not in LOADED_MODELS:
        model_path = BASE_DIR / model_name
        if not model_path.exists():
            print(f"Model yuklab olinmoqda: {model_name}")
        LOADED_MODELS[model_name] = YOLO(str(model_path) if model_path.exists() else model_name)
    return LOADED_MODELS[model_name]


class ProcessRequest(BaseModel):
    video_id: str
    polygon: List[List[int]]
    model_name: Optional[str] = "yolo11n.pt"
    conf_threshold: Optional[float] = 0.35


def run_video_processing(task_id: str, video_path: str, polygon_points: List[List[int]], model_name: str, conf: float):
    try:
        TASKS[task_id]["status"] = "processing"
        TASKS[task_id]["progress"] = 0

        polygon = np.array(polygon_points, dtype=np.int32)
        model = get_model(model_name)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            TASKS[task_id]["status"] = "failed"
            TASKS[task_id]["error"] = "Videoni ochib bo'lmadi"
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        out_video_path = OUTPUT_DIR / f"{task_id}_output.mp4"
        out_csv_path = OUTPUT_DIR / f"{task_id}_log.csv"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

        zone = sv.PolygonZone(polygon=polygon)
        zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.RED, thickness=2)
        box_annotator = sv.BoxAnnotator(color=sv.Color.RED)
        label_annotator = sv.LabelAnnotator()

        log_rows = []
        frame_idx = 0
        breach_frames = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = model(frame, conf=conf, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            people = detections[detections.class_id == PERSON_CLASS_ID]

            in_zone_mask = zone.trigger(detections=people)
            people_in_zone = int(in_zone_mask.sum())
            breach = people_in_zone > 0

            annotated = zone_annotator.annotate(scene=frame)
            if len(people) > 0:
                labels = [f"person {c:.2f}" for c in people.confidence]
                annotated = box_annotator.annotate(scene=annotated, detections=people)
                annotated = label_annotator.annotate(scene=annotated, detections=people, labels=labels)

            if breach:
                breach_frames += 1
                cv2.rectangle(annotated, (0, 0), (width, 50), (0, 0, 255), -1)
                cv2.putText(
                    annotated,
                    f"DIQQAT! XAVFLI HUDUDDA {people_in_zone} ODAM BOR",
                    (15, 33),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                log_rows.append(
                    {
                        "frame": frame_idx,
                        "time_sec": round(frame_idx / fps, 2),
                        "people_in_zone": people_in_zone,
                    }
                )

            writer.write(annotated)
            frame_idx += 1

            # Progress update
            progress_pct = min(100, int((frame_idx / total_frames) * 100))
            TASKS[task_id]["progress"] = progress_pct
            TASKS[task_id]["current_frame"] = frame_idx
            TASKS[task_id]["total_frames"] = total_frames
            TASKS[task_id]["breach_frames"] = breach_frames
            TASKS[task_id]["recent_breach"] = breach
            TASKS[task_id]["people_in_zone"] = people_in_zone

        cap.release()
        writer.release()

        # CSV log saqlash
        with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["frame", "time_sec", "people_in_zone"]
            csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
            csv_writer.writeheader()
            csv_writer.writerows(log_rows)

        TASKS[task_id]["status"] = "completed"
        TASKS[task_id]["progress"] = 100
        TASKS[task_id]["output_video"] = f"/api/download/{task_id}/video"
        TASKS[task_id]["output_log"] = f"/api/download/{task_id}/csv"
        TASKS[task_id]["log_summary"] = log_rows[:50]  # dastlabki loglar

    except Exception as e:
        TASKS[task_id]["status"] = "failed"
        TASKS[task_id]["error"] = str(e)


@app.get("/api/health")
def health_check():
    import torch

    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "No GPU (CPU Mode)"
    return {
        "status": "online",
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "active_tasks": len(TASKS),
    }


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename.endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="Faqat video fayllar (.mp4, .avi, .mov, .mkv) qabul qilinadi")

    video_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix
    saved_filename = f"{video_id}{ext}"
    video_path = UPLOAD_DIR / saved_filename

    with open(video_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        os.remove(video_path)
        raise HTTPException(status_code=400, detail="Video o'qishda xatolik yuz berdi")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    duration = round(total_frames / fps, 2)
    cap.release()

    return {
        "video_id": video_id,
        "filename": file.filename,
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
        "duration_seconds": duration,
    }


@app.get("/api/video/{video_id}/frame/{frame_idx}")
def get_video_frame(video_id: str, frame_idx: int):
    # video faylini izlash
    matches = list(UPLOAD_DIR.glob(f"{video_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Video topilmadi")

    video_path = matches[0]
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise HTTPException(status_code=400, detail="Kadrni o'qib bo'lmadi")

    ok_jpeg, jpeg_bytes = cv2.imencode(".jpg", frame)
    if not ok_jpeg:
        raise HTTPException(status_code=500, detail="Kadrni rasmlashda xatolik")

    return StreamingResponse(content=iter([jpeg_bytes.tobytes()]), media_type="image/jpeg")


@app.post("/api/process")
def start_processing(req: ProcessRequest, background_tasks: BackgroundTasks):
    matches = list(UPLOAD_DIR.glob(f"{req.video_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Video topilmadi")

    if len(req.polygon) < 3:
        raise HTTPException(status_code=400, detail="Xavfli hudud poligonida kamida 3 ta nuqta bo'lishi kerak")

    task_id = str(uuid.uuid4())
    video_path = str(matches[0])

    TASKS[task_id] = {
        "task_id": task_id,
        "video_id": req.video_id,
        "status": "queued",
        "progress": 0,
        "current_frame": 0,
        "total_frames": 0,
        "breach_frames": 0,
        "people_in_zone": 0,
        "error": None,
    }

    background_tasks.add_task(
        run_video_processing,
        task_id,
        video_path,
        req.polygon,
        req.model_name or "yolo11n.pt",
        req.conf_threshold or 0.35,
    )

    return {"task_id": task_id, "status": "queued"}


@app.get("/api/tasks/{task_id}")
def get_task_status(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Vazifa topilmadi")
    return TASKS[task_id]


@app.get("/api/download/{task_id}/video")
def download_video(task_id: str):
    file_path = OUTPUT_DIR / f"{task_id}_output.mp4"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Chiqish videosi topilmadi")
    return FileResponse(path=file_path, filename=f"danger_zone_{task_id}.mp4", media_type="video/mp4")


@app.get("/api/download/{task_id}/csv")
def download_csv(task_id: str):
    file_path = OUTPUT_DIR / f"{task_id}_log.csv"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Buzilishlar logi topilmadi")
    return FileResponse(path=file_path, filename=f"danger_log_{task_id}.csv", media_type="text/csv")


# Static fayllarni (Frontend HTML UI) ulash
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


def main():
    parser = argparse.ArgumentParser(description="Danger Zone Detection FastAPI Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host adresi (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Server porti (default: 8000)")
    args = parser.parse_args()

    print(f"🚀 Danger Zone FastAPI Server ishga tushmoqda: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
