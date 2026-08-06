"""
main.py - Elderly Fall Detection veb ilovasi (FastAPI) - Double Detector Test (v1 & v2)

Ishga tushirish:
    uvicorn main:app --reload

Keyin brauzerda: http://127.0.0.1:8000
"""

from __future__ import annotations

import uuid
from typing import Any
from pathlib import Path
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile

import fall_detector_v1
import fall_detector_v2

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_FILE_SIZE_MB = 300

app = FastAPI(title="Elderly Fall Detection v1 vs v2 Benchmark")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Oddiy xotiradagi job holati
JOBS: dict[str, dict[str, Any]] = {}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze")
async def analyze(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Qo'llab-quvvatlanmaydigan format: {ext or 'nomaʼlum'}")

    job_id = uuid.uuid4().hex
    input_path = UPLOAD_DIR / f"{job_id}{ext}"

    size = 0
    with input_path.open("wb") as out_file:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_FILE_SIZE_MB * 1024 * 1024:
                out_file.close()
                input_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail=f"Fayl {MAX_FILE_SIZE_MB}MB dan katta bo'lmasligi kerak")
            out_file.write(chunk)

    JOBS[job_id] = {"status": "queued", "progress": 0}
    background_tasks.add_task(_run_pipeline, job_id, input_path)
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Bunday job topilmadi")
    return job


def _run_pipeline(job_id: str, input_path: Path) -> None:
    JOBS[job_id]["status"] = "processing"

    v1_raw = OUTPUT_DIR / f"{job_id}_v1_raw.mp4"
    v1_web = OUTPUT_DIR / f"{job_id}_v1.mp4"
    v2_raw = OUTPUT_DIR / f"{job_id}_v2_raw.mp4"
    v2_web = OUTPUT_DIR / f"{job_id}_v2.mp4"

    def progress_cb_v1(current: int, total: int) -> None:
        pct = int((current / total) * 45) if total else 0
        JOBS[job_id]["progress"] = pct

    def progress_cb_v2(current: int, total: int) -> None:
        pct = 50 + (int((current / total) * 45) if total else 0)
        JOBS[job_id]["progress"] = pct

    try:
        # --- Detector V1 ---
        result_v1 = fall_detector_v1.process_video(str(input_path), str(v1_raw), progress_callback=progress_cb_v1)
        fall_detector_v1.reencode_for_web(str(v1_raw), str(v1_web))
        JOBS[job_id]["progress"] = 50

        # --- Detector V2 ---
        result_v2 = fall_detector_v2.process_video(str(input_path), str(v2_raw), progress_callback=progress_cb_v2)
        fall_detector_v2.reencode_for_web(str(v2_raw), str(v2_web))
        JOBS[job_id]["progress"] = 100

        JOBS[job_id] = {
            "status": "done",
            "progress": 100,
            "v1": {
                "video_url": f"/outputs/{v1_web.name}",
                "fall_detected": result_v1.fall_detected,
                "fall_events": [
                    {"frame": ev.frame_index, "time_sec": round(ev.timestamp_sec, 2)}
                    for ev in result_v1.fall_events
                ],
                "fps": result_v1.fps,
                "total_frames": result_v1.total_frames,
            },
            "v2": {
                "video_url": f"/outputs/{v2_web.name}",
                "fall_detected": result_v2.fall_detected,
                "fall_events": [
                    {
                        "frame": ev.frame_index,
                        "time_sec": round(ev.timestamp_sec, 2),
                        "track_id": getattr(ev, "track_id", None)
                    }
                    for ev in result_v2.fall_events
                ],
                "fps": result_v2.fps,
                "total_frames": result_v2.total_frames,
            }
        }
    except Exception as exc:
        JOBS[job_id] = {"status": "error", "error": str(exc)}
    finally:
        input_path.unlink(missing_ok=True)
        v1_raw.unlink(missing_ok=True)
        v2_raw.unlink(missing_ok=True)


@app.get("/health")
def health():
    return {"status": "ok"}
