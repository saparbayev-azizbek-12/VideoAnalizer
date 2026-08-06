"""
main.py - Yong'in, Tutun va Yiqilishni aniqlash veb ilovasi (FastAPI + ONNX)

Ishga tushirish:
    uvicorn main:app --reload --port 8002

Keyin brauzerda: http://127.0.0.1:8002
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import combined_detector_v2

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_FILE_SIZE_MB = 300

app = FastAPI(title="Multi-Object Detection Monitor v2 (Fire, Smoke & Fall)")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Xotiradagi job holati
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

    raw_output = OUTPUT_DIR / f"{job_id}_raw.mp4"
    web_output = OUTPUT_DIR / f"{job_id}.mp4"

    def progress_callback(current: int, total: int) -> None:
        pct = int((current / total) * 90) if total else 0
        JOBS[job_id]["progress"] = pct

    try:
        result = combined_detector_v2.process_video(
            str(input_path),
            str(raw_output),
            progress_callback=progress_callback,
        )

        JOBS[job_id]["progress"] = 92
        combined_detector_v2.reencode_for_web(str(raw_output), str(web_output))
        JOBS[job_id]["progress"] = 100

        JOBS[job_id] = {
            "status": "done",
            "progress": 100,
            "video_url": f"/outputs/{web_output.name}",
            "fire_detected": result.fire_detected,
            "smoke_detected": result.smoke_detected,
            "fall_detected": result.fall_detected,
            "events": [
                {
                    "frame": ev.frame_index,
                    "time_sec": round(ev.timestamp_sec, 2),
                    "type": ev.event_type,
                    "confidence": ev.confidence,
                    "track_id": ev.track_id,
                }
                for ev in result.events
            ],
            "fps": result.fps,
            "total_frames": result.total_frames,
        }
    except Exception as exc:
        JOBS[job_id] = {"status": "error", "error": str(exc)}
    finally:
        input_path.unlink(missing_ok=True)
        raw_output.unlink(missing_ok=True)


@app.get("/health")
def health():
    return {"status": "ok"}
