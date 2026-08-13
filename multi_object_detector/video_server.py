from __future__ import annotations
import os
import sys
import csv
import json
import time
import uuid
import shutil
import threading
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse, StreamingResponse
import cv2
import numpy as np

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, os.path.dirname(_PKG_DIR))

from multi_object_detector import config
from multi_object_detector.analysis.frame_filter import StreamCorruptionFilter
from multi_object_detector.analysis.detectors import fire_detector, ppe_detector, fall_detector

JOBS_DIR = os.path.join(config.BASE_DIR, "video_jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

VIDEO_SERVER_HOST = os.getenv("VIDEO_SERVER_HOST", "0.0.0.0")
VIDEO_SERVER_PORT = int(os.getenv("VIDEO_SERVER_PORT", "8001"))

app = FastAPI(
    title="AI Video Analyzer Server",
    description="Video yuklash, neyron tarmoqlar (Fire, PPE, Fall) orqali tahlil qilish va natijalarni yuklab olish API serveri",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class VideoJob:
    def __init__(
        self,
        job_id: str,
        filename: str,
        video_path: str,
        models_enabled: dict[str, bool],
        every_n_frames: int = 3,
        conf_threshold: float = 0.50,
    ):
        self.job_id = job_id
        self.filename = filename
        self.video_path = video_path
        self.models_enabled = models_enabled
        self.every_n_frames = max(1, every_n_frames)
        self.conf_threshold = conf_threshold

        self.job_dir = os.path.join(JOBS_DIR, job_id)
        os.makedirs(self.job_dir, exist_ok=True)

        self.output_video_path = os.path.join(self.job_dir, f"{job_id}_analyzed.mp4")
        self.csv_path = os.path.join(self.job_dir, f"{job_id}_events.csv")
        self.json_path = os.path.join(self.job_dir, f"{job_id}_events.json")

        self.status = "queued"
        self.progress_pct = 0.0
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 25.0
        self.width = 0
        self.height = 0
        self.duration_sec = 0.0

        self.events: list[dict] = []
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self.error_message: Optional[str] = None

        self.latest_jpeg: Optional[bytes] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "job_id": self.job_id,
                "filename": self.filename,
                "status": self.status,
                "progress_pct": round(self.progress_pct, 1),
                "current_frame": self.current_frame,
                "total_frames": self.total_frames,
                "fps": round(self.fps, 1),
                "width": self.width,
                "height": self.height,
                "duration_sec": round(self.duration_sec, 2),
                "events_count": len(self.events),
                "models_enabled": self.models_enabled,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "completed_at": self.completed_at,
                "error_message": self.error_message,
            }

    def cancel(self):
        self._stop_event.set()
        with self._lock:
            if self.status in ("queued", "processing"):
                self.status = "stopped"


_jobs: dict[str, VideoJob] = {}
_jobs_lock = threading.Lock()

_models_loaded = False
_fire_model = None
_ppe_model = None
_fall_model = None
_model_load_lock = threading.Lock()


def _ensure_models():
    global _models_loaded, _fire_model, _ppe_model, _fall_model
    with _model_load_lock:
        if not _models_loaded:
            try:
                _fire_model = fire_detector.get_model()
            except Exception as e:
                print(f"[VideoServer] Fire model yuklashda ogohlantirish: {e}")
            try:
                _ppe_model = ppe_detector.get_model()
            except Exception as e:
                print(f"[VideoServer] PPE model yuklashda ogohlantirish: {e}")
            try:
                _fall_model = fall_detector.get_model()
            except Exception as e:
                print(f"[VideoServer] Fall model yuklashda ogohlantirish: {e}")
            _models_loaded = True


def _analyze_frame_server(
    frame: np.ndarray,
    frame_idx: int,
    fps: float,
    models_enabled: dict[str, bool],
    fall_pose=None,
    fall_track_states=None,
) -> tuple[np.ndarray, list[dict]]:
    annotated = frame.copy()
    events = []
    ts = frame_idx / max(fps, 1.0)
    h_f, w_f = annotated.shape[:2]

    # 1. Fire / Smoke
    if models_enabled.get("fire") and _fire_model is not None:
        try:
            ann, det_events, has_fire, has_smoke = fire_detector.detect_fire_frame(
                annotated,
                model=_fire_model,
                conf_threshold=config.VIT_FIRE_CONF_THRESHOLD,
                fire_conf_threshold=config.VIT_FIRE_CONF_THRESHOLD,
                min_color_ratio=fire_detector._MIN_FIRE_PIXEL_RATIO,
            )
            annotated = ann
            if has_fire:
                events.append({"frame": frame_idx, "ts": round(ts, 3), "model": "fire", "label": "FIRE", "conf": None})
            if has_smoke:
                events.append({"frame": frame_idx, "ts": round(ts, 3), "model": "fire", "label": "SMOKE", "conf": None})
        except Exception:
            pass

    # 2. PPE Detection
    if models_enabled.get("ppe") and _ppe_model is not None:
        try:
            ann, violations, has_v = ppe_detector.analyze_ppe_frame(
                annotated, model=_ppe_model, conf=config.PPE_CONF_THRESHOLD
            )
            annotated = ann
            for v in violations:
                events.append({
                    "frame": frame_idx,
                    "ts": round(ts, 3),
                    "model": "ppe",
                    "label": "PPE_VIOLATION",
                    "conf": round(v.get("confidence", 0), 3),
                    "missing": v.get("missing", []),
                })
        except Exception:
            pass

    # 3. Fall Detection
    if models_enabled.get("fall") and _fall_model is not None and fall_pose is not None:
        any_fall_detected = False
        try:
            if not hasattr(_fall_model, "_tracker"):
                import supervision as sv
                _fall_model._tracker = sv.ByteTrack()
            _fall_tracker = _fall_model._tracker

            if hasattr(_fall_model, "predict") and not isinstance(_fall_model, YOLO):
                try:
                    detections = _fall_model.predict(annotated, threshold=config.FALL_PERSON_CONF_THRESHOLD)
                except Exception:
                    detections = _fall_model.predict(annotated)
            else:
                res = _fall_model(annotated, classes=[0], verbose=False)[0]
                import supervision as sv
                detections = sv.Detections.from_ultralytics(res)

            person_mask = (detections.class_id == 0)
            persons = detections[person_mask]
            tracked_persons = _fall_tracker.update_with_detections(persons)

            if len(tracked_persons) > 0 and tracked_persons.tracker_id is not None:
                for bbox, track_id_t in zip(tracked_persons.xyxy, tracked_persons.tracker_id):
                    track_id = int(track_id_t)
                    x1, y1, x2, y2 = map(int, bbox)
                    x1, y1 = max(x1, 0), max(y1, 0)
                    x2, y2 = min(x2, w_f), min(y2, h_f)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    bw, bh = x2 - x1, y2 - y1
                    if max(bw, bh) < 30 or (bw * bh) < 500:
                        continue

                    state = fall_track_states.setdefault(track_id, fall_detector.TrackState())
                    state.last_seen_frame = frame_idx
                    ar = bh / max(bw, 1)

                    # Padded crop for RTMPose context
                    pad_x = int(bw * 0.20)
                    pad_y = int(bh * 0.20)
                    cx1 = max(0, x1 - pad_x)
                    cy1 = max(0, y1 - pad_y)
                    cx2 = min(w_f, x2 + pad_x)
                    cy2 = min(h_f, y2 + pad_y)
                    crop = annotated[cy1:cy2, cx1:cx2].copy()
                    kpts, scs = fall_pose(crop)

                    posture = "Unknown"
                    smoothed_angle = 0.0

                    if len(kpts) > 0 and len(scs) > 0:
                        kpt = kpts[0]
                        sc = scs[0]

                        # Always draw RTMPose skeleton lines and coordinates on crop
                        fall_detector.draw_pose_skeleton(crop, kpts, scs, kpt_thr=0.20, draw_coords=True)
                        annotated[cy1:cy2, cx1:cx2] = crop

                        if fall_detector.is_pose_reliable(sc):
                            l_sh = kpt[fall_detector.KEYPOINT_LEFT_SHOULDER]
                            r_sh = kpt[fall_detector.KEYPOINT_RIGHT_SHOULDER]
                            l_hip = kpt[fall_detector.KEYPOINT_LEFT_HIP]
                            r_hip = kpt[fall_detector.KEYPOINT_RIGHT_HIP]

                            l_sh_sc = sc[fall_detector.KEYPOINT_LEFT_SHOULDER]
                            r_sh_sc = sc[fall_detector.KEYPOINT_RIGHT_SHOULDER]
                            l_hip_sc = sc[fall_detector.KEYPOINT_LEFT_HIP]
                            r_hip_sc = sc[fall_detector.KEYPOINT_RIGHT_HIP]

                            if l_sh_sc >= 0.20 and r_sh_sc >= 0.20:
                                sh_x = (l_sh[0] + r_sh[0]) * 0.5
                                sh_y = (l_sh[1] + r_sh[1]) * 0.5
                            elif l_sh_sc >= 0.20:
                                sh_x = float(l_sh[0])
                                sh_y = float(l_sh[1])
                            else:
                                sh_x = float(r_sh[0])
                                sh_y = float(r_sh[1])

                            if l_hip_sc >= 0.20 and r_hip_sc >= 0.20:
                                hip_x = (l_hip[0] + r_hip[0]) * 0.5
                                hip_y = (l_hip[1] + r_hip[1]) * 0.5
                            elif l_hip_sc >= 0.20:
                                hip_x = float(l_hip[0])
                                hip_y = float(l_hip[1])
                            else:
                                hip_x = float(r_hip[0])
                                hip_y = float(r_hip[1])

                            dy = hip_y - sh_y
                            dx = hip_x - sh_x
                            angle = abs(90.0 - np.degrees(math.atan2(dy, dx)))
                            state.angle_history.append(angle)
                            smoothed_angle = float(np.median(state.angle_history))
                            posture = fall_detector.classify_posture(smoothed_angle)

                            # Draw torso line and center points on full frame
                            g_sh = (int(cx1 + sh_x), int(cy1 + sh_y))
                            g_hip = (int(cx1 + hip_x), int(cy1 + hip_y))
                            cv2.line(annotated, g_sh, g_hip, (0, 0, 255), 3, cv2.LINE_AA)
                            cv2.circle(annotated, g_sh, 5, (0, 255, 255), -1, cv2.LINE_AA)
                            cv2.circle(annotated, g_hip, 5, (255, 0, 255), -1, cv2.LINE_AA)

                    # Geometric Aspect-Ratio Fallback
                    if posture == "Unknown":
                        if ar <= 0.85:
                            posture = "Lying Down"
                            smoothed_angle = max(65.0, 90.0 - (ar * 45.0))
                        elif ar <= 1.15:
                            posture = "Falling"
                            smoothed_angle = 45.0
                        else:
                            posture = "Standing"
                            smoothed_angle = 15.0

                    state.posture_history.append(posture)

                    # 5-frame window transition check (e.g., 2 standing + 3 falling, 1 falling + 4 lying, etc.)
                    is_transition_5f = fall_detector.check_5frame_transition(state.posture_history)

                    if posture in ("Falling", "Lying Down") or ar < 0.90:
                        state.falled_count += 1
                        state.standing_frames = 0
                    else:
                        state.falled_count = max(0, state.falled_count - 1)
                        state.standing_frames += 1

                    # Trigger "FALLED" on 5-frame transition or sustained fallen posture
                    if is_transition_5f or (state.is_falled and (posture in ("Falling", "Lying Down") or ar < 1.0)):
                        state.is_falled = True
                        if not state.fall_detected:
                            events.append({
                                "frame": frame_idx, "ts": round(ts, 3), "model": "fall",
                                "label": "FALLED", "conf": None
                            })
                            state.fall_detected = True

                    # Recovery reset when standing back up
                    if posture == "Standing" and state.standing_frames >= 8 and ar > 1.25:
                        state.is_falled = False
                        state.fall_detected = False
                        state.falled_count = 0

                    if state.is_falled or state.fall_detected:
                        any_fall_detected = True
                        box_color = (0, 0, 255)
                        lbl_text = f"ID{track_id}: FALLED ({smoothed_angle:.0f}deg) [x:{x1},y:{y1}]"
                    elif posture == "Falling":
                        box_color = (0, 140, 255)
                        lbl_text = f"ID{track_id}: Falling ({smoothed_angle:.0f}deg) [x:{x1},y:{y1}]"
                    elif posture == "Lying Down":
                        box_color = (0, 140, 255)
                        lbl_text = f"ID{track_id}: Lying ({smoothed_angle:.0f}deg) [x:{x1},y:{y1}]"
                    else:
                        box_color = (0, 200, 60)
                        lbl_text = f"ID{track_id}: Standing ({smoothed_angle:.0f}deg) [x:{x1},y:{y1}]"

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                    (tw, th), bl = cv2.getTextSize(lbl_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    ty = max(th + 4, y1)
                    cv2.rectangle(annotated, (x1, ty - th - 6), (x1 + tw + 6, ty + bl + 2), box_color, -1)
                    cv2.putText(annotated, lbl_text, (x1 + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

            if any_fall_detected:
                cv2.rectangle(annotated, (0, 0), (w_f, 42), (0, 0, 220), -1)
                cv2.putText(annotated, "ALARM: FALL DETECTED (YIQILISH)", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        except Exception:
            pass

    return annotated, events


def _process_video_job(job: VideoJob):
    _ensure_models()

    job.started_at = time.time()
    with job._lock:
        job.status = "processing"

    cap = cv2.VideoCapture(job.video_path)
    if not cap.isOpened():
        with job._lock:
            job.status = "failed"
            job.error_message = "Video faylni ochib bo'lmadi"
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    job.fps = fps
    job.width = w
    job.height = h
    job.total_frames = total
    job.duration_sec = total / fps if fps > 0 else 0.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(job.output_video_path, fourcc, fps, (w, h))

    filt = StreamCorruptionFilter(camera_id=job.job_id)
    fall_pose = None
    fall_states: dict = {}

    if job.models_enabled.get("fall"):
        fall_pose = fall_detector.create_pose_instance()

    frame_idx = 0

    try:
        while cap.isOpened():
            if job._stop_event.is_set():
                break
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            pct = (frame_idx / total) * 100.0

            valid, _ = filt.check_frame(frame)
            if not valid:
                writer.write(frame)
                with job._lock:
                    job.current_frame = frame_idx
                    job.progress_pct = pct
                continue

            if frame_idx % job.every_n_frames == 0:
                annotated, new_events = _analyze_frame_server(
                    frame,
                    frame_idx,
                    fps,
                    job.models_enabled,
                    fall_pose=fall_pose,
                    fall_track_states=fall_states,
                )
                writer.write(annotated)

                _, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                jpeg_bytes = buf.tobytes()

                with job._lock:
                    job.events.extend(new_events)
                    job.current_frame = frame_idx
                    job.progress_pct = pct
                    job.latest_jpeg = jpeg_bytes
            else:
                writer.write(frame)
                with job._lock:
                    job.current_frame = frame_idx
                    job.progress_pct = pct

    except Exception as e:
        with job._lock:
            job.status = "failed"
            job.error_message = str(e)
    finally:
        cap.release()
        writer.release()
        if fall_pose is not None:
            try:
                fall_pose.close()
            except Exception:
                pass

    # Save reports
    with job._lock:
        if not job._stop_event.is_set() and job.status != "failed":
            job.status = "completed"
            job.progress_pct = 100.0
        job.completed_at = time.time()

        # Write CSV
        try:
            with open(job.csv_path, "w", newline="", encoding="utf-8-sig") as f:
                csv_writer = csv.DictWriter(f, fieldnames=["frame", "ts", "model", "label", "conf"])
                csv_writer.writeheader()
                for ev in job.events:
                    csv_writer.writerow({
                        "frame": ev.get("frame"),
                        "ts": ev.get("ts"),
                        "model": ev.get("model"),
                        "label": ev.get("label"),
                        "conf": ev.get("conf"),
                    })
        except Exception:
            pass

        # Write JSON
        try:
            with open(job.json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "job_id": job.job_id,
                    "filename": job.filename,
                    "info": {
                        "fps": job.fps,
                        "width": job.width,
                        "height": job.height,
                        "total_frames": job.total_frames,
                        "duration_sec": job.duration_sec,
                    },
                    "models_enabled": job.models_enabled,
                    "events_count": len(job.events),
                    "events": job.events,
                }, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


# --- API Endpoints ---

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "AI Video Analyzer Server",
        "endpoints": {
            "upload": "POST /api/video/upload",
            "jobs": "GET /api/video/jobs",
            "status": "GET /api/video/status/{job_id}",
            "preview": "GET /api/video/preview/{job_id}",
            "stream": "GET /api/video/stream/{job_id}",
            "events": "GET /api/video/events/{job_id}",
            "download_video": "GET /api/video/download/{job_id}",
            "download_csv": "GET /api/video/download_csv/{job_id}",
            "download_json": "GET /api/video/download_json/{job_id}",
            "cancel": "POST /api/video/cancel/{job_id}",
        }
    }


@app.post("/api/video/upload")
async def upload_and_analyze(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    models: str = Form("fire,ppe,fall"),
    every_n: int = Form(3),
    conf: float = Form(0.50),
):
    job_id = uuid.uuid4().hex[:8]
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    file_ext = os.path.splitext(file.filename)[1] or ".mp4"
    saved_video_path = os.path.join(job_dir, f"input{file_ext}")

    with open(saved_video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    requested_models = [m.strip().lower() for m in models.split(",") if m.strip()]
    models_enabled = {
        "fire": "fire" in requested_models,
        "ppe": "ppe" in requested_models,
        "fall": "fall" in requested_models,
    }

    job = VideoJob(
        job_id=job_id,
        filename=file.filename,
        video_path=saved_video_path,
        models_enabled=models_enabled,
        every_n_frames=every_n,
        conf_threshold=conf,
    )

    with _jobs_lock:
        _jobs[job_id] = job

    background_tasks.add_task(_process_video_job, job)

    return {
        "job_id": job_id,
        "filename": file.filename,
        "status": "queued",
        "models_enabled": models_enabled,
        "every_n_frames": every_n,
    }


@app.get("/api/video/jobs")
def list_jobs():
    with _jobs_lock:
        return [job.to_dict() for job in _jobs.values()]


@app.get("/api/video/status/{job_id}")
def get_job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job topilmadi")
    return job.to_dict()


@app.get("/api/video/preview/{job_id}")
def get_job_preview(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job topilmadi")
    if job.latest_jpeg is None:
        raise HTTPException(status_code=204, detail="Kadr hali tayyor emas")
    return Response(content=job.latest_jpeg, media_type="image/jpeg")


@app.get("/api/video/stream/{job_id}")
def stream_job_preview(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job topilmadi")

    def frame_generator():
        while True:
            with job._lock:
                jpeg = job.latest_jpeg
                status = job.status
            if jpeg is not None:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            if status in ("completed", "stopped", "failed"):
                break
            time.sleep(0.1)

    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/video/events/{job_id}")
def get_job_events(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job topilmadi")
    with job._lock:
        return {"job_id": job_id, "total": len(job.events), "events": job.events}


@app.get("/api/video/download/{job_id}")
def download_video(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job topilmadi")
    if not os.path.exists(job.output_video_path):
        raise HTTPException(status_code=400, detail="Video hali to'liq ishlanmagan yoki mavjud emas")
    return FileResponse(
        path=job.output_video_path,
        filename=f"analyzed_{job.filename}",
        media_type="video/mp4",
    )


@app.get("/api/video/download_csv/{job_id}")
def download_csv(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job topilmadi")
    if not os.path.exists(job.csv_path):
        raise HTTPException(status_code=400, detail="CSV hisobot mavjud emas")
    return FileResponse(
        path=job.csv_path,
        filename=f"events_{job_id}.csv",
        media_type="text/csv",
    )


@app.get("/api/video/download_json/{job_id}")
def download_json(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job topilmadi")
    if not os.path.exists(job.json_path):
        raise HTTPException(status_code=400, detail="JSON hisobot mavjud emas")
    return FileResponse(
        path=job.json_path,
        filename=f"events_{job_id}.json",
        media_type="application/json",
    )


@app.post("/api/video/cancel/{job_id}")
def cancel_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job topilmadi")
    job.cancel()
    return {"job_id": job_id, "status": "stopped"}


@app.delete("/api/video/job/{job_id}")
def delete_job(job_id: str):
    with _jobs_lock:
        job = _jobs.pop(job_id, None)
    if not job:
        raise HTTPException(status_code=404, detail="Job topilmadi")
    job.cancel()
    try:
        shutil.rmtree(job.job_dir, ignore_errors=True)
    except Exception:
        pass
    return {"job_id": job_id, "status": "deleted"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("video_server:app", host=VIDEO_SERVER_HOST, port=VIDEO_SERVER_PORT, reload=False)
