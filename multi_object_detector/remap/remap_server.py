from __future__ import annotations
import os
import cv2
import time
import threading
import numpy as np
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

app = FastAPI(title="Camera Manual Remap Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StartManualRequest(BaseModel):
    source: str
    k1: float = -0.15
    k2: float = 0.02
    focal_scale: float = 1.0


class UpdateManualRequest(BaseModel):
    k1: float
    k2: float
    focal_scale: float = 1.0


class SaveManualRequest(BaseModel):
    save_path: str = "calib.npz"


class StartTestRequest(BaseModel):
    source: str
    calib_path: str = "calib.npz"
    alpha: float = 1.0


class RemapManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.mode = "idle"
        self.running = False
        self.thread: Optional[threading.Thread] = None

        self.source = ""
        self.calib_path = "calib.npz"

        self.manual_k1 = -0.15
        self.manual_k2 = 0.02
        self.manual_focal_scale = 1.0
        self.image_size: Optional[tuple[int, int]] = None

        self.test_frame: Optional[np.ndarray] = None
        self.last_error: Optional[str] = None

        self.map1: Optional[np.ndarray] = None
        self.map2: Optional[np.ndarray] = None

    def stop(self):
        with self.lock:
            self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)
        with self.lock:
            self.mode = "idle"
            self.test_frame = None
            self.last_error = None

    def start_manual_test(self, source: str, k1: float, k2: float, focal_scale: float):
        self.stop()
        with self.lock:
            self.mode = "manual"
            self.source = source
            self.manual_k1 = k1
            self.manual_k2 = k2
            self.manual_focal_scale = focal_scale
            self.running = True

        self.thread = threading.Thread(target=self._manual_loop, daemon=True)
        self.thread.start()

    def update_manual_params(self, k1: float, k2: float, focal_scale: float):
        with self.lock:
            self.manual_k1 = k1
            self.manual_k2 = k2
            self.manual_focal_scale = focal_scale

    def save_manual_calibration(self, save_path: str = "calib.npz") -> dict:
        with self.lock:
            w, h = self.image_size if self.image_size is not None else (1920, 1080)
            f = max(w, h) * self.manual_focal_scale
            cx, cy = w / 2.0, h / 2.0
            camera_matrix = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
            dist_coeffs = np.array([self.manual_k1, self.manual_k2, 0.0, 0.0, 0.0], dtype=np.float64)

            np.savez(
                save_path,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                image_size=np.array([w, h]),
                mean_error=0.0,
            )
        return {"ok": True, "save_path": save_path, "k1": self.manual_k1, "k2": self.manual_k2}

    def _draw_grid(self, img: np.ndarray, grid_step: int = 40, color: tuple[int, int, int] = (0, 0, 0)) -> None:
        h, w = img.shape[:2]
        for x in range(grid_step, w, grid_step):
            cv2.line(img, (x, 0), (x, h), color, 1, cv2.LINE_AA)
        for y in range(grid_step, h, grid_step):
            cv2.line(img, (0, y), (w, y), color, 1, cv2.LINE_AA)
        cv2.line(img, (w // 2, 0), (w // 2, h), color, 2, cv2.LINE_AA)
        cv2.line(img, (0, h // 2), (w, h // 2), color, 2, cv2.LINE_AA)

    def _manual_loop(self):
        try:
            try:
                src: int | str = int(self.source)
            except ValueError:
                src = self.source
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        except Exception as e:
            with self.lock:
                self.last_error = f"Kamera ochishda xatolik: {e}"
                self.running = False
            return

        while True:
            with self.lock:
                if not self.running:
                    break
                k1 = self.manual_k1
                k2 = self.manual_k2
                f_scale = self.manual_focal_scale

            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(1.0)
                continue

            h, w = frame.shape[:2]
            f = max(w, h) * f_scale
            cx, cy = w / 2.0, h / 2.0
            cam_mat = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
            dist_c = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)

            new_cam_mat, _ = cv2.getOptimalNewCameraMatrix(cam_mat, dist_c, (w, h), alpha=1.0)
            undistorted = cv2.undistort(frame, cam_mat, dist_c, None, new_cam_mat)

            h_disp = 480
            scale = h_disp / frame.shape[0]
            left = cv2.resize(frame, (int(frame.shape[1] * scale), h_disp))
            right = cv2.resize(undistorted, (int(undistorted.shape[1] * scale), h_disp))

            self._draw_grid(left)
            self._draw_grid(right)

            cv2.putText(left, "ASL KADR (Fisheye)", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(right, f"TO'G'IRLANGAN (k1={k1:.2f}, k2={k2:.2f})", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            combined = np.hstack([left, right])

            with self.lock:
                self.image_size = (w, h)
                self.test_frame = combined
                self.last_error = None

            time.sleep(0.03)

        cap.release()

    def start_test(self, source: str, calib_path: str, alpha: float):
        self.stop()
        if not os.path.exists(calib_path):
            raise HTTPException(status_code=400, detail=f"Kalibratsiya fayli topilmadi: {calib_path}")

        data = np.load(calib_path)
        camera_matrix = data["camera_matrix"]
        dist_coeffs = data["dist_coeffs"]
        w, h = int(data["image_size"][0]), int(data["image_size"][1])

        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), alpha=alpha
        )
        map1, map2 = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_16SC2
        )

        with self.lock:
            self.mode = "test"
            self.source = source
            self.calib_path = calib_path
            self.map1 = map1
            self.map2 = map2
            self.running = True

        self.thread = threading.Thread(target=self._test_loop, daemon=True)
        self.thread.start()

    def _test_loop(self):
        try:
            try:
                src: int | str = int(self.source)
            except ValueError:
                src = self.source
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        except Exception as e:
            with self.lock:
                self.last_error = f"Kamera ochishda xatolik: {e}"
                self.running = False
            return

        while True:
            with self.lock:
                if not self.running:
                    break

            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(1.0)
                continue

            undistorted = cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)

            h_disp = 480
            scale = h_disp / frame.shape[0]
            left = cv2.resize(frame, (int(frame.shape[1] * scale), h_disp))
            right = cv2.resize(undistorted, (int(undistorted.shape[1] * scale), h_disp))

            self._draw_grid(left)
            self._draw_grid(right)

            cv2.putText(left, "ASL KADR (Original)", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(right, "TO'G'IRLANGAN KADR (Undistorted)", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            combined = np.hstack([left, right])

            with self.lock:
                self.test_frame = combined
                self.last_error = None

            time.sleep(0.03)

        cap.release()


manager = RemapManager()


@app.get("/api/health")
def health():
    return {"status": "ok", "mode": manager.mode}


@app.get("/api/status")
def status():
    with manager.lock:
        return {
            "mode": manager.mode,
            "running": manager.running,
            "last_error": manager.last_error,
            "k1": manager.manual_k1,
            "k2": manager.manual_k2,
        }


@app.post("/api/start_manual")
def start_manual(req: StartManualRequest):
    try:
        manager.start_manual_test(req.source.strip(), req.k1, req.k2, req.focal_scale)
        return {"ok": True, "mode": "manual"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/update_manual")
def update_manual(req: UpdateManualRequest):
    manager.update_manual_params(req.k1, req.k2, req.focal_scale)
    return {"ok": True}


@app.post("/api/save_manual")
def save_manual(req: SaveManualRequest):
    return manager.save_manual_calibration(req.save_path.strip())


@app.post("/api/start_test")
def start_test(req: StartTestRequest):
    try:
        manager.start_test(req.source.strip(), req.calib_path.strip(), req.alpha)
        return {"ok": True, "mode": "test"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/stop")
def stop():
    manager.stop()
    return {"ok": True, "mode": "idle"}


@app.get("/api/download_calib")
def download_calib(filename: str = "calib.npz"):
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail="Kalibratsiya fayli topilmadi")
    return FileResponse(filename, media_type="application/octet-stream", filename=os.path.basename(filename))


def _gen_mjpeg_test():
    while True:
        with manager.lock:
            frame = manager.test_frame
        if frame is not None:
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
                )
        time.sleep(0.04)


@app.get("/api/test_feed")
def test_feed():
    return StreamingResponse(
        _gen_mjpeg_test(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
