from __future__ import annotations
import os
import time
import socket
import threading
import urllib.parse
from datetime import datetime
from multi_object_detector.core import config

_log_lock = threading.Lock()


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _mask_source_credentials(source: str) -> str:
    try:
        parsed = urllib.parse.urlparse(source)
        if parsed.password:
            masked_netloc = f"{parsed.username}:***@{parsed.hostname}"
            if parsed.port:
                masked_netloc += f":{parsed.port}"
            return urllib.parse.urlunparse((parsed.scheme, masked_netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))
    except Exception:
        pass
    return source


def diagnose_rtsp_connection(source: str) -> str:
    try:
        parsed = urllib.parse.urlparse(source)
        if parsed.scheme.lower() in ("rtsp", "rtsps", "http", "https"):
            host = parsed.hostname
            port = parsed.port or (554 if "rtsp" in parsed.scheme.lower() else 80)
            if not host:
                return "RTSP URL formati noto'g'ri (Host ko'rsatilmagan)"

            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1.5)
            try:
                s.connect((host, port))
                s.close()
                return f"Tarmoq porti ({host}:{port}) ochiq, lekin RTSP oqim ochilmadi (Login/parol yoki kanal manzili xato bo'lishi mumkin)"
            except socket.timeout:
                return f"Kameraga ulanishda timeout (IP: {host}:{port} javob bermadi - kamera offline yoki tarmoq uzilgan)"
            except ConnectionRefusedError:
                return f"Ulanish rad etildi (IP: {host}:{port} porti yopiq / kamera xizmati ishlamayapti)"
            except Exception as e:
                return f"Tarmoq ulanish xatosi (IP: {host}:{port}): {e}"
        else:
            return "Lokal kamera manbasi topilmadi yoki band qilingan"
    except Exception as e:
        return f"Diagnostika xatoligi: {e}"


class CameraStreamLogger:
    def __init__(self, log_path: str | None = None):
        self.log_path = log_path or getattr(config, "CAMERA_LOG_PATH", "camera_stream_errors.log")
        self._last_log_times: dict[str, float] = {}
        self._disconnect_start_time: dict[str, float] = {}
        self._was_disconnected: dict[str, bool] = {}

    def log_event(
        self,
        camera_id: str,
        camera_name: str,
        source: str,
        event_type: str,
        reason: str,
        throttle_sec: float = 0.0,
        extra: str = "",
    ) -> None:
        now = time.time()
        key = f"{camera_id}_{event_type}_{reason[:40]}"
        last_t = self._last_log_times.get(key, 0.0)

        if throttle_sec > 0 and (now - last_t < throttle_sec):
            return
        self._last_log_times[key] = now

        masked_src = _mask_source_credentials(source)
        extra_str = f" | {extra}" if extra else ""
        entry = (
            f"{_now_str()} | [CAM: {camera_name} ({camera_id})] | [{event_type}] | "
            f"Manba: {masked_src} | Sabab: {reason}{extra_str}\n"
        )

        with _log_lock:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(entry)
                    f.flush()
            except Exception:
                pass

    def log_connect_fail(self, camera_id: str, camera_name: str, source: str) -> str:
        diag = diagnose_rtsp_connection(source)
        if not self._was_disconnected.get(camera_id, False):
            self._disconnect_start_time[camera_id] = time.time()
            self._was_disconnected[camera_id] = True
        self.log_event(
            camera_id=camera_id,
            camera_name=camera_name,
            source=source,
            event_type="CONNECT_FAILED",
            reason=diag,
            throttle_sec=8.0,
        )
        return diag

    def log_stream_drop(self, camera_id: str, camera_name: str, source: str) -> str:
        if not self._was_disconnected.get(camera_id, False):
            self._disconnect_start_time[camera_id] = time.time()
            self._was_disconnected[camera_id] = True
        diag = diagnose_rtsp_connection(source)
        reason = f"Kadr o'qish uzildi (RTSP EOF / Packet timeout). Holat: {diag}"
        self.log_event(
            camera_id=camera_id,
            camera_name=camera_name,
            source=source,
            event_type="STREAM_DISCONNECTED",
            reason=reason,
            throttle_sec=6.0,
        )
        return reason

    def log_corruption(self, camera_id: str, camera_name: str, source: str, corrupt_reason: str) -> None:
        self.log_event(
            camera_id=camera_id,
            camera_name=camera_name,
            source=source,
            event_type="FRAME_CORRUPTED",
            reason=corrupt_reason,
            throttle_sec=3.0,
        )

    def log_reconnect_success(self, camera_id: str, camera_name: str, source: str) -> None:
        if self._was_disconnected.get(camera_id, False):
            start_t = self._disconnect_start_time.pop(camera_id, None)
            duration_str = ""
            if start_t is not None:
                dt = time.time() - start_t
                duration_str = f"Uzilish davomiyligi: {dt:.1f} soniya"
            self._was_disconnected[camera_id] = False
            self.log_event(
                camera_id=camera_id,
                camera_name=camera_name,
                source=source,
                event_type="RECONNECT_SUCCESS",
                reason="Kamera oqimi qayta tiklandi",
                throttle_sec=0.0,
                extra=duration_str,
            )

    def log_error(self, camera_id: str, camera_name: str, source: str, err_msg: str) -> None:
        self.log_event(
            camera_id=camera_id,
            camera_name=camera_name,
            source=source,
            event_type="ANALYSIS_ERROR",
            reason=err_msg,
            throttle_sec=5.0,
        )


stream_logger = CameraStreamLogger()
