from __future__ import annotations
import os
import traceback
import threading
from datetime import datetime
from multi_object_detector import config

LOG_FILE_PATH = os.path.join(config.BASE_DIR, "system_debug.log")
_log_lock = threading.Lock()


class SystemLogger:
    def __init__(self, log_path: str = LOG_FILE_PATH):
        self.log_path = log_path
        self._ensure_log_file()

    def _ensure_log_file(self):
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
            if not os.path.exists(self.log_path):
                with open(self.log_path, "w", encoding="utf-8") as f:
                    f.write(f"=== SYSTEM DEBUG LOG INITIALIZED AT {datetime.now()} ===\n")
        except Exception as e:
            pass

    def log(self, level: str, tag: str, message: str, exc: Exception | None = None, throttle_sec: float = 0.0, throttle_key: str = ""):
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        thread_name = threading.current_thread().name
        log_line = f"[{now_str}] [{level.upper():5s}] [{thread_name}] [{tag}] {message}\n"

        if exc is not None:
            tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            log_line += f"  TRACEBACK:\n{tb_str}\n"

        with _log_lock:
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(log_line)
                    f.flush()
            except Exception as e:
                pass

    def info(self, tag: str, message: str):
        self.log("INFO", tag, message)

    def warning(self, tag: str, message: str, exc: Exception | None = None):
        self.log("WARN", tag, message, exc=exc)

    def error(self, tag: str, message: str, exc: Exception | None = None):
        self.log("ERROR", tag, message, exc=exc)

    def debug(self, tag: str, message: str):
        self.log("DEBUG", tag, message)


sys_logger = SystemLogger()
