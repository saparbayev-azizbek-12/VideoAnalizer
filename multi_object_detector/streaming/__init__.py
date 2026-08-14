from __future__ import annotations
from multi_object_detector.streaming.camera_manager import (
    CameraWorker,
    CameraEvent,
    add_camera,
    remove_camera,
    get_camera,
    list_cameras,
    stop_all,
)

__all__ = [
    "CameraWorker",
    "CameraEvent",
    "add_camera",
    "remove_camera",
    "get_camera",
    "list_cameras",
    "stop_all",
]
