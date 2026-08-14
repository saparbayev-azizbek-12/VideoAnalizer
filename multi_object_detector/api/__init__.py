from __future__ import annotations
from multi_object_detector.api.live_server import app as live_app
from multi_object_detector.api.video_server import app as video_app

__all__ = ["live_app", "video_app"]
