from __future__ import annotations
import sys
import argparse

from multi_object_detector.core import config


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Video Monitoring & Analyzer Tizimi")
    parser.add_argument("--analyzer", action="store_true", help="Offline video fayl tahlil GUI dasturini ochish")
    parser.add_argument("--server", action="store_true", help="Jonli tahlil GPU serverini ishga tushirish")
    parser.add_argument("--video-server", action="store_true", help="Offline video tahlil serverini ishga tushirish")
    parser.add_argument("--host", default=None, help="Server host")
    parser.add_argument("--port", type=int, default=None, help="Server port")
    args, unknown = parser.parse_known_args()

    if args.analyzer:
        from multi_object_detector.ui.video_analyzer_ui import VideoAnalyzerApp
        app = VideoAnalyzerApp()
        app.mainloop()
    elif args.server:
        import uvicorn
        host = args.host or config.SERVER_HOST
        port = args.port or config.SERVER_PORT
        print(f"🚀 Live Server ishga tushmoqda: http://{host}:{port}")
        uvicorn.run("multi_object_detector.api.live_server:app", host=host, port=port)
    elif args.video_server:
        import uvicorn
        host = args.host or "0.0.0.0"
        port = args.port or 8001
        print(f"🎬 Video Server ishga tushmoqda: http://{host}:{port}")
        uvicorn.run("multi_object_detector.api.video_server:app", host=host, port=port)
    else:
        import threading
        from multi_object_detector.detectors import manager as models_manager
        threading.Thread(target=models_manager.preload_all, daemon=True, name="bg-preload").start()

        try:
            import uvicorn
            host = args.host or config.SERVER_HOST
            port = args.port or config.SERVER_PORT

            def _run_bg_server():
                try:
                    cfg = uvicorn.Config("multi_object_detector.api.live_server:app", host=host, port=port, log_level="warning")
                    server = uvicorn.Server(cfg)
                    server.run()
                except Exception:
                    pass

            threading.Thread(target=_run_bg_server, daemon=True, name="bg-api-server").start()
        except Exception:
            pass

        try:
            import tkinter as tk
            from multi_object_detector.ui.live_monitor_ui import MonitoringApp
        except ImportError as e:
            sys.exit(f"GUI interfeysi uchun Tkinter talab qilinadi: {e}\nServerni ishga tushirish uchun: python run_server.py yoki python main.py --server")
        root = tk.Tk()
        MonitoringApp(root)
        root.mainloop()


if __name__ == "__main__":
    main()