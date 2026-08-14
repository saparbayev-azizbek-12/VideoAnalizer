import os
import argparse
import uvicorn

VIDEO_SERVER_HOST = os.getenv("VIDEO_SERVER_HOST", "0.0.0.0")
VIDEO_SERVER_PORT = int(os.getenv("VIDEO_SERVER_PORT", "8001"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=VIDEO_SERVER_HOST, help="Server host")
    parser.add_argument("--port", type=int, default=VIDEO_SERVER_PORT, help="Server port")
    args = parser.parse_args()

    print(f"🎬 AI Video Analyzer Server ishga tushmoqda: http://{args.host}:{args.port}")
    uvicorn.run(
        "multi_object_detector.api.video_server:app",
        host=args.host,
        port=args.port,
        reload=False,
    )
