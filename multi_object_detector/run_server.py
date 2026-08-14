import argparse
import uvicorn
from multi_object_detector.core import config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=config.SERVER_HOST, help="Server host")
    parser.add_argument("--port", type=int, default=config.SERVER_PORT, help="Server port")
    args = parser.parse_args()

    print(f"🚀 AI Video Monitoring Server ishga tushmoqda: http://{args.host}:{args.port}")
    uvicorn.run(
        "multi_object_detector.api.live_server:app",
        host=args.host,
        port=args.port,
        reload=False,
    )
