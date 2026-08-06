"""
client.py
---------
Mahalliy kompyuterdan (Local PC) superkompyuterdagi FastAPI serverga video
va xavfli hudud (zone.json) koordinatalarini yuborish, ishlov jarayonini kuzatish
va natijaviy video ile loglarni yuklab olish uchun CLI vositasi.

Ishlatilishi:
    python client.py --server http://superkompyuter-ip:8000 --video camera.mp4 --zone zone.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description="Danger Zone Detector Local PC Client")
    parser.add_argument("--server", default="http://localhost:8000", help="Superkompyuter server URL adresi")
    parser.add_argument("--video", required=True, help="Mahalliy video fayl yo'li")
    parser.add_argument("--zone", default="zone.json", help="Poligon zonalari (zone.json)")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO modeli (yolo11n.pt / yolov8l.pt)")
    parser.add_argument("--conf", type=float, default=0.35, help="Ishonch darajasi (conf threshold)")
    parser.add_argument("--out-video", default="output_result.mp4", help="Saqlanadigan video nomi")
    parser.add_argument("--out-log", default="danger_log.csv", help="Saqlanadigan CSV log fayli")
    args = parser.parse_args()

    server_url = args.server.rstrip("/")
    video_path = Path(args.video)
    zone_path = Path(args.zone)

    if not video_path.exists():
        sys.exit(f"Xatolik: Video fayli topilmadi -> {video_path}")

    if not zone_path.exists():
        sys.exit(f"Xatolik: Zone JSON fayli topilmadi -> {zone_path}")

    # 1. Server holatini tekshirish
    try:
        r = requests.get(f"{server_url}/api/health", timeout=5)
        r.raise_for_status()
        health = r.json()
        print(f"✅ Serverga ulandi: {server_url}")
        print(f"🖥️ GPU Holati: {health.get('gpu_name')} (CUDA: {health.get('gpu_available')})")
    except Exception as e:
        sys.exit(f"❌ Serverga ulanib bo'lmadi ({server_url}): {e}")

    # 2. Zone poligonini o'qish
    with open(zone_path, "r", encoding="utf-8") as f:
        zone_data = json.load(f)
    polygon = zone_data.get("polygon", [])
    if len(polygon) < 3:
        sys.exit("Xatolik: zone.json poligonida kamida 3 nuqta bo'lishi kerak.")

    # 3. Videoni yuklash
    print(f"📤 Video serverga yuklanmoqda ({video_path.name})...")
    with open(video_path, "rb") as f:
        files = {"file": (video_path.name, f, "video/mp4")}
        r_upload = requests.post(f"{server_url}/api/upload", files=files)

    if r_upload.status_code != 200:
        sys.exit(f"❌ Video yuklashda xatolik: {r_upload.text}")

    upload_res = r_upload.json()
    video_id = upload_res["video_id"]
    print(f"✅ Video yuklandi. Video ID: {video_id} ({upload_res['width']}x{upload_res['height']}, {upload_res['duration_seconds']}s)")

    # 4. Qayta ishlash topshirig'ini yuborish
    print("⚙️ Superkompyuterda qayta ishlash boshlanmoqda...")
    payload = {
        "video_id": video_id,
        "polygon": polygon,
        "model_name": args.model,
        "conf_threshold": args.conf,
    }
    r_proc = requests.post(f"{server_url}/api/process", json=payload)
    if r_proc.status_code != 200:
        sys.exit(f"❌ Qayta ishlashni boshlashda xatolik: {r_proc.text}")

    task_id = r_proc.json()["task_id"]
    print(f"⏳ Topshiriq ID: {task_id}")

    # 5. Jarayonni kuzatish (Progress Polling)
    while True:
        r_status = requests.get(f"{server_url}/api/tasks/{task_id}")
        if r_status.status_code == 200:
            status_data = r_status.json()
            status = status_data.get("status")
            progress = status_data.get("progress", 0)
            breach_frames = status_data.get("breach_frames", 0)
            current_frame = status_data.get("current_frame", 0)
            total_frames = status_data.get("total_frames", 0)

            sys.stdout.write(f"\r progress: [{progress:3d}%] frame: {current_frame}/{total_frames} | buzilish kadrlar: {breach_frames}")
            sys.stdout.flush()

            if status == "completed":
                print("\n✅ Superkompyuterda qayta ishlash yakunlandi!")
                break
            elif status == "failed":
                print(f"\n❌ Xatolik yuz berdi: {status_data.get('error')}")
                sys.exit(1)

        time.sleep(1)

    # 6. Natija video va CSV logni yuklab olish
    print(f"📥 Natija videosi yuklanmoqda -> {args.out_video}")
    r_vid = requests.get(f"{server_url}/api/download/{task_id}/video", stream=True)
    if r_vid.status_code == 200:
        with open(args.out_video, "wb") as f:
            for chunk in r_vid.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"💾 Video saqlandi: {args.out_video}")

    print(f"📥 Log CSV yuklanmoqda -> {args.out_log}")
    r_csv = requests.get(f"{server_url}/api/download/{task_id}/csv")
    if r_csv.status_code == 200:
        with open(args.out_log, "wb") as f:
            f.write(r_csv.content)
        print(f"💾 Log saqlandi: {args.out_log}")


if __name__ == "__main__":
    main()
