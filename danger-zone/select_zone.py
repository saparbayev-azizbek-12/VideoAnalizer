"""
select_zone.py
---------------
Video kadridan sichqoncha bilan bosib, "xavfli hudud" (danger zone) poligonini
belgilash uchun interaktiv vosita.

Ishlatilishi:
    python select_zone.py --video kamera.mp4
    python select_zone.py --video kamera.mp4 --frame 50 --output zone.json

Boshqaruv:
    Chap tugma (klik)  -> yangi nuqta qo'shish
    'n'                -> 5 soniya OLDINGA o'tish (kerakli kadrni topish uchun)
    'b'                -> 5 soniya ORQAGA qaytish
    'z'                -> oxirgi nuqtani bekor qilish
    'r'                -> hamma nuqtalarni tozalash
    's'                -> saqlash va chiqish (kamida 3 nuqta kerak)
    'q' / ESC          -> saqlamasdan chiqish
"""

import argparse
import json
import sys

import cv2
import numpy as np

WINDOW_NAME = "Xavfli hududni belgilang (n/b=kadr, s=saqlash, r=tozalash, q=chiqish)"
SEEK_SECONDS = 5  # 'n' / 'b' bosilganda necha soniyaga siljish


def read_frame_at(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def draw_overlay(base_frame: np.ndarray, points: list, frame_index: int, fps: float) -> np.ndarray:
    frame = base_frame.copy()

    for i, pt in enumerate(points):
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        cv2.putText(frame, str(i + 1), (pt[0] + 8, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    if len(points) > 1:
        cv2.polylines(frame, [np.array(points)], isClosed=len(points) > 2,
                       color=(0, 0, 255), thickness=2)
    if len(points) >= 3:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [np.array(points)], color=(0, 0, 255))
        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

    time_sec = frame_index / fps if fps else 0
    info = f"Kadr: {frame_index}  |  Vaqt: {time_sec:.1f}s"
    (tw, th), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (5, 5), (15 + tw, 15 + th), (0, 0, 0), -1)
    cv2.putText(frame, info, (10, 10 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Xavfli hudud poligonini belgilash")
    parser.add_argument("--video", required=True, help="Video fayl yo'li")
    parser.add_argument("--frame", type=int, default=0, help="Qaysi kadrdan boshlash (default: 0)")
    parser.add_argument("--output", default="zone.json", help="Poligon saqlanadigan fayl")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Xatolik: video ochilmadi -> {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    seek_frames = max(1, int(round(SEEK_SECONDS * fps)))

    current_index = args.frame
    current_frame = read_frame_at(cap, current_index)
    if current_frame is None:
        sys.exit(f"Xatolik: {current_index}-kadrni o'qib bo'lmadi.")
    h, w = current_frame.shape[:2]

    points = []

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    print("Video kadrida xavfli hudud burchaklarini ketma-ket bosing.")
    print(f"Kerakli kadrni topish uchun 'n' (+{SEEK_SECONDS}s) / 'b' (-{SEEK_SECONDS}s) tugmalaridan foydalaning.")
    print("Kamida 3 nuqta belgilang, so'ng 's' tugmasi bilan saqlang.")

    while True:
        cv2.imshow(WINDOW_NAME, draw_overlay(current_frame, points, current_index, fps))
        key = cv2.waitKey(20) & 0xFF

        if key == ord('n') or key == ord('b'):
            step = seek_frames if key == ord('n') else -seek_frames
            new_index = current_index + step
            new_index = max(0, new_index)
            if total_frames:
                new_index = min(new_index, total_frames - 1)

            new_frame = read_frame_at(cap, new_index)
            if new_frame is None:
                print("Bu tomonda ko'proq kadr yo'q.")
            else:
                current_index = new_index
                current_frame = new_frame
        elif key == ord('z') and points:
            points.pop()
        elif key == ord('r'):
            points.clear()
        elif key == ord('s'):
            if len(points) >= 3:
                break
            print("Kamida 3 nuqta kerak!")
        elif key == ord('q') or key == 27:  # ESC
            print("Bekor qilindi, hech narsa saqlanmadi.")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

    cap.release()
    cv2.destroyAllWindows()

    data = {
        "video": args.video,
        "frame_size": [w, h],
        "selected_frame": current_index,
        "polygon": [[int(x), int(y)] for x, y in points],
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saqlandi -> {args.output}")
    print(f"Tanlangan kadr: {current_index}")
    print(f"Nuqtalar: {data['polygon']}")


if __name__ == "__main__":
    main()