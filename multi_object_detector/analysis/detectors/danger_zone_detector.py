from __future__ import annotations
import sys
import cv2
import csv
import json
import time
import argparse
import numpy as np
import supervision as sv
from typing import Optional
from ultralytics import YOLO
from multi_object_detector import config

PERSON_CLASS_ID = config.PERSON_CLASS_ID

def load_zone(zone_path: str) -> np.ndarray:
    with open(zone_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    polygon = np.array(data["polygon"], dtype=np.int32)
    if len(polygon) < 3:
        sys.exit("Xatolik: zone.json faylida kamida 3 nuqta bo'lishi kerak.")
    return polygon

class DangerZoneState:
    def __init__(self, polygon: np.ndarray, model: Optional[YOLO] = None):
        self.model = model or YOLO(config.PERSON_MODEL_PATH)
        self.polygon = polygon
        self.zone = sv.PolygonZone(polygon=polygon)
        self.zone_annotator = sv.PolygonZoneAnnotator(zone=self.zone, color=sv.Color.RED, thickness=2)
        self.box_annotator = sv.BoxAnnotator(color=sv.Color.RED)
        self.label_annotator = sv.LabelAnnotator()

    def update_polygon(self, polygon: np.ndarray) -> None:
        self.polygon = polygon
        self.zone = sv.PolygonZone(polygon=polygon)
        self.zone_annotator = sv.PolygonZoneAnnotator(zone=self.zone, color=sv.Color.RED, thickness=2)

    def analyze(self, frame: np.ndarray, conf: float = 0.35, draw_boxes: bool = True) -> tuple[np.ndarray, int, bool]:
        result = self.model(frame, conf=conf, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        people = detections[detections.class_id == PERSON_CLASS_ID]
        in_zone_mask = self.zone.trigger(detections=people)
        people_in_zone = int(in_zone_mask.sum())
        breach = people_in_zone > 0

        annotated = self.zone_annotator.annotate(scene=frame)
        if draw_boxes and len(people) > 0:
            labels = [f"person {c:.2f}" for c in people.confidence]
            annotated = self.box_annotator.annotate(scene=annotated, detections=people)
            annotated = self.label_annotator.annotate(scene=annotated, detections=people, labels=labels)

        if breach:
            width = frame.shape[1]
            cv2.rectangle(annotated, (0, 0), (width, 50), (0, 0, 255), -1)
            cv2.putText(annotated, f"DIQQAT! XAVFLI HUDUDDA {people_in_zone} ODAM BOR", (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return annotated, people_in_zone, breach

def main():
    parser = argparse.ArgumentParser(description="Xavfli hududda odam aniqlash tizimi")
    parser.add_argument("--video", required=True, help="Kirish video fayli")
    parser.add_argument("--zone", required=True, help="select_zone.py orqali saqlangan zone.json")
    parser.add_argument("--output", default="output.mp4", help="Chiqish (annotatsiyalangan) video")
    parser.add_argument("--model", default=config.PERSON_MODEL_PATH, help="Ultralytics YOLO model fayli")
    parser.add_argument("--conf", type=float, default=0.35, help="Aniqlash ishonch chegarasi")
    parser.add_argument("--show", action="store_true", help="Jonli oynada ko'rsatish")
    parser.add_argument("--log", default="danger_log.csv", help="Buzilishlar logi (CSV)")
    args = parser.parse_args()

    polygon = load_zone(args.zone)
    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Xatolik: video ochilmadi -> {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    zone = sv.PolygonZone(polygon=polygon)
    zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.RED, thickness=2)
    box_annotator = sv.BoxAnnotator(color=sv.Color.RED)
    label_annotator = sv.LabelAnnotator()

    log_rows = []
    frame_idx = 0
    breach_frames = 0
    t0 = time.time()

    print("Qayta ishlanmoqda...")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = model(frame, conf=args.conf, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        people = detections[detections.class_id == PERSON_CLASS_ID]
        in_zone_mask = zone.trigger(detections=people)
        people_in_zone = int(in_zone_mask.sum())
        breach = people_in_zone > 0
        annotated = zone_annotator.annotate(scene=frame)
        if len(people) > 0:
            labels = [f"person {conf:.2f}" for conf in people.confidence]
            annotated = box_annotator.annotate(scene=annotated, detections=people)
            annotated = label_annotator.annotate(scene=annotated, detections=people, labels=labels)

        if breach:
            breach_frames += 1
            cv2.rectangle(annotated, (0, 0), (width, 50), (0, 0, 255), -1)
            cv2.putText(annotated, f"DIQQAT! XAVFLI HUDUDDA {people_in_zone} ODAM BOR", (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            log_rows.append({"frame": frame_idx, "time_sec": round(frame_idx / fps, 2), "people_in_zone": people_in_zone})

        writer.write(annotated)
        if args.show:
            cv2.imshow("Xavfli hudud monitoring", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Foydalanuvchi tomonidan to'xtatildi.")
                break
        frame_idx += 1

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    with open(args.log, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["frame", "time_sec", "people_in_zone"]
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(log_rows)

    elapsed = time.time() - t0
    print(f"\nTayyor! {frame_idx} kadr {elapsed:.1f} soniyada qayta ishlandi.")
    print(f"Xavfli hududda odam bo'lgan kadrlar soni: {breach_frames}")
    print(f"Natija video: {args.output}")
    print(f"Log fayli: {args.log}")

if __name__ == "__main__":
    main()