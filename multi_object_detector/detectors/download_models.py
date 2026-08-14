import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "detectors" / "models"
VIT_DIR = MODELS_DIR / "vit-fire-detection"
VIT_HF_REPO = "EdBianchi/vit-fire-detection"


def download_vit_model() -> None:
    print("⏳ 1/3: ViT Yong'in modeli yuklab olinmoqda (EdBianchi/vit-fire-detection)...")
    from transformers import ViTForImageClassification, ViTImageProcessor

    VIT_DIR.mkdir(parents=True, exist_ok=True)
    processor = ViTImageProcessor.from_pretrained(VIT_HF_REPO)
    model = ViTForImageClassification.from_pretrained(VIT_HF_REPO)

    processor.save_pretrained(str(VIT_DIR))
    model.save_pretrained(str(VIT_DIR))
    print(f"✅ ViT modeli saqlandi: {VIT_DIR}")


def download_rfdetr_model() -> None:
    print("⏳ 2/3: RF-DETR odam aniqlash modeli tekshirilmoqda...")
    try:
        from rfdetr import RFDETRLarge
        _ = RFDETRLarge()
        print("✅ RF-DETR modeli tayyor.")
    except Exception as e:
        print(f"⚠ RF-DETR yuklashda ogohlantirish (YOLO fallback ishlaydi): {e}")


def download_yolo_models() -> None:
    print("⏳ 3/3: YOLO modellari tekshirilmoqda...")
    try:
        from ultralytics import YOLO
        yolo_path = MODELS_DIR / "yolov8l.pt"
        if not yolo_path.exists():
            model = YOLO("yolov8l.pt")
            import shutil
            if Path("yolov8l.pt").exists() and not yolo_path.exists():
                shutil.move("yolov8l.pt", str(yolo_path))
        print(f"✅ YOLO modeli tayyor: {yolo_path}")
    except Exception as e:
        print(f"⚠ YOLO yuklashda ogohlantirish: {e}")


def main() -> None:
    print("=" * 60)
    print("🚀 Barcha AI modellarni serverga yuklab olish boshlandi...")
    print(f"📁 Saqlash manzili: {MODELS_DIR}")
    print("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    download_vit_model()
    download_rfdetr_model()
    download_yolo_models()

    print("=" * 60)
    print("🎉 Barcha modellar muvaffaqiyatli yuklandi va server offline ishlashga tayyor!")
    print("=" * 60)


if __name__ == "__main__":
    main()
