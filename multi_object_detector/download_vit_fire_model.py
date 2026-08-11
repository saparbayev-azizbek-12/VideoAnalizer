"""
ViT fire detection modelini HuggingFace Hub'dan local papkaga yuklab olish.

Foydalanish:
    uv run python download_vit_fire_model.py
"""
from __future__ import annotations
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "analysis" / "models" / "vit-fire-detection"
HF_REPO = "EdBianchi/vit-fire-detection"

def main() -> None:
    bin_file = MODEL_DIR / "pytorch_model.bin"
    if MODEL_DIR.exists() and bin_file.exists() and bin_file.stat().st_size > 10_000_000:
        print(f"Model allaqachon mavjud: {MODEL_DIR}")
        return

    print(f"Model yuklanmoqda: {HF_REPO}")
    print(f"Manzil: {MODEL_DIR}")
    print("Bu bir martayin yuklanadi (~343 MB)...\n")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("XATOLIK: huggingface_hub o'rnatilmagan.")
        print("  uv add huggingface-hub")
        sys.exit(1)

    try:
        snapshot_download(
            repo_id=HF_REPO,
            local_dir=str(MODEL_DIR),
            ignore_patterns=["*.ot", "flax_model*", "tf_model*", "rust_model*"],
        )
        print(f"\nMuvaffaqiyatli yuklandi: {MODEL_DIR}")
    except Exception as e:
        print(f"\nXATOLIK: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
