"""
Run a small test: download 0.5MB from epfml/FineWeb2-HQ spa_Latn.
Usage: python rust/moe-mla/run_download.py
"""

import os
from datasets import load_dataset

HERE = os.path.dirname(__file__)
OUTPUT_PATH = os.path.join(HERE, "fineweb_spa_test.txt")
DATASET_ID = "epfml/FineWeb2-HQ"
DATASET_CONFIG = "spa_Latn"


def download_fineweb_spanish(output_path: str, block_mb: float = 0.5) -> int:
    mix_bytes = int(block_mb * 1024 * 1024)
    written = 0
    ds_iter = load_dataset(DATASET_ID, DATASET_CONFIG, split="train", streaming=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in ds_iter:
            if isinstance(item, dict):
                text = item.get("text", "")
            else:
                text = str(item)
            tam = len(text.encode("utf-8"))
            if written + tam > mix_bytes:
                break
            f.write(text)
            f.write("\n\n")
            written += tam

    return written


def main():
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)

    print(f"Downloading {DATASET_ID} {DATASET_CONFIG} ...")
    written = download_fineweb_spanish(OUTPUT_PATH, block_mb=0.5)
    print(f"Written {written} bytes to {OUTPUT_PATH}")
    print("Final size:", os.path.getsize(OUTPUT_PATH))

    with open(OUTPUT_PATH, "rb") as f:
        f.seek(max(0, os.path.getsize(OUTPUT_PATH) - 2048))
        print("--- TAIL START ---")
        print(f.read().decode("utf-8", errors="replace"))
        print("--- TAIL END ---")


if __name__ == "__main__":
    main()
