"""Spanish Tweets (pysentimiento/spanish-tweets): lectura de bloques de entrenamiento con skip y progreso, como wikipedia.py."""

import os
from typing import Iterator

from datasets import load_dataset

_DIR = os.path.dirname(os.path.abspath(__file__))
TWEETS_CONFIG = "pysentimiento/spanish-tweets"
TOKENIZER_TWEETS_PATH = os.path.join(_DIR, "tweets_tokenizer_10mb.txt")


def download_tweets_10mb(output_path: str = TOKENIZER_TWEETS_PATH) -> str:
    if os.path.exists(output_path) and os.path.getsize(output_path) >= 10_000_000:
        print(f"Tweets tokenizer data already at {output_path} ({os.path.getsize(output_path)} bytes)")
        return output_path
    print("Downloading 10MB Spanish Tweets for tokenizer...")
    ds = load_dataset(TWEETS_CONFIG, split="train", streaming=True)
    with open(output_path, "w", encoding="utf-8") as f:
        written = 0
        for item in ds:
            text = item.get("text") if isinstance(item, dict) else str(item)
            tam = len(text.encode("utf-8"))
            if written + tam > 10_000_000:
                break
            f.write(text)
            f.write("\n\n")
            written += tam
    print(f"Written {written} bytes to {output_path}")
    return output_path


def new_tweets_iter() -> Iterator:
    ds = load_dataset(TWEETS_CONFIG, split="train", streaming=True)
    return iter(ds)


def skip_blocks(iterator, skip_blocks: int, block_mb: float):
    """Salta skip_blocks bloques de tweets y imprime progreso, como wikipedia.py."""
    max_bytes = int(block_mb * 1024 * 1024)
    for b in range(skip_blocks):
        written = 0
        saw_item = False
        for item in iterator:
            saw_item = True
            text = item.get("text") if isinstance(item, dict) else str(item)
            tam = len(text.encode("utf-8"))
            if tam > max_bytes:
                print(f"  Skipping huge tweet of {tam} bytes while skipping blocks")
                continue
            if written + tam > max_bytes:
                break
            written += tam
        if not saw_item:
            print(f"  Tweets stream exhausted while skipping blocks")
            return 0, True
        if (b + 1) % 20 == 0 or b + 1 == skip_blocks:
            print(f"  tuit skip {b + 1}/{skip_blocks} bloques (~{(b + 1) * block_mb:.0f}MB descargados)")
    return None, False
