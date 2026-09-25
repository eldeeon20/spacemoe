"""Wikipedia ES: solo wikipedia. Corpus 50MB para el tokenizer y lectura de bloques de entrenamiento."""

import os
import threading
from typing import Iterator, Optional

from datasets import load_dataset

_DIR = os.path.dirname(os.path.abspath(__file__))
WIKI_CONFIG = ("wikimedia/wikipedia", "20231101.es")
TOKENIZER_DATA_PATH = os.path.join(_DIR, "wiki_tokenizer_50mb.txt")


def download_wikipedia_50mb(output_path: str = TOKENIZER_DATA_PATH) -> str:
    if os.path.exists(output_path) and os.path.getsize(output_path) >= 50_000_000:
        print(f"Tokenizer data already at {output_path} ({os.path.getsize(output_path)} bytes)")
        return output_path
    print("Downloading 50MB Wikipedia ES for tokenizer...")
    ds = load_dataset(*WIKI_CONFIG, split="train", streaming=True)
    with open(output_path, "w", encoding="utf-8") as f:
        written = 0
        for item in ds:
            text = f"--- {item['title']} ---\n{item['text']}\n\n"
            tam = len(text.encode("utf-8"))
            if written + tam > 50_000_000:
                break
            f.write(text)
            written += tam
    print(f"Written {written} bytes to {output_path}")
    return output_path


def new_wiki_iter() -> Iterator:
    ds = load_dataset(*WIKI_CONFIG, split="train", streaming=True)
    return iter(ds)


def skip_blocks(iterator, skip_blocks: int, block_mb: float):
    """Salta skip_blocks bloques de wiki y imprime progreso."""
    max_bytes = int(block_mb * 1024 * 1024)
    for b in range(skip_blocks):
        written = 0
        saw_item = False
        for item in iterator:
            saw_item = True
            text = f"--- {item['title']} ---\n{item['text']}\n\n"
            tam = len(text.encode("utf-8"))
            if tam > max_bytes:
                print(f"  Skipping huge Wikipedia article of {tam} bytes while skipping blocks")
                continue
            if written + tam > max_bytes:
                break
            written += tam
        if not saw_item:
            print(f"  Wikipedia stream exhausted while skipping blocks")
            return 0, True
        if (b + 1) % 20 == 0 or b + 1 == skip_blocks:
            print(f"  wiki skip {b + 1}/{skip_blocks} bloques (~{(b + 1) * block_mb:.0f}MB descargados)")
    return None, False


class WikiDataset:
    """Bloques de entrenamiento de Wikipedia ES (3MB por bloque) con skip y prints, solo wiki."""
    def __init__(self, block_mb: float = 3.0, block_idx: int = 0):
        self.block_mb = block_mb
        self.block_idx = block_idx
        self._path = os.path.join(_DIR, f"wiki_block_{block_idx}.txt")
        self._tokens = None
        self._tokenizer = None
        self._wiki_iter = None
        self._wiki_block_idx = 0
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_error: Exception | None = None

    def _ensure_wiki_iter(self):
        if self._wiki_iter is None:
            self._wiki_iter = new_wiki_iter()

    def _new_wiki_iter(self):
        return new_wiki_iter()

    def _download_block_from_iterator(self, iterator, skip_blocks: int, path: str) -> tuple[int, bool]:
        max_bytes = int(self.block_mb * 1024 * 1024)
        for b in range(skip_blocks):
            written = 0
            saw_item = False
            for item in iterator:
                saw_item = True
                text = f"--- {item['title']} ---\n{item['text']}\n\n"
                tam = len(text.encode("utf-8"))
                if tam > max_bytes:
                    print(f"  Skipping huge Wikipedia article of {tam} bytes while skipping blocks")
                    continue
                if written + tam > max_bytes:
                    break
                written += tam
            if not saw_item:
                print(f"  Wikipedia stream exhausted while skipping blocks")
                return 0, True
            if (b + 1) % 20 == 0 or b + 1 == skip_blocks:
                print(f"  wiki skip {b + 1}/{skip_blocks} bloques (~{(b + 1) * self.block_mb:.0f}MB descargados)")

        written = 0
        exhausted = True
        with open(path, "w", encoding="utf-8") as f:
            for item in iterator:
                exhausted = False
                text = f"--- {item['title']} ---\n{item['text']}\n\n"
                tam = len(text.encode("utf-8"))
                if tam > max_bytes:
                    print(f"  Skipping huge Wikipedia article of {tam} bytes while downloading")
                    continue
                if written + tam > max_bytes:
                    break
                f.write(text)
                written += tam
        return written, exhausted

    def download_block(self, iterator=None):
        max_bytes = int(self.block_mb * 1024 * 1024)
        if iterator is None:
            self._ensure_wiki_iter()
            if self._wiki_iter is None or self._wiki_block_idx > self.block_idx:
                self._wiki_iter = self._new_wiki_iter()
                self._wiki_block_idx = 0
            skip_blocks = max(0, self.block_idx - self._wiki_block_idx)
            print(f"  Descargando wiki (bloque {self.block_idx}, {self.block_mb}MB)...")
            written, exhausted = self._download_block_from_iterator(self._wiki_iter, skip_blocks, self._path)
            self._wiki_block_idx = self.block_idx + 1
            print(f"  Escrito wiki: {written} bytes")
        else:
            print(f"  Descargando wiki (prefetch bloque {self.block_idx}, {self.block_mb}MB)...")
            written, exhausted = self._download_block_from_iterator(iterator, self.block_idx, self._path)
            print(f"  Escrito wiki: {written} bytes")

        if exhausted:
            print(f"  Wikipedia stream exhausted while downloading block {self.block_idx}, wrapping on next block")
            self._wiki_iter = None
            self._wiki_block_idx = 0

        total = os.path.getsize(self._path)
        exp = self.block_mb
        print(f"  BLOQUE {self.block_idx} total: {total} bytes (~{total/2**20:.1f}MB esperado ~{exp:.0f}MB)")

    def _prefetch_worker(self, block_idx: int):
        old_block = self.block_idx
        old_path = self._path
        try:
            self._prefetch_error = None
            path = os.path.join(_DIR, f"wiki_block_{block_idx}.txt")
            if not os.path.exists(path):
                self.block_idx = block_idx
                self._path = path
                self.download_block()
        except Exception as e:
            self._prefetch_error = e
        finally:
            self.block_idx = old_block
            self._path = old_path

    def _wait_prefetch(self):
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            self._prefetch_thread.join()
            self._prefetch_thread = None
        if self._prefetch_error is not None:
            err = self._prefetch_error
            self._prefetch_error = None
            print(f"  Prefetch failed for block {self.block_idx + 1}: {err}")

    def _start_prefetch(self, block_idx: int):
        self._wait_prefetch()
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            print(f"  Prefetch previo aún activo; no se inicia otro")
            return
        print(f"  Entrando hilo prefetch: bloque {block_idx}")
        self._prefetch_thread = threading.Thread(target=self._prefetch_worker, args=(block_idx,), daemon=True)
        self._prefetch_thread.start()

    def _expected_bytes(self) -> int:
        return int(self.block_mb * 1024 * 1024)

    def _block_file_ok(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        return os.path.getsize(path) >= int(self._expected_bytes() * 0.8)

    def _load_tokens_from_file(self):
        with open(self._path, "r", encoding="utf-8") as f:
            text = f.read()
        self._tokens = self._tokenizer.encode(text)
        print(f"  Bytes bloque {self.block_idx}: {len(text.encode('utf-8'))} | tokens ids: {len(self._tokens)}")

    def load_tokens(self, tokenizer):
        self._tokenizer = tokenizer
        if not self._block_file_ok(self._path):
            self.download_block()
        self._load_tokens_from_file()
        print(f"Loaded {len(self._tokens)} tokens from block {self.block_idx}")
        self._start_prefetch(self.block_idx + 1)

    def next_block(self):
        self._wait_prefetch()
        old_path = os.path.join(_DIR, f"wiki_block_{self.block_idx}.txt")
        if os.path.exists(old_path):
            os.remove(old_path)
        self._tokens = None
        self.block_idx += 1
        self._path = os.path.join(_DIR, f"wiki_block_{self.block_idx}.txt")
        if not self._block_file_ok(self._path):
            self.download_block()
        self._load_tokens_from_file()
        if len(self._tokens) < 1000:
            os.remove(self._path)
            print(f"  Dataset exhausted at block {self.block_idx}, wrapping to block 0")
            self.block_idx = 0
            self._path = os.path.join(_DIR, f"wiki_block_{self.block_idx}.txt")
            self._wiki_iter = None
            if not self._block_file_ok(self._path):
                self.download_block()
            self._load_tokens_from_file()
        print(f"  Loaded block {self.block_idx}: {len(self._tokens)} tokens")
        self._start_prefetch(self.block_idx + 1)

    def get_tokens(self):
        if self._tokens is None:
            raise ValueError("Call load_tokens() first")
        return self._tokens
