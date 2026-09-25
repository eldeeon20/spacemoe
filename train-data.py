"""TrainData: arma bloques con los 3 datasets independientes (wiki 3MB + fine 2MB + tuit 1MB).

Cada fuente usa su propio archivo (wikipedia.py, fine-web.py, data-tuit.py) con su
skip_blocks por bloques. Cuando un bloque queda listo se lanza un hilo de prefetch
para el siguiente (+1). Al pedir un bloque se borran los archivos viejos, igual que
dataset.py. NO toca ni modifica dataset.py.
"""

import importlib
import os
import threading
from typing import Optional

import wikipedia

fine_web = importlib.import_module("fine-web")
data_tuit = importlib.import_module("data-tuit")

_DIR = os.path.dirname(os.path.abspath(__file__))

WIKI_MB = 3.0
FINE_MB = 2.0
TUIT_MB = 1.0


class TrainData:
    def __init__(self, block_idx: int = 0):
        self.block_idx = block_idx
        self.wiki_mb = WIKI_MB
        self.fine_mb = FINE_MB
        self.tuit_mb = TUIT_MB
        self._path = os.path.join(_DIR, f"wiki_block_{block_idx}.txt")
        self._tokens = None
        self._tokenizer = None
        self._iters: dict[str, Optional[object]] = {"wiki": None, "fine": None, "tuit": None}
        self._block_pos: dict[str, int] = {"wiki": 0, "fine": 0, "tuit": 0}
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_error: Exception | None = None

    def _mod_for(self, label: str):
        if label == "wiki":
            return wikipedia
        if label == "fine":
            return fine_web
        return data_tuit

    def _mb_for(self, label: str) -> float:
        if label == "wiki":
            return self.wiki_mb
        if label == "fine":
            return self.fine_mb
        return self.tuit_mb

    def _text_for(self, label: str, item) -> str:
        if label == "wiki":
            return f"--- {item['title']} ---\n{item['text']}\n\n"
        text = item.get("text") if isinstance(item, dict) else str(item)
        return text + "\n\n"

    def _ensure_iter(self, label: str):
        if self._iters[label] is not None:
            return
        if label == "wiki":
            self._iters[label] = wikipedia.new_wiki_iter()
        elif label == "fine":
            self._iters[label] = fine_web.new_fineweb_iter()
        elif label == "tuit":
            self._iters[label] = data_tuit.new_tweets_iter()
        self._block_pos[label] = 0

    def _skip_to_block(self, label: str) -> bool:
        self._ensure_iter(label)
        it = self._iters[label]
        if it is None:
            return False
        target = self.block_idx
        pos = self._block_pos[label]
        if pos > target:
            self._iters[label] = None
            self._block_pos[label] = 0
            return False
        skip = target - pos
        if skip > 0:
            res, exhausted = self._mod_for(label).skip_blocks(it, skip, self._mb_for(label))
            if exhausted:
                self._iters[label] = None
                self._block_pos[label] = 0
                return False
            self._block_pos[label] = target
        return True

    def _append_source(self, label: str, path: str, mode: str):
        if not self._skip_to_block(label):
            print(f"  {label}: stream agotado, se recicla en el próximo bloque")
            return
        it = self._iters[label]
        max_bytes = int(self._mb_for(label) * 1024 * 1024)
        print(f"  Descargando {label} (bloque {self.block_idx}, {self._mb_for(label)}MB)...")
        written = 0
        with open(path, mode, encoding="utf-8") as f:
            for item in it:
                text = self._text_for(label, item)
                tam = len(text.encode("utf-8"))
                if tam > max_bytes:
                    print(f"  Skipping huge {label} item of {tam} bytes")
                    continue
                if written + tam > max_bytes:
                    break
                f.write(text)
                written += tam
        print(f"  Escrito {label}: {written} bytes")

    def download_block(self):
        path = self._path
        self._append_source("wiki", path, "w")
        self._append_source("fine", path, "a")
        self._append_source("tuit", path, "a")
        total = os.path.getsize(path)
        exp = self.wiki_mb + self.fine_mb + self.tuit_mb
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
        return int((self.wiki_mb + self.fine_mb + self.tuit_mb) * 1024 * 1024)

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
        old_path = os.path.join(_DIR, f"wiki_block_{self.block_idx - 1}.txt")
        if os.path.exists(old_path):
            os.remove(old_path)
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
            self._iters = {"wiki": None, "fine": None, "tuit": None}
            self._block_pos = {"wiki": 0, "fine": 0, "tuit": 0}
            if not self._block_file_ok(self._path):
                self.download_block()
            self._load_tokens_from_file()
        print(f"  Loaded block {self.block_idx}: {len(self._tokens)} tokens")
        self._start_prefetch(self.block_idx + 1)

    def get_tokens(self):
        if self._tokens is None:
            raise ValueError("Call load_tokens() first")
        return self._tokens
