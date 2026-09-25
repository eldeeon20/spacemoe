"""Dataset handling: training blocks (Wiki ES 3MB + FineWeb2-HQ 2MB + Spanish Tweets 1MB)."""

import os, threading
from typing import Optional
from datasets import load_dataset

_DIR = os.path.dirname(os.path.abspath(__file__))
WIKI_CONFIG = ("wikimedia/wikipedia", "20231101.es")
FINEWEB_CONFIG = ("epfml/FineWeb2-HQ", "spa_Latn")
TWEETS_CONFIG = "pysentimiento/spanish-tweets"
TOKENIZER_DATA_PATH = os.path.join(_DIR, "wiki_tokenizer_50mb.txt")
TOKENIZER_FULL_PATH = os.path.join(_DIR, "tokenizer_70mb.txt")
TOKENIZER_FINEWEB_PATH = os.path.join(_DIR, "fineweb_tokenizer_10mb.txt")
TOKENIZER_TWEETS_PATH = os.path.join(_DIR, "tweets_tokenizer_10mb.txt")
TRAIN_DATA_PATH = os.path.join(_DIR, "wiki_train_data.txt")

# (dataset config, megabytes per block, label)
DEFAULT_MIXES = [
    (FINEWEB_CONFIG, 2.0, "fine"),
    (TWEETS_CONFIG, 1.0, "tuit"),
]


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


def _download_dataset_for_tokenizer(ds_config, max_bytes, output_path, label):
    if os.path.exists(output_path) and os.path.getsize(output_path) >= max_bytes:
        print(f"{label} tokenizer data already at {output_path} ({os.path.getsize(output_path)} bytes)")
        return output_path
    print(f"Downloading {label} {max_bytes // 2**20}MB for tokenizer...")
    if isinstance(ds_config, (tuple, list)):
        ds = load_dataset(*ds_config, split="train", streaming=True)
    else:
        ds = load_dataset(ds_config, split="train", streaming=True)
    with open(output_path, "w", encoding="utf-8") as f:
        written = 0
        for item in ds:
            text = item.get("text") if isinstance(item, dict) else str(item)
            tam = len(text.encode("utf-8"))
            if written + tam > max_bytes:
                break
            f.write(text)
            f.write("\n\n")
            written += tam
    print(f"Written {written} bytes to {output_path}")
    return output_path


def download_tokenizer_corpus(output_path: str = TOKENIZER_FULL_PATH) -> str:
    """Corpus de 70MB para el tokenizer: Wiki ES 50MB + FineWeb2-HQ 10MB + Tweets 10MB."""
    wiki = download_wikipedia_50mb()
    fineweb = _download_dataset_for_tokenizer(FINEWEB_CONFIG, 10_000_000, TOKENIZER_FINEWEB_PATH, "FineWeb2-HQ")
    tweets = _download_dataset_for_tokenizer(TWEETS_CONFIG, 10_000_000, TOKENIZER_TWEETS_PATH, "Spanish Tweets")
    with open(output_path, "w", encoding="utf-8") as fout:
        for p in (wiki, fineweb, tweets):
            with open(p, "r", encoding="utf-8") as fin:
                fout.write(fin.read())
    print(f"Combined tokenizer corpus at {output_path} ({os.path.getsize(output_path)} bytes, ~70MB)")
    return output_path


class StreamingDataset:
    """Stream training data in blocks via persistent iterators.
    
    WikiES is the main block (3MB). Each mix (FineWeb2-HQ 2MB, Spanish Tweets 1MB)
    appends its own bytes to the block via a persistent iterator. A mix block_idx is
    fixed per selected block, so requesting block N downloads block N of wiki, N of
    FineWeb and N of Tweets (no repetition within a pass), like the original design.
    Prefetch thread downloads the next block while training runs on current block.
    """
    def __init__(
        self,
        block_mb: float = 3.0,
        block_idx: int = 0,
        mezcla: bool = True,
        mix_mb: float = 1.0,
        mix_dataset: Optional[tuple[str, str] | str] = None,
        mixes: Optional[list] = None,
    ):
        self.block_mb = block_mb
        self.block_idx = block_idx
        self._path = os.path.join(_DIR, f"wiki_block_{block_idx}.txt")
        self._tokens = None
        self._tokenizer = None
        self.mezcla = mezcla
        self.mix_mb = mix_mb
        self.mix_dataset = mix_dataset if mix_dataset is not None else FINEWEB_CONFIG
        if mixes is None:
            if self.mix_dataset == FINEWEB_CONFIG:
                mixes = DEFAULT_MIXES
            else:
                mixes = [(self.mix_dataset, self.mix_mb, "mix")]
        self.mixes = mixes
        # label -> (iter, offset absoluto en bytes ya consumido)
        self._mix_iters: dict[str, Optional[object]] = {}
        self._mix_byte_pos: dict[str, int] = {}
        # Persistent streaming iterator for main block (created once, lives forever)
        self._wiki_iter = None
        self._wiki_block_idx = 0
        # Prefetch
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_error: Exception | None = None

    def _ensure_wiki_iter(self):
        if self._wiki_iter is None:
            ds = load_dataset(*WIKI_CONFIG, split="train", streaming=True)
            self._wiki_iter = iter(ds)

    def _ensure_mix_iter(self, label: str):
        if label in self._mix_iters:
            return
        ds_config = None
        for cfg, _, f in self.mixes:
            if f == label:
                ds_config = cfg
                break
        if ds_config is None:
            self._mix_iters[label] = None
            return
        if isinstance(ds_config, (tuple, list)):
            ds = load_dataset(*ds_config, split="train", streaming=True)
        else:
            ds = load_dataset(ds_config, split="train", streaming=True)
        self._mix_iters[label] = iter(ds)

    def _read_from_mix_iter(self, label: str, max_bytes: int):
        """Lee la ventana de bytes ABSOLUTA [block_idx*max_bytes, (block_idx+1)*max_bytes).

        Sin esto, reiniciar y pedir el bloque 500 directo devolvía el PRIMER mb del
        stream (el mix no hacía skip, a diferencia de wiki). Con el seek por offset
        absoluto, pedir el 500 siempre da el mismo contenido, directo o secuencial.
        Avanza solo hacia adelante (eficiente en corrida), y desde un stream fresco
        al reiniciar (reproducible entre entornos).
        """
        self._ensure_mix_iter(label)
        it = self._mix_iters.get(label)
        if it is None:
            return [], 0
        skip = self.block_idx * max_bytes
        pos = self._mix_byte_pos.get(label, 0)
        if pos > skip:
            # Wrap/restart (ej. epoch nuevo): stream viejo inservible, crear uno fresco
            # en vez de dejar None (si no, "no se pudo crear el stream" para siempre).
            del self._mix_iters[label]
            self._mix_byte_pos[label] = 0
            self._ensure_mix_iter(label)
            it = self._mix_iters.get(label)
            if it is None:
                return [], 0
            consumed = 0
        else:
            consumed = pos
        exhausted = False
        if consumed < skip:
            print(f"  {label} seek {pos // 2**20}MB -> {skip // 2**20}MB ({skip} bytes)...")
            for item in it:
                text = item.get("text") if isinstance(item, dict) else str(item)
                tam = len(text.encode("utf-8"))
                consumed += tam
                if consumed >= skip:
                    break
            else:
                exhausted = True
            if not exhausted:
                print(f"  {label} seek listo en {consumed // 2**20}MB")
        if exhausted:
            self._mix_iters[label] = None
            self._mix_byte_pos[label] = 0
            return [], 0
        texts = []
        appended = 0
        for item in it:
            text = item.get("text") if isinstance(item, dict) else str(item)
            tam = len(text.encode("utf-8"))
            if tam == 0:
                continue
            if appended + tam > max_bytes:
                break
            texts.append(text)
            appended += tam
        self._mix_iters[label] = it
        self._mix_byte_pos[label] = consumed + appended
        return texts, appended

    def _new_wiki_iter(self):
        ds = load_dataset(*WIKI_CONFIG, split="train", streaming=True)
        return iter(ds)

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

    def download_block(self, mix_path=None, iterator=None):
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

        if getattr(self, "mezcla", False) and self.mixes:
            self._append_mix_maybe(mix_path)
        total = os.path.getsize(self._path)
        exp = self.block_mb + sum(mb for _, mb, _ in self.mixes if mb > 0)
        print(f"  BLOQUE {self.block_idx} total: {total} bytes (~{total/2**20:.1f}MB esperado ~{exp:.0f}MB)")

    def _append_mix_maybe(self, mix_path=None):
        if not getattr(self, "mezcla", False) or not self.mixes:
            return
        for _, mb, label in self.mixes:
            if mb <= 0:
                continue
            mix_bytes = int(mb * 1024 * 1024)
            print(f"  Descargando {label} (bloque {self.block_idx}, {mb}MB)...")
            try:
                self._ensure_mix_iter(label)
                it = self._mix_iters.get(label)
                if it is None:
                    print(f"  {label}: no se pudo crear el stream")
                    continue
                texts, appended = self._read_from_mix_iter(label, mix_bytes)
                if texts:
                    out_path = mix_path or self._path
                    with open(out_path, "a", encoding="utf-8") as f:
                        for t in texts:
                            f.write(t)
                            f.write("\n\n")
                    print(f"  Appended {appended} bytes from {label} for block {self.block_idx}")
                else:
                    print(f"  {label}: 0 bytes (vacío/agotado)")
            except Exception as e:
                print(f"  Mixing skipped {label}: {e}")

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
        exp = int(self.block_mb * 1024 * 1024)
        if self.mezcla and self.mixes:
            exp += sum(int(mb * 1024 * 1024) for _, mb, _ in self.mixes if mb > 0)
        return exp

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
