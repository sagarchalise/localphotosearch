"""Photo index: incremental folder scan, embedding, SQLite storage and in-memory vector search."""

import logging
import os
import sqlite3
import threading
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from send2trash import send2trash

from .embedder import Embedder

register_heif_opener()
log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".heic", ".heif"}
THUMB_SIZE = 256
DECODE_SIZE = 1024  # JPEG draft target: large enough for 70-1120 soft tokens and thumbnails
COMMIT_EVERY = 25
EXIF_IFD, DATETIME_ORIGINAL, DATETIME = 0x8769, 36867, 306

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS roots (path TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    root TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime REAL NOT NULL,
    width INTEGER,
    height INTEGER,
    taken TEXT,
    vector BLOB,
    error TEXT
);
"""


@dataclass
class Hit:
    id: int
    path: str
    score: float
    taken: str | None


@dataclass
class Progress:
    done: int
    total: int
    failed: int
    current: str | None = None


@dataclass
class Decoded:
    path: str
    image: Image.Image | None
    width: int | None = None
    height: int | None = None
    taken: str | None = None
    error: str | None = None


def scan_folder(root: Path) -> Iterator[tuple[str, int, float]]:
    """Yield (path, size, mtime) for every image under `root`."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                path = os.path.join(dirpath, name)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                yield path, st.st_size, st.st_mtime


def decode(path: str, thumb_path: Path) -> Decoded:
    """Open, orient and downscale an image; write its thumbnail. Runs on a worker thread."""
    try:
        with Image.open(path) as im:
            width, height = im.size
            exif = im.getexif()
            taken = exif.get_ifd(EXIF_IFD).get(DATETIME_ORIGINAL) or exif.get(DATETIME)
            im.draft("RGB", (DECODE_SIZE, DECODE_SIZE))  # fast DCT-domain downscale for JPEG
            image = ImageOps.exif_transpose(im).convert("RGB")
        image.thumbnail((DECODE_SIZE, DECODE_SIZE), Image.Resampling.BICUBIC)
        thumb = image.copy()
        thumb.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.Resampling.BICUBIC)
        thumb.save(thumb_path, "JPEG", quality=82)
        return Decoded(path, image, width, height, str(taken) if taken else None)
    except Exception as e:  # noqa: BLE001 - corrupt/unsupported files are recorded, not fatal
        return Decoded(path, None, error=f"{type(e).__name__}: {e}")


def prefetch(items: Iterable, fn: Callable, depth: int, workers: int) -> Iterator:
    """Map `fn` over `items` on a thread pool, keeping at most `depth` results in flight, in order."""
    with ThreadPoolExecutor(workers) as pool:
        pending = deque()
        for item in items:
            pending.append(pool.submit(fn, *item))
            if len(pending) >= depth:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


class PhotoIndex:
    def __init__(self, data_dir: Path, embedder: Embedder, search_dim: int = 768):
        self.data_dir = data_dir
        self.thumb_dir = data_dir / "thumbs"
        self.thumb_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.search_dim = search_dim
        self.lock = threading.RLock()
        self.db = sqlite3.connect(data_dir / "index.db", check_same_thread=False)
        self.db.executescript(SCHEMA)
        self._check_model_settings()
        self._load_vectors()

    # ---- settings -------------------------------------------------------------------------

    def _model_signature(self) -> str:
        e = self.embedder
        return f"{e.paths['model'].name}|{e.paths['vision_encoder'].name}|soft_tokens={e.soft_tokens}"

    def _check_model_settings(self) -> None:
        """Drop stored vectors if they were produced by a different model configuration."""
        sig = self._model_signature()
        row = self.db.execute("SELECT value FROM meta WHERE key='model'").fetchone()
        if row and row[0] != sig:
            log.info("Model settings changed (%s -> %s); vectors will be rebuilt", row[0], sig)
            self.db.execute("UPDATE photos SET vector=NULL, error=NULL")
        self.db.execute("INSERT OR REPLACE INTO meta VALUES ('model', ?)", (sig,))
        self.db.commit()

    # ---- roots ----------------------------------------------------------------------------

    def roots(self) -> list[str]:
        with self.lock:
            return [r[0] for r in self.db.execute("SELECT path FROM roots ORDER BY path")]

    def add_root(self, path: str) -> None:
        with self.lock:
            self.db.execute("INSERT OR IGNORE INTO roots VALUES (?)", (str(Path(path).resolve()),))
            self.db.commit()

    def remove_root(self, path: str) -> None:
        with self.lock:
            self.db.execute("DELETE FROM roots WHERE path=?", (path,))
            ids = [r[0] for r in self.db.execute("SELECT id FROM photos WHERE root=?", (path,))]
            self.db.execute("DELETE FROM photos WHERE root=?", (path,))
            self.db.commit()
        for photo_id in ids:
            self.thumb_path(photo_id).unlink(missing_ok=True)
        self._load_vectors()

    # ---- indexing -------------------------------------------------------------------------

    def thumb_path(self, photo_id: int) -> Path:
        return self.thumb_dir / f"{photo_id}.jpg"

    def _plan(self) -> tuple[list[tuple[int, str]], int]:
        """Sync the photos table with disk. Returns (photos needing embedding, count removed)."""
        with self.lock:
            known = {
                path: (pid, size, mtime)
                for pid, path, size, mtime in self.db.execute("SELECT id, path, size, mtime FROM photos")
            }
            seen, todo = set(), []
            for root in self.roots():
                for path, size, mtime in scan_folder(Path(root)):
                    seen.add(path)
                    old = known.get(path)
                    if old is None:
                        cur = self.db.execute(
                            "INSERT INTO photos (path, root, size, mtime) VALUES (?, ?, ?, ?)",
                            (path, root, size, mtime),
                        )
                        todo.append((cur.lastrowid, path))
                    elif old[1] != size or old[2] != mtime:
                        self.db.execute(
                            "UPDATE photos SET size=?, mtime=?, vector=NULL, error=NULL WHERE id=?",
                            (size, mtime, old[0]),
                        )
                        todo.append((old[0], path))
            # Also pick up rows left unembedded by a cancelled run or a model change.
            queued = {pid for pid, _ in todo}
            todo += [
                (pid, path)
                for pid, path in self.db.execute("SELECT id, path FROM photos WHERE vector IS NULL AND error IS NULL")
                if pid not in queued and path in seen
            ]
            gone = [(pid, path) for path, (pid, _, _) in known.items() if path not in seen]
            for pid, _ in gone:
                self.db.execute("DELETE FROM photos WHERE id=?", (pid,))
                self.thumb_path(pid).unlink(missing_ok=True)
            self.db.commit()
        self._load_vectors()  # drop removed/changed photos from the in-memory search set
        return todo, len(gone)

    def update(
        self,
        on_progress: Callable[[Progress], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> Progress:
        """Scan all roots and embed new/changed photos. Safe to cancel and resume."""
        todo, removed = self._plan()
        log.info("Index update: %d to embed, %d removed", len(todo), removed)
        progress = Progress(0, len(todo), 0)
        if on_progress:
            on_progress(progress)
        id_by_path = {path: pid for pid, path in todo}
        jobs = ((path, self.thumb_path(pid)) for pid, path in todo)
        workers = max(2, min(8, (os.cpu_count() or 4) // 2))

        decoded_stream = prefetch(jobs, decode, depth=workers * 2, workers=workers)
        embedded = self.embedder.embed_stream(decoded_stream, lambda d: d.image)
        try:
            for decoded, vec, error in embedded:
                if cancel and cancel.is_set():
                    break
                pid = id_by_path[decoded.path]
                progress.current = decoded.path
                if decoded.image is None:
                    self._store(pid, None, decoded, decoded.error)
                    progress.failed += 1
                elif error is not None:
                    self._store(pid, None, decoded, f"embed: {error}")
                    progress.failed += 1
                else:
                    self._store(pid, vec, decoded, None)
                progress.done += 1
                if progress.done % COMMIT_EVERY == 0:
                    with self.lock:
                        self.db.commit()
                if on_progress:
                    on_progress(progress)
        finally:
            embedded.close()
            with self.lock:
                self.db.commit()
        return progress

    def _store(self, pid: int, vec: np.ndarray | None, d: Decoded, error: str | None) -> None:
        blob = vec.astype(np.float16).tobytes() if vec is not None else None
        with self.lock:
            self.db.execute(
                "UPDATE photos SET vector=?, width=?, height=?, taken=?, error=? WHERE id=?",
                (blob, d.width, d.height, d.taken, error, pid),
            )
            if blob is not None:  # searchable immediately, merged into the matrix on next search
                self._pending.append((pid, d.path, d.taken, blob))

    # ---- search ---------------------------------------------------------------------------

    def _load_vectors(self) -> None:
        with self.lock:
            rows = self.db.execute(
                "SELECT id, path, taken, vector FROM photos WHERE vector IS NOT NULL ORDER BY id"
            ).fetchall()
            self.ids = np.zeros(0, dtype=np.int64)
            self.paths, self.taken = [], []
            self.matrix = np.zeros((0, self.search_dim), dtype=np.float32)
            self._row_of: dict[int, int] = {}
            self._pending = rows
            self._merge_pending()

    def _merge_pending(self) -> None:
        """Append vectors stored since the last search to the in-memory matrix (call under lock)."""
        if not self._pending:
            return
        rows, self._pending = self._pending, []
        full = np.frombuffer(b"".join(r[3] for r in rows), dtype=np.float16).reshape(len(rows), -1)
        self.ids = np.concatenate([self.ids, np.array([r[0] for r in rows], dtype=np.int64)])
        self.paths += [r[1] for r in rows]
        self.taken += [r[2] for r in rows]
        self.matrix = np.concatenate([self.matrix, self._truncate(full.astype(np.float32))])
        self._row_of = {int(pid): i for i, pid in enumerate(self.ids)}

    def _truncate(self, vectors: np.ndarray) -> np.ndarray:
        """Matryoshka truncation to `search_dim`, then re-normalize."""
        v = vectors[..., : self.search_dim]
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    def __len__(self) -> int:
        with self.lock:
            return len(self.ids) + len(self._pending)

    def _top(self, query: np.ndarray, k: int, exclude: int | None = None) -> list[Hit]:
        with self.lock:
            self._merge_pending()
            ids, paths, taken, matrix = self.ids, self.paths, self.taken, self.matrix
        if not len(ids):
            return []
        scores = matrix @ self._truncate(query)
        k = min(k + (exclude is not None), len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        hits = [Hit(int(ids[i]), paths[i], float(scores[i]), taken[i]) for i in top if ids[i] != exclude]
        return hits[: k - (exclude is not None)]

    def search(self, query: str, k: int = 60) -> list[Hit]:
        return self._top(self.embedder.embed_texts([query])[0], k)

    def similar(self, photo_id: int, k: int = 60) -> list[Hit]:
        with self.lock:
            self._merge_pending()
            row = self._row_of.get(photo_id)
            if row is None:
                return []
            query = self.matrix[row].copy()
        return self._top(query, k, exclude=photo_id)

    def recent(self, k: int = 200) -> list[Hit]:
        """Indexed photos, newest first (EXIF date, else file mtime)."""
        with self.lock:
            rows = self.db.execute(
                "SELECT id, path, taken FROM photos WHERE vector IS NOT NULL "
                "ORDER BY COALESCE(REPLACE(SUBSTR(taken, 1, 10), ':', '-') || SUBSTR(taken, 11), "
                "DATETIME(mtime, 'unixepoch')) DESC LIMIT ?",
                (k,),
            ).fetchall()
        return [Hit(r[0], r[1], 0.0, r[2]) for r in rows]

    def delete_photos(self, photo_ids: list[int]) -> tuple[list[int], dict[int, str]]:
        """Move photos to the Recycle Bin and drop them from the index.

        Returns (deleted ids, {id: error}) — a photo that can't be recycled stays indexed.
        """
        deleted, failed = [], {}
        for photo_id in photo_ids:
            path = self.photo_path(photo_id)
            if path is None:
                continue
            try:
                if os.path.exists(path):
                    send2trash(path)
            except Exception as e:  # noqa: BLE001 - locked/permission/no recycle bin: report, keep indexed
                failed[photo_id] = str(e)
                continue
            deleted.append(photo_id)
        if deleted:
            with self.lock:
                self.db.executemany("DELETE FROM photos WHERE id=?", [(i,) for i in deleted])
                self.db.commit()
            for photo_id in deleted:
                self.thumb_path(photo_id).unlink(missing_ok=True)
            self._load_vectors()
        return deleted, failed

    def photo_path(self, photo_id: int) -> str | None:
        with self.lock:
            row = self.db.execute("SELECT path FROM photos WHERE id=?", (photo_id,)).fetchone()
        return row[0] if row else None

    def stats(self) -> dict:
        with self.lock:
            total, failed = self.db.execute(
                "SELECT COUNT(*), COUNT(error) FROM photos"
            ).fetchone()
        return {"photos": total, "indexed": len(self), "failed": failed, "roots": self.roots()}
