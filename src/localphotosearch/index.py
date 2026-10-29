"""Photo index: incremental folder scan, embedding, SQLite storage and in-memory vector search.

Photos come from local folders (source "local") or from Google Photos via the Chrome extension
(source "gphotos", path "gphotos:<mediaKey>", no local file).
"""

import logging
import os
import sqlite3
import threading
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
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
# Added after the first release; applied to existing databases by _migrate().
EXTRA_COLUMNS = {
    "source": "TEXT NOT NULL DEFAULT 'local'",
    "remote_id": "TEXT",  # Google Photos mediaKey
    "dedup_key": "TEXT",  # Google Photos dedupKey (used to trash/restore)
    "location": "TEXT",
    "trashed": "INTEGER NOT NULL DEFAULT 0",  # moved to Google Photos Trash; kept for undo
}
LOCAL, GPHOTOS = "local", "gphotos"
GPHOTOS_PREFIX = "gphotos:"


def google_photos_url(path: str) -> str | None:
    """photos.google.com URL for a Google Photos item's path, else None."""
    return f"https://photos.google.com/photo/{path[len(GPHOTOS_PREFIX):]}" if path.startswith(GPHOTOS_PREFIX) else None


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
        self.db = sqlite3.connect(data_dir / "index.db", check_same_thread=False, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")  # desktop app and extension host may share the db
        self.db.executescript(SCHEMA)
        self._migrate()
        self._check_model_settings()
        self._load_vectors()

    def _migrate(self) -> None:
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(photos)")}
        for name, ddl in EXTRA_COLUMNS.items():
            if name not in columns:
                self.db.execute(f"ALTER TABLE photos ADD COLUMN {name} {ddl}")
        self.db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS photos_remote_id ON photos(remote_id) WHERE remote_id IS NOT NULL"
        )
        self.db.commit()

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
                for pid, path, size, mtime in self.db.execute(
                    "SELECT id, path, size, mtime FROM photos WHERE source=?", (LOCAL,)
                )
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
                for pid, path in self.db.execute(
                    "SELECT id, path FROM photos WHERE vector IS NULL AND error IS NULL AND source=?", (LOCAL,)
                )
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
                "SELECT id, path, taken, vector FROM photos WHERE vector IS NOT NULL AND trashed=0 ORDER BY id"
            ).fetchall()
            self.ids = np.zeros(0, dtype=np.int64)
            self.paths, self.taken = [], []
            self.remote = np.zeros(0, dtype=bool)
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
        self.remote = np.concatenate([self.remote, np.array([r[1].startswith(GPHOTOS_PREFIX) for r in rows])])
        self.matrix = np.concatenate([self.matrix, self._truncate(full.astype(np.float32))])
        self._row_of = {int(pid): i for i, pid in enumerate(self.ids)}

    def _truncate(self, vectors: np.ndarray) -> np.ndarray:
        """Matryoshka truncation to `search_dim`, then re-normalize."""
        v = vectors[..., : self.search_dim]
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    def __len__(self) -> int:
        with self.lock:
            return len(self.ids) + len(self._pending)

    def _top(self, query: np.ndarray, k: int, exclude: int | None = None, source: str | None = None) -> list[Hit]:
        with self.lock:
            self._merge_pending()
            ids, paths, taken, remote, matrix = self.ids, self.paths, self.taken, self.remote, self.matrix
        valid = np.ones(len(ids), dtype=bool)
        if source is not None:
            valid &= remote if source == GPHOTOS else ~remote
        if exclude is not None:
            valid &= ids != exclude
        candidates = np.flatnonzero(valid)
        if not len(candidates):
            return []
        scores = matrix[candidates] @ self._truncate(query)
        k = min(k, len(candidates))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [
            Hit(int(ids[candidates[i]]), paths[candidates[i]], float(scores[i]), taken[candidates[i]]) for i in top
        ]

    def search(self, query: str, k: int = 60, source: str | None = None) -> list[Hit]:
        return self._top(self.embedder.embed_texts([query])[0], k, source=source)

    def similar(self, photo_id: int, k: int = 60, source: str | None = None) -> list[Hit]:
        with self.lock:
            self._merge_pending()
            row = self._row_of.get(photo_id)
            if row is None:
                return []
            query = self.matrix[row].copy()
        return self._top(query, k, exclude=photo_id, source=source)

    def recent(self, k: int = 200, source: str | None = None) -> list[Hit]:
        """Indexed photos, newest first (EXIF date, else file mtime)."""
        with self.lock:
            rows = self.db.execute(
                "SELECT id, path, taken FROM photos WHERE vector IS NOT NULL AND trashed=0 "
                "AND (? IS NULL OR source=?) "
                "ORDER BY COALESCE(REPLACE(SUBSTR(taken, 1, 10), ':', '-') || SUBSTR(taken, 11), "
                "DATETIME(mtime, 'unixepoch')) DESC LIMIT ?",
                (source, source, k),
            ).fetchall()
        return [Hit(r[0], r[1], 0.0, r[2]) for r in rows]

    # ---- Google Photos items (added by the Chrome extension) --------------------------------

    def remote_known(self, media_keys: list[str]) -> set[str]:
        """Media keys that already have a vector (trashed ones count as known)."""
        if not media_keys:
            return set()
        marks = ",".join("?" * len(media_keys))
        with self.lock:
            rows = self.db.execute(
                f"SELECT remote_id FROM photos WHERE vector IS NOT NULL AND remote_id IN ({marks})", media_keys
            ).fetchall()
        return {r[0] for r in rows}

    def add_remote(self, items: list[dict]) -> tuple[int, dict[str, str]]:
        """Embed and store Google Photos items.

        Each item: mediaKey, dedupKey, timestamp (ms), width, height, location, image (PIL).
        Returns (count stored, {mediaKey: error}).
        """
        stored, failed = 0, {}
        replaced = False  # re-adding an indexed item must not duplicate it in the search matrix
        stream = self.embedder.embed_stream(iter(items), lambda item: item["image"])
        try:
            for item, vec, error in stream:
                key = item["mediaKey"]
                if error is not None:
                    failed[key] = str(error)
                    continue
                taken = datetime.fromtimestamp(item["timestamp"] / 1000).strftime("%Y:%m:%d %H:%M:%S")
                path, blob = f"{GPHOTOS_PREFIX}{key}", vec.astype(np.float16).tobytes()
                with self.lock:
                    pid = self.db.execute(
                        "INSERT INTO photos (path, root, size, mtime, width, height, taken, vector, error, "
                        "source, remote_id, dedup_key, location, trashed) "
                        "VALUES (?, ?, 0, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, 0) "
                        "ON CONFLICT(path) DO UPDATE SET vector=excluded.vector, error=NULL, trashed=0, "
                        "dedup_key=excluded.dedup_key, location=excluded.location RETURNING id",
                        (path, GPHOTOS, item["timestamp"] / 1000, item.get("width"), item.get("height"), taken, blob,
                         GPHOTOS, key, item.get("dedupKey"), item.get("location")),
                    ).fetchone()[0]
                    if pid in self._row_of or any(row[0] == pid for row in self._pending):
                        replaced = True
                    else:
                        self._pending.append((pid, path, taken, blob))
                thumb = item["image"].copy()
                thumb.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.Resampling.BICUBIC)
                thumb.save(self.thumb_path(pid), "JPEG", quality=82)
                stored += 1
        finally:
            stream.close()
            with self.lock:
                self.db.commit()
            if replaced:
                self._load_vectors()
        return stored, failed

    def remote_info(self, photo_ids: list[int]) -> dict[int, dict]:
        """mediaKey, dedupKey and location for Google Photos items among `photo_ids`."""
        if not photo_ids:
            return {}
        marks = ",".join("?" * len(photo_ids))
        with self.lock:
            rows = self.db.execute(
                f"SELECT id, remote_id, dedup_key, location FROM photos WHERE source=? AND id IN ({marks})",
                [GPHOTOS, *photo_ids],
            ).fetchall()
        return {r[0]: {"mediaKey": r[1], "dedupKey": r[2], "location": r[3]} for r in rows}

    def set_remote_trashed(self, media_keys: list[str], trashed: bool) -> None:
        """Hide (or bring back) items moved to (or restored from) Google Photos Trash."""
        if not media_keys:
            return
        marks = ",".join("?" * len(media_keys))
        with self.lock:
            self.db.execute(f"UPDATE photos SET trashed=? WHERE remote_id IN ({marks})", [int(trashed), *media_keys])
            self.db.commit()
        self._load_vectors()

    def delete_photos(self, photo_ids: list[int]) -> tuple[list[int], dict[int, str]]:
        """Move photos to the Recycle Bin and drop them from the index.

        Returns (deleted ids, {id: error}) — a photo that can't be recycled stays indexed.
        """
        deleted, failed = [], {}
        for photo_id in photo_ids:
            path = self.photo_path(photo_id)
            if path is None:
                continue
            if path.startswith(GPHOTOS_PREFIX):
                failed[photo_id] = "Google Photos item: use Move to Trash in the Chrome side panel"
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
            total, failed, gphotos = self.db.execute(
                "SELECT COUNT(*), COUNT(error), "
                "SUM(source='gphotos' AND vector IS NOT NULL AND trashed=0) FROM photos"
            ).fetchone()
        return {
            "photos": total,
            "indexed": len(self),
            "failed": failed,
            "google_photos": gphotos or 0,
            "roots": self.roots(),
        }
