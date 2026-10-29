"""Chrome Native Messaging host: lets a browser extension use the local index and embedder.

Not used by anything shipped today; kept for extensions that bring photos from a web service
(the ops below were built for a Google Photos extension: remote items keyed by mediaKey/dedupKey,
with trash/undo).

To hook up an extension (Windows, per user, no admin):
1. A launcher, e.g. host.bat: `"<venv>\\Scripts\\python.exe" -m localphotosearch.native_host %*`
   (Chrome passes the caller origin and --parent-window as arguments; run() ignores them).
2. A host manifest JSON: {"name": "com.localphotosearch.host", "type": "stdio",
   "path": "<host.bat>", "allowed_origins": ["chrome-extension://<id>/"]}
   (pin the extension ID with a "key" in the extension's manifest).
3. Registry: HKCU\\Software\\Google\\Chrome\\NativeMessagingHosts\\com.localphotosearch.host,
   default value = path of the host manifest.

Chrome starts this process when the extension connects (`chrome.runtime.connectNative`) and talks
to it over stdin/stdout: each message is a 4-byte little-endian length + UTF-8 JSON. Messages to
Chrome must stay under 1 MB.

Requests: {"id": n, "op": ..., ...}; replies: {"id": n, "ok": true, "result": ...} or
{"id": n, "ok": false, "error": "..."}. Indexing ("add") runs on its own thread so searches stay
responsive during a library sync.
"""

import base64
import io
import json
import logging
import os
import queue
import struct
import sys
import threading
import traceback

from PIL import Image

from . import config

log = logging.getLogger(__name__)
HOST_NAME = "com.localphotosearch.host"
MAX_THUMBS_PER_REPLY = 24  # ~20 KB each as base64, well under Chrome's 1 MB reply limit


class Host:
    def __init__(self, stdin, stdout):
        self.stdin, self.stdout = stdin, stdout
        self.write_lock = threading.Lock()
        self.ready = threading.Event()
        self.index = None
        self.load_error = None

    # ---- framing --------------------------------------------------------------------------------

    def _read_exact(self, n: int) -> bytes | None:
        chunks, remaining = [], n
        while remaining:
            chunk = self.stdin.read(remaining)
            if not chunk:
                return None  # Chrome closed the port
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def read(self) -> dict | None:
        header = self._read_exact(4)
        if header is None:
            return None
        (length,) = struct.unpack("<I", header)
        body = self._read_exact(length)
        return None if body is None else json.loads(body.decode("utf-8"))

    def send(self, message: dict) -> None:
        data = json.dumps(message, separators=(",", ":")).encode("utf-8")
        with self.write_lock:
            self.stdout.write(struct.pack("<I", len(data)) + data)
            self.stdout.flush()

    # ---- lifecycle ------------------------------------------------------------------------------

    def load(self) -> None:
        try:
            from . import open_index

            self.index = open_index()
        except Exception as e:  # noqa: BLE001 - reported to the extension
            log.exception("Failed to load index")
            self.load_error = str(e)
        self.ready.set()

    def run(self) -> None:
        threading.Thread(target=self.load, daemon=True).start()
        queues = {"add": queue.Queue(), "other": queue.Queue()}
        for q in queues.values():
            threading.Thread(target=self.worker, args=(q,), daemon=True).start()
        while (message := self.read()) is not None:
            queues["add" if message.get("op") == "add" else "other"].put(message)
        log.info("Port closed; exiting")

    def worker(self, q: queue.Queue) -> None:
        while True:
            message = q.get()
            try:
                self.ready.wait()
                if self.load_error:
                    raise RuntimeError(f"Local Photo Search failed to start: {self.load_error}")
                handler = getattr(self, f"op_{message.get('op')}", None)
                if handler is None:
                    raise ValueError(f"unknown op {message.get('op')!r}")
                self.send({"id": message.get("id"), "ok": True, "result": handler(message)})
            except Exception as e:  # noqa: BLE001
                log.warning("op %s failed: %s", message.get("op"), traceback.format_exc())
                self.send({"id": message.get("id"), "ok": False, "error": str(e)})

    # ---- ops ------------------------------------------------------------------------------------

    def op_hello(self, _message) -> dict:
        return {"device": self.index.embedder.device, "stats": self.index.stats()}

    def op_known(self, message) -> dict:
        return {"known": sorted(self.index.remote_known(message["mediaKeys"]))}

    def op_add(self, message) -> dict:
        items = []
        for item in message["items"]:
            data = item.pop("image").split(",", 1)[-1]  # accept a data: URL or bare base64
            item["image"] = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
            items.append(item)
        stored, failed = self.index.add_remote(items)
        return {"stored": stored, "failed": failed, "total": self.index.stats()["google_photos"]}

    def _hits(self, hits) -> list[dict]:
        info = self.index.remote_info([h.id for h in hits])
        return [
            {"id": h.id, "score": round(h.score, 4), "taken": h.taken, **info[h.id]}
            for h in hits
            if h.id in info
        ]

    def op_search(self, message) -> dict:
        from .index import GPHOTOS

        return {"hits": self._hits(self.index.search(message["query"], message.get("k", 200), source=GPHOTOS))}

    def op_similar(self, message) -> dict:
        from .index import GPHOTOS

        with self.index.lock:
            row = self.index.db.execute("SELECT id FROM photos WHERE remote_id=?", (message["mediaKey"],)).fetchone()
        if row is None:
            return {"hits": []}
        return {"hits": self._hits(self.index.similar(row[0], message.get("k", 200), source=GPHOTOS))}

    def op_thumbs(self, message) -> dict:
        thumbs = {}
        for photo_id in message["ids"][:MAX_THUMBS_PER_REPLY]:
            path = self.index.thumb_path(int(photo_id))
            if path.exists():
                thumbs[photo_id] = "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
        return {"thumbs": thumbs}

    def op_trashed(self, message) -> dict:
        self.index.set_remote_trashed(message["mediaKeys"], bool(message.get("trashed", True)))
        return {"total": self.index.stats()["google_photos"]}

    def op_stats(self, _message) -> dict:
        return self.index.stats()


def run() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=config.DATA_DIR / "native-host.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Move Chrome's pipes off fds 0/1 and point those at NUL. On Windows, while one thread blocks
    # reading a synchronous pipe, any other thread that merely queries the same handle (isatty,
    # file type: libraries do this while loading) blocks too, so model loading would deadlock.
    # Stray prints then also go to NUL instead of corrupting the message stream.
    stdin = os.fdopen(os.dup(0), "rb", buffering=0)
    stdout = os.fdopen(os.dup(1), "wb", buffering=0)
    null = os.open(os.devnull, os.O_RDWR)
    os.dup2(null, 0)
    os.dup2(null, 1)
    sys.stdin = open(os.devnull)  # noqa: SIM115 - process-lifetime replacement
    sys.stdout = sys.stderr
    Host(stdin, stdout).run()


if __name__ == "__main__":
    run()
