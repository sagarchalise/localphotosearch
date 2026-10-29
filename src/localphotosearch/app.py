"""Desktop app: pywebview window + local media server + JS API over the photo index."""

import io
import logging
import os
import secrets
import subprocess
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import webview
from PIL import Image, ImageOps

from . import config
from .index import google_photos_url
from .watcher import FolderWatcher

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent / "web"
PREVIEW_SIZE = 1600
STATIC_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css", ".js": "text/javascript"}


class Api:
    """Methods callable from JS as `pywebview.api.<name>()`. Attributes are private so pywebview
    does not try to expose them."""

    def __init__(self, device: str, precision: str):
        self._device, self._precision = device, precision
        self._index = None
        self._window = None
        self._status, self._error = "loading", None
        self._progress = None
        self._rate = 0.0
        self._cancel = threading.Event()
        self._worker: threading.Thread | None = None
        self._worker_lock = threading.Lock()
        self._rerun = False
        self._watcher = FolderWatcher(self._start_update)
        threading.Thread(target=self._load, daemon=True).start()

    # ---- lifecycle --------------------------------------------------------------------------

    def _load(self) -> None:
        try:
            from . import open_index

            self._index = open_index(self._device, self._precision)
            self._status = "ready"
            self._watcher.watch(self._index.roots())
            if self._index.roots():
                self._start_update()
        except Exception as e:  # noqa: BLE001 - surfaced in the UI
            log.exception("Failed to load model/index")
            self._status, self._error = "error", str(e)

    def _start_update(self) -> None:
        """Start an incremental update, or queue one more pass if one is running."""
        if self._index is None:
            return
        with self._worker_lock:
            if self._worker and self._worker.is_alive():
                self._rerun = True
                return
            self._cancel.clear()
            self._worker = threading.Thread(target=self._update, daemon=True)
            self._worker.start()

    def _update(self) -> None:
        while True:
            self._status = "indexing"
            t0 = time.perf_counter()

            def on_progress(p):
                self._progress = p
                self._rate = p.done / max(time.perf_counter() - t0, 1e-6)

            try:
                self._index.update(on_progress=on_progress, cancel=self._cancel)
            except Exception as e:  # noqa: BLE001
                log.exception("Indexing failed")
                self._error = str(e)
            with self._worker_lock:
                if not self._rerun or self._cancel.is_set():
                    self._rerun = False
                    self._status = "ready"
                    return
                self._rerun = False

    # ---- JS API -----------------------------------------------------------------------------

    def state(self) -> dict:
        p = self._progress
        index = self._index
        return {
            "status": self._status,
            "error": self._error,
            "device": index.embedder.device if index is not None else None,
            "progress": p and {
                "done": p.done,
                "total": p.total,
                "failed": p.failed,
                "current": p.current and os.path.basename(p.current),
                "rate": round(self._rate, 1),
            },
            "stats": index.stats() if index is not None else None,
        }

    def add_folder(self) -> bool:
        result = self._window.create_file_dialog(webview.FileDialog.FOLDER)
        if not result or self._index is None:
            return False
        self._index.add_root(result[0])
        self._watcher.watch(self._index.roots())
        self._start_update()
        return True

    def remove_folder(self, path: str) -> None:
        if self._worker and self._worker.is_alive():
            self._cancel.set()
            self._worker.join()
        self._index.remove_root(path)
        self._watcher.watch(self._index.roots())
        self._start_update()

    def reindex(self) -> None:
        if self._index is not None:
            self._start_update()

    def cancel(self) -> None:
        self._cancel.set()

    def search(self, query: str, k: int = 120) -> list[dict]:
        return [self._hit(h) for h in self._index.search(query, k)] if self._index is not None else []

    def similar(self, photo_id: int, k: int = 120) -> list[dict]:
        return [self._hit(h) for h in self._index.similar(int(photo_id), k)] if self._index is not None else []

    def recent(self, k: int = 300) -> list[dict]:
        return [self._hit(h) for h in self._index.recent(k)] if self._index is not None else []

    def open_photo(self, photo_id: int) -> None:
        if path := self._index.photo_path(int(photo_id)):
            if url := google_photos_url(path):
                webbrowser.open(url)
            else:
                os.startfile(path)

    def reveal_photo(self, photo_id: int) -> None:
        if path := self._index.photo_path(int(photo_id)):
            if url := google_photos_url(path):
                webbrowser.open(url)
            else:
                subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])

    def delete_photos(self, photo_ids: list[int]) -> dict:
        """Move photos to the Recycle Bin and drop them from the index."""
        deleted, failed = self._index.delete_photos([int(i) for i in photo_ids])
        return {"deleted": deleted, "failed": {str(k): v for k, v in failed.items()}}

    def copy_text(self, text: str) -> None:
        """Clipboard fallback for when the WebView blocks navigator.clipboard."""
        subprocess.run(["clip"], input=b"\xff\xfe" + text.encode("utf-16-le"), check=False)

    @staticmethod
    def _hit(h) -> dict:
        return {"id": h.id, "name": os.path.basename(h.path), "path": h.path, "score": round(h.score, 4), "taken": h.taken}


def make_server(api: Api, token: str) -> ThreadingHTTPServer:
    """Serve the UI, thumbnails and previews on 127.0.0.1 under a per-session random token."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence per-request logging
            pass

        def _send(self, body: bytes, ctype: str, cache: bool = False) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parts = self.path.split("?")[0].strip("/").split("/")
            if not parts or parts[0] != token:
                return self.send_error(HTTPStatus.NOT_FOUND)
            route, rest = (parts[1] if len(parts) > 1 else ""), parts[2:]
            try:
                if route in ("", "index.html"):
                    return self._send((WEB_DIR / "index.html").read_bytes(), STATIC_TYPES[".html"])
                index = api._index
                if index is not None and route == "thumb" and rest:
                    thumb = index.thumb_path(int(rest[0].removesuffix(".jpg")))
                    if thumb.exists():
                        return self._send(thumb.read_bytes(), "image/jpeg", cache=True)
                if index is not None and route == "photo" and rest:
                    if path := index.photo_path(int(rest[0])):
                        if google_photos_url(path):  # no local file: show the stored thumbnail
                            return self._send(index.thumb_path(int(rest[0])).read_bytes(), "image/jpeg", cache=True)
                        return self._send(preview_jpeg(path), "image/jpeg", cache=True)
            except (ValueError, OSError) as e:
                log.warning("GET %s failed: %s", self.path, e)
            self.send_error(HTTPStatus.NOT_FOUND)

    return ThreadingHTTPServer(("127.0.0.1", 0), Handler)


def preview_jpeg(path: str) -> bytes:
    with Image.open(path) as im:
        im.draft("RGB", (PREVIEW_SIZE, PREVIEW_SIZE))
        image = ImageOps.exif_transpose(im).convert("RGB")
    image.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE), Image.Resampling.BICUBIC)
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=88)
    return buf.getvalue()


def run(device: str = "auto", precision: str | None = None) -> None:
    api = Api(device, precision)
    token = secrets.token_urlsafe(16)
    server = make_server(api, token)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/{token}/"
    api._window = webview.create_window(
        "Local Photo Search", url, js_api=api, width=1280, height=860, min_size=(640, 480)
    )
    webview.start()
    api.cancel()
    api._watcher.stop()
    server.shutdown()
