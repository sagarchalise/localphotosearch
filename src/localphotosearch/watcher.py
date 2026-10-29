"""Watch indexed folders and request an incremental index update after changes settle."""

import logging
import os
import threading
import time
from collections.abc import Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .index import IMAGE_EXTS

log = logging.getLogger(__name__)


class FolderWatcher(FileSystemEventHandler):
    def __init__(self, on_change: Callable[[], None], settle_seconds: float = 3.0):
        self.on_change = on_change
        self.settle = settle_seconds
        self._observer = None
        self._roots: list[str] = []
        self._last_event = 0.0
        self._dirty = threading.Event()
        threading.Thread(target=self._debounce, daemon=True).start()

    def watch(self, roots: list[str]) -> None:
        """(Re)start watching exactly `roots`."""
        if sorted(roots) == sorted(self._roots):
            return
        self.stop()
        self._roots = list(roots)
        if not roots:
            return
        self._observer = Observer()
        for root in roots:
            if os.path.isdir(root):
                self._observer.schedule(self, root, recursive=True)
        self._observer.start()
        log.info("Watching %d folder(s)", len(roots))

    def stop(self) -> None:
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.is_directory and event.event_type != "deleted":
            return
        paths = [event.src_path, getattr(event, "dest_path", "") or ""]
        # A deleted/moved directory may contain photos; image-file events are the common case.
        if event.is_directory or any(os.path.splitext(str(p))[1].lower() in IMAGE_EXTS for p in paths):
            self._last_event = time.monotonic()
            self._dirty.set()

    def _debounce(self) -> None:
        while True:
            self._dirty.wait()
            while time.monotonic() - self._last_event < self.settle:
                time.sleep(0.5)
            self._dirty.clear()
            try:
                self.on_change()
            except Exception:  # noqa: BLE001 - keep watching
                log.exception("Update after folder change failed")
