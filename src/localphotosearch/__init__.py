import argparse
import logging
import sys
import time

from . import config


def open_index(device: str = "auto", precision: str | None = None):
    from .embedder import Embedder
    from .index import PhotoIndex

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    embedder = Embedder(
        config.MODEL_DIR, device, precision, soft_tokens=config.SOFT_TOKENS, cache_dir=str(config.CACHE_DIR)
    )
    return PhotoIndex(config.DATA_DIR, embedder, search_dim=config.SEARCH_DIM)


def main() -> None:
    parser = argparse.ArgumentParser(prog="localphotosearch")
    parser.add_argument("--device", default="auto", help="auto, CUDA, NPU, GPU (Intel) or CPU")
    parser.add_argument("--precision", help="fp32, fp16 or int8 (default: per device)")
    sub = parser.add_subparsers(dest="cmd")
    p_index = sub.add_parser("index", help="add folders (optional) and update the index")
    p_index.add_argument("folders", nargs="*")
    p_search = sub.add_parser("search", help="text search")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=10)
    p_similar = sub.add_parser("similar", help="photos similar to a photo id")
    p_similar.add_argument("photo_id", type=int)
    p_similar.add_argument("-k", type=int, default=10)
    sub.add_parser("stats")
    sub.add_parser("ui", help="launch the desktop app (default)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.cmd in (None, "ui"):
        from .app import run

        run(device=args.device, precision=args.precision)
        return

    index = open_index(args.device, args.precision)
    if args.cmd == "index":
        for folder in args.folders:
            index.add_root(folder)
        t0 = time.perf_counter()

        def report(p):
            if p.done % 25 == 0 or p.done == p.total:
                rate = p.done / max(time.perf_counter() - t0, 1e-6)
                print(f"\r{p.done}/{p.total} ({p.failed} failed) {rate:.1f} img/s", end="", file=sys.stderr)

        p = index.update(on_progress=report)
        print(f"\nindexed {p.done - p.failed}, failed {p.failed}; total in index: {len(index)}")
    elif args.cmd in ("search", "similar"):
        t0 = time.perf_counter()
        hits = index.search(args.query, args.k) if args.cmd == "search" else index.similar(args.photo_id, args.k)
        print(f"{len(hits)} hits in {(time.perf_counter() - t0) * 1000:.0f} ms")
        for h in hits:
            print(f"{h.score:.3f}  #{h.id:<6} {h.path}")
    elif args.cmd == "stats":
        print(index.stats())
