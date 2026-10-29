"""Embed a folder of images on one device, run text queries, and report timings.

usage: uv run python scripts/spike.py <image_dir> [--device auto|NPU|GPU|CPU] [--precision fp16]
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image

from localphotosearch.embedder import Embedder

# query -> expected COCO image id (top-1) for the 10-image test set
EXPECTED = {
    "two cats sleeping on a couch": "000000039769",
    "a bear in the wild": "000000000285",
    "stop sign on a street": "000000000724",
    "person skiing down a snowy slope": "000000000785",
    "teddy bears": "000000000776",
    "tennis player hitting the ball": "000000000885",
    "baseball game": "000000000872",
    "kitchen with a refrigerator": "000000000802",
    "a bedroom": "000000000632",
    "living room with a television": "000000000139",
}
QUERIES = list(EXPECTED)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", help="default: per device")
    parser.add_argument("--soft-tokens", type=int, default=70)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save", type=Path, help="directory to save image vectors as <precision>_<device>.npy")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    t0 = time.perf_counter()
    emb = Embedder(device=args.device, precision=args.precision, soft_tokens=args.soft_tokens)
    print(f"device={emb.device} model={emb.paths['model'].name} load+compile={time.perf_counter() - t0:.1f}s")

    files = sorted(p for p in args.image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    emb.embed_images([Image.open(files[0])])  # warm-up

    t0 = time.perf_counter()
    vectors = []
    for i in range(0, len(files), args.batch):
        vectors.append(emb.embed_images([Image.open(f) for f in files[i : i + args.batch]]))
    image_vecs = np.concatenate(vectors)
    dt = time.perf_counter() - t0
    print(f"images: {len(files)} in {dt:.2f}s -> {len(files) / dt:.1f} img/s (batch={args.batch})")

    t0 = time.perf_counter()
    query_vecs = emb.embed_texts(QUERIES)
    print(f"queries: {len(QUERIES)} in {(time.perf_counter() - t0) * 1000:.0f} ms")

    scores = query_vecs @ image_vecs.T
    correct = 0
    for q, row in zip(QUERIES, scores):
        top = np.argsort(-row)[:3]
        correct += files[top[0]].stem == EXPECTED[q]
        if args.verbose:
            hits = ", ".join(f"{files[i].stem[-6:]}={row[i]:.3f}" for i in top)
            print(f"  {q:40s} {hits}")
    margin = np.mean(np.sort(scores, axis=1)[:, -1] - np.sort(scores, axis=1)[:, -2])
    print(f"top-1 correct: {correct}/{len(QUERIES)}  mean margin over 2nd: {margin:.3f}")
    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)
        np.save(args.save / f"{emb.paths['model'].stem}_{emb.device}.npy", image_vecs)


if __name__ == "__main__":
    main()
