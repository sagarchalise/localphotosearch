"""Download the 10 COCO val2017 images that scripts/spike.py's queries are labelled against.

usage: uv run python scripts/fetch_test_images.py [dest_dir]   (default: testimgs)
"""

import sys
import urllib.request
from pathlib import Path

IDS = [
    "000000039769", "000000000139", "000000000285", "000000000632", "000000000724",
    "000000000776", "000000000785", "000000000802", "000000000872", "000000000885",
]


def main() -> None:
    dest = Path(sys.argv[1] if len(sys.argv) > 1 else "testimgs")
    dest.mkdir(parents=True, exist_ok=True)
    for image_id in IDS:
        path = dest / f"{image_id}.jpg"
        if not path.exists():
            urllib.request.urlretrieve(f"http://images.cocodataset.org/val2017/{image_id}.jpg", path)
        print(path)


if __name__ == "__main__":
    main()
