"""App-wide locations and defaults."""

import os
from pathlib import Path

DATA_DIR = Path(
    os.environ.get("LOCALPHOTOSEARCH_DATA")
    or Path(os.environ.get("LOCALAPPDATA", Path.home())) / "localphotosearch"
)
MODEL_DIR = DATA_DIR / "models" / "embeddinggemma-2"
CACHE_DIR = DATA_DIR / "cache" / "openvino"

SOFT_TOKENS = 70
SEARCH_DIM = 768
