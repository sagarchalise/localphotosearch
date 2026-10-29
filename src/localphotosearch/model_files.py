"""Download the EmbeddingGemma 2 ONNX files needed for image + text embedding (no audio)."""

from pathlib import Path

from huggingface_hub import hf_hub_download

from .config import MODEL_DIR

REPO = "onnx-community/embeddinggemma-2-ONNX"

# Suffix used by the ONNX export for each precision.
PRECISIONS = {"fp32": "", "fp16": "_fp16", "int8": "_quantized", "q4": "_q4", "q4f16": "_q4f16"}
CONFIG_FILES = ["config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json"]


def _fetch(name: str, model_dir: Path) -> Path:
    local = model_dir / name
    if local.exists():
        return local
    return Path(hf_hub_download(REPO, name, local_dir=model_dir))


def ensure_model(
    precision: str = "fp16", model_dir: Path = MODEL_DIR, vision_precision: str | None = None
) -> dict[str, Path]:
    """Download (if missing) and return paths to the vision encoder, text model and tokenizer.

    `vision_precision` picks a different precision for the vision encoder only.
    """
    for name in CONFIG_FILES:
        _fetch(name, model_dir)

    paths = {}
    for graph, prec in (("vision_encoder", vision_precision or precision), ("model", precision)):
        suffix = PRECISIONS[prec]
        _fetch(f"onnx/{graph}{suffix}.onnx_data", model_dir)
        paths[graph] = _fetch(f"onnx/{graph}{suffix}.onnx", model_dir)
    paths["tokenizer"] = model_dir / "tokenizer.json"
    return paths


if __name__ == "__main__":
    import sys

    for key, path in ensure_model(sys.argv[1] if len(sys.argv) > 1 else "fp16").items():
        print(f"{key}: {path}")
