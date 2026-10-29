"""EmbeddingGemma 2 image/text embedder with per-device backends and automatic fallback.

Backends:
- OpenVINO: Intel NPU, GPU and CPU (always installed).
- ONNX Runtime: CUDA (NVIDIA, `--extra cuda`) and CPU.
"""

import json
import logging
import queue
import threading
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from tokenizers import Tokenizer

from .config import MODEL_DIR
from .model_files import ensure_model
from .preprocess import preprocess_image

log = logging.getLogger(__name__)

QUERY_PREFIX = "task: search result | query: "
FEATURE_DIM = 512


@dataclass(frozen=True)
class Device:
    name: str  # what the user picks: CUDA, NPU, GPU, CPU
    runtime: str  # "openvino" or "onnxruntime"
    precision: str  # which ONNX files to load (see model_files.PRECISIONS)
    vision_precision: str | None = None  # override for the vision encoder only


# Preference order for `device="auto"`.
# int8 is the default model: cos 0.996 vs fp16, ~1.3x faster on Intel GPU / ~2x on CPU,
# 510 MB vs 880 MB. q4 drifts more (cos 0.973) and is very slow on the Intel GPU.
# NPU runs only the vision encoder (patched by npu.make_npu_friendly), in fp16: int8 vision on NPU
# drifts (per-token cos down to 0.46); the text model runs on GPU, else CPU.
# CUDA uses fp32: the int8 graph's MatMulInteger runs on CPU under the CUDA provider, and fp16
# overflows to NaN (seen on the Intel GPU).
DEVICE_ORDER = (
    Device("CUDA", "onnxruntime", "fp32"),
    Device("NPU", "openvino", "int8", vision_precision="fp16"),
    Device("GPU", "openvino", "int8"),
    Device("CPU", "openvino", "int8"),
    Device("CPU", "onnxruntime", "int8"),
)
ORT_PROVIDERS = {"CUDA": "CUDAExecutionProvider", "CPU": "CPUExecutionProvider"}


def available_devices() -> list[Device]:
    """Devices whose runtime is installed and sees matching hardware, in preference order."""
    found = set()
    try:
        import openvino as ov

        found |= {("openvino", d) for d in ov.Core().available_devices}
    except ImportError:
        pass
    try:
        import onnxruntime as ort

        names = {v: k for k, v in ORT_PROVIDERS.items()}
        found |= {("onnxruntime", names[p]) for p in ort.get_available_providers() if p in names}
    except ImportError:
        pass
    return [d for d in DEVICE_ORDER if (d.runtime, d.name) in found]


class OpenVinoBackend:
    # fp16 GPU inference overflows to NaN in both graphs, so force f32. The CPU plugin's compile
    # cache returns NaN for this model when re-imported, so it is only used for GPU and NPU
    # (NPU compiles take 30-50 s cold, 0.1 s cached).
    CONFIG = {"GPU": {"INFERENCE_PRECISION_HINT": "f32"}}
    CACHED = {"GPU", "NPU"}

    def __init__(self, device: Device, paths: dict[str, Path], soft_tokens: int, cache_dir: str):
        import openvino as ov

        core = ov.Core()
        if cache_dir:
            for cached in self.CACHED & set(core.available_devices):
                core.set_property(cached, {"CACHE_DIR": cache_dir})

        vision = core.read_model(paths["vision_encoder"])
        max_patches = soft_tokens * 9  # always padded to this, so the shape can be static
        vision.reshape({"pixel_values": [1, max_patches, 768], "pixel_position_ids": [1, max_patches, 2]})
        text_device = device.name
        if device.name == "NPU":
            from .npu import make_npu_friendly

            vision = make_npu_friendly(vision)  # fixed [soft_tokens, 512] output; caller slices
            # The text model's sequence length varies per input, which the NPU can't compile.
            text_device = "GPU" if "GPU" in core.available_devices else "CPU"
        self._vision = core.compile_model(vision, device.name, self.CONFIG.get(device.name, {}))
        self._text = core.compile_model(paths["model"], text_device, self.CONFIG.get(text_device, {}))
        # Queries always run on CPU: the GPU recompiles for each new token length (~600 ms),
        # the CPU stays at 10-60 ms and doesn't queue behind indexing.
        self._query = self._text if text_device == "CPU" else core.compile_model(paths["model"], "CPU")

    def vision(self, feeds: dict) -> np.ndarray:
        return self._vision(feeds)[self._vision.output(0)]

    def text(self, feeds: dict) -> np.ndarray:
        return self._text(feeds)[self._text.output("sentence_embedding")]

    def query(self, feeds: dict) -> np.ndarray:
        return self._query(feeds)[self._query.output("sentence_embedding")]


class OnnxRuntimeBackend:
    def __init__(self, device: Device, paths: dict[str, Path], soft_tokens: int, cache_dir: str):
        import onnxruntime as ort

        if device.name == "CUDA" and hasattr(ort, "preload_dlls"):
            ort.preload_dlls()  # CUDA/cuDNN DLLs from the nvidia-* pip wheels
        providers = [ORT_PROVIDERS[device.name]] + (["CPUExecutionProvider"] if device.name != "CPU" else [])

        def session(path: Path, provider_list: list[str]):
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            return ort.InferenceSession(str(path), opts, providers=provider_list)

        self._vision = session(paths["vision_encoder"], providers)
        self._text = session(paths["model"], providers)
        self._query = self._text if device.name == "CPU" else session(paths["model"], ["CPUExecutionProvider"])
        if ORT_PROVIDERS[device.name] not in self._vision.get_providers():
            raise RuntimeError(f"{device.name} provider not active (got {self._vision.get_providers()})")

    def vision(self, feeds: dict) -> np.ndarray:
        return self._vision.run(["image_features"], feeds)[0]

    def text(self, feeds: dict) -> np.ndarray:
        return self._text.run(["sentence_embedding"], feeds)[0]

    def query(self, feeds: dict) -> np.ndarray:
        return self._query.run(["sentence_embedding"], feeds)[0]


BACKENDS = {"openvino": OpenVinoBackend, "onnxruntime": OnnxRuntimeBackend}


class Embedder:
    def __init__(
        self,
        model_dir: Path = MODEL_DIR,
        device: str = "auto",
        precision: str | None = None,
        soft_tokens: int = 70,
        cache_dir: str = "cache/openvino",
    ):
        """`device`: auto, CUDA, NPU, GPU or CPU. `precision` overrides each device's default."""
        self.model_dir = model_dir
        self.soft_tokens = soft_tokens
        self.cache_dir = cache_dir
        self._lock = threading.Lock()  # backend switching + text model
        self._vision_lock = threading.Lock()
        self._query_lock = threading.Lock()

        candidates = available_devices()
        if device != "auto":
            runtime, _, name = device.rpartition(":")  # e.g. "CPU" or "onnxruntime:CPU"
            candidates = [d for d in candidates if d.name == name.upper() and runtime in ("", d.runtime)]
            if not candidates:
                raise RuntimeError(f"Device {device} is not available")
        if precision:
            candidates = [Device(d.name, d.runtime, precision, d.vision_precision and precision) for d in candidates]
        self.candidates = candidates

        self.tokenizer = None
        self.backend = None
        self.device: str | None = None
        self.paths: dict[str, Path] = {}
        self._empty = np.zeros((0, FEATURE_DIM), dtype=np.float32)
        self._start_backend(0)

    def _start_backend(self, start: int) -> None:
        """Initialize the first working backend from `self.candidates[start:]`."""
        for i, device in enumerate(self.candidates[start:], start):
            try:
                paths = ensure_model(device.precision, self.model_dir, device.vision_precision)
                self.backend = BACKENDS[device.runtime](device, paths, self.soft_tokens, self.cache_dir)
                self._load_tokenizer(paths)
                self.paths, self.device, self._current = paths, device.name, i
                log.info("EmbeddingGemma 2 running on %s (%s, %s)", device.name, device.runtime, device.precision)
                return
            except Exception as e:  # noqa: BLE001 - any init failure means try the next device
                log.warning("Starting %s/%s failed: %s", device.runtime, device.name, e)
        raise RuntimeError("No device could run the model")

    def _load_tokenizer(self, paths: dict[str, Path]) -> None:
        if self.tokenizer is None:
            self.tokenizer = Tokenizer.from_file(str(paths["tokenizer"]))
            config = json.loads((paths["tokenizer"].parent / "config.json").read_text())
            self.pad_id = config["text_config"]["pad_token_id"]

    def _run(self, fn):
        """Run `fn`, falling back to the next device on inference failure.

        Serialized: the indexer thread and UI searches share the compiled models.
        """
        with self._lock:
            while True:
                try:
                    return fn()
                except Exception as e:  # noqa: BLE001
                    if self._current == len(self.candidates) - 1:
                        raise
                    log.warning("Inference on %s failed (%s); falling back", self.device, e)
                    self._start_backend(self._current + 1)

    def _encode_text(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        encodings = self.tokenizer.encode_batch(texts)
        length = max(len(e.ids) for e in encodings)
        ids = np.full((len(texts), length), self.pad_id, dtype=np.int64)
        mask = np.zeros((len(texts), length), dtype=np.int64)
        for i, e in enumerate(encodings):
            ids[i, : len(e.ids)] = e.ids
            mask[i, : len(e.ids)] = 1
        return ids, mask

    def _text_feeds(self, ids, mask, image_features) -> dict:
        return {
            "input_ids": ids,
            "attention_mask": mask,
            "image_features": image_features,
            "video_features": self._empty,
            "audio_features": self._empty,
        }

    @staticmethod
    def _normalize(emb: np.ndarray) -> np.ndarray:
        return emb / np.linalg.norm(emb, axis=-1, keepdims=True)

    def embed_texts(self, queries: list[str]) -> np.ndarray:
        ids, mask = self._encode_text([QUERY_PREFIX + q for q in queries])
        feeds = self._text_feeds(ids, mask, self._empty)
        with self._query_lock:
            return self._normalize(self.backend.query(feeds))

    def _vision_features(self, prepped: tuple) -> np.ndarray:
        patches, positions, n = prepped
        with self._vision_lock:
            # [:n]: the NPU-patched encoder returns all soft-token slots, real tokens first.
            return self.backend.vision({"pixel_values": patches[None], "pixel_position_ids": positions[None]})[:n]

    def _image_text_embedding(self, soft_tokens: list[int], features: np.ndarray) -> np.ndarray:
        ids, mask = self._encode_text([f"<|image>{'<|image|>' * n}<image|>" for n in soft_tokens])
        return self._normalize(self.backend.text(self._text_feeds(ids, mask, features)))

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        prepped = [preprocess_image(img, self.soft_tokens) for img in images]

        def run():
            features = np.concatenate([self._vision_features(p) for p in prepped])
            return self._image_text_embedding([p[2] for p in prepped], features)

        return self._run(run)

    def embed_stream(self, items: Iterable, image_of: Callable) -> Iterator[tuple]:
        """Embed `image_of(item)` for each item, in order, yielding (item, vector, error).

        The vision encoder for the next image runs on a worker thread while the text model runs
        for the current one; on NPU machines they are on different devices. Items whose image is
        None are yielded with (item, None, None). Failures retry on `embed_images`, which falls
        back to the next device.
        """
        handoff: queue.Queue = queue.Queue(maxsize=2)
        stop = threading.Event()
        done = object()

        def produce():
            try:
                for item in items:
                    if stop.is_set():
                        break
                    image, prepped, features, error = image_of(item), None, None, None
                    if image is not None:
                        try:
                            prepped = preprocess_image(image, self.soft_tokens)
                            features = self._vision_features(prepped)
                        except Exception as e:  # noqa: BLE001 - retried on the fallback path
                            error = e
                    handoff.put((item, image, prepped, features, error))
            finally:
                getattr(items, "close", lambda: None)()
                handoff.put(done)

        threading.Thread(target=produce, daemon=True).start()
        finished = False
        try:
            while (entry := handoff.get()) is not done:
                item, image, prepped, features, error = entry
                if image is None:
                    yield item, None, None
                    continue
                try:
                    if error is not None:
                        raise error
                    with self._lock:
                        vector = self._image_text_embedding([prepped[2]], features)[0]
                    yield item, vector, None
                except Exception:  # noqa: BLE001
                    try:
                        yield item, self.embed_images([image])[0], None
                    except Exception as e:  # noqa: BLE001
                        yield item, None, e
            finished = True
        finally:
            if not finished:  # closed early: unblock the producer and let it finish
                stop.set()
                while handoff.get() is not done:
                    pass
