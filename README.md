# Local Photo Search

Search the photos in your folders by describing them ("kids on a beach at sunset"), or find photos
similar to one you pick. Everything runs on this computer with
[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) on the NPU, GPU or CPU; no photo
or query leaves the machine.

## Run

```powershell
uv run localphotosearch            # desktop app: add folders, indexing starts automatically
```

On first launch the model files for the chosen device are downloaded from Hugging Face
(`onnx-community/embeddinggemma-2-ONNX`, vision encoder + text model only; see Devices).

Command line:

```powershell
uv run localphotosearch index D:\Photos E:\Camera   # add folders and index
uv run localphotosearch index                       # update (only new/changed files)
uv run localphotosearch search "dog in the snow" -k 10
uv run localphotosearch similar 42
uv run localphotosearch stats
```

`--device` overrides automatic device selection (`CUDA`, `NPU`, `GPU`, `CPU`, or
`runtime:DEVICE` such as `onnxruntime:CPU`); `--precision fp32|fp16|int8` overrides the per-device
model choice.

## Devices

| Device | Runtime | Model | Install |
|---|---|---|---|
| NVIDIA GPU (`CUDA`) | ONNX Runtime | fp32 (~1.75 GB) | `uv sync --extra cuda` (CUDA 13 + cuDNN from pip; driver ≥ 580) |
| Intel NPU (`NPU`) | OpenVINO | vision fp16 on NPU + text int8 on GPU/CPU (~820 MB) | default |
| Intel GPU (`GPU`) | OpenVINO | int8 (~510 MB) | default |
| CPU | OpenVINO | int8 | default |

`auto` tries them in the order above and falls back to the next one if a device fails to start or
fails during inference. Text queries always run on CPU (a GPU recompiles for each new query length).
Switching device can change the model files; the index notices and re-embeds photos automatically.

The NPU runs only the vision encoder, patched at load time (`npu.py`) so its output has a fixed
shape; the text model's input length varies, which the NPU can't compile. The first NPU launch
compiles for ~25 s, later launches use the cache.
With an extra installed, run with `uv run --extra cuda localphotosearch` so uv keeps it.

On an NVIDIA machine, check the CUDA path with:

```powershell
uv sync --extra cuda
uv run --extra cuda python scripts/fetch_test_images.py testimgs
uv run --extra cuda python scripts/spike.py testimgs --device CUDA --verbose
```

## Where things are stored

`%LOCALAPPDATA%\localphotosearch` (override with `LOCALPHOTOSEARCH_DATA`):

- `index.db`: SQLite, one row per photo (path, size, mtime, EXIF date, 768-d fp16 vector)
- `thumbs\`: 256 px JPEG thumbnails
- `models\`: downloaded ONNX files
- `cache\openvino\`: compiled GPU/NPU models (makes later launches faster)

## How it works

- **Indexing**: folders are walked; files whose size/mtime are unchanged are skipped. Images are
  decoded on worker threads (JPEG draft decoding), resized aspect-preserving to at most 630 16-px
  patches (70 vision tokens), embedded, and stored. Indexing can be cancelled and resumes where it
  stopped.
- **Watching**: added folders are watched; new, changed, moved or deleted images trigger an
  incremental update 3 s after changes settle. "Rescan" in the Folders panel forces one.
- **Finding and cleaning up**: the viewer shows the full path, with *Show in folder*, *Copy path*
  and *Delete*. In the grid, click a tile's check circle (or Ctrl/Shift+click) to select several,
  then *Show in folder* or *Delete*. Delete moves photos to the Recycle Bin and drops them from the
  index; keyboard: `Delete`, `Esc` clears the selection.
- **Search**: all vectors are kept in memory; a query is one matrix-vector product. Text queries
  use the `task: search result | query: ` prompt and always run on CPU (10–60 ms).

## Measured on Core Ultra 7 258V (Arc 140V)

163 real photos, including decode and thumbnails; vector agreement vs an fp16 CPU reference.

| | images/s | cos vs fp16 reference | notes |
|---|---|---|---|
| NPU (vision fp16) + GPU (text int8) | ~15 | 0.9999 | default |
| GPU, int8 | ~7 | 0.9955 | |
| CPU, int8 | ~1.7 | 0.996 | fallback |

Indexing is pipelined: while the text model embeds photo *i*, the vision encoder already runs on
photo *i+1* (different devices on NPU machines), and worker threads decode the photos after that.

## Known limitations

- **GPU fp16**: overflows to NaN, so the GPU runs with an f32 precision hint.
- **CPU compile cache**: OpenVINO's CPU cache returns NaN for this model when re-imported, so it
  is only enabled for GPU and NPU.
- **NVIDIA/CUDA**: implemented, verified only through ONNX Runtime's CPU provider with the same
  fp32 files; not yet run on an NVIDIA GPU.
- Video is not indexed.

## Development

- `scripts/fetch_test_images.py <dir>`: downloads the 10 labelled COCO test images.
- `scripts/spike.py <dir>`: accuracy/speed check on that set (`--device`, `--precision`,
  `--verbose`).
- `scripts/inspect_model.py`: lists OpenVINO devices and ONNX graph inputs/outputs.
