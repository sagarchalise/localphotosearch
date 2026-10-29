"""Print OpenVINO devices and the I/O signature of the EmbeddingGemma 2 ONNX graphs.

Downloads only the small graph files (not the .onnx_data weights).
"""

import onnx
import openvino as ov
from huggingface_hub import hf_hub_download

REPO = "onnx-community/embeddinggemma-2-ONNX"
GRAPHS = ["onnx/vision_encoder.onnx", "onnx/model.onnx"]


def describe(value_infos):
    for v in value_infos:
        t = v.type.tensor_type
        dims = [d.dim_param or d.dim_value for d in t.shape.dim]
        print(f"    {v.name}: {onnx.TensorProto.DataType.Name(t.elem_type)} {dims}")


def main() -> None:
    core = ov.Core()
    print("OpenVINO", ov.get_version())
    for dev in core.available_devices:
        print(f"  {dev}: {core.get_property(dev, 'FULL_DEVICE_NAME')}")

    for name in GRAPHS:
        path = hf_hub_download(REPO, name, local_dir="models/embeddinggemma-2")
        graph = onnx.load(path, load_external_data=False)
        print(f"\n{name}  ({len(graph.graph.node)} nodes)")
        print("  inputs:")
        describe(graph.graph.input)
        print("  outputs:")
        describe(graph.graph.output)


if __name__ == "__main__":
    main()
