"""OpenVINO graph patches that let the EmbeddingGemma 2 vision encoder run on Intel NPU.

1. The soft-token pooling ends in NonZero -> GatherND, keeping only tokens that received patches,
   so the output length depends on the data and the NPU compiler rejects it. GatherND is replaced
   by a Reshape to [-1, hidden]: the output becomes a fixed [70, 512] and the real tokens are the
   first N rows (NonZero returns ascending indices and real tokens are contiguous).
2. RoPE rotate_half is exported as Gather(x, permutation, axis=-1); the NPU runs those on slow
   cores (~10 ms each x32). Each becomes a MatMul by the permutation matrix (exact).

On CPU the patched graph matches the original exactly (per-token cos 1.0000).
"""

import numpy as np
import openvino as ov
import openvino.opset13 as ops


def make_npu_friendly(model: ov.Model) -> ov.Model:
    gather_nds = [op for op in model.get_ops() if op.get_type_name() == "GatherND"]
    if len(gather_nds) != 1:
        raise RuntimeError(f"expected 1 GatherND in vision encoder, found {len(gather_nds)}")
    data = gather_nds[0].input_value(0)
    hidden = data.get_partial_shape()[-1].get_length()
    flat = ops.reshape(data, ops.constant(np.array([-1, hidden], dtype=np.int64)), special_zero=False)
    for target in list(gather_nds[0].output(0).get_target_inputs()):
        target.replace_source_output(flat.output(0))

    for op in model.get_ops():
        if op.get_type_name() == "Gather" and (perm := _last_axis_permutation(op)) is not None:
            x = op.input_value(0)
            matrix = np.zeros((len(perm), len(perm)), dtype=x.get_element_type().to_dtype())
            matrix[perm, np.arange(len(perm))] = 1  # (x @ P)[..., j] == x[..., perm[j]]
            permuted = ops.matmul(x, ops.constant(matrix), False, False)
            for target in list(op.output(0).get_target_inputs()):
                target.replace_source_output(permuted.output(0))

    model.validate_nodes_and_infer_types()
    return model


def _last_axis_permutation(gather) -> np.ndarray | None:
    """Indices if `gather` is a constant full permutation along the last (static) axis."""
    indices, axis = gather.input_value(1).get_node(), gather.input_value(2).get_node()
    if indices.get_type_name() != "Constant" or axis.get_type_name() != "Constant":
        return None
    perm = indices.get_data()
    shape = gather.input_value(0).get_partial_shape()
    rank = shape.rank.get_length()
    ax = int(axis.get_data().reshape(-1)[0]) % rank
    if perm.ndim != 1 or ax != rank - 1 or not shape[ax].is_static or shape[ax].get_length() != len(perm):
        return None
    return perm if sorted(perm.tolist()) == list(range(len(perm))) else None
