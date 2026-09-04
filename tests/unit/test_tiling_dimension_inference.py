"""Fast guard on the unique-dimension inference that tiling generation depends on.

Parsing a real multi-layer model (resnet18) must resolve a stable set of unique loop dimensions with
correct sizes; a regression here silently corrupts every downstream tiling and allocation. This is
the fast, MILP-free counterpart to the slow resnet CO tests in ``tests/test_resnet_patterns.py``.
"""

from __future__ import annotations

import onnx
from onnx import TensorProto, helper

from stream.parser.onnx.model import ONNXModelParser

_RESNET18 = "stream/inputs/examples/workload/resnet18.onnx"
_FSRCNN = "stream/inputs/examples/workload/fsrcnn.onnx"


def test_resnet18_unique_dimension_inference():
    """Guard resnet18's affine dimension bookkeeping: the spatial pyramid (112->56->28->14->7), the
    channel widths (64/128/256/512) and the 7x7/3x3 kernels with 3 input channels all resolve, with a
    stable unique-dimension count and every size positive -- caught here before the (much slower) MILP
    allocation ever runs."""
    parser = ONNXModelParser(_RESNET18)
    parser.run()
    workload = parser.workload

    unique_dims, _ = workload.unique_dimensions()
    sizes = workload.get_dimension_sizes()

    # Every node dim resolves to a pure, tileable LayerDim: strided conv/pool axes are their own free
    # variable (not folded into an untileable compound), so the basis is wider than a naive identity
    # merge but every dimension is addressable.
    from stream.datatypes import LayerDim

    assert len(unique_dims) == 87, f"Expected 87 unique loop dimensions, got {len(unique_dims)}"
    assert all(
        isinstance(workload.get_dims(n)[i], LayerDim)
        for n in workload.get_computation_nodes()
        for i in range(len(workload.get_dims(n)))
    ), "Every node iteration dim must be a pure LayerDim (no compound expressions)"
    assert all(s > 0 for s in sizes), "Every inferred dimension size must be positive"

    distinct = set(sizes)
    assert {112, 56, 28, 14, 7}.issubset(distinct), f"Missing spatial pyramid extents in {sorted(distinct)}"
    assert {64, 128, 256, 512}.issubset(distinct), f"Missing channel widths in {sorted(distinct)}"
    assert {3, 7}.issubset(distinct), f"Missing input-channel / 7x7 stem-kernel extents in {sorted(distinct)}"


def test_onnx_parser_removes_dead_initializers():
    """FSRCNN contains one unreferenced initializer; it must not become an accessor-free InEdge."""

    parser = ONNXModelParser(_FSRCNN)
    parser.run()
    workload = parser.workload

    assert workload.get_in_edges()
    assert all(workload.out_degree(node) > 0 for node in workload.get_in_edges())
    assert parser.semantic_exclusions == []


def test_onnx_parser_reports_supplied_conv_bias_as_semantic_exclusion(tmp_path):
    """A real third Conv operand cannot disappear without a complete, declared audit record."""

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 4])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8, 4, 4])
    weight = helper.make_tensor("weight", TensorProto.FLOAT, [8, 3, 1, 1], [0.0] * (8 * 3))
    bias = helper.make_tensor("bias", TensorProto.FLOAT, [8], [0.0] * 8)
    conv = helper.make_node("Conv", ["x", "weight", "bias"], ["output"], name="conv")
    model = helper.make_model(
        helper.make_graph([conv], "conv-bias", [x], [output], [weight, bias]),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    path = tmp_path / "conv-bias.onnx"
    onnx.save(model, path)

    parser = ONNXModelParser(str(path))
    parser.run()

    assert parser.semantic_exclusions == [
        {
            "node": "conv",
            "operator_type": "Conv",
            "input_index": 2,
            "tensor": "bias",
            "shape": [8],
            "source": "initializer",
            "reason": "UNMODELED_ADDITIVE_OPERAND",
        }
    ]
    assert {node.name for node in parser.workload.get_in_edges()} == {"x", "weight"}


def test_semantic_exclusion_is_tracked_by_input_position_when_operands_alias(tmp_path):
    """Gemm B and C may name the same tensor; the C occurrence is still an excluded input position."""

    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 2])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 2])
    b = helper.make_tensor("b", TensorProto.FLOAT, [2, 2], [0.0] * 4)
    gemm = helper.make_node("Gemm", ["a", "b", "b"], ["output"], name="gemm")
    model = helper.make_model(
        helper.make_graph([gemm], "aliased-gemm-input", [a], [output], [b]),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    path = tmp_path / "aliased-gemm-input.onnx"
    onnx.save(model, path)

    parser = ONNXModelParser(str(path))
    parser.run()

    assert len(parser.semantic_exclusions) == 1
    exclusion = parser.semantic_exclusions[0]
    assert (exclusion["input_index"], exclusion["tensor"], exclusion["reason"]) == (
        2,
        "b",
        "UNMODELED_ADDITIVE_OPERAND",
    )


def test_onnx_parser_preserves_unused_declared_model_input(tmp_path):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    unused = helper.make_tensor_value_info("unused", TensorProto.FLOAT, [1, 4])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    add = helper.make_node("Add", ["x", "x"], ["output"], name="add")
    model = helper.make_model(
        helper.make_graph([add], "unused-input", [x, unused], [output]),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    path = tmp_path / "unused-input.onnx"
    onnx.save(model, path)

    parser = ONNXModelParser(str(path))
    parser.run()

    assert {node.name for node in parser.workload.get_in_edges()} == {"x", "unused"}
