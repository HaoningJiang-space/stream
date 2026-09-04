"""Deterministic dense-affine micro cases for Gate 1A conformance."""

from __future__ import annotations

from functools import lru_cache

import yaml
from xdsl.dialects.builtin import bf16
from xdsl.ir.affine import AffineMap

from stream.cost_model.core_cost import CoreCostEntry
from stream.cost_model.core_cost_lut import CoreCostLUT
from stream.cost_model.steady_state_scheduler import SteadyStateScheduler
from stream.datatypes import LayerDim
from stream.mapping.mapping import Mapping, NodeMapping
from stream.opt.solver import ConstraintSelection
from stream.parser.accelerator_factory import AcceleratorFactory
from stream.parser.accelerator_validator import AcceleratorValidator
from stream.workload.node import ComputationNode, InEdge, OutEdge
from stream.workload.tensor import Tensor
from stream.workload.workload import Workload

_HARDWARE = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"


def build_gate1a_scheduler(dag_class: str, *, backend: str = "ORTOOLS_GSCIP") -> SteadyStateScheduler:
    """Build a fresh real scheduler without invoking ZigZag or TTA."""

    workload = _workload(dag_class)
    accelerator = _accelerator()
    compute_core = next(core for core in accelerator.core_list if core.type == "compute")
    local_dim = LayerDim(position=0, prefix="d")
    mapping = Mapping()
    cost_lut = CoreCostLUT(load=False)
    for node in workload.get_computation_nodes():
        mapping.set(
            node,
            NodeMapping(
                resource_allocation=((compute_core,),),
                inter_core_tiling=(((local_dim, 1),),),
            ),
        )
        cost_lut.add_cost(node, compute_core, CoreCostEntry(1, 10, 10, 10))
    return SteadyStateScheduler(
        workload,
        accelerator,
        mapping,
        {},
        cost_lut,
        backend=backend,
        constraint_selection=ConstraintSelection(),
    )


@lru_cache(maxsize=1)
def _accelerator():
    with open(_HARDWARE) as file:
        data = yaml.safe_load(file)
    validator = AcceleratorValidator(data, _HARDWARE)
    validator.validate()
    return AcceleratorFactory(validator.normalized_data).create()


def _tensor(name: str) -> Tensor:
    return Tensor.create(name, bf16, (8,))


def _compute(name: str, inputs: tuple[Tensor, ...], output: Tensor, op_type: str) -> ComputationNode:
    identity = AffineMap.identity(1)
    return ComputationNode(
        name=name,
        inputs=inputs,
        outputs=(output,),
        operand_mapping=tuple(identity for _ in range(len(inputs) + 1)),
        type=op_type,
    )


def _linear(names: tuple[str, ...], op_type: str) -> Workload:
    tensors = tuple(_tensor(f"t{index}") for index in range(len(names) + 1))
    nodes = [InEdge(name="input", outputs=(tensors[0],))]
    nodes.extend(_compute(name, (tensors[index],), tensors[index + 1], op_type) for index, name in enumerate(names))
    nodes.append(OutEdge(name="output", inputs=(tensors[-1],)))
    return Workload(nodes)


def _workload(dag_class: str) -> Workload:
    if dag_class == "edge":
        return _linear(("A", "B"), "Elementwise")
    if dag_class == "chain":
        return _linear(("A", "B", "C"), "Elementwise")
    if dag_class == "gemm_chain":
        return _linear(("Gemm0", "Gemm1"), "Gemm")
    if dag_class == "conv_chain":
        return _linear(("Conv0", "Conv1"), "Conv")
    if dag_class == "residual_diamond":
        x, main_a, main_b, output = (_tensor(name) for name in ("x", "main_a", "main_b", "output"))
        return Workload(
            (
                InEdge(name="input", outputs=(x,)),
                _compute("A", (x,), main_a, "Elementwise"),
                _compute("B", (main_a,), main_b, "Elementwise"),
                _compute("Add", (main_b, x), output, "Add"),
                OutEdge(name="output", inputs=(output,)),
            )
        )
    if dag_class == "fork_join":
        x, fork, left, right, output = (_tensor(name) for name in ("x", "fork", "left", "right", "output"))
        return Workload(
            (
                InEdge(name="input", outputs=(x,)),
                _compute("A", (x,), fork, "Elementwise"),
                _compute("B", (fork,), left, "Elementwise"),
                _compute("C", (fork,), right, "Elementwise"),
                _compute("Join", (left, right), output, "Add"),
                OutEdge(name="output", inputs=(output,)),
            )
        )
    raise ValueError(f"unknown Gate 1A DAG class: {dag_class}")
