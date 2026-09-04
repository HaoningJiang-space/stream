"""Frozen micro/meso workloads for the Gate 1B predictive-value census."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files

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

_HARDWARE = "stream/inputs/examples/hardware/tpu_v7_ironwood.yaml"


def load_gate1b_contract() -> dict:
    resource = files("stream.structural.contracts").joinpath("gate1b_contract.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def build_gate1b_scheduler(dag_class: str, *, backend: str = "ORTOOLS_GSCIP") -> SteadyStateScheduler:
    """Build a fresh case with three independently restrictable staging tensors."""

    contract = load_gate1b_contract()
    try:
        spec = contract["case_specs"][dag_class]
    except KeyError as error:
        raise ValueError(f"unknown Gate 1B DAG class: {dag_class}") from error
    workload = _workload(dag_class, int(spec["tensor_elements"]))
    cycles = tuple(int(value) for value in spec["compute_cycles"])
    compute_nodes = tuple(workload.get_computation_nodes())
    if len(cycles) != len(compute_nodes):
        raise ValueError(f"{dag_class}: compute-cycle specification does not match the workload")

    accelerator = _accelerator()
    compute_core = next(core for core in accelerator.core_list if core.type == "compute")
    local_dim = LayerDim(position=0, prefix="d")
    mapping = Mapping()
    cost_lut = CoreCostLUT(load=False)
    for node, latency in zip(compute_nodes, cycles, strict=True):
        mapping.set(
            node,
            NodeMapping(
                resource_allocation=((compute_core,),),
                inter_core_tiling=(((local_dim, 1),),),
            ),
        )
        cost_lut.add_cost(node, compute_core, CoreCostEntry(1, latency, latency, latency))
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


def _tensor(name: str, elements: int) -> Tensor:
    return Tensor.create(name, bf16, (elements,))


def _compute(name: str, inputs: tuple[Tensor, ...], output: Tensor, op_type: str) -> ComputationNode:
    identity = AffineMap.identity(1)
    return ComputationNode(
        name=name,
        inputs=inputs,
        outputs=(output,),
        operand_mapping=tuple(identity for _ in range(len(inputs) + 1)),
        type=op_type,
    )


def _io_tensors(elements: int) -> tuple[Tensor, Tensor]:
    return _tensor("lhs", elements), _tensor("rhs", elements)


def _linear(dag_class: str, names: tuple[str, ...], op_type: str, elements: int) -> Workload:
    lhs, rhs = _io_tensors(elements)
    intermediates = tuple(_tensor(f"t{index}", elements) for index in range(len(names)))
    computations = []
    for index, name in enumerate(names):
        inputs = (lhs, rhs) if index == 0 else (intermediates[index - 1],)
        computations.append(_compute(name, inputs, intermediates[index], op_type))
    return Workload(
        (
            InEdge(name=f"{dag_class}_lhs", outputs=(lhs,)),
            InEdge(name=f"{dag_class}_rhs", outputs=(rhs,)),
            *computations,
            OutEdge(name=f"{dag_class}_output", inputs=(intermediates[-1],)),
        )
    )


def _workload(dag_class: str, elements: int) -> Workload:
    if dag_class == "edge":
        return _linear(dag_class, ("A", "B"), "Elementwise", elements)
    if dag_class == "chain":
        return _linear(dag_class, ("A", "B", "C"), "Elementwise", elements)
    if dag_class == "gemm_chain":
        return _linear(dag_class, ("Gemm0", "Gemm1"), "Gemm", elements)
    if dag_class == "conv_chain":
        return _linear(dag_class, ("Conv0", "Conv1"), "Conv", elements)
    if dag_class == "residual_diamond":
        lhs, rhs = _io_tensors(elements)
        main_a, main_b, output = (_tensor(name, elements) for name in ("main_a", "main_b", "output"))
        return Workload(
            (
                InEdge(name="residual_lhs", outputs=(lhs,)),
                InEdge(name="residual_rhs", outputs=(rhs,)),
                _compute("A", (lhs, rhs), main_a, "Elementwise"),
                _compute("B", (main_a,), main_b, "Elementwise"),
                _compute("Add", (main_b, lhs), output, "Add"),
                OutEdge(name="residual_output", inputs=(output,)),
            )
        )
    if dag_class == "fork_join":
        lhs, rhs = _io_tensors(elements)
        fork, left, right, output = (_tensor(name, elements) for name in ("fork", "left", "right", "output"))
        return Workload(
            (
                InEdge(name="fork_lhs", outputs=(lhs,)),
                InEdge(name="fork_rhs", outputs=(rhs,)),
                _compute("A", (lhs, rhs), fork, "Elementwise"),
                _compute("B", (fork,), left, "Elementwise"),
                _compute("C", (fork,), right, "Elementwise"),
                _compute("Join", (left, right), output, "Add"),
                OutEdge(name="fork_output", inputs=(output,)),
            )
        )
    raise ValueError(f"unknown Gate 1B DAG class: {dag_class}")
