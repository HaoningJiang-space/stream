from __future__ import annotations

import pytest
from xdsl.dialects.builtin import bf16
from xdsl.ir.affine import AffineMap

from stream.datatypes import LayerDim
from stream.hardware.architecture.core import Core
from stream.mapping.mapping import FusedGroup, Mapping, NodeMapping
from stream.stages.context import StageContext
from stream.stages.generation.operator_template_compilation import OperatorTemplateCompilationStage
from stream.stages.stage import LeafStage, MainStage
from stream.structural.operator_template_contract import (
    OperatorTemplate,
    OperatorTemplateAssignment,
    OperatorTemplateCompileError,
    OperatorTemplateLibrary,
    compile_operator_templates,
)
from stream.workload.node import ComputationNode, InEdge, OutEdge
from stream.workload.tensor import Tensor
from stream.workload.workload import Workload


def _case() -> tuple[Workload, Mapping, ComputationNode, tuple[Core, ...]]:
    source = Tensor.create("source", bf16, (8,))
    output = Tensor.create("output", bf16, (8,))
    identity = AffineMap.identity(1)
    node = ComputationNode(
        name="A",
        inputs=(source,),
        outputs=(output,),
        operand_mapping=(identity, identity),
        type="Elementwise",
    )
    workload = Workload((InEdge(name="input", outputs=(source,)), node, OutEdge(name="out", inputs=(output,))))
    cores = tuple(Core(core_id=index, name=f"core_{index}", core_type="compute") for index in range(4))
    dim = LayerDim(position=0, prefix="d")
    mapping = Mapping(
        {
            node: NodeMapping(
                resource_allocation=(cores,),
                inter_core_tiling=(((dim, 4),),),
            )
        },
        fused_groups=(FusedGroup("group", (node.name,), ((workload.get_dims(node)[0], 2),)),),
    )
    return workload, mapping, node, cores


def test_compiler_selects_one_paired_core_and_tiling_template_without_mutating_baseline():
    workload, mapping, node, cores = _case()
    selected = OperatorTemplate(node.name, (0, 1), ((0, 2),))
    library = OperatorTemplateLibrary(
        (
            OperatorTemplate(node.name, (0,), ()),
            selected,
            OperatorTemplate(node.name, (0, 1, 2, 3), ((0, 4),)),
        )
    )
    assignment = OperatorTemplateAssignment("pair-0", (selected,))

    compiled = compile_operator_templates(workload, mapping, assignment, library)

    assert mapping.get(node).resource_allocation == (cores,)
    assert mapping.get(node).inter_core_tiling == ((((LayerDim(position=0, prefix="d"), 4),)),)
    assert compiled.mapping.get(node).resource_allocation == ((cores[0], cores[1]),)
    assert compiled.mapping.get(node).inter_core_tiling == ((((LayerDim(position=0, prefix="d"), 2),)),)
    assert compiled.manifest["selections"] == [selected.key]


def test_compiler_rejects_unregistered_or_illegal_templates_without_repair():
    workload, mapping, node, _ = _case()
    allowed = OperatorTemplate(node.name, (0, 1), ((0, 2),))
    library = OperatorTemplateLibrary((allowed,))
    outside = OperatorTemplate(node.name, (2, 3), ((0, 2),))

    with pytest.raises(OperatorTemplateCompileError, match="outside the declared library"):
        compile_operator_templates(
            workload,
            mapping,
            OperatorTemplateAssignment("outside", (outside,)),
            library,
        )

    invalid_library = OperatorTemplateLibrary((OperatorTemplate(node.name, (0, 1, 2), ((0, 3),)),))
    with pytest.raises(OperatorTemplateCompileError, match="not divisible"):
        compile_operator_templates(
            workload,
            mapping,
            OperatorTemplateAssignment("illegal", invalid_library.templates),
            invalid_library,
        )


def test_pre_tiling_stage_is_identity_without_contract_and_requires_both_inputs():
    workload, mapping, _, _ = _case()
    baseline_context = StageContext.from_kwargs(workload=workload, mapping=mapping)

    baseline = MainStage([OperatorTemplateCompilationStage, LeafStage], baseline_context).run()

    assert len(baseline) == 1
    assert baseline[0].get("mapping") is mapping
    assert baseline[0].get("operator_template_compilation") is None

    broken_context = StageContext.from_kwargs(
        workload=workload,
        mapping=mapping,
        operator_template_assignment=OperatorTemplateAssignment("empty", ()),
    )
    with pytest.raises(ValueError, match="must be provided together"):
        MainStage([OperatorTemplateCompilationStage, LeafStage], broken_context).run()
