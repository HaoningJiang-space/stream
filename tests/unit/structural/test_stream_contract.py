from __future__ import annotations

from xdsl.dialects.builtin import bf16
from xdsl.ir.affine import AffineMap

from stream.cost_model.communication_manager import MulticastPathPlan
from stream.datatypes import LayerDim
from stream.hardware.architecture.core import Core
from stream.hardware.architecture.noc.communication_link import CommunicationLink
from stream.mapping.mapping import Mapping, NodeMapping
from stream.structural.conformance import audit_baseline_round_trip, audit_candidate_set, audit_determinism
from stream.structural.stream_contract import (
    CompileStage,
    CompileStatus,
    Gate1AEvalConfig,
    LiteralKind,
    PipelineSemanticManifest,
    StructuralLiteral,
    StructuralMappingContract,
    UnsupportedReason,
    compile_post_transfer,
    compile_pre_transfer,
    compile_structural_assignment,
    core_group_key,
    load_intended_space,
    path_key,
    tiling_key,
)
from stream.workload.node import ComputationNode, TransferNode, TransferType
from stream.workload.tensor import Tensor


def _core(core_id: int) -> Core:
    return Core(core_id=core_id, name=f"core_{core_id}", core_type="compute")


def _node(name: str = "A") -> ComputationNode:
    tensor_in = Tensor.create(f"{name}_in", bf16, (8,))
    tensor_out = Tensor.create(f"{name}_out", bf16, (8,))
    identity = AffineMap.identity(1)
    return ComputationNode(
        name=name,
        inputs=(tensor_in,),
        outputs=(tensor_out,),
        operand_mapping=(identity, identity),
        type="Elementwise",
    )


def _mapping() -> tuple[Mapping, ComputationNode, tuple[Core, ...], LayerDim]:
    node = _node()
    cores = (_core(0), _core(1), _core(2))
    dim = LayerDim(position=0, prefix="d")
    mapping = Mapping(
        {
            node: NodeMapping(
                resource_allocation=((cores[0],), (cores[1],), (cores[2],)),
                inter_core_tiling=(((dim, 1),), ((dim, 2),), ((dim, 4),)),
            )
        }
    )
    return mapping, node, cores, dim


def _config() -> Gate1AEvalConfig:
    return Gate1AEvalConfig(backend="ortools", cost_lut_digest="cost-lut-test")


def _pipeline_manifest() -> PipelineSemanticManifest:
    return PipelineSemanticManifest(
        workload_digest="workload-test",
        accelerator_digest="accelerator-test",
        cost_lut_digest="cost-lut-test",
        timeslots=(("A", 0),),
        reuse_option_domains=(("A_out", (-1, 0)),),
        constraint_families=("memory", "routing"),
    )


def test_pre_transfer_compiler_intersects_candidates_exactly_without_mutating_reference():
    mapping, node, cores, dim = _mapping()
    literals = (
        StructuralLiteral(
            "A.zone",
            LiteralKind.HARDWARE_ZONE,
            node.name,
            (core_group_key((cores[0],)), core_group_key((cores[2],))),
        ),
        StructuralLiteral(
            "A.tiling",
            LiteralKind.OPERATOR_TILING,
            node.name,
            (tiling_key(((dim, 2),)),),
        ),
    )

    reference = compile_pre_transfer(mapping, (), _config())
    compiled = compile_pre_transfer(mapping, literals, _config())

    assert [group[0].id for group in mapping.get(node).resource_allocation] == [0, 1, 2]
    assert [group[0].id for group in compiled.mapping.get(node).resource_allocation] == [0, 2]
    assert compiled.mapping.get(node).inter_core_tiling == (((dim, 2),),)
    assert compiled.status is CompileStatus.EXACT
    assert all(audit_candidate_set(reference, compiled, literal).exact for literal in literals)


def test_compiler_is_deterministic_and_baseline_round_trip_is_semantically_identical():
    mapping, _, cores, _ = _mapping()
    literal = StructuralLiteral(
        "A.zone",
        LiteralKind.HARDWARE_ZONE,
        "A",
        (core_group_key((cores[1],)),),
    )
    runs = tuple(compile_pre_transfer(mapping, (literal,), _config()) for _ in range(3))
    baseline = compile_post_transfer(mapping, (), _config(), pipeline_manifest=_pipeline_manifest())
    round_trip = compile_post_transfer(mapping, (), _config(), pipeline_manifest=_pipeline_manifest())

    assert audit_determinism(runs)
    assert audit_baseline_round_trip(baseline, round_trip)


def test_unrepresentable_literals_are_explicitly_unsupported():
    mapping, _, _, _ = _mapping()
    literals = (
        StructuralLiteral("T.mode", LiteralKind.MATERIALIZATION, "T", ("streaming",)),
        StructuralLiteral("T.layout", LiteralKind.OUTPUT_LAYOUT, "T", ("0,1",)),
        StructuralLiteral("T.reuse", LiteralKind.EXACT_REUSE, "T", ("2",)),
    )
    result = compile_pre_transfer(mapping, literals, _config())

    assert result.status is CompileStatus.UNSUPPORTED
    assert [item.reason for item in result.classifications] == [
        UnsupportedReason.NO_PIPELINE_LITERAL,
        UnsupportedReason.NO_LAYOUT_CONSTRAINT,
        UnsupportedReason.NO_EXACT_REUSE_LITERAL,
    ]
    assert all(item.stage is None for item in result.classifications)


def test_intended_space_denominator_is_versioned_and_complete():
    manifest = load_intended_space()

    assert manifest["version"] == 1
    assert manifest["classification"] == ["EXACT", "UNSUPPORTED"]
    assert manifest["minimum_deterministic_assignments"] == 1000
    assert manifest["compile_repetitions"] == 3
    assert manifest["tensor_literals"]["materialization"] == ["full", "partial", "streaming"]
    assert manifest["tensor_literals"]["distribution"] == ["local", "block", "replicated", "shared"]
    assert len(manifest["dag_classes"]) == 6
    assert CompileStage.POST_TRANSFER.value == "post_transfer"


def test_two_stage_contract_filters_paths_before_downstream_scheduling():
    source, target = _core(0), _core(1)
    tensor_in = Tensor.create("T.in", bf16, (8,))
    tensor_out = Tensor.create("T.out", bf16, (8,))
    transfer = TransferNode(
        name="Transfer(T)",
        inputs=(tensor_in,),
        outputs=(tensor_out,),
        operand_mapping=(AffineMap.identity(1), AffineMap.identity(1)),
        transfer_type=TransferType.COMPUTE_TO_COMPUTE,
    )
    direct_link = CommunicationLink(source, target, bandwidth=32, unit_energy_cost=1)
    reverse_link = CommunicationLink(target, source, bandwidth=32, unit_energy_cost=1)
    direct = MulticastPathPlan((source,), (target,), 1, (direct_link,))
    reverse = MulticastPathPlan((target,), (source,), 1, (reverse_link,))
    mapping = Mapping({transfer: NodeMapping(resource_allocation=(direct, reverse))})
    path_literal = StructuralLiteral(
        "T.path",
        LiteralKind.TRANSFER_PATH,
        transfer.name,
        (path_key(direct),),
    )
    unsupported_literal = StructuralLiteral(
        "T.mode",
        LiteralKind.MATERIALIZATION,
        "T",
        ("streaming",),
    )
    contract = StructuralMappingContract("assignment-0", (path_literal, unsupported_literal))

    pre = compile_structural_assignment(mapping, contract, _config(), CompileStage.PRE_TRANSFER)
    post = compile_structural_assignment(
        mapping,
        contract,
        _config(),
        CompileStage.POST_TRANSFER,
        prior_classifications=pre.classifications,
    )

    assert pre.classifications == ()
    assert post.mapping.get(transfer).resource_allocation == (direct,)
    assert post.classifications[0].status is CompileStatus.EXACT
    assert post.classifications[0].stage is CompileStage.POST_TRANSFER
    assert post.classifications[1].reason is UnsupportedReason.NO_PIPELINE_LITERAL
    assert post.status is CompileStatus.UNSUPPORTED
