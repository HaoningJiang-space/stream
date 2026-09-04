"""Exact finite-state structural mapping primitives."""

from stream.structural.direct_cost import DirectEvent, assert_factor_accounting, direct_cost
from stream.structural.elimination import EliminationResult, variable_elimination
from stream.structural.exhaustive import ExhaustiveResult, exhaustive_minimize
from stream.structural.factors import FactorGraph, OwnedEventFactor, PhysicalEvent, Variable
from stream.structural.states import (
    DistributionKind,
    DistributionPlan,
    DistributionPlanKind,
    DistributionTemplate,
    MaterializationMode,
    OperatorState,
    TensorRealization,
)
from stream.structural.stream_contract import (
    CompiledStructuralAssignment,
    CompileStage,
    CompileStatus,
    Gate1AEvalConfig,
    LiteralClassification,
    LiteralKind,
    PipelineSemanticManifest,
    StructuralLiteral,
    StructuralMappingContract,
    UnsupportedReason,
    compile_post_transfer,
    compile_pre_transfer,
    compile_structural_assignment,
    load_intended_space,
)

__all__ = [
    "DirectEvent",
    "CompileStage",
    "CompileStatus",
    "CompiledStructuralAssignment",
    "DistributionKind",
    "DistributionPlan",
    "DistributionPlanKind",
    "DistributionTemplate",
    "EliminationResult",
    "ExhaustiveResult",
    "FactorGraph",
    "Gate1AEvalConfig",
    "LiteralClassification",
    "LiteralKind",
    "MaterializationMode",
    "OperatorState",
    "OwnedEventFactor",
    "PhysicalEvent",
    "PipelineSemanticManifest",
    "TensorRealization",
    "StructuralLiteral",
    "StructuralMappingContract",
    "UnsupportedReason",
    "Variable",
    "assert_factor_accounting",
    "compile_post_transfer",
    "compile_pre_transfer",
    "compile_structural_assignment",
    "direct_cost",
    "exhaustive_minimize",
    "load_intended_space",
    "variable_elimination",
]
