from __future__ import annotations

import json
from importlib.resources import files

from stream.structural.tensor_restriction_gate import _independent_solution_satisfies


def test_gate1a_v2_denominator_is_frozen_to_the_minimal_exact_subset():
    resource = files("stream.structural.contracts").joinpath("gate1a_intended_space_v2.json")
    manifest = json.loads(resource.read_text(encoding="utf-8"))

    assert manifest["version"] == 2
    assert manifest["exact_tensor_literals"] == ["tensor_placement", "transfer_path"]
    assert manifest["unsupported_literals"] == ["partial", "streaming", "exact_reuse", "layout"]
    assert manifest["minimum_deterministic_assignments"] == 1000
    assert manifest["compile_repetitions"] == 3
    assert len(manifest["dag_classes"]) == 6
    assert len(manifest["allowed_placement_index_sets"]) == 6


def test_independent_oracle_predicate_detects_relaxation_and_overrestriction():
    solution = (("tensor:T", "cores:9"), ("path:Transfer(T)", "path-a"))
    intended = {
        "tensor:T": frozenset({"cores:9"}),
        "path:Transfer(T)": frozenset({"path-a"}),
    }

    assert _independent_solution_satisfies(solution, intended)
    assert not _independent_solution_satisfies(
        solution,
        {**intended, "tensor:T": frozenset({"cores:19"})},
    )
    assert not _independent_solution_satisfies(
        solution,
        {**intended, "path:Transfer(T)": frozenset({"path-b"})},
    )
