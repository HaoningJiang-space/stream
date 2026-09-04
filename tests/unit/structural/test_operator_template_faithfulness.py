from __future__ import annotations

import pytest

from stream.structural.gate1a_cases import load_gate1a_accelerator
from stream.structural.operator_template_faithfulness import (
    _assignment_specs,
    _run_assignment,
    load_operator_template_faithfulness_contract,
)


def test_gate1a_v3_denominator_is_derived_from_frozen_dags_and_paired_templates():
    contract = load_operator_template_faithfulness_contract()
    specs = _assignment_specs(contract)

    assert len(specs) == contract["expected_assignment_count"] == 112
    assert len(set(specs)) == len(specs)
    assert len(contract["template_family"]) == 7


def test_compiled_template_matches_independent_downstream_reference(tmp_path):
    contract = load_operator_template_faithfulness_contract()
    result = _run_assignment(
        "edge",
        "A",
        4,
        tmp_path,
        load_gate1a_accelerator(contract["hardware"]),
        contract,
    )

    assert result["paired_candidate_set_exact"] is True
    assert result["compiled_reference_outcome_exact"] is True
    assert result["exact_executable"] is True
    assert result["compiler_deterministic"] is True
    assert result["silent_relaxations"] == 0


def test_shared_input_tiling_mismatch_is_exactly_localized_and_not_silently_relaxed(tmp_path):
    contract = load_operator_template_faithfulness_contract()
    result = _run_assignment(
        "residual_diamond",
        "A",
        4,
        tmp_path,
        load_gate1a_accelerator(contract["hardware"]),
        contract,
    )

    assert result["paired_candidate_set_exact"] is True
    assert result["compiled_reference_outcome_exact"] is True
    assert result["compiled_outcome"]["reason"] == "SHARED_INPUT_TILING_INCOMPATIBLE"
    assert "A=((z0, 2),), Add=((z0, 4),)" in result["compiled_outcome"]["failure_signature"]
    assert result["exact_executable"] is False
    assert result["silent_relaxations"] == 0


def test_gate_rejects_a_worker_override_that_bypasses_the_frozen_parallel_replay(tmp_path):
    from stream.structural.operator_template_faithfulness import run_operator_template_faithfulness

    with pytest.raises(ValueError, match="requires exactly 8"):
        run_operator_template_faithfulness(tmp_path / "report.json", max_workers=1)
