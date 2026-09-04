from __future__ import annotations

import pytest

from stream.structural.operator_template_census import (
    OperatorTemplateCensusError,
    _divisors_up_to,
    _operator_census,
    _parse_core_group,
    _validate_contract,
    load_operator_template_contract,
)


def _node() -> dict:
    return {
        "id": "ComputationNode:Gemm",
        "operation": "Gemm",
        "dimensions": [
            {"global_position": 0, "size": 8},
            {"global_position": 1, "size": 16},
            {"global_position": 2, "size": 4},
        ],
        "outputs": [{"name": "O"}],
        "operand_maps": ["(d0, d1, d2) -> (d0, d1)", "(d0, d1, d2) -> (d0, d2)"],
    }


def _mapping() -> dict:
    return {
        "Gemm": {
            "node_type": "ComputationNode",
            "resource_options": ["cores:0,1,2,3"],
            "tiling_options": ["tiling:D0=4"],
        }
    }


def test_operator_census_uses_only_output_indexed_dimensions_and_bounded_chunks():
    result = _operator_census("test", 0, _node(), _mapping())

    assert result["output_indexed_dimensions"] == [
        {"dimension": "D0", "extent": 8},
        {"dimension": "D2", "extent": 4},
    ]
    assert result["candidate_state_count"] == 11
    assert result["candidate_templates_unique"] is True
    assert all(
        template["core_count"] == len(template["core_group"])
        and 4 % template["core_count"] == 0
        and all(split["dimension"] != "D1" for split in template["splits"])
        for template in result["candidate_templates"]
    )


def test_core_groups_and_divisors_fail_closed():
    assert _parse_core_group("cores:3,1,2") == (1, 2, 3)
    assert _divisors_up_to(12, 4) == (1, 2, 3, 4)
    with pytest.raises(OperatorTemplateCensusError, match="duplicate core"):
        _parse_core_group("cores:1,1")
    with pytest.raises(OperatorTemplateCensusError, match="positive"):
        _divisors_up_to(0, 4)


def test_contract_rejects_unreported_criteria():
    contract = load_operator_template_contract()
    contract["pass_criteria"]["unknown"] = True

    with pytest.raises(OperatorTemplateCensusError, match="pass criteria"):
        _validate_contract(contract)
