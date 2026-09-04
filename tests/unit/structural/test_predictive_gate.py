from __future__ import annotations

import json
from importlib.resources import files

import pytest

from stream.structural.gate1b_cases import load_gate1b_contract
from stream.structural.predictive_gate import _average_ranks, enumerate_structural_candidates, predictive_metrics
from stream.structural.stream_contract import Gate1AEvalConfig


def _evaluation(candidate_id: str, score: int, latency: int) -> dict:
    return {"candidate_id": candidate_id, "structural_score": score, "latency_cycles": latency}


def test_gate1b_contract_has_non_saturating_preregistered_census():
    resource = files("stream.structural.contracts").joinpath("gate1b_contract.json")
    contract = json.loads(resource.read_text(encoding="utf-8"))

    assert contract["candidate_space"]["required_candidates_per_class"] == 64
    assert contract["candidate_space"]["required_staging_tensors_per_class"] == 3
    assert max(contract["metrics"]["top_n"]) == 16
    assert contract["candidate_space"]["required_candidates_per_class"] > max(contract["metrics"]["top_n"])
    assert contract["pass_criteria"]["minimum_passing_dag_classes"] == 5
    assert contract["evaluation"]["minimum_ortools_version"] == "9.15"


def test_predictive_metrics_compute_recall_regret_and_spearman():
    evaluations = tuple(
        _evaluation(candidate_id, score, latency)
        for candidate_id, score, latency in (
            ("a", 1, 10),
            ("b", 2, 12),
            ("c", 3, 20),
            ("d", 4, 30),
            ("e", 5, 40),
        )
    )

    metrics = predictive_metrics(evaluations, (1, 4), (1, 5))

    assert metrics["recall_at_n"]["1"]["1"]["value"] == 1.0
    assert metrics["recall_at_n"]["4"]["5"]["value"] == 0.8
    assert metrics["regret_at_n"]["1"] == 0.0
    assert metrics["spearman"] == pytest.approx(1.0)


def test_recall_expands_the_tetra_cutoff_for_latency_ties():
    evaluations = tuple(
        _evaluation(candidate_id, score, latency)
        for candidate_id, score, latency in (
            ("a", 1, 10),
            ("b", 2, 10),
            ("c", 3, 10),
            ("d", 4, 20),
        )
    )

    metrics = predictive_metrics(evaluations, (1,), (1,))

    assert metrics["recall_at_n"]["1"]["1"]["value"] == 1.0
    assert metrics["recall_at_n"]["1"]["1"]["tie_expanded_good_set_size"] == 3


def test_average_ranks_use_midranks_for_ties():
    assert _average_ranks((30.0, 10.0, 10.0, 20.0)) == (4.0, 1.5, 1.5, 3.0)


@pytest.mark.parametrize("dag_class", load_gate1b_contract()["dag_classes"])
def test_each_gate1b_case_has_complete_nonconstant_candidate_space(dag_class: str):
    contract = load_gate1b_contract()
    evaluation = contract["evaluation"]
    config = Gate1AEvalConfig(
        backend=evaluation["backend"],
        constraints=tuple(evaluation["constraints"]),
        timeslot_policy=evaluation["timeslot_policy"],
    )

    candidates = enumerate_structural_candidates(dag_class, config, contract)

    assert len(candidates) == 64
    assert len({candidate.candidate_id for candidate in candidates}) == 64
    assert len({candidate.structural_score for candidate in candidates}) > 1
    assert all(len(candidate.restrictions) == 3 for candidate in candidates)
