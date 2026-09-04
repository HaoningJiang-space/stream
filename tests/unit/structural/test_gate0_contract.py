import pytest

from stream.structural.gate0 import run_gate0


@pytest.mark.slow
def test_quantitative_gate0_contract():
    report = run_gate0(configs_per_class=20, base_seed=20260904)
    summary = report.summary()

    assert report.passed
    assert summary["dag_classes"] == 6
    assert summary["configuration_count"] == 120
    assert summary["unique_configuration_count"] == 120
    assert summary["direct_factor_match_rate"] == 1.0
    assert summary["ve_optimum_match_count"] == 120
    assert summary["ve_optimum_match_rate"] == 1.0
    assert summary["canonical_optimum_hit_count"] == 120
    assert summary["canonical_optimum_hit_rate"] == 1.0
    assert summary["baseline_coverage_count"] == 120
    assert summary["baseline_coverage_rate"] == 1.0
    assert summary["heuristic_prunes"] == 0
    assert summary["reduction_accounting_count"] == 120
