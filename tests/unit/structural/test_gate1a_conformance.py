from stream.structural.conformance import (
    CandidateSetAudit,
    Gate1AReport,
    Gate1AVerdict,
    SolutionSetAudit,
    ViolationWitnessAudit,
    audit_solution_sets,
    audit_violation_witness,
)
from stream.structural.stream_contract import CompileStage, CompileStatus, LiteralClassification


def _classification(literal_id: str, status: CompileStatus) -> LiteralClassification:
    if status is CompileStatus.EXACT:
        return LiteralClassification(literal_id, status, stage=CompileStage.PRE_TRANSFER)
    raise AssertionError("test helper only creates exact classifications")


def test_finite_witness_and_solution_set_audits_cover_soundness_and_completeness():
    from stream.structural.stream_contract import LiteralKind, StructuralLiteral

    literal = StructuralLiteral("A.zone", LiteralKind.HARDWARE_ZONE, "A", ("cores:0", "cores:1"))
    valid = frozenset({(("A.zone", "cores:0"),), (("A.zone", "cores:1"),)})
    too_narrow = frozenset({(("A.zone", "cores:0"),)})

    assert audit_violation_witness(literal, valid).passed
    assert not audit_solution_sets(too_narrow, valid).complete
    assert not audit_solution_sets(too_narrow, valid).exact


def test_gate_remains_not_run_until_all_three_layers_and_sample_count_exist():
    report = Gate1AReport(
        intended_literal_count=1,
        classifications=(_classification("A.zone", CompileStatus.EXACT),),
        candidate_audits=(CandidateSetAudit("A.zone", frozenset({"cores:0"}), frozenset({"cores:0"})),),
        audited_assignment_count=999,
    )
    assert report.verdict is Gate1AVerdict.NOT_RUN


def test_complete_clean_evidence_distinguishes_pass_and_narrow():
    classification = _classification("A.zone", CompileStatus.EXACT)
    candidate = CandidateSetAudit("A.zone", frozenset({"cores:0"}), frozenset({"cores:0"}))
    witness = ViolationWitnessAudit("A.zone", ())
    solutions = SolutionSetAudit(frozenset({(("A.zone", "cores:0"),)}), frozenset({(("A.zone", "cores:0"),)}))
    common = {
        "classifications": (classification,),
        "candidate_audits": (candidate,),
        "witness_audits": (witness,),
        "solution_set_audits": (solutions,),
        "audited_assignment_count": 1000,
    }

    assert Gate1AReport(intended_literal_count=1, **common).verdict is Gate1AVerdict.PASS
    assert (
        Gate1AReport(intended_literal_count=2, meaningful_exact_subset=True, **common).verdict is Gate1AVerdict.NARROW
    )


def test_any_semantic_mismatch_is_fail_after_evidence_is_complete():
    classification = _classification("A.zone", CompileStatus.EXACT)
    report = Gate1AReport(
        intended_literal_count=1,
        classifications=(classification,),
        candidate_audits=(CandidateSetAudit("A.zone", frozenset({"cores:0", "cores:1"}), frozenset({"cores:0"})),),
        witness_audits=(ViolationWitnessAudit("A.zone", ()),),
        solution_set_audits=(audit_solution_sets(frozenset(), frozenset()),),
        audited_assignment_count=1000,
    )
    assert report.false_exact == 1
    assert report.verdict is Gate1AVerdict.FAIL
