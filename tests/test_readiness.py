"""Tests that AGI readiness remains conservative and fail-closed."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena.readiness import (  # noqa: E402
    EvidenceLevel,
    assess_agi_readiness,
)


def test_current_athena_is_not_reported_as_agi():
    report = assess_agi_readiness()
    assert report.agi_ready is False
    assert len(report.blockers) == len(report.gates)
    assert report.counts()[EvidenceLevel.NARROW.value] == 4
    assert report.counts()[EvidenceLevel.NOT_DEMONSTRATED.value] == 6


def test_narrow_laboratory_success_never_counts_as_a_pass():
    evidence = {
        gate.key: (EvidenceLevel.NARROW, "toy benchmark passed")
        for gate in assess_agi_readiness().gates
    }
    report = assess_agi_readiness(evidence)
    assert report.agi_ready is False
    assert len(report.blockers) == len(report.gates)


def test_missing_evidence_fails_closed():
    report = assess_agi_readiness({})
    assert report.agi_ready is False
    assert all(
        gate.evidence is EvidenceLevel.NOT_DEMONSTRATED for gate in report.gates
    )


def test_only_broad_evidence_for_every_gate_can_mark_agi_ready():
    evidence = {
        gate.key: (EvidenceLevel.DEMONSTRATED, "independent broad evaluation")
        for gate in assess_agi_readiness().gates
    }
    report = assess_agi_readiness(evidence)
    assert report.agi_ready is True
    assert report.blockers == ()


if __name__ == "__main__":
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failed = 0
    for name, function in tests:
        try:
            function()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {exc}")
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
