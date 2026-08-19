"""Behavioral tests for Athena's protected neural plasticity layer."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena.plasticity import (  # noqa: E402
    NeuralExample,
    PlasticityConfig,
    ProtectedPlasticity,
    make_reasoning_cases,
)


def _flipped(cases):
    return tuple(NeuralExample(item.inputs, 1 - item.target) for item in cases)


def _learn_balance(system=None, name="balance"):
    system = system or ProtectedPlasticity()
    report = system.learn(
        name,
        make_reasoning_cases("relative_balance", 128, 101),
        make_reasoning_cases("relative_balance", 96, 102),
    )
    return system, report


def test_verified_experience_changes_neural_weights_and_transfers():
    system, report = _learn_balance()

    assert report.promoted
    assert report.recruited
    assert report.weight_delta > 1.0
    assert report.checksum_before != report.checksum_after
    assert report.candidate_accuracy >= 0.95
    # Different seed and magnitude are unavailable during training. The expert
    # must learn the relational boundary rather than replay examples.
    transfer = make_reasoning_cases("relative_balance", 256, 103, scale=5.0)
    assert system.evaluate("balance", transfer) >= 0.98


def test_new_operator_recruits_capacity_without_changing_old_expert():
    system, _ = _learn_balance()
    balance_before = system.get("balance")
    balance_probe = make_reasoning_cases("relative_balance", 128, 104)
    accuracy_before = system.evaluate("balance", balance_probe)

    report = system.learn(
        "same-sign",
        make_reasoning_cases("same_sign", 128, 105),
        make_reasoning_cases("same_sign", 96, 106),
    )

    assert report.promoted and report.recruited
    assert len(system) == 2
    assert system.get("balance").checksum == balance_before.checksum
    assert system.evaluate("balance", balance_probe) == accuracy_before
    assert system.evaluate(
        "same-sign", make_reasoning_cases("same_sign", 256, 107, scale=1.5)
    ) >= 0.98


def test_incompatible_update_is_rolled_back_before_retained_mind_changes():
    system, _ = _learn_balance()
    before = system.get("balance")
    protected_probe = make_reasoning_cases("relative_balance", 128, 108)
    prediction_before = tuple(
        system.predict("balance", item.inputs) for item in protected_probe
    )

    report = system.learn(
        "balance",
        _flipped(make_reasoning_cases("relative_balance", 128, 109)),
        _flipped(make_reasoning_cases("relative_balance", 96, 110)),
    )

    assert not report.promoted
    assert report.rolled_back
    assert report.checksum_after == report.checksum_before == before.checksum
    assert system.get("balance") == before
    assert tuple(system.predict("balance", item.inputs) for item in protected_probe) == (
        prediction_before
    )


def test_compatible_update_replays_old_cases_and_advances_version():
    system, _ = _learn_balance()
    report = system.learn(
        "balance",
        make_reasoning_cases("relative_balance", 64, 111),
        make_reasoning_cases("relative_balance", 96, 112),
    )

    assert report.promoted
    assert not report.recruited
    assert not report.rolled_back
    assert report.version == 2
    assert report.regression_accuracy == 1.0
    assert system.get("balance").examples_seen == 192


def test_neural_skills_survive_restart_with_exact_predictions():
    system, _ = _learn_balance()
    probe = make_reasoning_cases("relative_balance", 64, 113)
    before = tuple(system.predict_probability("balance", item.inputs) for item in probe)
    checksum = system.get("balance").checksum

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "plasticity.npz"
        system.save(checkpoint)
        restored = ProtectedPlasticity.load(checkpoint)

    after = tuple(restored.predict_probability("balance", item.inputs) for item in probe)
    assert after == before
    assert restored.get("balance").checksum == checksum
    assert restored.get("balance").replay_examples == system.get(
        "balance"
    ).replay_examples


def test_unverified_candidate_never_enters_registry():
    system = ProtectedPlasticity()
    try:
        system.learn(
            "under-tested",
            make_reasoning_cases("relative_balance", 32, 114),
            make_reasoning_cases("relative_balance", 8, 115),
        )
    except ValueError as exc:
        assert "held-out" in str(exc)
        assert len(system) == 0
        return
    raise AssertionError("under-tested neural candidate entered the registry")


def test_learning_is_deterministic_given_seed_and_experience():
    first, first_report = _learn_balance()
    second, second_report = _learn_balance()

    assert first_report == second_report
    assert first.get("balance") == second.get("balance")


def test_rejects_invalid_neural_configuration_and_examples():
    for kwargs in (
        {"hidden_dim": 1},
        {"epochs": 0},
        {"validation_threshold": 0.5},
        {"replay_capacity": 8, "minimum_validation_cases": 16},
    ):
        try:
            PlasticityConfig(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid config accepted: {kwargs}")

    try:
        NeuralExample((1.0, 2.0), 2)
    except ValueError:
        return
    raise AssertionError("invalid neural target was accepted")


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
