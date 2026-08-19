"""Behavioral tests for verified procedural skill acquisition."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena.skills import (  # noqa: E402
    DEFAULT_VERIFICATION_INPUTS,
    Example,
    NovelTaskLearner,
    Program,
    ProgramCatalog,
    SkillRegistry,
)


def _cases(program, inputs=DEFAULT_VERIFICATION_INPUTS):
    return tuple(Example(tuple(item), program.apply(item)) for item in inputs)


def test_unknown_rule_is_identified_by_active_experiment_and_transfers():
    target = Program(("take_even", "reverse"))
    learner = NovelTaskLearner()

    report = learner.learn_by_experiment("outside-in", target.apply)

    assert report.gap_before.status == "unresolved"
    assert report.gap_before.hypotheses_remaining > 50
    assert report.experiments
    assert report.experiments[0].information_gain_bits > 0.0
    assert report.gap_after.hypotheses_remaining == 1
    assert report.candidate_program == target
    assert report.consolidation.accepted
    # None of these numbers occur in the letter-only discovery probes.  The
    # acquired procedure must execute structurally instead of replaying memory.
    transfer_input = (90, 20, 70, 10, 50, 30)
    assert learner.registry.run("outside-in", transfer_input) == target.apply(
        transfer_input
    )


def test_instruction_is_not_consolidated_unless_behavior_verifies():
    learner = NovelTaskLearner()
    target = Program(("reverse", "unique"))

    rejected = learner.learn_from_instruction(
        "dedupe-backwards",
        ("rotate_left",),
        verifier=target.apply,
    )
    assert not rejected.consolidation.accepted
    assert len(learner.registry) == 0

    accepted = learner.learn_from_instruction(
        "dedupe-backwards",
        target.steps,
        verifier=target.apply,
    )
    assert accepted.consolidation.accepted
    assert learner.registry.run("dedupe-backwards", ("A", "B", "A")) == (
        "A",
        "B",
    )


def test_skill_checkpoint_retains_executable_behavior_after_restart():
    target = Program(("swap_pairs", "rotate_right"))
    learner = NovelTaskLearner()
    assert learner.learn_by_experiment("pair-wheel", target.apply).consolidation.accepted

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "skills.json"
        learner.registry.save(path)
        restored = SkillRegistry.load(path)

    novel_input = ("q", "w", "e", "r", "t")
    assert restored.run("pair-wheel", novel_input) == target.apply(novel_input)
    assert restored.get("pair-wheel") == learner.registry.get("pair-wheel")


def test_regression_gate_rejects_replacement_that_forgets_old_cases():
    registry = SkillRegistry()
    original = Program(("reverse",))
    first = registry.consolidate(
        name="turn-around",
        domain="sequence-transformation",
        program=original,
        acquired_via="instruction",
        verification_cases=_cases(original),
    )
    assert first.accepted

    incompatible = Program(("rotate_left",))
    replacement = registry.consolidate(
        name="turn-around",
        domain="sequence-transformation",
        program=incompatible,
        acquired_via="new-feedback",
        # The new evidence alone supports the incompatible replacement.  It
        # must still pass the established skill's protected regression cases.
        verification_cases=_cases(incompatible),
    )
    assert not replacement.accepted
    assert not replacement.verification.regression_passed
    assert registry.get("turn-around").version == 1
    assert registry.run("turn-around", (1, 2, 3)) == (3, 2, 1)


def test_learned_skills_can_be_composed_into_a_larger_verified_skill():
    registry = SkillRegistry()
    learner = NovelTaskLearner(registry=registry)
    reverse = Program(("reverse",))
    unique = Program(("unique",))
    learner.learn_from_instruction("reverse", reverse.steps, verifier=reverse.apply)
    learner.learn_from_instruction("unique", unique.steps, verifier=unique.apply)

    combined = Program(("reverse", "unique"))
    report = registry.compose(
        "reverse-then-unique",
        ("reverse", "unique"),
        verifier=combined.apply,
    )
    assert report.accepted
    assert report.skill is not None
    assert report.skill.components == ("reverse", "unique")
    assert registry.run("reverse-then-unique", ("A", "B", "A", "C")) == (
        "C",
        "A",
        "B",
    )


def test_learning_a_second_skill_does_not_change_the_first():
    learner = NovelTaskLearner()
    first_target = Program(("sort_asc", "rotate_left"))
    second_target = Program(("duplicate_each", "take_odd"))
    probe = ("D", "A", "C", "B")

    learner.learn_by_experiment("first", first_target.apply)
    before = learner.registry.run("first", probe)
    learner.learn_by_experiment("second", second_target.apply)
    after = learner.registry.run("first", probe)

    assert before == after == first_target.apply(probe)
    assert len(learner.registry) == 2


def test_out_of_language_observation_is_reported_instead_of_memorized():
    learner = NovelTaskLearner(catalog=ProgramCatalog())

    def impossible(_values):
        return ("not", "in", "the", "dsl", "at", "all", "!")

    try:
        learner.learn_by_experiment("impossible", impossible)
    except ValueError as exc:
        assert "contradicts every hypothesis" in str(exc)
        assert len(learner.registry) == 0
        return
    raise AssertionError("an unsupported rule was falsely reported as learned")


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
