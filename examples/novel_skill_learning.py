"""Reproducible proof of Athena learning, retaining, and transferring skills."""

from __future__ import annotations

from pathlib import Path
import tempfile

from athena import NovelTaskLearner, Program, SkillRegistry


def main() -> None:
    learner = NovelTaskLearner()
    target = Program(("take_even", "reverse"))
    report = learner.learn_by_experiment("outside-in", target.apply)

    print("Athena v0.5 novel-skill trial")
    print(f"hypotheses before: {report.gap_before.hypotheses_remaining}")
    for index, experiment in enumerate(report.experiments, 1):
        print(
            f"experiment {index}: {experiment.hypotheses_before} -> "
            f"{experiment.hypotheses_after} hypotheses "
            f"({experiment.information_gain_bits:.2f} bits)"
        )
    learned = report.consolidation.skill
    assert learned is not None
    verification = report.consolidation.verification
    print(f"learned program: {learned.program.expression}")
    print(
        f"held-out verification: {verification.passed_cases}/"
        f"{verification.total_cases}"
    )

    transfer_input = (90, 20, 70, 10, 50, 30)
    transfer_output = learner.registry.run("outside-in", transfer_input)
    print(f"novel-token transfer: {transfer_input} -> {transfer_output}")

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "skills.json"
        learner.registry.save(checkpoint)
        restored = SkillRegistry.load(checkpoint)
    retained = restored.run("outside-in", transfer_input) == transfer_output
    print(f"retained after restart: {retained}")

    second = Program(("sort_desc", "rotate_right"))
    learner.learn_by_experiment("rank-wheel", second.apply)
    no_forgetting = learner.registry.run("outside-in", transfer_input) == transfer_output
    print(f"first skill after learning second: {no_forgetting}")


if __name__ == "__main__":
    main()
