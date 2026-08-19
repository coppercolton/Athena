"""Exhaustive benchmark over Athena's first procedural hypothesis language."""

from __future__ import annotations

from statistics import mean

from athena import NovelTaskLearner, ProgramCatalog, SkillRegistry


def main() -> None:
    catalog = ProgramCatalog()
    registry = SkillRegistry()
    experiment_counts: list[int] = []
    verified_cases = 0
    induced = 0
    transfers = 0
    transfer_input = (90, 20, 70, 10, 50, 30)

    for index, target in enumerate(catalog.programs):
        learner = NovelTaskLearner(registry=registry, catalog=catalog)
        name = f"skill-{index:02d}"
        report = learner.learn_by_experiment(name, target.apply)
        experiment_counts.append(len(report.experiments))
        if report.consolidation.accepted:
            induced += 1
            verified_cases += report.consolidation.verification.passed_cases
        if registry.run(name, transfer_input) == target.apply(transfer_input):
            transfers += 1

    retained = sum(
        registry.run(f"skill-{index:02d}", transfer_input)
        == target.apply(transfer_input)
        for index, target in enumerate(catalog.programs)
    )
    print("Athena v0.5 exhaustive skill benchmark")
    print(f"canonical unknown programs: {len(catalog.programs)}")
    print(f"successfully induced: {induced}/{len(catalog.programs)}")
    print(f"mean active experiments: {mean(experiment_counts):.2f}")
    print(f"maximum active experiments: {max(experiment_counts)}")
    print(f"held-out cases passed: {verified_cases}/{len(catalog.programs) * 6}")
    print(f"novel-token transfers: {transfers}/{len(catalog.programs)}")
    print(f"retained after sequential learning: {retained}/{len(catalog.programs)}")


if __name__ == "__main__":
    main()
