"""Procedural multi-world benchmark for Athena's tool-learning agent."""

from __future__ import annotations

from statistics import mean

from athena import (
    OpaqueKVWorld,
    ToolGoal,
    ToolLearningAgent,
    make_validation_cases,
)


def main() -> None:
    trials = []
    for seed in range(30):
        for kind in ("store", "update", "delete"):
            key = f"training-{seed}-{kind}"
            value = None if kind == "delete" else f"value-{seed}-{kind}"
            initial = {key: "old"} if kind in ("update", "delete") else {}
            agent = ToolLearningAgent()
            report = agent.learn(
                f"{kind}-workflow",
                OpaqueKVWorld(seed, initial),
                ToolGoal(kind, key, value),
                validation_cases=make_validation_cases(seed, kind),
            )
            transfer_key = f"transfer-{seed}-{kind}"
            transfer_value = None if kind == "delete" else f"new-{seed}-{kind}"
            transfer_initial = (
                {transfer_key: "different-old"}
                if kind in ("update", "delete")
                else {}
            )
            transfer = agent.execute_skill(
                f"{kind}-workflow",
                OpaqueKVWorld(seed + 50_000, transfer_initial),
                ToolGoal(kind, transfer_key, transfer_value),
            )
            trials.append((report, transfer))

    acquisitions = sum(report.acquisition.success for report, _ in trials)
    validations = sum(
        item.passed for report, _ in trials for item in report.validations
    )
    consolidations = sum(report.consolidated for report, _ in trials)
    transfers = sum(transfer.success for _, transfer in trials)
    transfer_reasoning = sum(transfer.reasoner_steps for _, transfer in trials)
    acquisition_steps = [report.acquisition.reasoner_steps for report, _ in trials]
    print("Athena v0.6 unfamiliar-tool benchmark")
    print(f"training worlds solved: {acquisitions}/{len(trials)}")
    print(f"held-out worlds passed: {validations}/{len(trials) * 2}")
    print(f"workflows consolidated: {consolidations}/{len(trials)}")
    print(f"renamed-world transfers: {transfers}/{len(trials)}")
    print(f"mean acquisition decisions: {mean(acquisition_steps):.2f}")
    print(f"foundation decisions during retained transfer: {transfer_reasoning}")


if __name__ == "__main__":
    main()
