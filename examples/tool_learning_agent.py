"""One transparent unfamiliar-tool learning and transfer demonstration."""

from __future__ import annotations

from pathlib import Path
import tempfile

from athena import (
    OpaqueKVWorld,
    ToolGoal,
    ToolLearningAgent,
    ToolSkillRegistry,
    make_validation_cases,
)


def main() -> None:
    agent = ToolLearningAgent()
    goal = ToolGoal("store", "new-customer", "follow-up-booked")
    report = agent.learn(
        "store-and-verify",
        OpaqueKVWorld(17),
        goal,
        validation_cases=make_validation_cases(17, "store"),
    )

    print("Athena v0.6 unfamiliar-tool mission")
    print(f"knowledge gap: {report.knowledge_gap}")
    for experience in report.acquisition.trace:
        observation = experience.result.output or {"error": experience.result.error}
        print(
            f"{experience.index}. predict {experience.decision.tool_name} -> "
            f"observe {observation} [{experience.permission}]"
        )
    print(f"training verified: {report.acquisition.success}")
    print(
        "held-out worlds: "
        f"{sum(item.passed for item in report.validations)}/{len(report.validations)}"
    )
    print(f"consolidated: {report.consolidated}")
    assert report.skill is not None
    print(
        "compiled workflow: "
        + " -> ".join(step.capability for step in report.skill.steps)
    )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "tool-skills.json"
        agent.registry.save(path)
        restored = ToolLearningAgent(registry=ToolSkillRegistry.load(path))

    transfer_goal = ToolGoal("store", "different-key", "different-value")
    transfer = restored.execute_skill(
        "store-and-verify",
        OpaqueKVWorld(91_337),
        transfer_goal,
    )
    print(f"transfer after restart: {transfer.success}")
    print(f"new foundation decisions during transfer: {transfer.reasoner_steps}")


if __name__ == "__main__":
    main()
