"""Behavioral tests for Athena's permissioned unfamiliar-tool agent."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena.foundation import (  # noqa: E402
    FoundationError,
    OpenAIResponsesToolReasoner,
    OpenRouterChatToolReasoner,
)
from athena.tool_learning import (  # noqa: E402
    OpaqueKVWorld,
    ProcedureStep,
    ToolGoal,
    ToolLearningAgent,
    ToolPolicy,
    ToolResult,
    ToolSkillRegistry,
    make_validation_cases,
)


def _learn_store(seed=7):
    agent = ToolLearningAgent()
    goal = ToolGoal("store", "lead-42", "booked")
    report = agent.learn(
        "store-and-verify",
        OpaqueKVWorld(seed),
        goal,
        validation_cases=make_validation_cases(seed, "store"),
    )
    return agent, report


def test_agent_learns_unfamiliar_tools_and_consolidates_verified_procedure():
    agent, report = _learn_store()

    assert report.acquisition.success
    assert report.acquisition.reasoner_steps > 0
    assert all(item.prediction_error == 0.0 for item in report.acquisition.trace)
    assert report.consolidated
    assert len(report.validations) == 2
    assert all(item.passed for item in report.validations)
    assert [item.capability for item in report.candidate_steps] == [
        "write_value",
        "read_value",
    ]
    assert report.skill is not None
    assert report.skill.required_capabilities == ("write_value", "read_value")
    assert len(agent.registry) == 1


def test_retained_skill_transfers_to_new_tool_names_and_goal_values():
    agent, report = _learn_store(seed=11)
    assert report.skill is not None
    training_tools = {item.decision.tool_name for item in report.acquisition.trace}

    world = OpaqueKVWorld(9_999)
    goal = ToolGoal("store", "completely-new-key", "completely-new-value")
    run = agent.execute_skill("store-and-verify", world, goal)
    transfer_tools = {item.decision.tool_name for item in run.trace}

    assert run.success
    assert run.reasoner_steps == 0, "retained procedure still needed foundation reasoning"
    assert world.data == {"completely-new-key": "completely-new-value"}
    assert training_tools != transfer_tools, "test world did not actually rename tools"


def test_update_and_delete_workflows_are_learned_and_transferred():
    for seed, kind in ((101, "update"), (202, "delete")):
        key = f"{kind}-key"
        goal = ToolGoal(kind, key, None if kind == "delete" else "new-value")
        agent = ToolLearningAgent()
        report = agent.learn(
            f"{kind}-workflow",
            OpaqueKVWorld(seed, {key: "old-value"}),
            goal,
            validation_cases=make_validation_cases(seed, kind),
        )
        assert report.consolidated

        transfer_key = f"transfer-{kind}"
        transfer_goal = ToolGoal(
            kind,
            transfer_key,
            None if kind == "delete" else "transferred-value",
        )
        transfer = agent.execute_skill(
            f"{kind}-workflow",
            OpaqueKVWorld(seed + 10_000, {transfer_key: "different-old"}),
            transfer_goal,
        )
        assert transfer.success
        assert transfer.reasoner_steps == 0


def test_procedure_survives_restart_and_remains_executable():
    agent, _ = _learn_store(seed=22)
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "tool-skills.json"
        agent.registry.save(checkpoint)
        restored_registry = ToolSkillRegistry.load(checkpoint)

    restored = ToolLearningAgent(registry=restored_registry)
    world = OpaqueKVWorld(23)
    goal = ToolGoal("store", "after-restart", "retained")
    run = restored.execute_skill("store-and-verify", world, goal)
    assert run.success
    assert run.used_skill == "store-and-verify"
    assert restored.registry.get("store-and-verify") == agent.registry.get(
        "store-and-verify"
    )


def test_external_write_is_denied_even_when_reasoner_requests_it():
    world = OpaqueKVWorld(31, external_write_capability="write_value")
    agent = ToolLearningAgent(policy=ToolPolicy(max_calls=8))
    report = agent.solve(world, ToolGoal("store", "protected", "nope"))

    assert not report.success
    assert report.denied_calls > 0
    assert world.data == {}


class FailingWriteWorld(OpaqueKVWorld):
    def invoke(self, name, arguments):
        result = super().invoke(name, arguments)
        if result.state_changed and result.output.get("stored"):
            return ToolResult(
                False,
                {"mutated_before_failure": True},
                state_changed=True,
                error="injected write failure",
            )
        return result


def test_failed_reversible_write_rolls_back_world_state():
    world = FailingWriteWorld(41)
    agent = ToolLearningAgent(policy=ToolPolicy(max_calls=8))
    report = agent.solve(world, ToolGoal("store", "rollback-me", "unsafe"))

    assert not report.success
    assert report.rollback_count > 0
    assert world.data == {}


class ExplodingWriteWorld(OpaqueKVWorld):
    def invoke(self, name, arguments):
        result = super().invoke(name, arguments)
        if result.state_changed and result.output.get("stored"):
            raise RuntimeError("injected crash after mutation")
        return result


def test_exception_after_mutation_is_caught_and_rolled_back():
    world = ExplodingWriteWorld(42)
    agent = ToolLearningAgent(policy=ToolPolicy(max_calls=8))
    report = agent.solve(world, ToolGoal("store", "crash-safe", "value"))

    assert not report.success
    assert report.rollback_count > 0
    assert any("RuntimeError" in (item.result.error or "") for item in report.trace)
    assert world.data == {}


def test_skill_is_not_consolidated_without_independent_worlds():
    agent = ToolLearningAgent()
    report = agent.learn(
        "under-tested",
        OpaqueKVWorld(50),
        ToolGoal("store", "x", "y"),
        validation_cases=(),
    )
    assert report.acquisition.success
    assert not report.consolidated
    assert "at least 2" in report.reason
    assert len(agent.registry) == 0


def test_existing_procedure_cannot_be_silently_rewritten():
    agent, report = _learn_store(seed=55)
    assert report.skill is not None
    accepted, reason, retained = agent.registry.consolidate(
        name="store-and-verify",
        task_kind="store",
        steps=(
            ProcedureStep("read_value", {"slot": "$goal.key"}),
            ProcedureStep(
                "write_value",
                {"slot": "$goal.key", "payload": "$goal.value"},
            ),
        ),
        acquired_from="untrusted-replacement",
        validations=report.validations,
    )
    assert not accepted
    assert "overwrite" in reason
    assert retained == report.skill
    assert agent.registry.get("store-and-verify").version == 1


class CapturingToolTransport:
    def __init__(self, name="inspect_manual", finish=False):
        self.calls = []
        self.name = "finish_task" if finish else name

    def __call__(self, url, headers, payload, timeout):
        self.calls.append((url, headers, payload, timeout))
        arguments = {
            "_hypothesis": "The manual will reveal the operation.",
            "_expected_observation": "A capability identifier.",
            "_expected_success": True,
            "_confidence": 0.8,
        }
        if self.name == "inspect_manual":
            arguments["tool"] = "aegis"
        if self.name == "finish_task":
            arguments["summary"] = "Observed state matches the goal."
        return {
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_test",
                    "name": self.name,
                    "arguments": json.dumps(arguments),
                }
            ],
        }


class CapturingOpenRouterToolTransport(CapturingToolTransport):
    def __init__(self, name="inspect_manual", finish=False, argument_overrides=None):
        super().__init__(name=name, finish=finish)
        self.argument_overrides = argument_overrides or {}

    def __call__(self, url, headers, payload, timeout):
        self.calls.append((url, headers, payload, timeout))
        arguments = {
            "_hypothesis": "The manual will reveal the operation.",
            "_expected_observation": "A capability identifier.",
            "_expected_success": True,
            "_confidence": 0.8,
        }
        if self.name == "inspect_manual":
            arguments["tool"] = "aegis"
        if self.name == "finish_task":
            arguments["summary"] = "Observed state matches the goal."
        arguments.update(self.argument_overrides)
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_test",
                                "type": "function",
                                "function": {
                                    "name": self.name,
                                    "arguments": json.dumps(arguments),
                                },
                            }
                        ],
                    }
                }
            ]
        }


def test_openai_reasoner_uses_strict_function_tools_behind_policy_boundary():
    transport = CapturingToolTransport()
    reasoner = OpenAIResponsesToolReasoner(
        api_key="secret-tool-key",
        model="test-model",
        transport=transport,
    )
    world = OpaqueKVWorld(61)
    decision = reasoner.next_step(
        ToolGoal("store", "x", "y"),
        tools=world.tools(),
        trace=(),
        known_skills=(),
    )

    assert decision.action == "call"
    assert decision.tool_name == "inspect_manual"
    assert decision.arguments == {"tool": "aegis"}
    url, headers, payload, timeout = transport.calls[0]
    assert url.endswith("/v1/responses")
    assert headers["Authorization"] == "Bearer secret-tool-key"
    assert "secret-tool-key" not in json.dumps(payload)
    assert payload["tool_choice"] == "required"
    assert payload["parallel_tool_calls"] is False
    assert payload["store"] is False
    assert all(item["strict"] is True for item in payload["tools"])
    assert all(
        item["parameters"]["additionalProperties"] is False
        for item in payload["tools"]
    )
    assert timeout == 60.0


def test_openai_reasoner_finish_is_still_checked_by_environment_verifier():
    transport = CapturingToolTransport(finish=True)
    reasoner = OpenAIResponsesToolReasoner(api_key="test", transport=transport)
    decision = reasoner.next_step(
        ToolGoal("delete", "x"),
        tools=OpaqueKVWorld(70).tools(),
        trace=(),
        known_skills=(),
    )
    assert decision.action == "finish"
    assert decision.tool_name is None
    assert decision.expected_success is True


def test_openai_reasoner_cannot_select_unregistered_tool():
    reasoner = OpenAIResponsesToolReasoner(
        api_key="test",
        transport=CapturingToolTransport(name="shell_exec"),
    )
    try:
        reasoner.next_step(
            ToolGoal("store", "x", "y"),
            tools=OpaqueKVWorld(80).tools(),
            trace=(),
            known_skills=(),
        )
    except FoundationError as exc:
        assert "unavailable tool" in str(exc)
        return
    raise AssertionError("unregistered tool escaped the reasoner boundary")


def test_openrouter_reasoner_uses_chat_tools_behind_local_policy_boundary():
    transport = CapturingOpenRouterToolTransport()
    reasoner = OpenRouterChatToolReasoner(
        api_key="secret-openrouter-tool-key",
        transport=transport,
    )
    world = OpaqueKVWorld(81)
    decision = reasoner.next_step(
        ToolGoal("store", "x", "y"),
        tools=world.tools(),
        trace=(),
        known_skills=(),
    )
    assert decision.tool_name == "inspect_manual"
    assert decision.arguments == {"tool": "aegis"}
    url, headers, payload, timeout = transport.calls[0]
    assert url == "https://openrouter.ai/api/v1/chat/completions"
    assert headers["Authorization"] == "Bearer secret-openrouter-tool-key"
    assert "secret-openrouter-tool-key" not in json.dumps(payload)
    assert payload["model"] == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert payload["tool_choice"] == "required"
    assert payload["parallel_tool_calls"] is False
    assert all("function" in item for item in payload["tools"])
    assert all(item["function"]["strict"] is True for item in payload["tools"])
    assert timeout == 120.0


def test_openrouter_reasoner_cannot_select_unregistered_tool():
    reasoner = OpenRouterChatToolReasoner(
        api_key="test",
        transport=CapturingOpenRouterToolTransport(name="shell_exec"),
    )
    try:
        reasoner.next_step(
            ToolGoal("store", "x", "y"),
            tools=OpaqueKVWorld(82).tools(),
            trace=(),
            known_skills=(),
        )
    except FoundationError as exc:
        assert "unavailable tool" in str(exc)
        return
    raise AssertionError("unregistered OpenRouter tool escaped the reasoner boundary")


def test_openrouter_reasoner_rejects_arguments_outside_registered_schema():
    reasoner = OpenRouterChatToolReasoner(
        api_key="test",
        transport=CapturingOpenRouterToolTransport(
            argument_overrides={"shell_command": "do-not-run"}
        ),
    )
    try:
        reasoner.next_step(
            ToolGoal("store", "x", "y"),
            tools=OpaqueKVWorld(83).tools(),
            trace=(),
            known_skills=(),
        )
    except FoundationError as exc:
        assert "invalid tool arguments" in str(exc)
        return
    raise AssertionError("extra OpenRouter argument escaped local schema validation")


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
