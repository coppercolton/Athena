"""End-to-end tests for Athena's runnable browser agent."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from urllib import request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena import AgentConfig  # noqa: E402
from athena.foundation import (  # noqa: E402
    DemoFoundation,
    FoundationError,
    FoundationRefusal,
    OpenAIResponsesFoundation,
    OpenRouterChatFoundation,
)
from athena.playground import (  # noqa: E402
    PlaygroundService,
    _foundation,
    _is_loopback,
    make_server,
)


class CapturingTransport:
    def __init__(self, candidates=None, response=None):
        self.calls = []
        self.candidates = candidates or [
            {"action": "inspect", "response": "Inspect the interface.", "prior": 0.8},
            {"action": "test", "response": "Run a safe test.", "prior": 0.6},
        ]
        self.response = response

    def __call__(self, url, headers, payload, timeout):
        self.calls.append((url, headers, payload, timeout))
        if self.response is not None:
            return self.response
        return {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps({"candidates": self.candidates}),
                        }
                    ],
                }
            ],
        }


class CapturingOpenRouterTransport:
    def __init__(self, candidates=None, function_name="submit_candidates"):
        self.calls = []
        self.candidates = candidates or [
            {"action": "inspect", "response": "Inspect the interface.", "prior": 0.8},
            {"action": "test", "response": "Run a safe test.", "prior": 0.6},
        ]
        self.function_name = function_name

    def __call__(self, url, headers, payload, timeout):
        self.calls.append((url, headers, payload, timeout))
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
                                    "name": self.function_name,
                                    "arguments": json.dumps(
                                        {"candidates": self.candidates}
                                    ),
                                },
                            }
                        ],
                    }
                }
            ]
        }


def test_openai_adapter_uses_structured_output_without_leaking_key():
    transport = CapturingTransport()
    foundation = OpenAIResponsesFoundation(
        api_key="secret-test-key",
        model="test-model",
        transport=transport,
    )
    candidates = foundation.propose(
        "Learn a tool nobody has seen before",
        memories=(),
        facts=(),
        strategies=(),
        n=2,
    )
    assert [item.action for item in candidates] == ["inspect", "test"]
    url, headers, payload, timeout = transport.calls[0]
    assert url.endswith("/v1/responses")
    assert headers["Authorization"] == "Bearer secret-test-key"
    assert "secret-test-key" not in json.dumps(payload)
    assert payload["model"] == "test-model"
    output_format = payload["text"]["format"]
    assert output_format["type"] == "json_schema"
    assert output_format["strict"] is True
    assert output_format["schema"]["additionalProperties"] is False
    assert timeout == 60.0


def test_openai_adapter_exposes_refusal_as_a_typed_failure():
    transport = CapturingTransport(
        response={
            "status": "completed",
            "output": [
                {
                    "content": [
                        {"type": "refusal", "refusal": "Cannot help with that."}
                    ]
                }
            ],
        }
    )
    foundation = OpenAIResponsesFoundation(api_key="test", transport=transport)
    try:
        foundation.propose(
            "request",
            memories=(),
            facts=(),
            strategies=(),
            n=1,
        )
    except FoundationRefusal as exc:
        assert "Cannot help" in str(exc)
        return
    raise AssertionError("foundation refusal was treated as a candidate response")


def test_openrouter_adapter_uses_free_nemotron_tool_call_without_leaking_key():
    transport = CapturingOpenRouterTransport()
    foundation = OpenRouterChatFoundation(
        api_key="secret-openrouter-key",
        transport=transport,
    )
    candidates = foundation.propose(
        "Learn a tool nobody has seen before",
        memories=(),
        facts=(),
        strategies=(),
        n=2,
    )
    assert [item.action for item in candidates] == ["inspect", "test"]
    url, headers, payload, timeout = transport.calls[0]
    assert url == "https://openrouter.ai/api/v1/chat/completions"
    assert headers["Authorization"] == "Bearer secret-openrouter-key"
    assert "secret-openrouter-key" not in json.dumps(payload)
    assert payload["model"] == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert "response_format" not in payload
    assert payload["parallel_tool_calls"] is False
    assert payload["tool_choice"]["function"]["name"] == "submit_candidates"
    function = payload["tools"][0]["function"]
    assert function["strict"] is True
    assert function["parameters"]["additionalProperties"] is False
    assert function["parameters"]["properties"]["candidates"]["minItems"] == 2
    assert timeout == 120.0


def test_openrouter_candidate_contract_is_validated_locally():
    transport = CapturingOpenRouterTransport(
        candidates=[{"action": "unsafe", "response": "Guess.", "prior": 1.5}]
    )
    foundation = OpenRouterChatFoundation(api_key="test", transport=transport)
    try:
        foundation.propose(
            "request",
            memories=(),
            facts=(),
            strategies=(),
            n=1,
        )
    except FoundationError as exc:
        assert "invalid candidate" in str(exc)
        return
    raise AssertionError("invalid OpenRouter candidate escaped local validation")


def test_playground_auto_selects_openrouter_from_server_environment():
    previous_openrouter = os.environ.get("OPENROUTER_API_KEY")
    previous_openai = os.environ.get("OPENAI_API_KEY")
    try:
        os.environ["OPENROUTER_API_KEY"] = "test-openrouter"
        os.environ["OPENAI_API_KEY"] = "test-openai"
        foundation = _foundation("auto", None)
        assert isinstance(foundation, OpenRouterChatFoundation)
        assert foundation.model == "nvidia/nemotron-3-ultra-550b-a55b:free"
    finally:
        if previous_openrouter is None:
            os.environ.pop("OPENROUTER_API_KEY", None)
        else:
            os.environ["OPENROUTER_API_KEY"] = previous_openrouter
        if previous_openai is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = previous_openai


def test_demo_foundation_is_immediately_usable_offline():
    foundation = DemoFoundation()
    candidates = foundation.propose(
        "I need to learn an unknown software tool",
        memories=(),
        facts=(),
        strategies=(),
        n=3,
    )
    assert len(candidates) == 3
    assert candidates[0].action == "research"
    assert all(0.0 <= item.prior <= 1.0 for item in candidates)


def test_service_checkpoints_decisions_outcomes_and_taught_facts():
    with tempfile.TemporaryDirectory() as directory:
        state_path = Path(directory) / "state" / "athena.npz"
        service = PlaygroundService(
            DemoFoundation(),
            state_path,
            config=AgentConfig(feature_dim=24),
        )
        decision = service.decide(
            {"situation": "An urgent lead wants a reply", "context_key": "urgent"}
        )["decision"]
        pending = PlaygroundService(DemoFoundation(), state_path).state()
        assert pending["pending_decisions"][0]["id"] == decision["id"]
        service.learn(
            {
                "decision_id": decision["id"],
                "reward": 1.0,
                "observation": "The lead booked",
            }
        )
        for index in range(5):
            service.learn_fact(
                {
                    "key": "office closes",
                    "value": "6 PM",
                    "source": f"calendar-{index}",
                    "reliability": 1.0,
                }
            )
        assert state_path.exists()
        assert not state_path.with_name(".athena.npz.tmp").exists()

        restored = PlaygroundService(DemoFoundation(), state_path)
        state = restored.state()
        assert state["total_experiences"] == 1
        assert state["episodes"][0]["observation"] == "The lead booked"
        assert state["beliefs"][0]["consolidated"] is True


def test_service_discovers_persists_and_executes_a_verified_skill():
    with tempfile.TemporaryDirectory() as directory:
        state_path = Path(directory) / "state" / "athena.npz"
        service = PlaygroundService(DemoFoundation(), state_path)
        result = service.discover_skill(
            {
                "name": "unknown-rule",
                "seed": 42,
                "transfer_input": [9, 2, 7, 1, 5],
            }
        )
        learning = result["learning"]
        assert learning["gap_before"]["hypotheses_remaining"] > 50
        assert learning["gap_after"]["hypotheses_remaining"] == 1
        assert learning["consolidation"]["accepted"] is True
        assert result["transfer"]["passed"] is True
        assert result["state"]["skill_count"] == 1
        assert service.skill_path.exists()

        restored = PlaygroundService(DemoFoundation(), state_path)
        assert restored.state()["skills"][0]["name"] == "unknown-rule"
        executed = restored.run_skill(
            {"name": "unknown-rule", "input": ["unseen", "after", "restart"]}
        )
        assert isinstance(executed["output"], tuple)


def test_service_learns_persists_and_transfers_an_unfamiliar_tool_workflow():
    with tempfile.TemporaryDirectory() as directory:
        state_path = Path(directory) / "state" / "athena.npz"
        service = PlaygroundService(DemoFoundation(), state_path)
        result = service.learn_tool_workflow(
            {
                "name": "store-workflow",
                "kind": "store",
                "key": "new-lead",
                "value": "booked",
                "seed": 123,
            }
        )
        learning = result["learning"]
        assert learning["acquisition"]["success"] is True
        assert learning["consolidated"] is True
        assert len(learning["validations"]) == 2
        assert result["transfer"]["success"] is True
        assert result["transfer"]["reasoner_steps"] == 0
        assert result["state"]["tool_skill_count"] == 1
        assert service.tool_skill_path.exists()

        restored = PlaygroundService(DemoFoundation(), state_path)
        assert restored.state()["tool_skills"][0]["name"] == "store-workflow"
        replay = restored.run_tool_skill(
            {
                "name": "store-workflow",
                "key": "restart-key",
                "value": "restart-value",
                "seed": 456,
            }
        )["run"]
        assert replay["success"] is True
        assert replay["reasoner_steps"] == 0


def test_service_grows_persists_and_runs_verified_neural_operator():
    with tempfile.TemporaryDirectory() as directory:
        state_path = Path(directory) / "state" / "athena.npz"
        service = PlaygroundService(DemoFoundation(), state_path)
        result = service.grow_neural_skill(
            {
                "name": "relational-balance",
                "rule": "relative_balance",
                "seed": 901,
            }
        )
        report = result["report"]
        assert report["promoted"] is True
        assert report["weight_delta"] > 1.0
        assert report["checksum_before"] != report["checksum_after"]
        assert report["candidate_accuracy"] >= 0.95
        assert result["transfer"]["passed"] is True
        assert service.plasticity_path.exists()
        assert result["state"]["plastic_skill_count"] == 1

        restored = PlaygroundService(DemoFoundation(), state_path)
        assert restored.state()["plastic_skills"][0]["name"] == "relational-balance"
        prediction = restored.run_neural_skill(
            {
                "name": "relational-balance",
                "input": [0.9, 0.8, -0.4, -0.3],
            }
        )
        assert prediction["prediction"] == 1
        assert prediction["probability"] > 0.5


def test_service_learns_persists_and_reuses_raw_visual_representation():
    with tempfile.TemporaryDirectory() as directory:
        state_path = Path(directory) / "state" / "athena.npz"
        service = PlaygroundService(DemoFoundation(), state_path)
        result = service.ground_visual_operator(
            {
                "name": "visual-right-of",
                "rule": "horizontal_order",
                "seed": 0,
            }
        )

        representation = result["representation_report"]
        operator = result["operator_report"]
        assert representation["promoted"] is True
        assert representation["candidate_loss"] < representation["before_loss"]
        assert operator["promoted"] is True
        assert operator["candidate_accuracy"] > operator["untrained_representation_accuracy"]
        assert result["representation_reused"] is True
        assert result["transfer"]["passed"] is True
        assert result["state"]["representation"]["sensor_dim"] == 128
        assert result["state"]["representation"]["latent_dim"] == 16
        assert result["state"]["grounded_operator_count"] == 1
        assert service.representation_path.exists()

        restored = PlaygroundService(DemoFoundation(), state_path)
        state = restored.state()
        assert state["representation"] == result["state"]["representation"]
        assert state["grounded_operators"][0]["name"] == "visual-right-of"


def _json_request(url, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    outgoing = request.Request(
        f"{url}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    with request.urlopen(outgoing, timeout=5) as response:
        return response.status, response.headers, json.loads(response.read())


def test_browser_api_runs_the_full_prediction_feedback_loop():
    with tempfile.TemporaryDirectory() as directory:
        service = PlaygroundService(DemoFoundation(), Path(directory) / "state.npz")
        server = make_server(service, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address[:2]
            url = f"http://{host}:{port}"
            status, headers, initial = _json_request(url, "/api/state")
            assert status == 200
            assert initial["total_experiences"] == 0
            assert headers["Cache-Control"] == "no-store"

            _, _, proposed = _json_request(
                url,
                "/api/decide",
                {"situation": "Help a lead reply tonight", "context_key": "urgent"},
            )
            decision = proposed["decision"]
            assert decision["selected"]["action"] == "email"
            assert proposed["state"]["pending_count"] == 1

            _, _, learned = _json_request(
                url,
                "/api/learn",
                {
                    "decision_id": decision["id"],
                    "reward": 0,
                    "observation": "No reply",
                },
            )
            assert learned["report"]["adapted"] is True
            assert learned["state"]["total_experiences"] == 1

            _, _, tool_learning = _json_request(
                url,
                "/api/tools/learn",
                {
                    "name": "http-store-workflow",
                    "kind": "store",
                    "key": "http-key",
                    "value": "http-value",
                    "seed": 789,
                },
            )
            assert tool_learning["learning"]["consolidated"] is True
            assert tool_learning["transfer"]["success"] is True
            assert tool_learning["state"]["tool_skill_count"] == 1

            _, _, neural_learning = _json_request(
                url,
                "/api/plasticity/learn",
                {
                    "name": "http-neural-operator",
                    "rule": "relative_balance",
                    "seed": 902,
                },
            )
            assert neural_learning["report"]["promoted"] is True
            assert neural_learning["transfer"]["passed"] is True
            assert neural_learning["state"]["plastic_skill_count"] == 1

            _, _, representation_learning = _json_request(
                url,
                "/api/representations/learn",
                {
                    "name": "http-visual-operator",
                    "rule": "horizontal_order",
                    "seed": 0,
                },
            )
            assert representation_learning["representation_report"]["promoted"] is True
            assert representation_learning["operator_report"]["promoted"] is True
            assert representation_learning["transfer"]["passed"] is True
            assert representation_learning["state"]["grounded_operator_count"] == 1

            with request.urlopen(f"{url}/", timeout=5) as response:
                page = response.read().decode("utf-8")
            assert "Continual Intelligence Lab" in page
            assert "Enter an unfamiliar workspace" in page
            assert "Grow a neural reasoning operator" in page
            assert "Learn concepts from raw pixels" in page
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


def test_playground_defaults_to_local_only():
    assert _is_loopback("127.0.0.1")
    assert _is_loopback("::1")
    assert _is_loopback("localhost")
    assert not _is_loopback("0.0.0.0")


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
