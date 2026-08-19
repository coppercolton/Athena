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
    FoundationRefusal,
    OpenAIResponsesFoundation,
)
from athena.playground import PlaygroundService, _is_loopback, make_server  # noqa: E402


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

            with request.urlopen(f"{url}/", timeout=5) as response:
                page = response.read().decode("utf-8")
            assert "Continual Agent Lab" in page
            assert "Give Athena a problem" in page
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
