"""Local browser playground for Athena's continual-learning agent.

Launch with ``python -m athena.playground`` or the installed
``athena-playground`` command. The server binds to loopback by default, keeps
credentials out of the browser, and checkpoints after every state transition.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
import ipaddress
import json
import os
from pathlib import Path
import random
import threading
from typing import Any
import webbrowser

from .agent import AgentConfig, AthenaAgent, Decision, FoundationModel
from .foundation import (
    DemoFoundation,
    FoundationError,
    OpenAIResponsesFoundation,
    OpenAIResponsesToolReasoner,
    OpenRouterChatFoundation,
    OpenRouterChatToolReasoner,
)
from .plasticity import ProtectedPlasticity, make_reasoning_cases
from .skills import NovelTaskLearner, SkillRegistry
from .tool_learning import (
    DemoToolReasoner,
    OpaqueKVWorld,
    ToolGoal,
    ToolLearningAgent,
    ToolReasoner,
    ToolSkillRegistry,
    make_validation_cases,
)


MAX_REQUEST_BYTES = 1_000_000
CONTENT_SECURITY_POLICY = "default-src 'self'; style-src 'self'; script-src 'self'"


def _decision_state(decision: Decision) -> dict[str, object]:
    return asdict(decision)


class PlaygroundService:
    """Thread-safe application service shared by HTTP handlers and tests."""

    def __init__(
        self,
        foundation: FoundationModel,
        state_path: str | Path,
        *,
        config: AgentConfig | None = None,
        tool_reasoner: ToolReasoner | None = None,
    ) -> None:
        self.foundation = foundation
        self.state_path = Path(state_path).expanduser().resolve()
        self.skill_path = self.state_path.with_name(
            f"{self.state_path.stem}.skills.json"
        )
        self.tool_skill_path = self.state_path.with_name(
            f"{self.state_path.stem}.tool-skills.json"
        )
        self.plasticity_path = self.state_path.with_name(
            f"{self.state_path.stem}.plasticity.npz"
        )
        self._lock = threading.RLock()
        if self.state_path.exists():
            self.agent = AthenaAgent.load(self.state_path, foundation=foundation)
        else:
            self.agent = AthenaAgent(foundation=foundation, config=config)
        registry = (
            SkillRegistry.load(self.skill_path)
            if self.skill_path.exists()
            else SkillRegistry()
        )
        self.skill_learner = NovelTaskLearner(registry=registry)
        tool_registry = (
            ToolSkillRegistry.load(self.tool_skill_path)
            if self.tool_skill_path.exists()
            else ToolSkillRegistry()
        )
        if tool_reasoner is None:
            if isinstance(foundation, OpenAIResponsesFoundation):
                tool_reasoner = OpenAIResponsesToolReasoner.from_foundation(foundation)
            elif isinstance(foundation, OpenRouterChatFoundation):
                tool_reasoner = OpenRouterChatToolReasoner.from_foundation(foundation)
            else:
                tool_reasoner = DemoToolReasoner()
        self.tool_agent = ToolLearningAgent(
            reasoner=tool_reasoner,
            registry=tool_registry,
        )
        self.plasticity = (
            ProtectedPlasticity.load(self.plasticity_path)
            if self.plasticity_path.exists()
            else ProtectedPlasticity()
        )

    @property
    def backend_name(self) -> str:
        return str(getattr(self.foundation, "name", type(self.foundation).__name__))

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_name(f".{self.state_path.name}.tmp")
        self.agent.save(temporary)
        os.replace(temporary, self.state_path)

    def _save_skills(self) -> None:
        self.skill_learner.registry.save(self.skill_path)

    def _save_tool_skills(self) -> None:
        self.tool_agent.registry.save(self.tool_skill_path)

    def _save_plasticity(self) -> None:
        self.plasticity.save(self.plasticity_path)

    def decide(self, payload: dict[str, Any]) -> dict[str, object]:
        situation = str(payload.get("situation", ""))
        context_key = str(payload.get("context_key", "general"))
        with self._lock:
            decision = self.agent.decide(situation, context_key=context_key)
            self._save()
            return {
                "decision": _decision_state(decision),
                "state": self.state(),
            }

    def learn(self, payload: dict[str, Any]) -> dict[str, object]:
        decision_id = str(payload.get("decision_id", ""))
        reward = float(payload.get("reward"))
        observation = str(payload.get("observation", ""))
        reliability = float(payload.get("reliability", 1.0))
        adapt = bool(payload.get("adapt", True))
        with self._lock:
            report = self.agent.learn(
                decision_id,
                reward,
                observation=observation,
                reliability=reliability,
                adapt=adapt,
            )
            self._save()
            return {"report": asdict(report), "state": self.state()}

    def learn_fact(self, payload: dict[str, Any]) -> dict[str, object]:
        with self._lock:
            belief = self.agent.learn_fact(
                str(payload.get("key", "")),
                str(payload.get("value", "")),
                source=str(payload.get("source", "")),
                reliability=float(payload.get("reliability", 1.0)),
            )
            self._save()
            return {"belief": asdict(belief), "state": self.state()}

    def discover_skill(self, payload: dict[str, Any]) -> dict[str, object]:
        """Run a real active-learning trial against an unrevealed rule."""

        with self._lock:
            default_name = f"novel-skill-{len(self.skill_learner.registry) + 1}"
            name = str(payload.get("name", default_name)).strip()
            if not name:
                raise ValueError("skill name is required")
            if len(name) > 120:
                raise ValueError("skill name must be at most 120 characters")
            seed = int(payload.get("seed", len(self.skill_learner.registry) + 1))
            target_pool = tuple(
                program
                for program in self.skill_learner.catalog.programs
                if len(program.steps) == 2
            )
            if not target_pool:
                raise RuntimeError("no multi-step target programs are available")
            target = random.Random(seed).choice(target_pool)
            report = self.skill_learner.learn_by_experiment(name, target.apply)
            if report.consolidation.accepted:
                self._save_skills()

            raw_transfer = payload.get("transfer_input", [90, 20, 70, 10, 50, 30])
            if not isinstance(raw_transfer, list):
                raise ValueError("transfer_input must be a JSON array")
            transfer_input = tuple(raw_transfer)
            learned_output = (
                self.skill_learner.registry.run(name, transfer_input)
                if report.consolidation.accepted
                else ()
            )
            expected_output = target.apply(transfer_input)
            return {
                "learning": asdict(report),
                # The world rule is revealed only after all predictions,
                # observations, and held-out verification have completed.
                "revealed_world_program": target.expression,
                "transfer": {
                    "input": transfer_input,
                    "output": learned_output,
                    "expected": expected_output,
                    "passed": learned_output == expected_output,
                },
                "state": self.state(),
            }

    def run_skill(self, payload: dict[str, Any]) -> dict[str, object]:
        name = str(payload.get("name", "")).strip()
        values = payload.get("input")
        if not isinstance(values, list):
            raise ValueError("input must be a JSON array")
        with self._lock:
            return {
                "name": name,
                "input": values,
                "output": self.skill_learner.registry.run(name, values),
            }

    def grow_neural_skill(self, payload: dict[str, Any]) -> dict[str, object]:
        """Change neural weights, then gate consolidation on unseen experience."""

        with self._lock:
            rule = str(payload.get("rule", "relative_balance"))
            if rule not in ("relative_balance", "same_sign"):
                raise ValueError("rule must be relative_balance or same_sign")
            seed = int(payload.get("seed", len(self.plasticity) + 701))
            default_name = f"{rule.replace('_', '-')}-{len(self.plasticity) + 1}"
            name = str(payload.get("name", default_name)).strip()
            training = make_reasoning_cases(rule, 128, seed)
            validation = make_reasoning_cases(rule, 96, seed + 1)
            report = self.plasticity.learn(name, training, validation)
            if report.promoted:
                self._save_plasticity()
                transfer = make_reasoning_cases(rule, 256, seed + 2, scale=1.5)
                transfer_accuracy = self.plasticity.evaluate(name, transfer)
                skill = asdict(self.plasticity.get(name))
            else:
                transfer_accuracy = 0.0
                skill = None
            return {
                "rule": rule,
                "report": asdict(report),
                "transfer": {
                    "cases": 256,
                    "different_magnitude": True,
                    "accuracy": transfer_accuracy,
                    "passed": transfer_accuracy >= 0.95,
                },
                "skill": skill,
                "state": self.state(),
            }

    def run_neural_skill(self, payload: dict[str, Any]) -> dict[str, object]:
        name = str(payload.get("name", "")).strip()
        values = payload.get("input")
        if not isinstance(values, list):
            raise ValueError("input must be a JSON array")
        with self._lock:
            probability = self.plasticity.predict_probability(name, values)
            return {
                "name": name,
                "input": values,
                "probability": probability,
                "prediction": int(probability >= 0.5),
            }

    @staticmethod
    def _tool_goal(payload: dict[str, Any], *, prefix: str = "") -> ToolGoal:
        kind = str(payload.get(f"{prefix}kind", payload.get("kind", "store")))
        if kind not in ("store", "update", "delete"):
            raise ValueError("kind must be store, update, or delete")
        key = str(payload.get(f"{prefix}key", f"{prefix or 'training-'}record"))
        value = (
            None
            if kind == "delete"
            else str(payload.get(f"{prefix}value", f"{prefix or 'training-'}value"))
        )
        return ToolGoal(kind, key, value)

    def learn_tool_workflow(self, payload: dict[str, Any]) -> dict[str, object]:
        """Learn a workflow in one opaque world and prove cross-world transfer."""

        with self._lock:
            seed = int(payload.get("seed", len(self.tool_agent.registry) + 101))
            goal = self._tool_goal(payload)
            default_name = f"{goal.kind}-workflow-{len(self.tool_agent.registry) + 1}"
            name = str(payload.get("name", default_name)).strip()
            if not name:
                raise ValueError("tool skill name is required")
            if len(name) > 120:
                raise ValueError("tool skill name must be at most 120 characters")
            initial = (
                {goal.key: "previous-value"}
                if goal.kind in ("update", "delete")
                else {}
            )
            training_world = OpaqueKVWorld(seed, initial)
            report = self.tool_agent.learn(
                name,
                training_world,
                goal,
                validation_cases=make_validation_cases(seed, goal.kind),
            )
            if report.consolidated:
                self._save_tool_skills()

            transfer_goal = self._tool_goal(payload, prefix="transfer_")
            # The stored procedure is specific to one task family. Make the
            # default transfer match it even when only `kind` was supplied.
            if f"transfer_kind" not in payload:
                transfer_goal = ToolGoal(
                    goal.kind,
                    str(payload.get("transfer_key", "unseen-transfer-record")),
                    None
                    if goal.kind == "delete"
                    else str(payload.get("transfer_value", "unseen-transfer-value")),
                )
            transfer_initial = (
                {transfer_goal.key: "different-previous-value"}
                if transfer_goal.kind in ("update", "delete")
                else {}
            )
            transfer_world = OpaqueKVWorld(seed + 10_000, transfer_initial)
            transfer = (
                self.tool_agent.execute_skill(
                    name,
                    transfer_world,
                    transfer_goal,
                )
                if report.consolidated
                else None
            )
            return {
                "learning": asdict(report),
                "transfer": None if transfer is None else asdict(transfer),
                "state": self.state(),
            }

    def run_tool_skill(self, payload: dict[str, Any]) -> dict[str, object]:
        with self._lock:
            name = str(payload.get("name", "")).strip()
            skill = self.tool_agent.registry.get(name)
            goal_payload = dict(payload)
            goal_payload["kind"] = skill.task_kind
            goal = self._tool_goal(goal_payload)
            initial = (
                {goal.key: str(payload.get("previous_value", "previous-value"))}
                if goal.kind in ("update", "delete")
                else {}
            )
            world = OpaqueKVWorld(int(payload.get("seed", 50_001)), initial)
            return {"run": asdict(self.tool_agent.execute_skill(name, world, goal))}

    def state(self) -> dict[str, object]:
        with self._lock:
            episodes = [asdict(item) for item in reversed(self.agent.memory.episodes[-20:])]
            beliefs = [asdict(item) for item in self.agent.beliefs.all()]
            strategies = [asdict(item) for item in self.agent.knowledge()]
            pending = [
                _decision_state(item.decision)
                for _, item in sorted(self.agent._pending.items())
            ]
            skills = [
                {
                    "name": item.name,
                    "domain": item.domain,
                    "program": item.program.expression,
                    "acquired_via": item.acquired_via,
                    "version": item.version,
                    "confidence": item.confidence,
                    "verification_cases": len(item.verification_cases),
                    "components": item.components,
                }
                for item in self.skill_learner.registry.all()
            ]
            tool_skills = [
                {
                    "name": item.name,
                    "task_kind": item.task_kind,
                    "steps": [asdict(step) for step in item.steps],
                    "required_capabilities": item.required_capabilities,
                    "version": item.version,
                    "confidence": item.confidence,
                    "validation_worlds": len(item.validations),
                }
                for item in self.tool_agent.registry.all()
            ]
            plastic_skills = [asdict(item) for item in self.plasticity.all()]
            return {
                "backend": self.backend_name,
                "state_path": str(self.state_path),
                "episodes": episodes,
                "beliefs": beliefs,
                "strategies": strategies,
                "pending_count": len(self.agent._pending),
                "pending_decisions": pending,
                "total_experiences": len(self.agent.memory),
                "skills": skills,
                "skill_count": len(skills),
                "tool_skills": tool_skills,
                "tool_skill_count": len(tool_skills),
                "plastic_skills": plastic_skills,
                "plastic_skill_count": len(plastic_skills),
            }


class PlaygroundHTTPServer(ThreadingHTTPServer):
    daemon_threads = True


def make_handler(service: PlaygroundService):
    """Create an isolated request-handler class bound to one service."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "AthenaPlayground/0.7"

        def _json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
            encoded = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", CONTENT_SECURITY_POLICY)
            self.end_headers()
            self.wfile.write(encoded)

        def _asset(self, name: str, content_type: str) -> None:
            data = resources.files("athena").joinpath("static", name).read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", CONTENT_SECURITY_POLICY)
            self.end_headers()
            self.wfile.write(data)

        def _read_json(self) -> dict[str, Any]:
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                raise ValueError("Content-Length is required")
            length = int(raw_length)
            if length < 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("request body is too large")
            decoded = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(decoded, dict):
                raise ValueError("JSON body must be an object")
            return decoded

        def do_GET(self) -> None:  # noqa: N802
            try:
                if self.path == "/":
                    self._asset("index.html", "text/html; charset=utf-8")
                elif self.path == "/app.css":
                    self._asset("app.css", "text/css; charset=utf-8")
                elif self.path == "/app.js":
                    self._asset("app.js", "text/javascript; charset=utf-8")
                elif self.path == "/api/state":
                    self._json(service.state())
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except Exception as exc:  # noqa: BLE001
                self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_POST(self) -> None:  # noqa: N802
            try:
                payload = self._read_json()
                routes = {
                    "/api/decide": service.decide,
                    "/api/learn": service.learn,
                    "/api/facts": service.learn_fact,
                    "/api/skills/discover": service.discover_skill,
                    "/api/skills/run": service.run_skill,
                    "/api/tools/learn": service.learn_tool_workflow,
                    "/api/tools/run": service.run_tool_skill,
                    "/api/plasticity/learn": service.grow_neural_skill,
                    "/api/plasticity/run": service.run_neural_skill,
                }
                if self.path not in routes:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                    return
                self._json(routes[self.path](payload))
            except (ValueError, TypeError, KeyError, FoundationError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:  # noqa: BLE001
                self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def log_message(self, format: str, *args: object) -> None:
            # Keep the terminal useful without logging request bodies, memories,
            # observations, or credentials.
            print(f"[athena] {self.address_string()} {format % args}")

    return Handler


def make_server(
    service: PlaygroundService,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> PlaygroundHTTPServer:
    return PlaygroundHTTPServer((host, port), make_handler(service))


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _foundation(kind: str, model: str | None):
    if kind == "demo":
        return DemoFoundation()
    if kind == "openai":
        return OpenAIResponsesFoundation.from_env(model=model)
    if kind == "openrouter":
        return OpenRouterChatFoundation.from_env(model=model)
    if os.getenv("OPENROUTER_API_KEY"):
        return OpenRouterChatFoundation.from_env(model=model)
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIResponsesFoundation.from_env(model=model)
    return DemoFoundation()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--state",
        default=str(Path.home() / ".athena" / "playground.npz"),
        help="persistent agent checkpoint",
    )
    parser.add_argument(
        "--foundation",
        choices=("auto", "demo", "openai", "openrouter"),
        default="auto",
    )
    parser.add_argument(
        "--model",
        help="provider model (or OPENROUTER_MODEL / OPENAI_MODEL)",
    )
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="allow binding beyond the local machine",
    )
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("port must be between 0 and 65535")
    if not _is_loopback(args.host) and not args.allow_network:
        parser.error("non-loopback binding requires --allow-network")
    try:
        foundation = _foundation(args.foundation, args.model)
    except ValueError as exc:
        parser.error(str(exc))
    service = PlaygroundService(foundation, args.state)
    server = make_server(service, args.host, args.port)
    address, port = server.server_address[:2]
    url = f"http://{address}:{port}"
    print(f"Athena playground: {url}")
    print(f"Foundation: {service.backend_name}")
    print(f"Learning state: {service.state_path}")
    print("Press Ctrl+C to stop. Learning is checkpointed after every update.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Athena playground.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
