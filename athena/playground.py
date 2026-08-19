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
import threading
from typing import Any
import webbrowser

from .agent import AgentConfig, AthenaAgent, Decision, FoundationModel
from .foundation import DemoFoundation, FoundationError, OpenAIResponsesFoundation


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
    ) -> None:
        self.foundation = foundation
        self.state_path = Path(state_path).expanduser().resolve()
        self._lock = threading.RLock()
        if self.state_path.exists():
            self.agent = AthenaAgent.load(self.state_path, foundation=foundation)
        else:
            self.agent = AthenaAgent(foundation=foundation, config=config)

    @property
    def backend_name(self) -> str:
        return str(getattr(self.foundation, "name", type(self.foundation).__name__))

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_name(f".{self.state_path.name}.tmp")
        self.agent.save(temporary)
        os.replace(temporary, self.state_path)

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

    def state(self) -> dict[str, object]:
        with self._lock:
            episodes = [asdict(item) for item in reversed(self.agent.memory.episodes[-20:])]
            beliefs = [asdict(item) for item in self.agent.beliefs.all()]
            strategies = [asdict(item) for item in self.agent.knowledge()]
            pending = [
                _decision_state(item.decision)
                for _, item in sorted(self.agent._pending.items())
            ]
            return {
                "backend": self.backend_name,
                "state_path": str(self.state_path),
                "episodes": episodes,
                "beliefs": beliefs,
                "strategies": strategies,
                "pending_count": len(self.agent._pending),
                "pending_decisions": pending,
                "total_experiences": len(self.agent.memory),
            }


class PlaygroundHTTPServer(ThreadingHTTPServer):
    daemon_threads = True


def make_handler(service: PlaygroundService):
    """Create an isolated request-handler class bound to one service."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "AthenaPlayground/0.4"

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
        choices=("auto", "demo", "openai"),
        default="auto",
    )
    parser.add_argument("--model", help="OpenAI model (or OPENAI_MODEL)")
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
