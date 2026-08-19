"""Command-line interface for Athena's persistent apprenticeship worker."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shlex
import signal
from typing import Sequence

from .apprentice import (
    ApprenticeConfig,
    ApprenticeRuntime,
    OpenRouterRepositoryReasoner,
    ScriptedRepositoryReasoner,
)
from .foundation import OpenRouterChatFoundation


DEFAULT_ROOT = Path.home() / ".athena"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="athena-apprentice",
        description=(
            "Run Athena's persistent, permissioned repository apprenticeship worker."
        ),
    )
    parser.add_argument("--state", default=str(DEFAULT_ROOT / "apprentice.sqlite3"))
    parser.add_argument("--workspace-root", default=str(DEFAULT_ROOT / "workspaces"))
    parser.add_argument("--artifact-root", default=str(DEFAULT_ROOT / "artifacts"))
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--command-timeout", type=float, default=120.0)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Initialize persistent state and print status.")

    submit = subparsers.add_parser("submit", help="Queue one repository task.")
    submit.add_argument("--repo", required=True, help="Path to a local source repository.")
    submit.add_argument("--goal", required=True, help="Observable task goal.")
    submit.add_argument(
        "--kind",
        required=True,
        help="Stable task signature used for procedural transfer.",
    )
    submit.add_argument(
        "--check",
        action="append",
        required=True,
        help="Exact verifier command; repeat for multiple checks.",
    )
    submit.add_argument(
        "--allow-no-change",
        action="store_true",
        help="Allow success without a repository diff.",
    )

    run = subparsers.add_parser("run", help="Process at most one queued task.")
    run.add_argument(
        "--model",
        default=os.getenv("OPENROUTER_MODEL", OpenRouterChatFoundation.DEFAULT_MODEL),
    )

    daemon = subparsers.add_parser("daemon", help="Continuously process queued tasks.")
    daemon.add_argument(
        "--model",
        default=os.getenv("OPENROUTER_MODEL", OpenRouterChatFoundation.DEFAULT_MODEL),
    )
    daemon.add_argument("--poll", type=float, default=2.0)
    daemon.add_argument("--max-tasks", type=int)

    subparsers.add_parser("status", help="Print queue, learning, and ledger status.")
    tasks = subparsers.add_parser("tasks", help="List recent tasks.")
    tasks.add_argument("--limit", type=int, default=20)
    retry = subparsers.add_parser("retry", help="Requeue a failed task with its lessons retained.")
    retry.add_argument("task_id")
    teach = subparsers.add_parser("teach", help="Teach a failed task, then requeue it.")
    teach.add_argument("task_id")
    teach.add_argument("--instruction", required=True)
    return parser


def _config(arguments: argparse.Namespace) -> ApprenticeConfig:
    return ApprenticeConfig(
        database_path=arguments.state,
        workspace_root=arguments.workspace_root,
        artifact_root=arguments.artifact_root,
        max_steps=arguments.max_steps,
        command_timeout=arguments.command_timeout,
    )


def _offline_runtime(config: ApprenticeConfig) -> ApprenticeRuntime:
    return ApprenticeRuntime(config, ScriptedRepositoryReasoner(()))


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = _config(arguments)

    if arguments.command in {"init", "status", "tasks", "submit", "retry", "teach"}:
        runtime = _offline_runtime(config)
        if arguments.command in {"init", "status"}:
            _print(asdict(runtime.store.status()))
            return 0
        if arguments.command == "tasks":
            _print([asdict(item) for item in runtime.store.list_tasks(limit=arguments.limit)])
            return 0
        if arguments.command == "retry":
            _print(asdict(runtime.store.retry(arguments.task_id)))
            return 0
        if arguments.command == "teach":
            _print(
                asdict(
                    runtime.store.teach_and_retry(
                        arguments.task_id, arguments.instruction
                    )
                )
            )
            return 0
        checks = [shlex.split(item) for item in arguments.check]
        if any(not item for item in checks):
            raise SystemExit("each --check must contain a command")
        task = runtime.submit(
            goal=arguments.goal,
            source_path=arguments.repo,
            checks=checks,
            task_kind=arguments.kind,
            require_change=not arguments.allow_no_change,
        )
        _print(asdict(task))
        return 0

    reasoner = OpenRouterRepositoryReasoner(model=arguments.model)
    runtime = ApprenticeRuntime(config, reasoner)
    if arguments.command == "run":
        report = runtime.run_once()
        _print(None if report is None else asdict(report))
        return 0 if report is None or report.success else 1

    def stop(_signum: int, _frame: object) -> None:
        runtime.stop()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    processed = runtime.run_forever(
        poll_interval=arguments.poll,
        max_tasks=arguments.max_tasks,
    )
    _print({"processed": processed, "status": asdict(runtime.store.status())})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
