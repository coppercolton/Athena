"""Offline end-to-end demonstration of Athena's live apprenticeship loop."""

from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile

from athena import (
    ApprenticeConfig,
    ApprenticeRuntime,
    ScriptedRepositoryReasoner,
    ToolDecision,
)


def decision(name: str, arguments: dict[str, object]) -> ToolDecision:
    return ToolDecision(
        "call",
        name,
        arguments,
        f"I predict {name} will move the task toward verified success.",
        f"The {name} observation should agree with the task goal.",
        True,
        0.9,
    )


def reasoner() -> ScriptedRepositoryReasoner:
    return ScriptedRepositoryReasoner(
        (
            decision("read_file", {"path": "greeting.py"}),
            decision(
                "replace_text",
                {
                    "path": "greeting.py",
                    "old": "return 'hello'",
                    "new": "return 'hello, Athena'",
                },
            ),
            decision("run_check", {"check_index": 0}),
            ToolDecision(
                "finish",
                None,
                {},
                "The requested behavior now passes its independent check.",
                "The verifier will pass and the patch will contain one edit.",
                True,
                0.98,
            ),
        )
    )


def make_repository(root: Path) -> Path:
    repository = root / "source"
    repository.mkdir()
    (repository / "greeting.py").write_text(
        "def greeting():\n    return 'hello'\n", encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "-c", "user.email=athena@example.test", "-c", "user.name=Athena", "add", "greeting.py"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "-c", "user.email=athena@example.test", "-c", "user.name=Athena", "commit", "-qm", "fixture"],
        cwd=repository,
        check=True,
    )
    return repository


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = make_repository(root)
        config = ApprenticeConfig(
            root / "athena.sqlite3",
            root / "workspaces",
            root / "artifacts",
        )
        runtime = ApprenticeRuntime(config, reasoner())

        for experience in range(1, 4):
            runtime.submit(
                goal="Make greeting() return 'hello, Athena'.",
                source_path=source,
                checks=(
                    (
                        "python",
                        "-c",
                        "from greeting import greeting; assert greeting() == 'hello, Athena'",
                    ),
                ),
                task_kind="personalize-greeting",
            )
            report = runtime.run_once()
            assert report is not None and report.success
            print(
                f"experience {experience}: verified={report.success}, "
                f"foundation steps={report.reasoner_steps}, "
                f"retained procedure={report.used_procedure is not None}"
            )
            if experience < 3:
                runtime.reasoner = reasoner()
            else:
                assert report.reasoner_steps == 0

        status = runtime.store.status()
        print(f"active procedures: {status.active_procedures}")
        print(f"tamper-evident ledger valid: {status.event_chain_valid}")
        print("source repository unchanged:", "return 'hello'" in (source / "greeting.py").read_text())


if __name__ == "__main__":
    main()
