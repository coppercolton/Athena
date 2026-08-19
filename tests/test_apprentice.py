"""Behavioral tests for Athena's persistent repository apprenticeship runtime."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena.apprentice import (  # noqa: E402
    ApprenticeConfig,
    ApprenticeRuntime,
    ApprenticeStore,
    CheckCommand,
    DeveloperTask,
    OpenRouterRepositoryReasoner,
    RepositorySandbox,
    ScriptedRepositoryReasoner,
)
from athena.foundation import FoundationError  # noqa: E402
from athena.tool_learning import ToolDecision  # noqa: E402


def _decision(name, arguments, *, expected=True):
    return ToolDecision(
        "call",
        name,
        arguments,
        f"I predict {name} is the next evidence-producing action.",
        f"The {name} observation should match the task hypothesis.",
        expected,
        0.85,
    )


def _finish():
    return ToolDecision(
        "finish",
        None,
        {},
        "The edit and independent check now satisfy the task.",
        "Every configured verifier should pass and a diff should exist.",
        True,
        0.95,
    )


def _reasoner(old="return 'old'", new="return 'new'"):
    return ScriptedRepositoryReasoner(
        (
            _decision("read_file", {"path": "app.py"}),
            _decision(
                "replace_text",
                {"path": "app.py", "old": old, "new": new},
            ),
            _decision("run_check", {"check_index": 0}),
            _finish(),
        )
    )


def _repository(root: Path, *, value="old") -> Path:
    source = root / "source"
    source.mkdir()
    (source / "app.py").write_text(
        f"def value():\n    return {value!r}\n", encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.email", "athena@example.test"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.name", "Athena Test"], cwd=source, check=True)
    subprocess.run(["git", "add", "app.py"], cwd=source, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=source, check=True)
    return source


def _config(root: Path, *, promotion_successes=2) -> ApprenticeConfig:
    return ApprenticeConfig(
        root / "athena.sqlite3",
        root / "workspaces",
        root / "artifacts",
        command_timeout=20.0,
        promotion_successes=promotion_successes,
    )


def _submit(runtime: ApprenticeRuntime, source: Path, *, kind="change-return"):
    return runtime.submit(
        goal="Make value() return the string new.",
        source_path=source,
        checks=(("python", "-c", "from app import value; assert value() == 'new'"),),
        task_kind=kind,
    )


def test_worker_edits_only_disposable_clone_and_exports_verified_patch():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = _repository(root)
        runtime = ApprenticeRuntime(_config(root), _reasoner())
        task = _submit(runtime, source)
        report = runtime.run_once()

        assert report is not None and report.success
        assert report.task_id == task.id
        assert report.reasoner_steps == 3
        assert report.verification.passed
        assert report.patch_path is not None
        patch = Path(report.patch_path).read_text(encoding="utf-8")
        assert "return 'new'" in patch
        assert "return 'old'" in (source / "app.py").read_text(encoding="utf-8")
        assert not any(Path(runtime.config.workspace_root).iterdir())


def test_prediction_is_recorded_before_observation_and_ledger_survives_restart():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = _repository(root)
        runtime = ApprenticeRuntime(_config(root), _reasoner())
        _submit(runtime, source)
        report = runtime.run_once()
        assert report is not None and report.success
        assert runtime.store.verify_event_chain()

        with sqlite3.connect(runtime.store.path) as db:
            event_types = [
                row[0]
                for row in db.execute("SELECT event_type FROM events ORDER BY sequence")
            ]
        first_prediction = event_types.index("action_predicted")
        first_observation = event_types.index("action_observed")
        assert first_prediction < first_observation
        assert event_types.index("action_predicted", first_prediction + 1) < event_types.index(
            "finish_observed"
        )

        restored = ApprenticeStore(runtime.store.path)
        assert restored.verify_event_chain()
        assert restored.status().succeeded == 1


def test_ledger_detects_tampering():
    with tempfile.TemporaryDirectory() as directory:
        store = ApprenticeStore(Path(directory) / "state.sqlite3")
        store.append_event("first", {"truth": 1})
        store.append_event("second", {"truth": 2})
        assert store.verify_event_chain()
        with sqlite3.connect(store.path) as db:
            db.execute("UPDATE events SET payload_json = '{\"truth\":99}' WHERE sequence = 1")
        assert not store.verify_event_chain()


def test_expired_task_lease_is_recovered_after_worker_restart():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = _repository(root)
        runtime = ApprenticeRuntime(_config(root), _reasoner())
        task = _submit(runtime, source)
        claimed = runtime.store.claim_next("crashed-worker", lease_seconds=60)
        assert claimed is not None and claimed.id == task.id
        with sqlite3.connect(runtime.store.path) as db:
            db.execute("UPDATE tasks SET lease_until = ? WHERE id = ?", (time.time() - 1, task.id))
        recovered = runtime.store.recover_expired_leases()
        assert recovered == (task.id,)
        assert runtime.store.get_task(task.id).status == "queued"


def test_unapproved_verification_program_is_rejected_before_queueing():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = _repository(root)
        runtime = ApprenticeRuntime(_config(root), _reasoner())
        try:
            runtime.submit(
                goal="unsafe",
                source_path=source,
                checks=(("bash", "-c", "echo not-allowed"),),
                task_kind="unsafe",
            )
        except PermissionError as exc:
            assert "allowlist" in str(exc)
        else:
            raise AssertionError("unapproved verifier escaped the command boundary")
        assert runtime.store.status().queued == 0


def test_dirty_source_repository_is_rejected_instead_of_silently_ignored():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = _repository(root)
        (source / "app.py").write_text(
            "def value():\n    return 'uncommitted'\n", encoding="utf-8"
        )
        runtime = ApprenticeRuntime(_config(root), _reasoner())
        try:
            _submit(runtime, source)
        except ValueError as exc:
            assert "clean" in str(exc)
        else:
            raise AssertionError("dirty source changes were silently ignored")
        assert runtime.store.status().queued == 0


def test_failed_task_can_receive_human_teaching_and_retry():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = _repository(root)
        failing = ScriptedRepositoryReasoner((_finish(),))
        runtime = ApprenticeRuntime(_config(root), failing)
        task = _submit(runtime, source, kind="teachable")
        failed = runtime.run_once()
        assert failed is not None and not failed.success

        taught = runtime.store.teach_and_retry(
            task.id,
            "The exact return statement is in app.py; replace old with new.",
        )
        assert taught.status == "queued"
        assert "Human teaching" in taught.goal
        runtime.reasoner = _reasoner()
        recovered = runtime.run_once()
        assert recovered is not None and recovered.success
        assert runtime.store.get_task(task.id).attempt_count == 2


def test_verifier_process_does_not_inherit_provider_secret():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = _repository(root)
        runtime = ApprenticeRuntime(_config(root), _reasoner())
        previous = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = "must-not-reach-repository-code"
        try:
            runtime.submit(
                goal="Make value() return the string new without receiving provider secrets.",
                source_path=source,
                checks=(
                    (
                        "python",
                        "-c",
                        "import os; from app import value; assert value() == 'new'; "
                        "assert 'OPENROUTER_API_KEY' not in os.environ",
                    ),
                ),
                task_kind="secret-isolation",
            )
            report = runtime.run_once()
        finally:
            if previous is None:
                os.environ.pop("OPENROUTER_API_KEY", None)
            else:
                os.environ["OPENROUTER_API_KEY"] = previous
        assert report is not None and report.success


def test_parent_path_and_external_symlink_are_denied():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = _repository(root)
        outside = root / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        (source / "escape.txt").symlink_to(outside)
        subprocess.run(["git", "add", "escape.txt"], cwd=source, check=True)
        subprocess.run(["git", "commit", "-qm", "symlink fixture"], cwd=source, check=True)
        task = DeveloperTask(
            "paths",
            "inspect",
            str(source),
            (CheckCommand(("python", "-c", "assert True")),),
            "paths",
            "queued",
            "now",
            "now",
        )
        with RepositorySandbox(task, _config(root)) as sandbox:
            for path in ("../outside.txt", "escape.txt"):
                try:
                    sandbox.read_file(path)
                except PermissionError:
                    pass
                else:
                    raise AssertionError(f"unsafe path was readable: {path}")
            assert sandbox.search_text("secret", ".")["matches"] == []


def test_repeated_independent_successes_promote_and_reuse_procedure():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = _repository(root)
        config = _config(root)
        runtime = ApprenticeRuntime(config, _reasoner())

        _submit(runtime, source)
        first = runtime.run_once()
        assert first is not None and first.success
        assert runtime.store.status().shadow_procedures == 1
        assert runtime.store.status().active_procedures == 0

        runtime.reasoner = _reasoner()
        _submit(runtime, source)
        second = runtime.run_once()
        assert second is not None and second.success
        assert runtime.store.status().active_procedures == 1

        unused_reasoner = ScriptedRepositoryReasoner(())
        runtime.reasoner = unused_reasoner
        _submit(runtime, source)
        transfer = runtime.run_once()
        assert transfer is not None and transfer.success
        assert transfer.used_procedure is not None
        assert transfer.reasoner_steps == 0
        assert unused_reasoner.calls == 0
        status = runtime.store.status()
        assert status.attempts == 3
        assert status.verified_success_rate == 1.0
        assert status.reasoner_steps == first.reasoner_steps + second.reasoner_steps
        assert status.procedure_reuses == 1


def test_failed_active_procedure_rolls_back_then_reasoner_recovers():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = _repository(root)
        config = _config(root)
        runtime = ApprenticeRuntime(config, _reasoner())
        for _ in range(2):
            _submit(runtime, source)
            report = runtime.run_once()
            assert report is not None and report.success
            runtime.reasoner = _reasoner()

        changed_source = root / "different"
        subprocess.run(["git", "clone", "-q", "--", str(source), str(changed_source)], check=True)
        (changed_source / "app.py").write_text(
            "def value():\n    return 'different'\n", encoding="utf-8"
        )
        subprocess.run(["git", "add", "app.py"], cwd=changed_source, check=True)
        subprocess.run(
            ["git", "-c", "user.email=athena@example.test", "-c", "user.name=Athena Test", "commit", "-qm", "different"],
            cwd=changed_source,
            check=True,
        )
        runtime.reasoner = _reasoner("return 'different'", "return 'new'")
        _submit(runtime, changed_source)
        recovered = runtime.run_once()
        assert recovered is not None and recovered.success
        assert recovered.used_procedure is None
        assert runtime.store.status().active_procedures == 0


def test_failed_replacement_procedure_restores_previous_active_version():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = _repository(root)
        runtime = ApprenticeRuntime(_config(root), _reasoner())
        tasks = [_submit(runtime, source, kind="versioned") for _ in range(5)]
        first_steps = (
            {"tool_name": "replace_text", "arguments": {"path": "a", "old": "x", "new": "y"}},
        )
        second_steps = (
            {"tool_name": "replace_text", "arguments": {"path": "b", "old": "x", "new": "z"}},
        )
        for task in tasks[:2]:
            first = runtime.store.observe_procedure(
                signature="versioned",
                steps=first_steps,
                task_id=task.id,
                passed=True,
                promotion_successes=2,
            )
        assert first is not None and first.status == "active"
        for task in tasks[2:4]:
            second = runtime.store.observe_procedure(
                signature="versioned",
                steps=second_steps,
                task_id=task.id,
                passed=True,
                promotion_successes=2,
            )
        assert second is not None and second.status == "active"
        assert runtime.store.active_procedure("versioned").id == second.id

        runtime.store.record_procedure_failure(second, tasks[4].id)
        restored = runtime.store.active_procedure("versioned")
        assert restored is not None and restored.id == first.id


class CapturingRepositoryTransport:
    def __init__(self, *, function_name="list_files", arguments=None):
        self.calls = []
        self.function_name = function_name
        self.arguments = arguments or {"pattern": "**/*"}

    def __call__(self, url, headers, payload, timeout):
        self.calls.append((url, headers, payload, timeout))
        arguments = {
            **self.arguments,
            "_hypothesis": "Listing files will reveal repository structure.",
            "_expected_observation": "A bounded set of repository-relative files.",
            "_expected_success": True,
            "_confidence": 0.8,
        }
        return {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": self.function_name,
                                    "arguments": json.dumps(arguments),
                                }
                            }
                        ]
                    }
                }
            ]
        }


def test_openrouter_repository_reasoner_uses_strict_tools_without_leaking_key():
    transport = CapturingRepositoryTransport()
    reasoner = OpenRouterRepositoryReasoner(api_key="secret-test-key", transport=transport)
    task = DeveloperTask(
        "model",
        "inspect repository",
        "/tmp/example",
        (CheckCommand(("python", "-c", "assert True")),),
        "inspect",
        "queued",
        "now",
        "now",
    )
    decision = reasoner.next_step(
        task,
        tools=ApprenticeRuntime.tools(),
        trace=(),
        lessons=(),
    )
    assert decision.tool_name == "list_files"
    _, headers, payload, _ = transport.calls[0]
    assert headers["Authorization"] == "Bearer secret-test-key"
    assert "secret-test-key" not in json.dumps(payload)
    assert payload["parallel_tool_calls"] is False
    assert all(
        item["function"]["parameters"]["additionalProperties"] is False
        for item in payload["tools"]
    )


def test_openrouter_repository_reasoner_rejects_unregistered_tool():
    reasoner = OpenRouterRepositoryReasoner(
        api_key="test",
        transport=CapturingRepositoryTransport(function_name="shell"),
    )
    task = DeveloperTask(
        "model",
        "unsafe request",
        "/tmp/example",
        (CheckCommand(("python", "-c", "assert True")),),
        "inspect",
        "queued",
        "now",
        "now",
    )
    try:
        reasoner.next_step(task, tools=ApprenticeRuntime.tools(), trace=(), lessons=())
    except FoundationError as exc:
        assert "unavailable tool" in str(exc)
    else:
        raise AssertionError("unregistered model-selected tool escaped validation")


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
