"""Permissioned tool learning and verified procedural memory for Athena.

The v0.5 skill lab proved active induction inside a fixed symbolic language.
This module moves the same contracts into an agent loop: tools are unfamiliar,
calls have permissions and side effects, every reversible write is snapshotted,
the task is independently verified, and successful experience is compiled into
a parameterized procedure that can bind to differently named tools later.

The bundled ``OpaqueKVWorld`` is a safe virtual environment, not an arbitrary
shell.  It provides a falsifiable bridge toward real tool use while keeping
authorization, rollback, verification, and transfer observable in tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random
from typing import Any, Callable, Literal, Protocol, Sequence


JSONScalar = str | int | float | bool | None
Permission = Literal["read", "reversible_write", "external_write"]
GoalKind = Literal["store", "update", "delete"]


@dataclass(frozen=True)
class ToolParameter:
    name: str
    type: str
    description: str


@dataclass(frozen=True)
class ToolSpec:
    """A callable tool boundary with an explicit authorization class."""

    name: str
    description: str
    permission: Permission
    parameters: tuple[ToolParameter, ...] = ()

    def strict_schema(self, *, predictive_fields: bool = False) -> dict[str, object]:
        properties: dict[str, object] = {
            item.name: {"type": item.type, "description": item.description}
            for item in self.parameters
        }
        if predictive_fields:
            properties.update(
                {
                    "_hypothesis": {
                        "type": "string",
                        "description": "Current falsifiable belief about what this call will do.",
                    },
                    "_expected_observation": {
                        "type": "string",
                        "description": "What should be observed if the hypothesis is correct.",
                    },
                    "_expected_success": {
                        "type": "boolean",
                        "description": "Whether the call is predicted to succeed.",
                    },
                    "_confidence": {
                        "type": "number",
                        "description": "Confidence in the prediction from 0 to 1.",
                    },
                }
            )
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }

    def openai_tool(self) -> dict[str, object]:
        return {
            "type": "function",
            "name": self.name,
            "description": (
                f"{self.description} Permission class: {self.permission}. "
                "Include a falsifiable hypothesis and expected observation before calling."
            ),
            "parameters": self.strict_schema(predictive_fields=True),
            "strict": True,
        }


@dataclass(frozen=True)
class ToolGoal:
    kind: GoalKind
    key: str
    value: str | None = None

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("goal key is required")
        if self.kind in ("store", "update") and self.value is None:
            raise ValueError(f"{self.kind} goal requires a value")

    @property
    def description(self) -> str:
        if self.kind == "delete":
            return f"Delete the record at key {self.key!r} and verify it is absent."
        verb = "Store" if self.kind == "store" else "Update"
        return (
            f"{verb} value {self.value!r} at key {self.key!r} and verify it can "
            "be read back exactly."
        )


@dataclass(frozen=True)
class ToolDecision:
    action: Literal["call", "finish"]
    tool_name: str | None
    arguments: dict[str, JSONScalar]
    hypothesis: str
    expected_observation: str
    expected_success: bool
    confidence: float


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: dict[str, Any]
    state_changed: bool = False
    error: str | None = None


@dataclass(frozen=True)
class ToolExperience:
    index: int
    decision: ToolDecision
    permission: Permission
    result: ToolResult
    prediction_error: float
    rolled_back: bool


@dataclass(frozen=True)
class ToolVerification:
    passed: bool
    details: str


@dataclass(frozen=True)
class ToolRunReport:
    world_id: str
    goal: ToolGoal
    success: bool
    verification: ToolVerification
    trace: tuple[ToolExperience, ...]
    reasoner_steps: int
    denied_calls: int
    rollback_count: int
    used_skill: str | None = None


class ToolEnvironment(Protocol):
    """Environment contract; real integrations can implement this boundary."""

    world_id: str
    documentation_tool: str

    def tools(self) -> Sequence[ToolSpec]: ...

    def invoke(self, name: str, arguments: dict[str, JSONScalar]) -> ToolResult: ...

    def snapshot(self) -> object: ...

    def restore(self, snapshot: object) -> None: ...

    def verify(self, goal: ToolGoal) -> ToolVerification: ...


class ToolReasoner(Protocol):
    """Provider-neutral decision boundary for unfamiliar tool use."""

    name: str

    def next_step(
        self,
        goal: ToolGoal,
        *,
        tools: Sequence[ToolSpec],
        trace: Sequence[ToolExperience],
        known_skills: Sequence["ToolSkill"],
    ) -> ToolDecision: ...


@dataclass(frozen=True)
class ToolPolicy:
    """Authorization and resource limits enforced outside the reasoner."""

    allowed_permissions: tuple[Permission, ...] = ("read", "reversible_write")
    max_calls: int = 12
    rollback_on_failure: bool = True

    def __post_init__(self) -> None:
        if self.max_calls < 1:
            raise ValueError("max_calls must be >= 1")

    def allows(self, permission: Permission) -> bool:
        return permission in self.allowed_permissions


@dataclass(frozen=True)
class ProcedureStep:
    capability: str
    arguments: dict[str, JSONScalar]


@dataclass(frozen=True)
class SkillValidation:
    world_id: str
    goal: ToolGoal
    passed: bool
    calls: int
    reasoner_steps: int


@dataclass(frozen=True)
class ToolSkill:
    """A verified workflow expressed in semantic capabilities, not tool names."""

    name: str
    task_kind: GoalKind
    steps: tuple[ProcedureStep, ...]
    required_capabilities: tuple[str, ...]
    acquired_from: str
    version: int
    confidence: float
    validations: tuple[SkillValidation, ...]


@dataclass(frozen=True)
class ToolLearningReport:
    skill_name: str
    knowledge_gap: str
    acquisition: ToolRunReport
    candidate_steps: tuple[ProcedureStep, ...]
    validations: tuple[SkillValidation, ...]
    consolidated: bool
    reason: str
    skill: ToolSkill | None


class ToolSkillRegistry:
    """Persistent procedures protected by multi-world validation and regression."""

    CHECKPOINT_VERSION = 1

    def __init__(self, *, minimum_validation_worlds: int = 2) -> None:
        if minimum_validation_worlds < 1:
            raise ValueError("minimum_validation_worlds must be >= 1")
        self.minimum_validation_worlds = int(minimum_validation_worlds)
        self._skills: dict[str, ToolSkill] = {}

    def __len__(self) -> int:
        return len(self._skills)

    def all(self) -> tuple[ToolSkill, ...]:
        return tuple(self._skills[name] for name in sorted(self._skills))

    def get(self, name: str) -> ToolSkill:
        try:
            return self._skills[name]
        except KeyError as exc:
            raise KeyError(f"unknown tool skill: {name}") from exc

    def consolidate(
        self,
        *,
        name: str,
        task_kind: GoalKind,
        steps: Sequence[ProcedureStep],
        acquired_from: str,
        validations: Sequence[SkillValidation],
    ) -> tuple[bool, str, ToolSkill | None]:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("skill name is required")
        checks = tuple(validations)
        if len(checks) < self.minimum_validation_worlds:
            return (
                False,
                f"needs at least {self.minimum_validation_worlds} independent worlds",
                self._skills.get(clean_name),
            )
        if not all(item.passed for item in checks):
            return False, "candidate failed held-out world verification", self._skills.get(clean_name)
        candidate_steps = tuple(steps)
        if not candidate_steps:
            return False, "candidate has no executable steps", self._skills.get(clean_name)

        existing = self._skills.get(clean_name)
        if existing is not None and existing.task_kind != task_kind:
            return False, "replacement changes the protected task kind", existing
        if existing is not None and existing.steps != candidate_steps:
            return False, "replacement would overwrite the protected procedure", existing
        required = tuple(dict.fromkeys(step.capability for step in candidate_steps))
        version = 1 if existing is None else existing.version + 1
        skill = ToolSkill(
            name=clean_name,
            task_kind=task_kind,
            steps=candidate_steps,
            required_capabilities=required,
            acquired_from=acquired_from,
            version=version,
            confidence=len(checks) / (len(checks) + 1.0),
            validations=checks,
        )
        self._skills[clean_name] = skill
        return True, "verified across held-out worlds and consolidated", skill

    def to_state(self) -> dict[str, object]:
        return {
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "minimum_validation_worlds": self.minimum_validation_worlds,
            "skills": [asdict(item) for item in self.all()],
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(
            json.dumps(self.to_state(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "ToolSkillRegistry":
        source = Path(path).expanduser().resolve()
        state = json.loads(source.read_text(encoding="utf-8"))
        if state.get("checkpoint_version") != cls.CHECKPOINT_VERSION:
            raise ValueError("unsupported tool-skill checkpoint version")
        registry = cls(
            minimum_validation_worlds=int(state["minimum_validation_worlds"])
        )
        for raw in state.get("skills", []):
            validations = tuple(
                SkillValidation(
                    world_id=str(item["world_id"]),
                    goal=ToolGoal(**item["goal"]),
                    passed=bool(item["passed"]),
                    calls=int(item["calls"]),
                    reasoner_steps=int(item["reasoner_steps"]),
                )
                for item in raw["validations"]
            )
            skill = ToolSkill(
                name=str(raw["name"]),
                task_kind=raw["task_kind"],
                steps=tuple(
                    ProcedureStep(
                        capability=str(item["capability"]),
                        arguments=dict(item["arguments"]),
                    )
                    for item in raw["steps"]
                ),
                required_capabilities=tuple(raw["required_capabilities"]),
                acquired_from=str(raw["acquired_from"]),
                version=int(raw["version"]),
                confidence=float(raw["confidence"]),
                validations=validations,
            )
            registry._skills[skill.name] = skill
        return registry


class OpaqueKVWorld:
    """Safe virtual record store whose operation names change every deployment."""

    _ALIASES = (
        "aegis",
        "brontes",
        "cirrus",
        "daedalus",
        "eidos",
        "flux",
        "gaia",
        "helios",
        "iris",
        "janus",
        "kronos",
        "lyra",
    )
    _CAPABILITIES = ("read_value", "write_value", "delete_value", "list_keys")

    def __init__(
        self,
        seed: int,
        initial: dict[str, str] | None = None,
        *,
        external_write_capability: str | None = None,
    ) -> None:
        self.seed = int(seed)
        self.world_id = f"opaque-kv-{self.seed}"
        self.documentation_tool = "inspect_manual"
        self._data = dict(initial or {})
        names = random.Random(self.seed).sample(self._ALIASES, len(self._CAPABILITIES))
        self._capability_by_tool = dict(zip(names, self._CAPABILITIES))
        self._tool_by_capability = {
            capability: name for name, capability in self._capability_by_tool.items()
        }
        self._external_write_capability = external_write_capability

    @property
    def data(self) -> dict[str, str]:
        return dict(self._data)

    def tools(self) -> tuple[ToolSpec, ...]:
        specs = [
            ToolSpec(
                self.documentation_tool,
                "Read the manual for one unfamiliar tool without changing world state.",
                "read",
                (ToolParameter("tool", "string", "Exact unfamiliar tool name."),),
            )
        ]
        for name in sorted(self._capability_by_tool):
            capability = self._capability_by_tool[name]
            if capability == "write_value":
                parameters = (
                    ToolParameter("slot", "string", "Record slot."),
                    ToolParameter("payload", "string", "Record payload."),
                )
            elif capability in ("read_value", "delete_value"):
                parameters = (ToolParameter("slot", "string", "Record slot."),)
            else:
                parameters = ()
            permission: Permission = (
                "external_write"
                if capability == self._external_write_capability
                else (
                    "reversible_write"
                    if capability in ("write_value", "delete_value")
                    else "read"
                )
            )
            specs.append(
                ToolSpec(
                    name,
                    "Unfamiliar operation. Inspect its manual before use.",
                    permission,
                    parameters,
                )
            )
        return tuple(specs)

    def snapshot(self) -> object:
        return dict(self._data)

    def restore(self, snapshot: object) -> None:
        if not isinstance(snapshot, dict):
            raise TypeError("invalid world snapshot")
        self._data = {str(key): str(value) for key, value in snapshot.items()}

    def _manual(self, tool: str) -> ToolResult:
        capability = self._capability_by_tool.get(tool)
        if capability is None:
            return ToolResult(False, {}, error=f"unknown tool: {tool}")
        usage = {
            "read_value": "Pass slot. Returns found and value.",
            "write_value": "Pass slot and payload. Stores payload at slot.",
            "delete_value": "Pass slot. Removes the record if present.",
            "list_keys": "Takes no arguments. Returns every current slot.",
        }[capability]
        return ToolResult(
            True,
            {"tool": tool, "capability": capability, "usage": usage},
        )

    def invoke(self, name: str, arguments: dict[str, JSONScalar]) -> ToolResult:
        if name == self.documentation_tool:
            return self._manual(str(arguments.get("tool", "")))
        capability = self._capability_by_tool.get(name)
        if capability is None:
            return ToolResult(False, {}, error=f"unknown tool: {name}")
        if capability == "list_keys":
            if arguments:
                return ToolResult(False, {}, error="list operation takes no arguments")
            return ToolResult(True, {"keys": sorted(self._data)})
        slot = arguments.get("slot")
        if not isinstance(slot, str) or not slot:
            return ToolResult(False, {}, error="slot must be a non-empty string")
        if capability == "read_value":
            return ToolResult(
                True,
                {"found": slot in self._data, "value": self._data.get(slot)},
            )
        if capability == "write_value":
            payload = arguments.get("payload")
            if not isinstance(payload, str):
                return ToolResult(False, {}, error="payload must be a string")
            previous = self._data.get(slot)
            self._data[slot] = payload
            return ToolResult(
                True,
                {"stored": True, "previous": previous},
                state_changed=True,
            )
        existed = slot in self._data
        previous = self._data.pop(slot, None)
        return ToolResult(
            True,
            {"deleted": existed, "previous": previous},
            state_changed=existed,
        )

    def verify(self, goal: ToolGoal) -> ToolVerification:
        if goal.kind == "delete":
            passed = goal.key not in self._data
            return ToolVerification(
                passed,
                "record is absent" if passed else "record still exists",
            )
        actual = self._data.get(goal.key)
        passed = actual == goal.value
        return ToolVerification(
            passed,
            "stored value matches exactly"
            if passed
            else f"expected {goal.value!r}, observed {actual!r}",
        )


def _bindings(trace: Sequence[ToolExperience], documentation_tool: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for experience in trace:
        if experience.decision.tool_name != documentation_tool or not experience.result.ok:
            continue
        capability = experience.result.output.get("capability")
        tool = experience.result.output.get("tool")
        if isinstance(capability, str) and isinstance(tool, str):
            result[capability] = tool
    return result


class DemoToolReasoner:
    """Deterministic reasoner that learns only from the manuals it observes."""

    name = "Offline unfamiliar-tool reasoner"

    @staticmethod
    def _required(goal: ToolGoal) -> tuple[str, ...]:
        if goal.kind == "delete":
            return "delete_value", "read_value"
        return "write_value", "read_value"

    def next_step(
        self,
        goal: ToolGoal,
        *,
        tools: Sequence[ToolSpec],
        trace: Sequence[ToolExperience],
        known_skills: Sequence[ToolSkill],
    ) -> ToolDecision:
        del known_skills
        documentation = next(
            (item.name for item in tools if "manual" in item.name),
            "inspect_manual",
        )
        learned = _bindings(trace, documentation)
        required = self._required(goal)
        inspected = {
            str(item.decision.arguments.get("tool"))
            for item in trace
            if item.decision.tool_name == documentation
        }
        if any(capability not in learned for capability in required):
            target = next(
                item.name
                for item in tools
                if item.name != documentation and item.name not in inspected
            )
            return ToolDecision(
                "call",
                documentation,
                {"tool": target},
                "This unfamiliar tool may provide a capability required by the goal.",
                "The manual will identify its capability and required arguments.",
                True,
                0.65,
            )

        capability_by_tool = {tool: capability for capability, tool in learned.items()}
        operational = [
            item
            for item in trace
            if item.decision.tool_name in capability_by_tool and item.result.ok
        ]
        change_capability = "delete_value" if goal.kind == "delete" else "write_value"
        changed = any(
            capability_by_tool.get(item.decision.tool_name) == change_capability
            for item in operational
        )
        if not changed:
            arguments: dict[str, JSONScalar] = {"slot": goal.key}
            if change_capability == "write_value":
                arguments["payload"] = goal.value
            return ToolDecision(
                "call",
                learned[change_capability],
                arguments,
                f"The manual identifies this as {change_capability}.",
                "The requested state change should succeed in the reversible sandbox.",
                True,
                0.92,
            )

        reads_after_change = [
            item
            for item in operational
            if capability_by_tool.get(item.decision.tool_name) == "read_value"
            and item.index > max(
                changed_item.index
                for changed_item in operational
                if capability_by_tool.get(changed_item.decision.tool_name)
                == change_capability
            )
        ]
        if not reads_after_change:
            return ToolDecision(
                "call",
                learned["read_value"],
                {"slot": goal.key},
                "A read-after-write check can test the predicted final state.",
                "The observation should exactly match the goal.",
                True,
                0.95,
            )
        observed = reads_after_change[-1].result.output
        correct = (
            not observed.get("found")
            if goal.kind == "delete"
            else observed.get("found") and observed.get("value") == goal.value
        )
        if correct:
            return ToolDecision(
                "finish",
                None,
                {},
                "The action result and independent read both match the goal.",
                "The environment verifier should pass.",
                True,
                0.99,
            )
        return ToolDecision(
            "finish",
            None,
            {},
            "The observed state contradicts the goal; stop without claiming success.",
            "The environment verifier should fail.",
            False,
            0.99,
        )


def _replace_goal_values(
    value: JSONScalar, goal: ToolGoal
) -> JSONScalar:
    if value == goal.key:
        return "$goal.key"
    if goal.value is not None and value == goal.value:
        return "$goal.value"
    return value


def _render_goal_value(value: JSONScalar, goal: ToolGoal) -> JSONScalar:
    if value == "$goal.key":
        return goal.key
    if value == "$goal.value":
        return goal.value
    return value


class ToolLearningAgent:
    """Solve, verify, compile, retain, and transfer unfamiliar-tool workflows."""

    def __init__(
        self,
        reasoner: ToolReasoner | None = None,
        registry: ToolSkillRegistry | None = None,
        policy: ToolPolicy | None = None,
    ) -> None:
        self.reasoner = reasoner or DemoToolReasoner()
        self.registry = registry if registry is not None else ToolSkillRegistry()
        self.policy = policy or ToolPolicy()

    @staticmethod
    def _tool_map(environment: ToolEnvironment) -> dict[str, ToolSpec]:
        return {item.name: item for item in environment.tools()}

    def _invoke(
        self,
        environment: ToolEnvironment,
        decision: ToolDecision,
        index: int,
    ) -> tuple[ToolExperience, bool]:
        specs = self._tool_map(environment)
        if decision.tool_name not in specs:
            denied = ToolExperience(
                index,
                decision,
                "external_write",
                ToolResult(False, {}, error="tool is not registered"),
                float(decision.expected_success),
                False,
            )
            return denied, True
        spec = specs[decision.tool_name]
        if not self.policy.allows(spec.permission):
            denied = ToolExperience(
                index,
                decision,
                spec.permission,
                ToolResult(False, {}, error=f"permission denied: {spec.permission}"),
                float(decision.expected_success),
                False,
            )
            return denied, True

        snapshot = (
            environment.snapshot()
            if spec.permission in ("reversible_write", "external_write")
            else None
        )
        rolled_back = False
        try:
            result = environment.invoke(decision.tool_name, decision.arguments)
        except Exception as exc:  # noqa: BLE001 - environment boundary
            if snapshot is not None and self.policy.rollback_on_failure:
                environment.restore(snapshot)
                rolled_back = True
            result = ToolResult(
                False,
                {},
                error=f"tool raised {type(exc).__name__}",
            )
        if (
            snapshot is not None
            and not result.ok
            and self.policy.rollback_on_failure
            and not rolled_back
        ):
            environment.restore(snapshot)
            rolled_back = True
        experience = ToolExperience(
            index=index,
            decision=decision,
            permission=spec.permission,
            result=result,
            prediction_error=float(decision.expected_success != result.ok),
            rolled_back=rolled_back,
        )
        return experience, False

    def solve(self, environment: ToolEnvironment, goal: ToolGoal) -> ToolRunReport:
        trace: list[ToolExperience] = []
        denied_calls = 0
        reasoner_steps = 0
        for index in range(1, self.policy.max_calls + 1):
            decision = self.reasoner.next_step(
                goal,
                tools=environment.tools(),
                trace=trace,
                known_skills=self.registry.all(),
            )
            reasoner_steps += 1
            if decision.action == "finish":
                break
            experience, denied = self._invoke(environment, decision, index)
            trace.append(experience)
            denied_calls += int(denied)
        verification = environment.verify(goal)
        return ToolRunReport(
            world_id=environment.world_id,
            goal=goal,
            success=verification.passed,
            verification=verification,
            trace=tuple(trace),
            reasoner_steps=reasoner_steps,
            denied_calls=denied_calls,
            rollback_count=sum(item.rolled_back for item in trace),
        )

    @staticmethod
    def _compile(
        run: ToolRunReport,
        documentation_tool: str,
    ) -> tuple[ProcedureStep, ...]:
        if not run.success:
            return ()
        learned = _bindings(run.trace, documentation_tool)
        capability_by_tool = {tool: capability for capability, tool in learned.items()}
        required = (
            {"delete_value", "read_value"}
            if run.goal.kind == "delete"
            else {"write_value", "read_value"}
        )
        steps: list[ProcedureStep] = []
        for item in run.trace:
            capability = capability_by_tool.get(item.decision.tool_name or "")
            if capability not in required:
                continue
            arguments = {
                key: _replace_goal_values(value, run.goal)
                for key, value in item.decision.arguments.items()
            }
            steps.append(ProcedureStep(capability, arguments))
        return tuple(steps)

    def _discover_bindings(
        self,
        environment: ToolEnvironment,
        capabilities: Sequence[str],
        trace: list[ToolExperience],
    ) -> dict[str, str]:
        specs = self._tool_map(environment)
        candidates = [
            item.name
            for item in environment.tools()
            if item.name != environment.documentation_tool
        ]
        for tool_name in candidates:
            if all(capability in _bindings(trace, environment.documentation_tool) for capability in capabilities):
                break
            decision = ToolDecision(
                "call",
                environment.documentation_tool,
                {"tool": tool_name},
                "This manual may reveal one required semantic capability.",
                "A capability identifier and usage contract should be returned.",
                True,
                0.90,
            )
            if environment.documentation_tool not in specs:
                break
            experience, _ = self._invoke(environment, decision, len(trace) + 1)
            trace.append(experience)
        return _bindings(trace, environment.documentation_tool)

    def execute_skill(
        self,
        skill: ToolSkill | str,
        environment: ToolEnvironment,
        goal: ToolGoal,
    ) -> ToolRunReport:
        learned_skill = self.registry.get(skill) if isinstance(skill, str) else skill
        if learned_skill.task_kind != goal.kind:
            raise ValueError("skill task kind does not match the requested goal")
        trace: list[ToolExperience] = []
        bindings = self._discover_bindings(
            environment,
            learned_skill.required_capabilities,
            trace,
        )
        denied_calls = 0
        for step in learned_skill.steps:
            tool_name = bindings.get(step.capability)
            if tool_name is None:
                break
            arguments = {
                key: _render_goal_value(value, goal)
                for key, value in step.arguments.items()
            }
            decision = ToolDecision(
                "call",
                tool_name,
                arguments,
                f"Retained skill maps this step to capability {step.capability}.",
                "The parameterized procedure should reproduce verified behavior.",
                True,
                learned_skill.confidence,
            )
            experience, denied = self._invoke(environment, decision, len(trace) + 1)
            trace.append(experience)
            denied_calls += int(denied)
            if not experience.result.ok:
                break
        verification = environment.verify(goal)
        return ToolRunReport(
            world_id=environment.world_id,
            goal=goal,
            success=verification.passed,
            verification=verification,
            trace=tuple(trace),
            reasoner_steps=0,
            denied_calls=denied_calls,
            rollback_count=sum(item.rolled_back for item in trace),
            used_skill=learned_skill.name,
        )

    def learn(
        self,
        skill_name: str,
        environment: ToolEnvironment,
        goal: ToolGoal,
        *,
        validation_cases: Sequence[tuple[ToolEnvironment, ToolGoal]],
    ) -> ToolLearningReport:
        acquisition = self.solve(environment, goal)
        steps = self._compile(acquisition, environment.documentation_tool)
        validations: list[SkillValidation] = []
        provisional = ToolSkill(
            name=skill_name,
            task_kind=goal.kind,
            steps=steps,
            required_capabilities=tuple(
                dict.fromkeys(item.capability for item in steps)
            ),
            acquired_from=environment.world_id,
            version=1,
            confidence=0.5,
            validations=(),
        )
        if acquisition.success and steps:
            for validation_environment, validation_goal in validation_cases:
                run = self.execute_skill(
                    provisional,
                    validation_environment,
                    validation_goal,
                )
                validations.append(
                    SkillValidation(
                        world_id=run.world_id,
                        goal=validation_goal,
                        passed=run.success,
                        calls=len(run.trace),
                        reasoner_steps=run.reasoner_steps,
                    )
                )
        if not acquisition.success:
            accepted, reason, skill = False, "training world was not solved", None
        else:
            accepted, reason, skill = self.registry.consolidate(
                name=skill_name,
                task_kind=goal.kind,
                steps=steps,
                acquired_from=environment.world_id,
                validations=validations,
            )
        return ToolLearningReport(
            skill_name=skill_name,
            knowledge_gap=(
                "Tool names and operational semantics are unknown; documentation "
                "must be inspected before acting."
            ),
            acquisition=acquisition,
            candidate_steps=steps,
            validations=tuple(validations),
            consolidated=accepted,
            reason=reason,
            skill=skill,
        )


def make_validation_cases(
    seed: int,
    kind: GoalKind,
    count: int = 2,
) -> tuple[tuple[OpaqueKVWorld, ToolGoal], ...]:
    """Create held-out worlds with new aliases, keys, and values."""

    cases: list[tuple[OpaqueKVWorld, ToolGoal]] = []
    for offset in range(1, count + 1):
        world_seed = seed + 100 * offset
        key = f"held-out-{world_seed}"
        value = f"value-{world_seed}"
        initial = {key: f"old-{world_seed}"} if kind in ("update", "delete") else {}
        cases.append(
            (
                OpaqueKVWorld(world_seed, initial),
                ToolGoal(kind, key, None if kind == "delete" else value),
            )
        )
    return tuple(cases)
