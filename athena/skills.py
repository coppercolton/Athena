"""Verified procedural skill acquisition for Athena.

This module is intentionally smaller than the long-term research claim.  It
provides a concrete, falsifiable continual-learning environment in which Athena
can encounter an unknown sequence transformation, expose its uncertainty,
choose informative experiments, induce an executable program, verify it on
held-out cases, and retain the resulting skill.

The task language is a constrained DSL, not open-ended human reasoning.  That
constraint is useful: it lets tests distinguish actual rule induction from
remembering prose or trusting a model's self-assessment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from typing import Callable, Iterable, Sequence


Token = str | int | float | bool | None
TokenSequence = tuple[Token, ...]
Oracle = Callable[[TokenSequence], Sequence[Token]]


def _tokens(value: Sequence[Token]) -> TokenSequence:
    if isinstance(value, (str, bytes)):
        raise TypeError("skill input must be a sequence of tokens, not text")
    return tuple(value)


def _sort_key(value: Token) -> tuple[str, str]:
    """Give heterogeneous JSON scalars a deterministic total ordering."""

    return type(value).__name__, repr(value)


def _identity(values: TokenSequence) -> TokenSequence:
    return values


def _reverse(values: TokenSequence) -> TokenSequence:
    return values[::-1]


def _rotate_left(values: TokenSequence) -> TokenSequence:
    return values[1:] + values[:1] if values else values


def _rotate_right(values: TokenSequence) -> TokenSequence:
    return values[-1:] + values[:-1] if values else values


def _sort_asc(values: TokenSequence) -> TokenSequence:
    return tuple(sorted(values, key=_sort_key))


def _sort_desc(values: TokenSequence) -> TokenSequence:
    return tuple(sorted(values, key=_sort_key, reverse=True))


def _unique(values: TokenSequence) -> TokenSequence:
    result: list[Token] = []
    for value in values:
        if value not in result:
            result.append(value)
    return tuple(result)


def _take_even(values: TokenSequence) -> TokenSequence:
    return values[::2]


def _take_odd(values: TokenSequence) -> TokenSequence:
    return values[1::2]


def _duplicate_each(values: TokenSequence) -> TokenSequence:
    return tuple(value for item in values for value in (item, item))


def _swap_pairs(values: TokenSequence) -> TokenSequence:
    result = list(values)
    for index in range(0, len(result) - 1, 2):
        result[index], result[index + 1] = result[index + 1], result[index]
    return tuple(result)


PRIMITIVES: dict[str, Callable[[TokenSequence], TokenSequence]] = {
    "identity": _identity,
    "reverse": _reverse,
    "rotate_left": _rotate_left,
    "rotate_right": _rotate_right,
    "sort_asc": _sort_asc,
    "sort_desc": _sort_desc,
    "unique": _unique,
    "take_even": _take_even,
    "take_odd": _take_odd,
    "duplicate_each": _duplicate_each,
    "swap_pairs": _swap_pairs,
}


DEFAULT_PROBES: tuple[TokenSequence, ...] = (
    ("D", "B", "A", "C"),
    ("A", "A", "C", "B"),
    ("E", "B", "D", "A", "C"),
    ("C", "A", "B"),
    ("B", "D", "A", "D", "C", "A"),
    ("A", "B"),
    ("Z",),
    (),
)


DEFAULT_VERIFICATION_INPUTS: tuple[TokenSequence, ...] = (
    ("H", "F", "G", "E"),
    ("K", "J", "K", "I", "L"),
    ("Q", "S", "P", "R", "T", "U"),
    (8, 3, 9, 1, 4),
    (True, False, True, False),
    ("new", "tokens", "never", "queried"),
)


@dataclass(frozen=True)
class Program:
    """An inspectable executable procedure in Athena's first skill DSL."""

    steps: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("a program needs at least one step")
        unknown = [step for step in self.steps if step not in PRIMITIVES]
        if unknown:
            raise ValueError(f"unknown primitive(s): {', '.join(unknown)}")

    @property
    def expression(self) -> str:
        return " -> ".join(self.steps)

    def apply(self, values: Sequence[Token]) -> TokenSequence:
        result = _tokens(values)
        for step in self.steps:
            result = PRIMITIVES[step](result)
        return result


class ProgramCatalog:
    """Finite hypothesis language used by the active rule learner."""

    def __init__(
        self,
        programs: Iterable[Program] | None = None,
        *,
        max_depth: int = 2,
        diagnostic_inputs: Sequence[Sequence[Token]] = DEFAULT_PROBES,
    ) -> None:
        if not 1 <= max_depth <= 2:
            raise ValueError("this catalog supports max_depth 1 or 2")
        if programs is None:
            primitive_names = tuple(name for name in PRIMITIVES if name != "identity")
            generated = [Program(("identity",))]
            generated.extend(Program((name,)) for name in primitive_names)
            if max_depth >= 2:
                generated.extend(
                    Program((first, second))
                    for first in primitive_names
                    for second in primitive_names
                )
            programs = generated

        # Remove programs that are observationally equivalent over a diverse
        # diagnostic basis.  Without this, no amount of active querying could
        # distinguish, for example, identity from reverse -> reverse.
        unique: dict[tuple[TokenSequence, ...], Program] = {}
        for program in programs:
            signature = tuple(program.apply(item) for item in diagnostic_inputs)
            existing = unique.get(signature)
            if existing is None or (len(program.steps), program.expression) < (
                len(existing.steps),
                existing.expression,
            ):
                unique[signature] = program
        self.programs = tuple(
            sorted(unique.values(), key=lambda item: (len(item.steps), item.expression))
        )
        if not self.programs:
            raise ValueError("catalog cannot be empty")

    def find(self, expression: str) -> Program:
        for program in self.programs:
            if program.expression == expression:
                return program
        raise KeyError(expression)


@dataclass(frozen=True)
class Example:
    input: TokenSequence
    output: TokenSequence


@dataclass(frozen=True)
class Experiment:
    input: TokenSequence
    observation: TokenSequence
    hypotheses_before: int
    hypotheses_after: int
    information_gain_bits: float


@dataclass(frozen=True)
class KnowledgeGap:
    task: str
    status: str
    hypotheses_remaining: int
    confidence: float
    missing_information: str
    next_experiment: TokenSequence | None


@dataclass(frozen=True)
class VerificationReport:
    passed: bool
    passed_cases: int
    total_cases: int
    failures: tuple[Example, ...]
    regression_passed: bool


@dataclass(frozen=True)
class Skill:
    """A durable, executable capability rather than a text memory."""

    name: str
    domain: str
    program: Program
    acquired_via: str
    version: int
    confidence: float
    verification_cases: tuple[Example, ...]
    components: tuple[str, ...] = ()

    def run(self, values: Sequence[Token]) -> TokenSequence:
        return self.program.apply(values)


@dataclass(frozen=True)
class ConsolidationReport:
    accepted: bool
    reason: str
    skill: Skill | None
    verification: VerificationReport


@dataclass(frozen=True)
class LearningReport:
    skill_name: str
    acquired_via: str
    gap_before: KnowledgeGap
    gap_after: KnowledgeGap
    experiments: tuple[Experiment, ...]
    candidate_program: Program | None
    consolidation: ConsolidationReport


def _example(value: Sequence[Token], output: Sequence[Token]) -> Example:
    return Example(_tokens(value), _tokens(output))


class SkillRegistry:
    """Versioned skill memory guarded by verification and regression cases."""

    CHECKPOINT_VERSION = 1

    def __init__(self, *, minimum_verification_cases: int = 4) -> None:
        if minimum_verification_cases < 1:
            raise ValueError("minimum_verification_cases must be >= 1")
        self.minimum_verification_cases = int(minimum_verification_cases)
        self._skills: dict[str, Skill] = {}

    def __len__(self) -> int:
        return len(self._skills)

    def all(self) -> tuple[Skill, ...]:
        return tuple(self._skills[name] for name in sorted(self._skills))

    def get(self, name: str) -> Skill:
        try:
            return self._skills[name]
        except KeyError as exc:
            raise KeyError(f"unknown skill: {name}") from exc

    def run(self, name: str, values: Sequence[Token]) -> TokenSequence:
        return self.get(name).run(values)

    @staticmethod
    def _evaluate(program: Program, cases: Sequence[Example]) -> tuple[Example, ...]:
        return tuple(case for case in cases if program.apply(case.input) != case.output)

    def consolidate(
        self,
        *,
        name: str,
        domain: str,
        program: Program,
        acquired_via: str,
        verification_cases: Sequence[Example],
        components: Sequence[str] = (),
    ) -> ConsolidationReport:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("skill name is required")
        cases = tuple(verification_cases)
        existing = self._skills.get(clean_name)
        regression_cases = () if existing is None else existing.verification_cases
        failures = self._evaluate(program, cases)
        regression_failures = self._evaluate(program, regression_cases)
        enough = len(cases) >= self.minimum_verification_cases
        passed = enough and not failures and not regression_failures
        verification = VerificationReport(
            passed=passed,
            passed_cases=len(cases) - len(failures),
            total_cases=len(cases),
            failures=failures,
            regression_passed=not regression_failures,
        )
        if not enough:
            return ConsolidationReport(
                False,
                f"needs at least {self.minimum_verification_cases} held-out cases",
                existing,
                verification,
            )
        if failures:
            return ConsolidationReport(
                False,
                "candidate failed independent verification",
                existing,
                verification,
            )
        if regression_failures:
            return ConsolidationReport(
                False,
                "candidate would break the previous skill version",
                existing,
                verification,
            )

        version = 1 if existing is None else existing.version + 1
        skill = Skill(
            name=clean_name,
            domain=domain,
            program=program,
            acquired_via=acquired_via,
            version=version,
            confidence=len(cases) / (len(cases) + 1.0),
            verification_cases=cases,
            components=tuple(components),
        )
        self._skills[clean_name] = skill
        return ConsolidationReport(True, "verified and consolidated", skill, verification)

    def compose(
        self,
        name: str,
        component_names: Sequence[str],
        *,
        verifier: Oracle,
        verification_inputs: Sequence[Sequence[Token]] = DEFAULT_VERIFICATION_INPUTS,
    ) -> ConsolidationReport:
        if len(component_names) < 2:
            raise ValueError("skill composition needs at least two components")
        components = tuple(self.get(item) for item in component_names)
        steps = tuple(step for skill in components for step in skill.program.steps)
        program = Program(steps)
        cases = tuple(
            _example(item, verifier(_tokens(item))) for item in verification_inputs
        )
        return self.consolidate(
            name=name,
            domain="composed-sequence-transformation",
            program=program,
            acquired_via="skill-composition",
            verification_cases=cases,
            components=component_names,
        )

    def to_state(self) -> dict[str, object]:
        return {
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "minimum_verification_cases": self.minimum_verification_cases,
            "skills": [asdict(skill) for skill in self.all()],
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
    def load(cls, path: str | Path) -> "SkillRegistry":
        source = Path(path).expanduser().resolve()
        state = json.loads(source.read_text(encoding="utf-8"))
        if state.get("checkpoint_version") != cls.CHECKPOINT_VERSION:
            raise ValueError("unsupported skill checkpoint version")
        registry = cls(
            minimum_verification_cases=int(state["minimum_verification_cases"])
        )
        for item in state.get("skills", []):
            program = Program(tuple(item["program"]["steps"]))
            cases = tuple(
                Example(tuple(case["input"]), tuple(case["output"]))
                for case in item["verification_cases"]
            )
            skill = Skill(
                name=str(item["name"]),
                domain=str(item["domain"]),
                program=program,
                acquired_via=str(item["acquired_via"]),
                version=int(item["version"]),
                confidence=float(item["confidence"]),
                verification_cases=cases,
                components=tuple(item.get("components", ())),
            )
            registry._skills[skill.name] = skill
        return registry


class NovelTaskLearner:
    """Active program induction with an explicit epistemic state."""

    def __init__(
        self,
        registry: SkillRegistry | None = None,
        catalog: ProgramCatalog | None = None,
    ) -> None:
        self.registry = registry if registry is not None else SkillRegistry()
        self.catalog = catalog or ProgramCatalog()

    @staticmethod
    def _partition(
        programs: Sequence[Program], probe: TokenSequence
    ) -> dict[TokenSequence, list[Program]]:
        groups: dict[TokenSequence, list[Program]] = {}
        for program in programs:
            groups.setdefault(program.apply(probe), []).append(program)
        return groups

    def _best_probe(
        self,
        programs: Sequence[Program],
        probes: Sequence[TokenSequence],
    ) -> TokenSequence | None:
        if len(programs) <= 1 or not probes:
            return None

        def score(probe: TokenSequence) -> tuple[float, int, int]:
            groups = self._partition(programs, probe)
            total = len(programs)
            entropy = -sum(
                (len(group) / total) * math.log2(len(group) / total)
                for group in groups.values()
            )
            largest = max(len(group) for group in groups.values())
            return entropy, -largest, len(groups)

        return max(probes, key=score)

    def gap(
        self,
        task: str,
        programs: Sequence[Program] | None = None,
        probes: Sequence[TokenSequence] = DEFAULT_PROBES,
    ) -> KnowledgeGap:
        candidates = tuple(programs or self.catalog.programs)
        remaining = len(candidates)
        resolved = remaining == 1
        return KnowledgeGap(
            task=task,
            status="resolved" if resolved else "unresolved",
            hypotheses_remaining=remaining,
            confidence=1.0 if resolved else 1.0 / max(remaining, 1),
            missing_information=(
                "none; one executable hypothesis remains"
                if resolved
                else "the transformation rule is underdetermined"
            ),
            next_experiment=self._best_probe(candidates, tuple(probes)),
        )

    def learn_by_experiment(
        self,
        skill_name: str,
        oracle: Oracle,
        *,
        probes: Sequence[Sequence[Token]] = DEFAULT_PROBES,
        verification_inputs: Sequence[Sequence[Token]] = DEFAULT_VERIFICATION_INPUTS,
        maximum_experiments: int = 8,
    ) -> LearningReport:
        if maximum_experiments < 1:
            raise ValueError("maximum_experiments must be >= 1")
        available = tuple(_tokens(item) for item in probes)
        candidates = tuple(self.catalog.programs)
        before = self.gap(skill_name, candidates, available)
        experiments: list[Experiment] = []

        for _ in range(maximum_experiments):
            probe = self._best_probe(candidates, available)
            if probe is None:
                break
            available = tuple(item for item in available if item != probe)
            observation = _tokens(oracle(probe))
            prior_count = len(candidates)
            candidates = tuple(
                program
                for program in candidates
                if program.apply(probe) == observation
            )
            if not candidates:
                raise ValueError(
                    "observation contradicts every hypothesis in the current skill language"
                )
            experiments.append(
                Experiment(
                    input=probe,
                    observation=observation,
                    hypotheses_before=prior_count,
                    hypotheses_after=len(candidates),
                    information_gain_bits=math.log2(prior_count / len(candidates)),
                )
            )
            if len(candidates) == 1:
                break

        after = self.gap(skill_name, candidates, available)
        candidate = candidates[0] if len(candidates) == 1 else None
        if candidate is None:
            empty_verification = VerificationReport(False, 0, 0, (), True)
            consolidation = ConsolidationReport(
                False,
                "skill remains underdetermined; more informative experience is required",
                None,
                empty_verification,
            )
        else:
            cases = tuple(
                _example(item, oracle(_tokens(item))) for item in verification_inputs
            )
            consolidation = self.registry.consolidate(
                name=skill_name,
                domain="sequence-transformation",
                program=candidate,
                acquired_via="active-experimentation",
                verification_cases=cases,
            )

        return LearningReport(
            skill_name=skill_name,
            acquired_via="active-experimentation",
            gap_before=before,
            gap_after=after,
            experiments=tuple(experiments),
            candidate_program=candidate,
            consolidation=consolidation,
        )

    def learn_from_instruction(
        self,
        skill_name: str,
        steps: Sequence[str],
        *,
        verifier: Oracle,
        verification_inputs: Sequence[Sequence[Token]] = DEFAULT_VERIFICATION_INPUTS,
    ) -> LearningReport:
        program = Program(tuple(steps))
        before = self.gap(skill_name)
        cases = tuple(
            _example(item, verifier(_tokens(item))) for item in verification_inputs
        )
        consolidation = self.registry.consolidate(
            name=skill_name,
            domain="sequence-transformation",
            program=program,
            acquired_via="instruction",
            verification_cases=cases,
        )
        after = KnowledgeGap(
            task=skill_name,
            status="resolved" if consolidation.accepted else "unresolved",
            hypotheses_remaining=1 if consolidation.accepted else len(self.catalog.programs),
            confidence=1.0 if consolidation.accepted else 0.0,
            missing_information=(
                "none; instruction passed independent verification"
                if consolidation.accepted
                else consolidation.reason
            ),
            next_experiment=None,
        )
        return LearningReport(
            skill_name=skill_name,
            acquired_via="instruction",
            gap_before=before,
            gap_after=after,
            experiments=(),
            candidate_program=program,
            consolidation=consolidation,
        )
