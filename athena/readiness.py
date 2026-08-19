"""Fail-closed AGI readiness audit for Athena.

The audit deliberately has no percentage score.  A percentage would make a
small synthetic success look commensurable with open-world generality.  Every
required gate must instead have broad, adversarial evidence before `agi_ready`
can become true.  Narrow laboratory evidence remains explicitly narrow.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class EvidenceLevel(str, Enum):
    NOT_DEMONSTRATED = "not demonstrated"
    NARROW = "narrow laboratory evidence"
    DEMONSTRATED = "broadly demonstrated"


@dataclass(frozen=True)
class ReadinessGate:
    key: str
    question: str
    evidence: EvidenceLevel
    current_evidence: str


@dataclass(frozen=True)
class AGIReadinessReport:
    gates: tuple[ReadinessGate, ...]

    @property
    def agi_ready(self) -> bool:
        return bool(self.gates) and all(
            gate.evidence is EvidenceLevel.DEMONSTRATED for gate in self.gates
        )

    @property
    def blockers(self) -> tuple[ReadinessGate, ...]:
        return tuple(
            gate
            for gate in self.gates
            if gate.evidence is not EvidenceLevel.DEMONSTRATED
        )

    def counts(self) -> dict[str, int]:
        return {
            level.value: sum(gate.evidence is level for gate in self.gates)
            for level in EvidenceLevel
        }


_REQUIRED_GATES: tuple[tuple[str, str], ...] = (
    (
        "continual_learning",
        "Can it keep learning after deployment without catastrophic forgetting?",
    ),
    (
        "novel_problem_learning",
        "Can it identify a genuine knowledge gap and learn an unseen task?",
    ),
    (
        "representation_transfer",
        "Can it learn useful representations and transfer them across tasks?",
    ),
    (
        "cross_domain_generality",
        "Can the same system solve novel problems across many unrelated domains?",
    ),
    (
        "autonomous_experimentation",
        "Can it design safe experiments and learn from real consequences?",
    ),
    (
        "long_horizon_planning",
        "Can it pursue changing goals over long horizons and recover from failure?",
    ),
    (
        "multimodal_grounding",
        "Can it ground language, vision, action, causality, and common sense together?",
    ),
    (
        "general_reasoning_improvement",
        "Can experience improve general reasoning rather than one task-specific head?",
    ),
    (
        "retention_at_scale",
        "Can it accumulate thousands of abilities for long periods without interference?",
    ),
    (
        "real_world_reliability",
        "Is it robust, corrigible, resource-bounded, and safe in open environments?",
    ),
)


_CURRENT_EVIDENCE: dict[str, tuple[EvidenceLevel, str]] = {
    "continual_learning": (
        EvidenceLevel.NARROW,
        "online learning, replay, checkpoints, protected neural updates, and persistent verified procedures",
    ),
    "novel_problem_learning": (
        EvidenceLevel.NARROW,
        "active induction, unfamiliar-tool learning, and model-guided disposable repository tasks",
    ),
    "representation_transfer": (
        EvidenceLevel.NARROW,
        "raw-grid latent learning and two reusable spatial reasoning operators",
    ),
    "cross_domain_generality": (
        EvidenceLevel.NOT_DEMONSTRATED,
        "no independent broad cross-domain evaluation",
    ),
    "autonomous_experimentation": (
        EvidenceLevel.NARROW,
        "bounded experiments and repository checks exist behind local permission policies",
    ),
    "long_horizon_planning": (
        EvidenceLevel.NOT_DEMONSTRATED,
        "no long-duration changing-goal benchmark",
    ),
    "multimodal_grounding": (
        EvidenceLevel.NOT_DEMONSTRATED,
        "numeric streams and synthetic grids are not general multimodal grounding",
    ),
    "general_reasoning_improvement": (
        EvidenceLevel.NOT_DEMONSTRATED,
        "task-specific modules improve; the hosted foundation does not",
    ),
    "retention_at_scale": (
        EvidenceLevel.NOT_DEMONSTRATED,
        "retention is tested over a small number of abilities and short deployments",
    ),
    "real_world_reliability": (
        EvidenceLevel.NOT_DEMONSTRATED,
        "review-only repository patches exist, but hostile-code containment and open-world safety are unevaluated",
    ),
}


def assess_agi_readiness(
    evidence: Mapping[str, tuple[EvidenceLevel, str]] | None = None,
) -> AGIReadinessReport:
    """Assess every mandatory gate; missing evidence fails closed."""

    supplied = _CURRENT_EVIDENCE if evidence is None else evidence
    gates = []
    for key, question in _REQUIRED_GATES:
        level, detail = supplied.get(
            key,
            (EvidenceLevel.NOT_DEMONSTRATED, "no evidence supplied"),
        )
        if not isinstance(level, EvidenceLevel):
            raise TypeError(f"invalid evidence level for {key}")
        gates.append(ReadinessGate(key, question, level, str(detail)))
    return AGIReadinessReport(tuple(gates))
