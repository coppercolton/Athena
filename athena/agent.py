"""A foundation-model shell that learns from deployment experience.

``AthenaAgent`` does not fine-tune a language model after every message.  That
would make transient feedback, data poisoning, and catastrophic forgetting
part of the main model.  Instead it keeps broad pretrained intelligence stable
and adds a transparent continual-learning loop around it:

1. retrieve relevant experiences and consolidated facts;
2. ask a foundation model for candidate actions;
3. predict the reward of each action before the outcome is visible;
4. choose, then record the real outcome; and
5. update an online contextual value model with source-weighted evidence.

The foundation-model boundary is a small protocol rather than a provider SDK.
OpenAI, Anthropic, a local model, or a deterministic test double can implement
the same ``propose`` method without making Athena's memory provider-specific.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

from .memory import Belief, BeliefStore, Episode, EpisodicMemory, HashingEncoder


@dataclass
class AgentConfig:
    """Controls the continual-experience layer."""

    feature_dim: int = 64
    memory_capacity: int = 1000
    retrieval_k: int = 5
    candidate_count: int = 3
    prior_strength: float = 5.0
    exploration_bonus: float = 0.08
    # A value just below one preserves a plasticity floor: old experience is
    # retained, but enough contradictory outcomes can still overturn it.
    forgetting: float = 0.995
    ridge: float = 1.0
    minimum_strategy_evidence: float = 6.0
    strategy_threshold: float = 0.60
    confidence_z: float = 1.0
    fact_minimum_support: float = 3.0
    fact_threshold: float = 0.70

    def __post_init__(self) -> None:
        if self.feature_dim < 8:
            raise ValueError("feature_dim must be >= 8")
        if self.memory_capacity < 1:
            raise ValueError("memory_capacity must be >= 1")
        if self.retrieval_k < 0 or self.candidate_count < 1:
            raise ValueError("retrieval_k must be >= 0 and candidate_count >= 1")
        if self.prior_strength <= 0.0:
            raise ValueError("prior_strength must be > 0")
        if self.exploration_bonus < 0.0:
            raise ValueError("exploration_bonus must be >= 0")
        if not 0.0 < self.forgetting <= 1.0:
            raise ValueError("forgetting must be in (0, 1]")
        if self.ridge <= 0.0:
            raise ValueError("ridge must be > 0")
        if self.minimum_strategy_evidence <= 0.0:
            raise ValueError("minimum_strategy_evidence must be > 0")
        if not 0.5 < self.strategy_threshold < 1.0:
            raise ValueError("strategy_threshold must be in (0.5, 1)")
        if self.confidence_z <= 0.0:
            raise ValueError("confidence_z must be > 0")


@dataclass(frozen=True)
class Candidate:
    """One action proposed by broad pretrained knowledge."""

    action: str
    response: str
    prior: float = 0.5


@dataclass(frozen=True)
class StrategyKnowledge:
    """Evidence summary for one action in one deployment context."""

    context_key: str
    action: str
    effective_samples: float
    mean_reward: float
    lower_bound: float
    upper_bound: float
    status: str

    @property
    def consolidated(self) -> bool:
        return self.status != "provisional"


class FoundationModel(Protocol):
    """Provider-neutral interface to a pretrained reasoner."""

    def propose(
        self,
        situation: str,
        *,
        memories: Sequence[Episode],
        facts: Sequence[Belief],
        strategies: Sequence[StrategyKnowledge],
        n: int,
    ) -> Sequence[Candidate]:
        """Return candidate actions without seeing the future outcome."""


class CallableFoundation:
    """Adapt an ordinary callable to :class:`FoundationModel`."""

    def __init__(self, function: Callable[..., Sequence[Candidate]]) -> None:
        self.function = function

    def propose(
        self,
        situation: str,
        *,
        memories: Sequence[Episode],
        facts: Sequence[Belief],
        strategies: Sequence[StrategyKnowledge],
        n: int,
    ) -> Sequence[Candidate]:
        return self.function(
            situation,
            memories=memories,
            facts=facts,
            strategies=strategies,
            n=n,
        )


@dataclass(frozen=True)
class ScoredCandidate:
    """A candidate after pretrained and experiential evidence are fused."""

    candidate: Candidate
    predicted_reward: float
    learned_reward: float
    uncertainty: float
    experience_weight: float
    selection_score: float


@dataclass(frozen=True)
class Decision:
    """A prediction contract created before its outcome exists."""

    id: str
    situation: str
    context_key: str
    selected: Candidate
    predicted_reward: float
    selection_score: float
    alternatives: tuple[ScoredCandidate, ...]
    memory_ids: tuple[str, ...]
    fact_keys: tuple[str, ...]


@dataclass(frozen=True)
class OutcomeReport:
    """Score and learning result for one resolved decision."""

    decision_id: str
    action: str
    predicted_reward: float
    reward: float
    prediction_error: float
    squared_error: float
    reliability: float
    adapted: bool
    knowledge: StrategyKnowledge


@dataclass
class _OutcomeEvidence:
    effective_samples: float = 0.0
    successes: float = 0.0
    failures: float = 0.0

    def update(self, reward: float, weight: float, forgetting: float) -> None:
        self.effective_samples = forgetting * self.effective_samples + weight
        self.successes = forgetting * self.successes + weight * reward
        self.failures = forgetting * self.failures + weight * (1.0 - reward)


class _RecursiveValueModel:
    """Weighted RLS model of reward conditioned on hashed situation text."""

    def __init__(self, dimension: int, forgetting: float, ridge: float) -> None:
        self.dimension = int(dimension)
        self.forgetting = float(forgetting)
        self.ridge = float(ridge)
        self.theta = np.zeros(self.dimension, dtype=float)
        self.covariance = np.eye(self.dimension, dtype=float) / self.ridge
        self.effective_samples = 0.0

    def predict(self, features: np.ndarray) -> tuple[float, float]:
        centered = float(self.theta @ features)
        reward = float(np.clip(0.5 + centered, 0.0, 1.0))
        raw_variance = max(0.0, float(features @ self.covariance @ features))
        uncertainty = math.sqrt(raw_variance / (1.0 + raw_variance))
        return reward, float(np.clip(uncertainty, 0.0, 1.0))

    def update(self, features: np.ndarray, reward: float, weight: float) -> None:
        if weight <= 0.0:
            return
        scale = math.sqrt(weight)
        x = scale * features
        target = scale * (reward - 0.5)
        projected = self.covariance @ x
        denominator = self.forgetting + float(x @ projected)
        gain = projected / max(denominator, 1e-12)
        error = target - float(x @ self.theta)
        self.theta += gain * error
        self.covariance = (
            self.covariance - np.outer(gain, x) @ self.covariance
        ) / self.forgetting
        # Roundoff can make an analytically symmetric covariance slightly
        # asymmetric across long runs.
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self.effective_samples = self.forgetting * self.effective_samples + weight


@dataclass
class _PendingDecision:
    decision: Decision
    features: np.ndarray


class AthenaAgent:
    """Wrap broad pretrained intelligence in a continual experience loop."""

    CHECKPOINT_VERSION = 1

    def __init__(
        self,
        foundation: FoundationModel | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        self.cfg = config or AgentConfig()
        self.foundation = foundation
        self.encoder = HashingEncoder(self.cfg.feature_dim)
        self.memory = EpisodicMemory(self.encoder, self.cfg.memory_capacity)
        self.beliefs = BeliefStore(
            self.encoder,
            minimum_support=self.cfg.fact_minimum_support,
            consolidation_threshold=self.cfg.fact_threshold,
        )
        self._models: dict[str, _RecursiveValueModel] = {}
        self._evidence: dict[tuple[str, str], _OutcomeEvidence] = {}
        self._pending: dict[str, _PendingDecision] = {}
        self._next_decision = 1
        self._next_episode = 1

    def _model(self, action: str) -> _RecursiveValueModel:
        if action not in self._models:
            self._models[action] = _RecursiveValueModel(
                self.cfg.feature_dim,
                self.cfg.forgetting,
                self.cfg.ridge,
            )
        return self._models[action]

    @staticmethod
    def _validate_candidates(candidates: Sequence[Candidate]) -> tuple[Candidate, ...]:
        result = tuple(candidates)
        if not result:
            raise ValueError("at least one candidate is required")
        identities: set[tuple[str, str]] = set()
        for candidate in result:
            if not isinstance(candidate, Candidate):
                raise TypeError("foundation must return Candidate objects")
            if not candidate.action.strip() or not candidate.response.strip():
                raise ValueError("candidate action and response must be non-empty")
            if not math.isfinite(candidate.prior) or not 0.0 <= candidate.prior <= 1.0:
                raise ValueError("candidate prior must be finite and in [0, 1]")
            identity = (candidate.action, candidate.response)
            if identity in identities:
                raise ValueError("duplicate candidate action/response")
            identities.add(identity)
        return result

    def decide(
        self,
        situation: str,
        *,
        context_key: str = "general",
        candidates: Sequence[Candidate] | None = None,
    ) -> Decision:
        """Choose an action and record its outcome prediction before acting."""
        situation = situation.strip()
        context_key = context_key.strip()
        if not situation or not context_key:
            raise ValueError("situation and context_key must be non-empty")

        memories = self.memory.retrieve(
            situation,
            context_key,
            limit=self.cfg.retrieval_k,
        )
        facts = self.beliefs.relevant(
            f"{context_key} {situation}",
            limit=self.cfg.retrieval_k,
        )
        strategies = tuple(
            item
            for item in self.knowledge(context_key=context_key)
            if item.consolidated
        )
        if candidates is None:
            if self.foundation is None:
                raise ValueError("no foundation model or explicit candidates supplied")
            candidates = self.foundation.propose(
                situation,
                memories=memories,
                facts=facts,
                strategies=strategies,
                n=self.cfg.candidate_count,
            )
        proposals = self._validate_candidates(candidates)

        features = self.encoder.encode(f"{context_key} {situation}")
        scored: list[ScoredCandidate] = []
        for candidate in proposals:
            model = self._models.get(candidate.action)
            if model is None:
                learned_reward, uncertainty = 0.5, 1.0
            else:
                learned_reward, uncertainty = model.predict(features)
            local = self._evidence.get((context_key, candidate.action))
            samples = 0.0 if local is None else local.effective_samples
            experience_weight = samples / (samples + self.cfg.prior_strength)
            predicted = (
                (1.0 - experience_weight) * candidate.prior
                + experience_weight * learned_reward
            )
            selection_score = predicted + self.cfg.exploration_bonus * (
                1.0 - experience_weight
            ) * uncertainty
            scored.append(
                ScoredCandidate(
                    candidate=candidate,
                    predicted_reward=float(np.clip(predicted, 0.0, 1.0)),
                    learned_reward=learned_reward,
                    uncertainty=uncertainty,
                    experience_weight=experience_weight,
                    selection_score=selection_score,
                )
            )

        selected_index = max(
            range(len(scored)),
            key=lambda i: (scored[i].selection_score, scored[i].candidate.prior, -i),
        )
        selected = scored[selected_index]
        decision_id = f"athena-{self._next_decision:08d}"
        self._next_decision += 1
        decision = Decision(
            id=decision_id,
            situation=situation,
            context_key=context_key,
            selected=selected.candidate,
            predicted_reward=selected.predicted_reward,
            selection_score=selected.selection_score,
            alternatives=tuple(scored),
            memory_ids=tuple(episode.decision_id for episode in memories),
            fact_keys=tuple(fact.key for fact in facts),
        )
        self._pending[decision_id] = _PendingDecision(decision, features.copy())
        return decision

    def learn(
        self,
        decision_id: str,
        reward: float,
        *,
        observation: str = "",
        reliability: float = 1.0,
        adapt: bool = True,
    ) -> OutcomeReport:
        """Reveal an outcome once, score the prediction, and optionally adapt.

        ``adapt=False`` is the agent-level frozen evaluation contract.  The
        pending prediction is resolved and scored, but models, facts, strategy
        evidence, and episodic memory remain unchanged.
        """
        if decision_id not in self._pending:
            raise KeyError(f"unknown or already resolved decision {decision_id!r}")
        if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
            raise ValueError("reward must be finite and in [0, 1]")
        if not math.isfinite(reliability) or not 0.0 <= reliability <= 1.0:
            raise ValueError("reliability must be finite and in [0, 1]")

        pending = self._pending.pop(decision_id)
        decision = pending.decision
        prediction_error = float(reward - decision.predicted_reward)
        adapted = bool(adapt and reliability > 0.0)
        if adapted:
            action = decision.selected.action
            self._model(action).update(pending.features, reward, reliability)
            evidence = self._evidence.setdefault(
                (decision.context_key, action),
                _OutcomeEvidence(),
            )
            evidence.update(reward, reliability, self.cfg.forgetting)
            self.memory.add(
                Episode(
                    sequence=self._next_episode,
                    decision_id=decision.id,
                    situation=decision.situation,
                    context_key=decision.context_key,
                    action=action,
                    response=decision.selected.response,
                    predicted_reward=decision.predicted_reward,
                    reward=float(reward),
                    reliability=float(reliability),
                    observation=observation,
                )
            )
            self._next_episode += 1

        knowledge = self.strategy(decision.context_key, decision.selected.action)
        return OutcomeReport(
            decision_id=decision.id,
            action=decision.selected.action,
            predicted_reward=decision.predicted_reward,
            reward=float(reward),
            prediction_error=prediction_error,
            squared_error=prediction_error * prediction_error,
            reliability=float(reliability),
            adapted=adapted,
            knowledge=knowledge,
        )

    def strategy(self, context_key: str, action: str) -> StrategyKnowledge:
        evidence = self._evidence.get((context_key, action), _OutcomeEvidence())
        alpha = 1.0 + evidence.successes
        beta = 1.0 + evidence.failures
        mean = alpha / (alpha + beta)
        variance = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1.0))
        radius = self.cfg.confidence_z * math.sqrt(max(0.0, variance))
        lower = max(0.0, mean - radius)
        upper = min(1.0, mean + radius)
        status = "provisional"
        if evidence.effective_samples >= self.cfg.minimum_strategy_evidence:
            if lower >= self.cfg.strategy_threshold:
                status = "preferred"
            elif upper <= 1.0 - self.cfg.strategy_threshold:
                status = "avoid"
        return StrategyKnowledge(
            context_key=context_key,
            action=action,
            effective_samples=evidence.effective_samples,
            mean_reward=mean,
            lower_bound=lower,
            upper_bound=upper,
            status=status,
        )

    def knowledge(
        self,
        *,
        context_key: str | None = None,
    ) -> tuple[StrategyKnowledge, ...]:
        keys = sorted(self._evidence)
        if context_key is not None:
            keys = [key for key in keys if key[0] == context_key]
        return tuple(self.strategy(context, action) for context, action in keys)

    def learn_fact(
        self,
        key: str,
        value: str,
        *,
        source: str,
        reliability: float = 1.0,
    ) -> Belief:
        """Add source-labelled evidence without pretending one claim is truth."""
        return self.beliefs.observe(
            key,
            value,
            source=source,
            reliability=reliability,
        )

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    @staticmethod
    def _candidate_state(candidate: Candidate) -> dict[str, object]:
        return asdict(candidate)

    @classmethod
    def _scored_state(cls, scored: ScoredCandidate) -> dict[str, object]:
        state = asdict(scored)
        state["candidate"] = cls._candidate_state(scored.candidate)
        return state

    @classmethod
    def _decision_state(cls, decision: Decision) -> dict[str, object]:
        return {
            "id": decision.id,
            "situation": decision.situation,
            "context_key": decision.context_key,
            "selected": cls._candidate_state(decision.selected),
            "predicted_reward": decision.predicted_reward,
            "selection_score": decision.selection_score,
            "alternatives": [cls._scored_state(item) for item in decision.alternatives],
            "memory_ids": list(decision.memory_ids),
            "fact_keys": list(decision.fact_keys),
        }

    @staticmethod
    def _decision_from_state(state: dict[str, object]) -> Decision:
        alternatives = []
        for item in state["alternatives"]:
            item = dict(item)
            item["candidate"] = Candidate(**item["candidate"])
            alternatives.append(ScoredCandidate(**item))
        return Decision(
            id=str(state["id"]),
            situation=str(state["situation"]),
            context_key=str(state["context_key"]),
            selected=Candidate(**state["selected"]),
            predicted_reward=float(state["predicted_reward"]),
            selection_score=float(state["selection_score"]),
            alternatives=tuple(alternatives),
            memory_ids=tuple(state["memory_ids"]),
            fact_keys=tuple(state["fact_keys"]),
        )

    def save(self, path: str | Path) -> Path:
        """Checkpoint memories, beliefs, pending predictions, and value models."""
        target = Path(path)
        arrays: dict[str, np.ndarray] = {}
        models = []
        for index, action in enumerate(sorted(self._models)):
            model = self._models[action]
            arrays[f"theta_{index}"] = model.theta
            arrays[f"covariance_{index}"] = model.covariance
            models.append(
                {
                    "action": action,
                    "effective_samples": model.effective_samples,
                }
            )
        metadata = {
            "version": self.CHECKPOINT_VERSION,
            "config": asdict(self.cfg),
            "next_decision": self._next_decision,
            "next_episode": self._next_episode,
            "memory": self.memory.to_state(),
            "beliefs": self.beliefs.to_state(),
            "models": models,
            "evidence": [
                {
                    "context_key": context,
                    "action": action,
                    "effective_samples": item.effective_samples,
                    "successes": item.successes,
                    "failures": item.failures,
                }
                for (context, action), item in sorted(self._evidence.items())
            ],
            "pending": [
                {
                    "decision": self._decision_state(item.decision),
                    "features": item.features.tolist(),
                }
                for _, item in sorted(self._pending.items())
            ],
        }
        with target.open("wb") as handle:
            np.savez_compressed(
                handle,
                metadata=np.asarray(json.dumps(metadata)),
                **arrays,
            )
        return target

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        foundation: FoundationModel | None = None,
    ) -> "AthenaAgent":
        """Restore learned experience while injecting a runtime model adapter."""
        with np.load(Path(path), allow_pickle=False) as checkpoint:
            metadata = json.loads(str(checkpoint["metadata"].item()))
            version = int(metadata.get("version", -1))
            if version != cls.CHECKPOINT_VERSION:
                raise ValueError(
                    f"unsupported AthenaAgent checkpoint version {version}; "
                    f"expected {cls.CHECKPOINT_VERSION}"
                )
            agent = cls(foundation=foundation, config=AgentConfig(**metadata["config"]))
            agent._next_decision = int(metadata["next_decision"])
            agent._next_episode = int(metadata["next_episode"])
            agent.memory = EpisodicMemory.from_state(
                agent.encoder,
                agent.cfg.memory_capacity,
                metadata["memory"],
            )
            agent.beliefs = BeliefStore.from_state(agent.encoder, metadata["beliefs"])
            for index, state in enumerate(metadata["models"]):
                model = _RecursiveValueModel(
                    agent.cfg.feature_dim,
                    agent.cfg.forgetting,
                    agent.cfg.ridge,
                )
                model.theta = checkpoint[f"theta_{index}"].copy()
                model.covariance = checkpoint[f"covariance_{index}"].copy()
                model.effective_samples = float(state["effective_samples"])
                agent._models[str(state["action"])] = model
            for state in metadata["evidence"]:
                agent._evidence[(state["context_key"], state["action"])] = _OutcomeEvidence(
                    effective_samples=float(state["effective_samples"]),
                    successes=float(state["successes"]),
                    failures=float(state["failures"]),
                )
            for state in metadata["pending"]:
                decision = cls._decision_from_state(state["decision"])
                agent._pending[decision.id] = _PendingDecision(
                    decision,
                    np.asarray(state["features"], dtype=float),
                )
            return agent
