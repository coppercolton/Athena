"""Protected neural plasticity for Athena.

This module is the first Athena subsystem whose neural parameters improve after
deployment. It deliberately uses small NumPy networks rather than pretending to
fine-tune a hosted foundation model. Experiences become supervised prediction
examples, candidate weights train in isolation, and an independent promotion
gate checks unseen cases plus protected replay before the retained system can
change.

The scope is intentionally falsifiable: fixed-width binary reasoning operators,
not open-ended AGI. The architecture establishes the mechanisms a broader
lifelong learner needs—plastic weights, modular recruitment, replay,
consolidation, regression protection, rollback, transfer tests, and checkpoints.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Literal, Sequence

import numpy as np


ReasoningRule = Literal["relative_balance", "same_sign"]


@dataclass(frozen=True)
class NeuralExample:
    """One outcome-labelled experience for a neural reasoning operator."""

    inputs: tuple[float, ...]
    target: int

    def __post_init__(self) -> None:
        if not self.inputs:
            raise ValueError("neural example inputs cannot be empty")
        if any(not math.isfinite(float(item)) for item in self.inputs):
            raise ValueError("neural example inputs must be finite")
        if self.target not in (0, 1):
            raise ValueError("neural example target must be 0 or 1")


@dataclass(frozen=True)
class PlasticityConfig:
    """Controls bounded neural adaptation and its promotion gate."""

    input_dim: int = 4
    hidden_dim: int = 16
    learning_rate: float = 0.025
    epochs: int = 700
    l2: float = 1e-4
    replay_capacity: int = 256
    minimum_validation_cases: int = 32
    validation_threshold: float = 0.95
    regression_tolerance: float = 0.0
    seed: int = 23

    def __post_init__(self) -> None:
        if self.input_dim < 1:
            raise ValueError("input_dim must be >= 1")
        if self.hidden_dim < 2:
            raise ValueError("hidden_dim must be >= 2")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.l2 < 0.0:
            raise ValueError("l2 must be >= 0")
        if self.replay_capacity < self.minimum_validation_cases:
            raise ValueError("replay_capacity must cover minimum validation cases")
        if self.minimum_validation_cases < 1:
            raise ValueError("minimum_validation_cases must be >= 1")
        if not 0.5 < self.validation_threshold <= 1.0:
            raise ValueError("validation_threshold must be in (0.5, 1]")
        if not 0.0 <= self.regression_tolerance < 0.5:
            raise ValueError("regression_tolerance must be in [0, 0.5)")


@dataclass(frozen=True)
class PlasticSkill:
    """Inspectable summary of one consolidated neural expert."""

    name: str
    version: int
    input_dim: int
    hidden_dim: int
    examples_seen: int
    replay_examples: int
    validation_accuracy: float
    checksum: str


@dataclass(frozen=True)
class PlasticityReport:
    """Evidence for accepting or rolling back a candidate neural update."""

    name: str
    promoted: bool
    recruited: bool
    rolled_back: bool
    version: int
    before_accuracy: float
    candidate_accuracy: float
    regression_accuracy: float
    weight_delta: float
    training_examples: int
    validation_examples: int
    replay_examples: int
    checksum_before: str
    checksum_after: str
    reason: str


class _NeuralExpert:
    """Small two-layer classifier trained with deterministic full-batch Adam."""

    def __init__(self, config: PlasticityConfig, rng: np.random.Generator) -> None:
        self.config = config
        first_scale = math.sqrt(2.0 / (config.input_dim + config.hidden_dim))
        second_scale = math.sqrt(2.0 / (config.hidden_dim + 1))
        self.w1 = rng.normal(
            0.0, first_scale, (config.input_dim, config.hidden_dim)
        )
        self.b1 = np.zeros(config.hidden_dim, dtype=float)
        self.w2 = rng.normal(0.0, second_scale, (config.hidden_dim, 1))
        self.b2 = np.zeros(1, dtype=float)

    def copy(self) -> "_NeuralExpert":
        duplicate = object.__new__(_NeuralExpert)
        duplicate.config = self.config
        duplicate.w1 = self.w1.copy()
        duplicate.b1 = self.b1.copy()
        duplicate.w2 = self.w2.copy()
        duplicate.b2 = self.b2.copy()
        return duplicate

    def arrays(self) -> tuple[np.ndarray, ...]:
        return self.w1, self.b1, self.w2, self.b2

    def checksum(self) -> str:
        digest = hashlib.sha256()
        for item in self.arrays():
            digest.update(np.asarray(item, dtype="<f8").tobytes())
        return digest.hexdigest()[:16]

    def distance(self, other: "_NeuralExpert") -> float:
        total = sum(
            float(np.sum(np.square(left - right)))
            for left, right in zip(self.arrays(), other.arrays())
        )
        return math.sqrt(total)

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        return np.where(
            values >= 0.0,
            1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0))),
            np.exp(np.clip(values, -60.0, 60.0))
            / (1.0 + np.exp(np.clip(values, -60.0, 60.0))),
        )

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        hidden = np.tanh(features @ self.w1 + self.b1)
        return self._sigmoid(hidden @ self.w2 + self.b2).reshape(-1)

    def train(self, cases: Sequence[NeuralExample], *, epochs: int) -> None:
        features, targets = _matrix(cases, self.config.input_dim)
        targets = targets.reshape(-1, 1)
        parameters = [self.w1, self.b1, self.w2, self.b2]
        first_moment = [np.zeros_like(item) for item in parameters]
        second_moment = [np.zeros_like(item) for item in parameters]
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8

        for step in range(1, epochs + 1):
            hidden = np.tanh(features @ self.w1 + self.b1)
            probabilities = self._sigmoid(hidden @ self.w2 + self.b2)
            output_error = (probabilities - targets) / len(cases)
            grad_w2 = hidden.T @ output_error + self.config.l2 * self.w2
            grad_b2 = np.sum(output_error, axis=0)
            hidden_error = (output_error @ self.w2.T) * (1.0 - np.square(hidden))
            grad_w1 = features.T @ hidden_error + self.config.l2 * self.w1
            grad_b1 = np.sum(hidden_error, axis=0)
            gradients = [grad_w1, grad_b1, grad_w2, grad_b2]

            for index, (parameter, gradient) in enumerate(
                zip(parameters, gradients)
            ):
                np.clip(gradient, -5.0, 5.0, out=gradient)
                first_moment[index] = (
                    beta1 * first_moment[index] + (1.0 - beta1) * gradient
                )
                second_moment[index] = (
                    beta2 * second_moment[index]
                    + (1.0 - beta2) * np.square(gradient)
                )
                corrected_first = first_moment[index] / (1.0 - beta1**step)
                corrected_second = second_moment[index] / (1.0 - beta2**step)
                parameter -= self.config.learning_rate * corrected_first / (
                    np.sqrt(corrected_second) + epsilon
                )


@dataclass
class _ExpertRecord:
    expert: _NeuralExpert
    version: int
    examples_seen: int
    replay: tuple[NeuralExample, ...]
    validation_accuracy: float


def _matrix(
    cases: Sequence[NeuralExample], input_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    if not cases:
        raise ValueError("at least one neural example is required")
    if any(len(item.inputs) != input_dim for item in cases):
        raise ValueError(f"every neural example must have {input_dim} inputs")
    features = np.asarray([item.inputs for item in cases], dtype=float)
    targets = np.asarray([item.target for item in cases], dtype=float)
    return features, targets


def _accuracy(
    expert: _NeuralExpert,
    cases: Sequence[NeuralExample],
    input_dim: int,
) -> float:
    features, targets = _matrix(cases, input_dim)
    predictions = expert.probabilities(features) >= 0.5
    return float(np.mean(predictions == targets.astype(bool)))


def _merge_replay(
    existing: Sequence[NeuralExample],
    additions: Sequence[NeuralExample],
    capacity: int,
) -> tuple[NeuralExample, ...]:
    unique: dict[tuple[tuple[float, ...], int], NeuralExample] = {}
    for item in tuple(existing) + tuple(additions):
        unique[(item.inputs, item.target)] = item
    merged = tuple(unique.values())
    if len(merged) <= capacity:
        return merged
    # Preserve examples across the entire lifetime instead of keeping only the
    # most recent experiences.
    indices = np.linspace(0, len(merged) - 1, capacity, dtype=int)
    return tuple(merged[int(index)] for index in indices)


class ProtectedPlasticity:
    """Expandable neural skill registry with protected candidate promotion."""

    CHECKPOINT_VERSION = 1

    def __init__(self, config: PlasticityConfig | None = None) -> None:
        self.config = config or PlasticityConfig()
        self._skills: dict[str, _ExpertRecord] = {}

    def __len__(self) -> int:
        return len(self._skills)

    def _seed_for(self, name: str) -> int:
        digest = hashlib.sha256(
            f"{self.config.seed}:{name}".encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:8], "big", signed=False)

    def _new_expert(self, name: str) -> _NeuralExpert:
        return _NeuralExpert(
            self.config,
            np.random.default_rng(self._seed_for(name)),
        )

    @staticmethod
    def _clean_name(name: str) -> str:
        clean = name.strip()
        if not clean:
            raise ValueError("plastic skill name is required")
        if len(clean) > 120:
            raise ValueError("plastic skill name must be at most 120 characters")
        return clean

    def all(self) -> tuple[PlasticSkill, ...]:
        return tuple(self.get(name) for name in sorted(self._skills))

    def get(self, name: str) -> PlasticSkill:
        clean = self._clean_name(name)
        try:
            record = self._skills[clean]
        except KeyError as exc:
            raise KeyError(f"unknown plastic skill: {clean}") from exc
        return PlasticSkill(
            name=clean,
            version=record.version,
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            examples_seen=record.examples_seen,
            replay_examples=len(record.replay),
            validation_accuracy=record.validation_accuracy,
            checksum=record.expert.checksum(),
        )

    def predict_probability(self, name: str, inputs: Sequence[float]) -> float:
        record = self._skills[self._clean_name(name)]
        example = NeuralExample(tuple(float(item) for item in inputs), 0)
        features, _ = _matrix((example,), self.config.input_dim)
        return float(record.expert.probabilities(features)[0])

    def predict(self, name: str, inputs: Sequence[float]) -> int:
        return int(self.predict_probability(name, inputs) >= 0.5)

    def evaluate(self, name: str, cases: Sequence[NeuralExample]) -> float:
        record = self._skills[self._clean_name(name)]
        return _accuracy(record.expert, tuple(cases), self.config.input_dim)

    def learn(
        self,
        name: str,
        training_cases: Sequence[NeuralExample],
        validation_cases: Sequence[NeuralExample],
        *,
        epochs: int | None = None,
    ) -> PlasticityReport:
        """Train candidate weights and promote only independently verified updates."""

        clean = self._clean_name(name)
        training = tuple(training_cases)
        validation = tuple(validation_cases)
        _matrix(training, self.config.input_dim)
        _matrix(validation, self.config.input_dim)
        if len(validation) < self.config.minimum_validation_cases:
            raise ValueError(
                "needs at least "
                f"{self.config.minimum_validation_cases} held-out validation cases"
            )
        training_epochs = self.config.epochs if epochs is None else int(epochs)
        if training_epochs < 1:
            raise ValueError("epochs must be >= 1")

        existing = self._skills.get(clean)
        recruited = existing is None
        starting = self._new_expert(clean) if recruited else existing.expert
        candidate = starting.copy()
        checksum_before = starting.checksum()
        before_accuracy = _accuracy(starting, validation, self.config.input_dim)
        old_replay = () if existing is None else existing.replay
        replay_batch = _merge_replay(
            old_replay,
            training,
            self.config.replay_capacity,
        )
        candidate.train(replay_batch, epochs=training_epochs)
        candidate_accuracy = _accuracy(
            candidate, validation, self.config.input_dim
        )
        regression_accuracy = (
            1.0
            if not old_replay
            else _accuracy(candidate, old_replay, self.config.input_dim)
        )
        required_regression = 1.0 - self.config.regression_tolerance
        passed_new = (
            candidate_accuracy >= self.config.validation_threshold
            and candidate_accuracy >= before_accuracy
        )
        passed_regression = regression_accuracy >= required_regression
        promoted = passed_new and passed_regression
        weight_delta = candidate.distance(starting)

        if promoted:
            protected = _merge_replay(
                old_replay,
                training + validation,
                self.config.replay_capacity,
            )
            version = 1 if existing is None else existing.version + 1
            examples_seen = len(training) + (
                0 if existing is None else existing.examples_seen
            )
            self._skills[clean] = _ExpertRecord(
                expert=candidate,
                version=version,
                examples_seen=examples_seen,
                replay=protected,
                validation_accuracy=candidate_accuracy,
            )
            checksum_after = candidate.checksum()
            reason = (
                "recruited, verified, and consolidated"
                if recruited
                else "updated, replay-verified, and consolidated"
            )
            replay_examples = len(protected)
        else:
            version = 0 if existing is None else existing.version
            checksum_after = checksum_before
            replay_examples = 0 if existing is None else len(existing.replay)
            if not passed_new:
                reason = "candidate failed held-out neural verification"
            else:
                reason = "candidate would regress a protected ability"

        return PlasticityReport(
            name=clean,
            promoted=promoted,
            recruited=recruited,
            rolled_back=not promoted,
            version=version,
            before_accuracy=before_accuracy,
            candidate_accuracy=candidate_accuracy,
            regression_accuracy=regression_accuracy,
            weight_delta=weight_delta,
            training_examples=len(training),
            validation_examples=len(validation),
            replay_examples=replay_examples,
            checksum_before=checksum_before,
            checksum_after=checksum_after,
            reason=reason,
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        skills = []
        for index, name in enumerate(sorted(self._skills)):
            record = self._skills[name]
            for suffix, value in zip(
                ("w1", "b1", "w2", "b2"), record.expert.arrays()
            ):
                arrays[f"expert_{index}_{suffix}"] = value
            skills.append(
                {
                    "name": name,
                    "version": record.version,
                    "examples_seen": record.examples_seen,
                    "validation_accuracy": record.validation_accuracy,
                    "replay": [asdict(item) for item in record.replay],
                }
            )
        metadata = {
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "config": asdict(self.config),
            "skills": skills,
        }
        temporary = target.with_name(f".{target.name}.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                metadata=np.asarray(json.dumps(metadata)),
                **arrays,
            )
        os.replace(temporary, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "ProtectedPlasticity":
        source = Path(path).expanduser().resolve()
        with np.load(source, allow_pickle=False) as checkpoint:
            metadata = json.loads(str(checkpoint["metadata"].item()))
            if metadata.get("checkpoint_version") != cls.CHECKPOINT_VERSION:
                raise ValueError("unsupported plasticity checkpoint version")
            system = cls(PlasticityConfig(**metadata["config"]))
            for index, state in enumerate(metadata.get("skills", [])):
                expert = system._new_expert(str(state["name"]))
                expert.w1 = checkpoint[f"expert_{index}_w1"].copy()
                expert.b1 = checkpoint[f"expert_{index}_b1"].copy()
                expert.w2 = checkpoint[f"expert_{index}_w2"].copy()
                expert.b2 = checkpoint[f"expert_{index}_b2"].copy()
                replay = tuple(
                    NeuralExample(
                        tuple(float(value) for value in item["inputs"]),
                        int(item["target"]),
                    )
                    for item in state["replay"]
                )
                system._skills[str(state["name"])] = _ExpertRecord(
                    expert=expert,
                    version=int(state["version"]),
                    examples_seen=int(state["examples_seen"]),
                    replay=replay,
                    validation_accuracy=float(state["validation_accuracy"]),
                )
            return system


def make_reasoning_cases(
    rule: ReasoningRule,
    count: int,
    seed: int,
    *,
    scale: float = 1.0,
) -> tuple[NeuralExample, ...]:
    """Generate independently seeded experiences for transparent benchmarks."""

    if rule not in ("relative_balance", "same_sign"):
        raise ValueError(f"unknown reasoning rule: {rule}")
    if count < 1:
        raise ValueError("count must be >= 1")
    if scale <= 0.0:
        raise ValueError("scale must be > 0")
    rng = np.random.default_rng(seed)
    cases: list[NeuralExample] = []
    while len(cases) < count:
        values = rng.uniform(-scale, scale, 4)
        normalized = values / scale
        if rule == "relative_balance":
            score = float(normalized[0] + normalized[1] - normalized[2] - normalized[3])
            if abs(score) < 0.20:
                continue
            target = int(score > 0.0)
        else:
            if abs(float(normalized[0])) < 0.15 or abs(float(normalized[1])) < 0.15:
                continue
            target = int((normalized[0] >= 0.0) == (normalized[1] >= 0.0))
        cases.append(
            NeuralExample(tuple(float(item) for item in values), target)
        )
    return tuple(cases)
