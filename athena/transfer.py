"""Skills that make the next skill cheaper.

Athena's protected-expert design gives every skill its own independently
initialised network and freezes the shared representation while operators
train. That buys perfect retention -- learning a new skill cannot disturb an
old one -- but it buys it by making the skills mutually invisible. Measured on
the existing registries, learning two skills changes the third skill's
accuracy by exactly 0.000000. Nothing carries over, because structurally
nothing can.

That is one half of the stated goal ("retain the acquired skill", "without
forgetting earlier abilities") at the cost of the other half ("get smarter the
more it learns", "transfer it to a different problem"). The two halves are not
independent features to be built separately; they are the two ends of the same
tradeoff, and freezing sits at one extreme of it.

This module takes the middle. It keeps every earlier expert bit-for-bit frozen
-- so retention stays exactly perfect, and every promote/rollback guarantee
still holds -- but lets a *new* expert read the internal features of the
experts already learned. Old skills cannot be damaged because their weights are
never written. New skills get cheaper because they start from whatever earlier
skills already worked out, instead of from noise.

The approach is progressive networks (Rusu et al., 2016), which was designed
for exactly this pair of requirements. The cost is honest and worth stating:
each expert's input grows with the number of skills retained, so lateral
sources are capped, and transfer between genuinely unrelated tasks can be
negative rather than zero. The benchmark measures both.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Example:
    """One labelled case: a feature vector and a binary target."""

    inputs: tuple[float, ...]
    target: int

    def __post_init__(self) -> None:
        if not self.inputs:
            raise ValueError("inputs must not be empty")
        if self.target not in (0, 1):
            raise ValueError("target must be 0 or 1")


@dataclass
class TransferConfig:
    input_dim: int = 4
    hidden_dim: int = 12
    epochs: int = 400
    learning_rate: float = 0.15
    max_lateral_sources: int = 6
    validation_threshold: float = 0.51
    seed: int = 0

    def __post_init__(self) -> None:
        if self.input_dim < 1 or self.hidden_dim < 1:
            raise ValueError("input_dim and hidden_dim must be >= 1")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if not 0.0 < self.learning_rate:
            raise ValueError("learning_rate must be > 0")
        if self.max_lateral_sources < 0:
            raise ValueError("max_lateral_sources must be >= 0")


@dataclass
class TransferReport:
    name: str
    accuracy: float
    promoted: bool
    lateral_sources: tuple[str, ...]
    frozen_checksums: dict[str, str] = field(default_factory=dict)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class _Expert:
    """One skill: a hidden layer over its own inputs plus frozen prior features."""

    def __init__(self, in_dim: int, hidden: int, rng: np.random.Generator) -> None:
        self.w1 = rng.normal(0.0, 1.0 / np.sqrt(in_dim), size=(hidden, in_dim))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0.0, 1.0 / np.sqrt(hidden), size=hidden)
        self.b2 = 0.0

    def hidden(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x @ self.w1.T + self.b1)

    def probability(self, x: np.ndarray) -> np.ndarray:
        return _sigmoid(self.hidden(x) @ self.w2 + self.b2)

    def checksum(self) -> str:
        digest = hashlib.sha256()
        for array in (self.w1, self.b1, self.w2, np.asarray([self.b2])):
            digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
        return digest.hexdigest()

    def train(self, x: np.ndarray, y: np.ndarray, *, epochs: int, lr: float) -> None:
        n = max(len(x), 1)
        for _ in range(epochs):
            h = self.hidden(x)
            p = _sigmoid(h @ self.w2 + self.b2)
            err = p - y
            g_w2 = h.T @ err / n
            g_b2 = float(err.mean())
            delta = np.outer(err, self.w2) * (1.0 - h * h)
            g_w1 = delta.T @ x / n
            g_b1 = delta.mean(axis=0)
            self.w2 -= lr * g_w2
            self.b2 -= lr * g_b2
            self.w1 -= lr * g_w1
            self.b1 -= lr * g_b1


class ProgressiveRegistry:
    """Skills that keep every earlier skill frozen and still learn from them.

    ``lateral=False`` reproduces the current isolated-expert behaviour, so the
    two can be compared under identical data, seeds, and budgets.
    """

    def __init__(self, config: TransferConfig | None = None, *, lateral: bool = True) -> None:
        self.config = config or TransferConfig()
        self.lateral = bool(lateral)
        self._order: list[str] = []
        self._experts: dict[str, _Expert] = {}
        # Frozen at promotion time. Recomputing an expert's sources later would
        # let skills learned *after* it feed into it, which is both circular
        # and a retention violation: the expert's own behaviour would change
        # as unrelated skills arrived.
        self._source_of: dict[str, tuple[str, ...]] = {}

    def __len__(self) -> int:
        return len(self._experts)

    def _seed_for(self, name: str) -> int:
        digest = hashlib.sha256(f"{self.config.seed}:{name}".encode()).digest()
        return int.from_bytes(digest[:8], "big", signed=False)

    def _next_sources(self) -> tuple[str, ...]:
        """Which existing skills a newly arriving skill may read."""
        if not self.lateral:
            return ()
        return tuple(self._order[-self.config.max_lateral_sources :])

    def _augment(self, raw: np.ndarray, sources: Sequence[str]) -> np.ndarray:
        """Append the frozen hidden features of earlier skills to the input.

        This is the only channel through which one skill can help another, and
        it is strictly read-only: the source experts are evaluated, never
        updated. Retention is therefore preserved by construction rather than
        by a promotion gate that has to catch damage after the fact.
        """
        if not sources:
            return raw
        features = [raw]
        for name in sources:
            expert = self._experts[name]
            features.append(expert.hidden(self._augment(raw, self._source_of[name])))
        return np.concatenate(features, axis=1)

    def _matrix(self, cases: Sequence[Example]) -> tuple[np.ndarray, np.ndarray]:
        if not cases:
            raise ValueError("need at least one case")
        x = np.asarray([c.inputs for c in cases], dtype=float)
        if x.shape[1] != self.config.input_dim:
            raise ValueError(
                f"cases have {x.shape[1]} features, config expects {self.config.input_dim}"
            )
        y = np.asarray([c.target for c in cases], dtype=float)
        return x, y

    def accuracy(self, name: str, cases: Sequence[Example]) -> float:
        x, y = self._matrix(cases)
        expert = self._experts[name]
        p = expert.probability(self._augment(x, self._source_of[name]))
        return float(((p >= 0.5).astype(float) == y).mean())

    def learn(
        self,
        name: str,
        training: Sequence[Example],
        validation: Sequence[Example],
        *,
        epochs: int | None = None,
    ) -> TransferReport:
        """Learn one skill, reading earlier skills but never writing to them."""
        if name in self._experts:
            raise ValueError(f"skill already learned: {name}")
        x_train, y_train = self._matrix(training)
        x_val, y_val = self._matrix(validation)

        sources = self._next_sources()
        before = {n: self._experts[n].checksum() for n in self._experts}

        augmented_train = self._augment(x_train, sources)
        expert = _Expert(
            augmented_train.shape[1],
            self.config.hidden_dim,
            np.random.default_rng(self._seed_for(name)),
        )
        expert.train(
            augmented_train,
            y_train,
            epochs=self.config.epochs if epochs is None else int(epochs),
            lr=self.config.learning_rate,
        )

        p = expert.probability(self._augment(x_val, sources))
        accuracy = float(((p >= 0.5).astype(float) == y_val).mean())
        promoted = accuracy >= self.config.validation_threshold
        if promoted:
            self._experts[name] = expert
            self._source_of[name] = sources
            self._order.append(name)

        after = {n: self._experts[n].checksum() for n in before}
        if after != before:
            raise AssertionError("an earlier skill was modified; retention is broken")

        return TransferReport(
            name=name,
            accuracy=accuracy,
            promoted=promoted,
            lateral_sources=sources,
            frozen_checksums=after,
        )
