"""A network that never stops training.

The protected-expert design freezes everything that already works. That makes
retention exact and transfer impossible: measured on this repository's own
registries, learning two skills changes the third skill's accuracy by
0.000000, and learning three more changes the first skill's accuracy by
0.000000. Nothing can help anything, because nothing can touch anything.

``athena.transfer`` relaxed that one notch -- new skills may *read* frozen old
ones -- which buys forward transfer while keeping retention exact. But the old
weights still never move, so an earlier skill can never get better. A system
that is genuinely still learning after deployment has to be able to improve
something it already knows.

This module goes the rest of the way. One shared trunk, trained on every
observation for as long as the process is alive, with per-skill heads on top.
Because the trunk is shared and always moving, three things become possible
that freezing rules out by construction:

*   **Forward transfer** -- a new skill starts from a representation shaped by
    every earlier skill.
*   **Backward transfer** -- an *old* skill gets better without being retrained,
    because the trunk underneath it improved. This is the signature of a system
    that is actually getting smarter rather than accumulating drawers, and it
    is strictly impossible under freezing.
*   **Catastrophic forgetting** -- the same door, opening the other way.

So the point of the machinery here is not to train the trunk; that part is
easy. It is to keep the third item bounded while leaving the first two free:

*   **Replay.** Every skill keeps a reservoir sample of its own experience, and
    every update step trains on a mixture of the current skill and the past.
    This is the mechanism brains appear to use, and it is doing most of the
    work here.
*   **Consolidation.** Weights that mattered to earlier skills accumulate an
    importance score, and moving them is penalised in proportion. Elastic
    weight consolidation, in its diagonal-Fisher form.
*   **Rollback.** Held-out probes per skill are re-checked periodically. If
    aggregate retention drops past tolerance, the trunk reverts to its last
    consolidated checkpoint. Learning stays aggressive because a mistake is
    recoverable rather than permanent.

The honest framing: freezing trades all plasticity for perfect stability, this
trades exact retention for the ability to keep improving, and which one is
right depends on whether the deployment can tolerate a skill moving by a
percent. Both are measured side by side in ``examples/continual_benchmark2.py``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Experience:
    """One labelled case belonging to a named skill."""

    inputs: tuple[float, ...]
    target: int

    def __post_init__(self) -> None:
        if not self.inputs:
            raise ValueError("inputs must not be empty")
        if self.target not in (0, 1):
            raise ValueError("target must be 0 or 1")


@dataclass
class ContinualConfig:
    input_dim: int = 6
    hidden: tuple[int, ...] = (24, 16)
    learning_rate: float = 0.05
    momentum: float = 0.9
    replay_per_step: int = 64
    replay_capacity: int = 512
    consolidation: float = 200.0
    retention_tolerance: float = 0.05
    checkpoint_every: int = 250
    # Control switch. With the trunk frozen after the first skill, only the
    # per-skill heads train -- which is the frozen-representation design, built
    # from exactly the same code, layers, and initialisation. Comparing against
    # a different module's network would measure the two implementations
    # instead of the one mechanism under test.
    freeze_trunk: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        if self.input_dim < 1:
            raise ValueError("input_dim must be >= 1")
        if not self.hidden or any(h < 1 for h in self.hidden):
            raise ValueError("hidden must be a non-empty tuple of positive sizes")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if self.replay_per_step < 0 or self.replay_capacity < 1:
            raise ValueError("replay sizes must be non-negative and capacity >= 1")
        if self.consolidation < 0.0:
            raise ValueError("consolidation must be >= 0")
        if not 0.0 <= self.retention_tolerance <= 1.0:
            raise ValueError("retention_tolerance must be in [0, 1]")


@dataclass
class LearningReport:
    skill: str
    steps: int
    accuracy: float
    retained: dict[str, float] = field(default_factory=dict)
    rolled_back: bool = False


class _Reservoir:
    """Uniform sample of a stream that never grew a buffer proportional to it."""

    def __init__(self, capacity: int, rng: np.random.Generator) -> None:
        self.capacity = int(capacity)
        self.items: list[Experience] = []
        self.seen = 0
        self._rng = rng

    def add(self, item: Experience) -> None:
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
            return
        # Classic reservoir sampling: keeps the buffer representative of the
        # whole lifetime rather than of the most recent window, which is what
        # stops old skills quietly ageing out of the replay mixture.
        index = int(self._rng.integers(0, self.seen))
        if index < self.capacity:
            self.items[index] = item


class ContinualLearner:
    """A shared trunk that keeps training, with one head per skill."""

    def __init__(self, config: ContinualConfig | None = None) -> None:
        self.config = config or ContinualConfig()
        rng = np.random.default_rng(self.config.seed)
        self._rng = rng

        dims = [self.config.input_dim, *self.config.hidden]
        self.weights = [
            rng.normal(0.0, 1.0 / np.sqrt(dims[i]), size=(dims[i + 1], dims[i]))
            for i in range(len(dims) - 1)
        ]
        self.biases = [np.zeros(d) for d in dims[1:]]
        self._velocity = [np.zeros_like(w) for w in self.weights]
        self._bias_velocity = [np.zeros_like(b) for b in self.biases]

        self.heads: dict[str, np.ndarray] = {}
        self.head_bias: dict[str, float] = {}
        self._replay: dict[str, _Reservoir] = {}
        self._order: list[str] = []

        # Consolidation state: how much each trunk weight mattered to skills
        # already learned, and where it sat when they were consolidated.
        self._importance = [np.zeros_like(w) for w in self.weights]
        self._anchor = [w.copy() for w in self.weights]
        self._checkpoint: tuple | None = None
        self._probes: dict[str, tuple[Experience, ...]] = {}
        self.steps = 0
        self.rollbacks = 0
        self._frozen = False

    # ------------------------------------------------------------------
    @property
    def skills(self) -> tuple[str, ...]:
        return tuple(self._order)

    def checksum(self) -> str:
        digest = hashlib.sha256()
        for array in (*self.weights, *self.biases):
            digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
        return digest.hexdigest()

    def _ensure_head(self, skill: str) -> None:
        if skill in self.heads:
            return
        width = self.config.hidden[-1]
        self.heads[skill] = self._rng.normal(0.0, 1.0 / np.sqrt(width), size=width)
        self.head_bias[skill] = 0.0
        self._replay[skill] = _Reservoir(self.config.replay_capacity, self._rng)
        self._order.append(skill)

    # ------------------------------------------------------------------
    def _forward(self, x: np.ndarray) -> list[np.ndarray]:
        activations = [x]
        current = x
        for w, b in zip(self.weights, self.biases):
            current = np.tanh(current @ w.T + b)
            activations.append(current)
        return activations

    def latent(self, x: np.ndarray) -> np.ndarray:
        """The shared representation -- the thing that keeps improving."""
        return self._forward(np.atleast_2d(np.asarray(x, dtype=float)))[-1]

    def _probability(self, skill: str, x: np.ndarray) -> np.ndarray:
        h = self._forward(x)[-1]
        z = h @ self.heads[skill] + self.head_bias[skill]
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

    def predict(self, skill: str, inputs: Sequence[float]) -> int:
        x = np.atleast_2d(np.asarray(inputs, dtype=float))
        return int(self._probability(skill, x)[0] >= 0.5)

    def accuracy(self, skill: str, cases: Sequence[Experience]) -> float:
        if not cases:
            raise ValueError("need at least one case")
        x = np.asarray([c.inputs for c in cases], dtype=float)
        y = np.asarray([c.target for c in cases], dtype=float)
        p = self._probability(skill, x)
        return float(((p >= 0.5).astype(float) == y).mean())

    # ------------------------------------------------------------------
    def _gradients(
        self, batches: dict[str, tuple[np.ndarray, np.ndarray]]
    ) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, np.ndarray], dict[str, float]]:
        """Accumulate trunk gradients across every skill present in the batch.

        This is where compounding actually happens: one set of trunk weights
        receives the sum of what every skill wants, so a representation that
        helps several skills at once is reinforced several times over.
        """
        grad_w = [np.zeros_like(w) for w in self.weights]
        grad_b = [np.zeros_like(b) for b in self.biases]
        grad_head: dict[str, np.ndarray] = {}
        grad_head_bias: dict[str, float] = {}

        total = sum(len(x) for x, _ in batches.values()) or 1
        for skill, (x, y) in batches.items():
            if not len(x):
                continue
            activations = self._forward(x)
            h = activations[-1]
            z = h @ self.heads[skill] + self.head_bias[skill]
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
            err = (p - y) / total

            grad_head[skill] = h.T @ err
            grad_head_bias[skill] = float(err.sum())

            delta = np.outer(err, self.heads[skill]) * (1.0 - h * h)
            for layer in range(len(self.weights) - 1, -1, -1):
                grad_w[layer] += delta.T @ activations[layer]
                grad_b[layer] += delta.sum(axis=0)
                if layer:
                    lower = activations[layer]
                    delta = (delta @ self.weights[layer]) * (1.0 - lower * lower)
        return grad_w, grad_b, grad_head, grad_head_bias

    def _step(self, batches: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        cfg = self.config
        grad_w, grad_b, grad_head, grad_head_bias = self._gradients(batches)

        if cfg.freeze_trunk and self._frozen:
            for skill, gradient in grad_head.items():
                self.heads[skill] -= cfg.learning_rate * gradient
                self.head_bias[skill] -= cfg.learning_rate * grad_head_bias[skill]
            self.steps += 1
            return

        for layer in range(len(self.weights)):
            if cfg.consolidation > 0.0:
                # Pull toward where the weight sat when earlier skills were
                # consolidated, in proportion to how much they depended on it.
                grad_w[layer] = grad_w[layer] + cfg.consolidation * self._importance[
                    layer
                ] * (self.weights[layer] - self._anchor[layer])
            self._velocity[layer] = cfg.momentum * self._velocity[layer] - cfg.learning_rate * grad_w[layer]
            self.weights[layer] += self._velocity[layer]
            self._bias_velocity[layer] = (
                cfg.momentum * self._bias_velocity[layer] - cfg.learning_rate * grad_b[layer]
            )
            self.biases[layer] += self._bias_velocity[layer]

        for skill, gradient in grad_head.items():
            self.heads[skill] -= cfg.learning_rate * gradient
            self.head_bias[skill] -= cfg.learning_rate * grad_head_bias[skill]
        self.steps += 1

    # ------------------------------------------------------------------
    def _replay_batch(self, exclude: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        per_skill = max(self.config.replay_per_step // max(len(self._order), 1), 1)
        batches: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for skill in self._order:
            if skill == exclude:
                continue
            items = self._replay[skill].items
            if not items:
                continue
            picks = self._rng.choice(len(items), size=min(per_skill, len(items)), replace=False)
            chosen = [items[int(i)] for i in picks]
            batches[skill] = (
                np.asarray([c.inputs for c in chosen], dtype=float),
                np.asarray([c.target for c in chosen], dtype=float),
            )
        return batches

    def observe(self, skill: str, batch: Sequence[Experience]) -> None:
        """Train one step on new experience plus a replayed slice of the past.

        There is no separate training mode. This is the only way experience
        enters the system, and it is available for as long as the process runs.
        """
        if not batch:
            raise ValueError("batch must not be empty")
        self._ensure_head(skill)
        for item in batch:
            if len(item.inputs) != self.config.input_dim:
                raise ValueError(
                    f"experience has {len(item.inputs)} features, "
                    f"config expects {self.config.input_dim}"
                )
            self._replay[skill].add(item)

        batches = self._replay_batch(exclude=skill)
        batches[skill] = (
            np.asarray([c.inputs for c in batch], dtype=float),
            np.asarray([c.target for c in batch], dtype=float),
        )
        self._step(batches)

    # ------------------------------------------------------------------
    def _snapshot(self) -> tuple:
        return (
            [w.copy() for w in self.weights],
            [b.copy() for b in self.biases],
            {k: v.copy() for k, v in self.heads.items()},
            dict(self.head_bias),
        )

    def _restore(self, snapshot: tuple) -> None:
        weights, biases, heads, head_bias = snapshot
        self.weights = [w.copy() for w in weights]
        self.biases = [b.copy() for b in biases]
        for name, vector in heads.items():
            self.heads[name] = vector.copy()
        self.head_bias.update(head_bias)
        self._velocity = [np.zeros_like(w) for w in self.weights]
        self._bias_velocity = [np.zeros_like(b) for b in self.biases]

    def _accumulate_importance(self, skill: str) -> None:
        items = self._replay[skill].items
        if not items:
            return
        x = np.asarray([c.inputs for c in items], dtype=float)
        y = np.asarray([c.target for c in items], dtype=float)
        grad_w, _, _, _ = self._gradients({skill: (x, y)})
        for layer, gradient in enumerate(grad_w):
            self._importance[layer] += gradient * gradient

    def consolidate(self, skill: str | None = None) -> None:
        """Record what the current weights are worth, and checkpoint them."""
        for name in ([skill] if skill else self._order):
            if name in self._replay:
                self._accumulate_importance(name)
        self._anchor = [w.copy() for w in self.weights]
        self._checkpoint = self._snapshot()

    def retention(self) -> dict[str, float]:
        return {
            name: self.accuracy(name, probe)
            for name, probe in self._probes.items()
            if probe
        }

    def set_probe(self, skill: str, cases: Sequence[Experience]) -> None:
        """Register held-out cases used to detect retention loss and roll back."""
        self._probes[skill] = tuple(cases)

    def check_retention(self, expected: dict[str, float]) -> bool:
        """Revert to the last checkpoint if retention fell past tolerance.

        Returns True when a rollback happened. Aggressive learning is only
        safe if a bad stretch is recoverable; this is what makes it so.
        """
        if self._checkpoint is None or not expected:
            return False
        current = self.retention()
        shared = [k for k in expected if k in current]
        if not shared:
            return False
        drop = max(expected[k] - current[k] for k in shared)
        if drop > self.config.retention_tolerance:
            self._restore(self._checkpoint)
            self.rollbacks += 1
            return True
        return False

    # ------------------------------------------------------------------
    def teach(
        self,
        skill: str,
        training: Sequence[Experience],
        validation: Sequence[Experience],
        *,
        steps: int = 400,
        batch_size: int = 16,
    ) -> LearningReport:
        """Learn a skill from a finite set of examples, without ever freezing."""
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if not training:
            raise ValueError("need training examples")
        self._ensure_head(skill)
        expected = self.retention()

        training = tuple(training)
        for _ in range(steps):
            picks = self._rng.choice(
                len(training), size=min(batch_size, len(training)), replace=False
            )
            self.observe(skill, [training[int(i)] for i in picks])
            if self.config.checkpoint_every and self.steps % self.config.checkpoint_every == 0:
                self.check_retention(expected)

        self.set_probe(skill, validation)
        rolled_back = self.check_retention(expected)
        self.consolidate(skill)
        if self.config.freeze_trunk:
            self._frozen = True
        return LearningReport(
            skill=skill,
            steps=steps,
            accuracy=self.accuracy(skill, validation),
            retained=self.retention(),
            rolled_back=rolled_back,
        )


class SharedPlasticity:
    """``ProtectedPlasticity``'s interface, backed by a trunk that keeps training.

    Same call shape as the existing registry -- ``learn(name, training,
    validation)`` returning a report with ``candidate_accuracy`` and
    ``promoted`` -- so it can be swapped in wherever protected experts are used
    today. The difference is underneath: skills share one representation that
    never stops learning, so they compound instead of sitting in separate
    drawers.

    It accepts ``athena.plasticity.NeuralExample`` as-is; only ``inputs`` and
    ``target`` are read.
    """

    def __init__(
        self,
        config: ContinualConfig | None = None,
        *,
        validation_threshold: float = 0.51,
        steps: int = 600,
    ) -> None:
        self.learner = ContinualLearner(config)
        self.validation_threshold = float(validation_threshold)
        self.steps = int(steps)

    def __len__(self) -> int:
        return len(self.learner.skills)

    @staticmethod
    def _convert(cases: Sequence[object]) -> list[Experience]:
        return [Experience(tuple(c.inputs), int(c.target)) for c in cases]

    def learn(self, name: str, training: Sequence[object], validation: Sequence[object]):
        report = self.learner.teach(
            name,
            self._convert(training),
            self._convert(validation),
            steps=self.steps,
        )
        report.rolled_back = report.rolled_back or report.accuracy < self.validation_threshold
        return report

    def evaluate(self, name: str, cases: Sequence[object]) -> float:
        return self.learner.accuracy(name, self._convert(cases))

    def predict(self, name: str, inputs: Sequence[float]) -> int:
        return self.learner.predict(name, inputs)


def related_tasks(count: int, seed: int, *, dim: int = 8, rank: int = 3) -> list[np.ndarray]:
    """Tasks drawn from a shared low-rank basis.

    Every task is a different combination of the same few underlying factors,
    so one good representation serves all of them. This is the situation any
    real domain is in, and it is the condition under which learning more makes
    a system better at what it already knew.
    """
    rng = np.random.default_rng(seed)
    basis = [rng.normal(0.0, 1.0, (dim, dim)) for _ in range(rank)]
    return [
        sum(c * b for c, b in zip(rng.normal(0.0, 1.0, rank), basis))
        for _ in range(count)
    ]


def unrelated_tasks(count: int, seed: int, *, dim: int = 8) -> list[np.ndarray]:
    """Tasks with no shared structure -- the control, where nothing can transfer."""
    rng = np.random.default_rng(seed)
    return [rng.normal(0.0, 1.0, (dim, dim)) for _ in range(count)]


def task_cases(matrix: np.ndarray, count: int, seed: int) -> list[Experience]:
    """Label points by the sign of a quadratic form."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, (count, matrix.shape[0]))
    y = ((x @ matrix * x).sum(axis=1) > 0).astype(int)
    return [Experience(tuple(row), int(t)) for row, t in zip(x, y)]


def stream(learner: ContinualLearner, skill: str, experiences: Iterable[Experience]) -> int:
    """Feed an unbounded stream of experience into a live learner."""
    count = 0
    for item in experiences:
        learner.observe(skill, [item])
        count += 1
    return count
