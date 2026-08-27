"""How many timescales does continual learning actually need?

Every method that works in this literature has the same shape: keep a slower
copy of yourself and be pulled toward it. EWC anchors weights to their earlier
values. DER++ anchors the function to a frozen snapshot of its own outputs.
Self-distillation makes the previous model the teacher. SuRe pairs a fast and a
slow adapter merged by EMA. Nested Learning generalises the idea to a continuum
of memory modules each updating at its own rate.

One anchor, two anchors, a continuum. The principle is widely asserted and, as
far as the published record goes, never isolated: each paper proposes a whole
architecture, so the number of timescales always varies alongside everything
else. This module holds everything else fixed.

Anchors available here, at three different lags:

``snapshot``  the logits recorded when an example entered the buffer. For an
              old example this is a very long lag -- the function as it stood
              many tasks ago.
``slow``      an exponential moving average of the live weights, decay 0.999.
``fast``      the same at decay 0.99, a few hundred steps of lag.

The control that makes this a test of *timescales* rather than of *strength*:
the total anchoring weight is held constant and divided equally among whichever
anchors are active. Without that, three anchors would pull three times as hard
as one and would win for a reason that has nothing to do with having three
timescales.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .continual import ContinualConfig, Sample
from .der import DERLearner

DECAY = {"slow": 0.999, "fast": 0.99}


class _EMACopy:
    """A lagging copy of the network, updated toward it every step."""

    def __init__(self, learner: "TimescaleLearner", decay: float) -> None:
        self.decay = float(decay)
        self.weights = [w.copy() for w in learner.weights]
        self.biases = [b.copy() for b in learner.biases]
        self.heads: dict[str, np.ndarray] = {}
        self.head_bias: dict[str, np.ndarray] = {}

    def update(self, learner: "TimescaleLearner") -> None:
        d = self.decay
        for i, w in enumerate(learner.weights):
            self.weights[i] = d * self.weights[i] + (1.0 - d) * w
            self.biases[i] = d * self.biases[i] + (1.0 - d) * learner.biases[i]
        for name, head in learner.heads.items():
            if name not in self.heads:
                self.heads[name] = head.copy()
                self.head_bias[name] = np.asarray(learner.head_bias[name]).copy()
            else:
                self.heads[name] = d * self.heads[name] + (1.0 - d) * head
                self.head_bias[name] = (
                    d * self.head_bias[name] + (1.0 - d) * np.asarray(learner.head_bias[name])
                )

    def logits(self, skill: str, x: np.ndarray) -> np.ndarray | None:
        if skill not in self.heads:
            return None
        h = x
        for w, b in zip(self.weights, self.biases):
            h = np.tanh(h @ w.T + b)
        return h @ self.heads[skill].T + self.head_bias[skill]


class TimescaleLearner(DERLearner):
    """DER++ generalised to an arbitrary set of anchor timescales."""

    def __init__(
        self,
        config: ContinualConfig | None = None,
        *,
        classes: int = 10,
        anchors: Sequence[str] = ("snapshot",),
        alpha: float = 0.1,
        beta: float = 0.5,
    ) -> None:
        for anchor in anchors:
            if anchor not in ("snapshot", "slow", "fast"):
                raise ValueError(f"unknown anchor: {anchor}")
        self.anchors = tuple(anchors)
        mode = "der++" if anchors else "hard"
        super().__init__(config, classes=classes, mode=mode, alpha=alpha, beta=beta)
        # Equal split of a fixed budget: the experiment is about how many
        # timescales, not how hard they pull.
        self.share = alpha / max(len(self.anchors), 1)
        self.emas = {
            name: _EMACopy(self, DECAY[name]) for name in self.anchors if name in DECAY
        }
        self._replay_x: np.ndarray | None = None

    # ------------------------------------------------------------------
    def _head_signal(self, skill: str, h: np.ndarray, y: np.ndarray, total: int):
        z = self._logits(skill, h)
        err = np.zeros_like(z)
        n = min(self._fresh.get(skill, len(y)), len(y))
        stored = self._targets.get(skill)
        if stored is not None and len(stored) != len(y) - n:
            n, stored = len(y), None

        p_new = self._softmax(z[:n])
        onehot = np.zeros_like(p_new)
        onehot[np.arange(n), y[:n].astype(int)] = 1.0
        err[:n] = p_new - onehot

        if n < len(y):
            p_old = self._softmax(z[n:])
            hot = np.zeros_like(p_old)
            hot[np.arange(len(p_old)), y[n:].astype(int)] = 1.0
            if not self.anchors:
                err[n:] = p_old - hot
            else:
                err[n:] = self.beta * (p_old - hot)
                if "snapshot" in self.anchors and stored is not None:
                    err[n:] += self.share * (z[n:] - stored)
                for name, ema in self.emas.items():
                    if self._replay_x is None:
                        continue
                    target = ema.logits(skill, self._replay_x)
                    if target is not None:
                        err[n:] += self.share * (z[n:] - target)

        err /= total
        return err.T @ h, err.sum(axis=0), err @ self.heads[skill]

    # ------------------------------------------------------------------
    def observe(self, skill: str, batch: Sequence[Sample]) -> None:
        if not batch:
            raise ValueError("batch must not be empty")
        self._ensure_head(skill)

        fresh_x = np.asarray([c.inputs for c in batch], dtype=float)
        fresh_y = np.asarray([c.target for c in batch], dtype=float)
        snapshot = self._logits(skill, self._forward(fresh_x)[-1])
        for item, row in zip(batch, snapshot):
            self._replay[skill].add(item, row.copy())

        replayed = self._replay_with_targets(skill, self.config.replay_per_step)
        self._fresh[skill] = len(batch)
        if replayed is None:
            self._targets[skill] = None
            self._replay_x = None
            self._step({skill: (fresh_x, fresh_y)})
        else:
            past_x, past_y, past_z = replayed
            self._targets[skill] = past_z
            self._replay_x = past_x
            self._step(
                {skill: (np.concatenate([fresh_x, past_x]), np.concatenate([fresh_y, past_y]))}
            )
        for ema in self.emas.values():
            ema.update(self)
