"""What to store per slot, once you know you cannot choose better slots.

The previous experiment established that replay is limited by *coverage*: every
policy that ranked which examples to keep lost to uniform reservoir sampling,
because ranking concentrates the buffer on whatever is hard now and evicts
everything older. If selection cannot be improved, the remaining lever is how
much each retained slot carries.

A hard label is a very weak constraint. "This image is a 7" is satisfied by
enormous numbers of different functions, so rehearsing it pins down almost
nothing about the network that produced it. The logits the network computed
when it first saw that image are a far tighter constraint: they encode the
whole similarity structure it had learned -- how 7 relates to 1, to 9, to 4 --
and rehearsing them asks the network to still compute what it used to compute,
not merely to still get the answer right.

That is the difference between rehearsing an *answer* and rehearsing a
*function*, and it is why Dark Experience Replay (Buzzega et al., NeurIPS 2020)
stores logits. It matters most exactly where this repository operates: domain-
incremental streams with small buffers, where coverage is scarce and each slot
has to do as much work as possible.

Three modes, identical in buffer contents, sampling, capacity, and every
hyperparameter -- differing only in what is rehearsed:

``hard``    cross-entropy against the stored label (ordinary replay)
``logits``  squared error against the stored logits (DER)
``der++``   both terms (DER++)
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

from .continual import ContinualConfig, MultiClassLearner, Sample

Mode = Literal["hard", "logits", "der++"]
Probe = Literal["real", "noise", "shuffled"]


class _LogitReservoir:
    """Reservoir sampling that keeps each example's logits alongside it."""

    def __init__(self, capacity: int, rng: np.random.Generator) -> None:
        self.capacity = int(capacity)
        self.items: list[Sample] = []
        self.logits: list[np.ndarray] = []
        self.seen = 0
        self._rng = rng

    def add(self, item: Sample, logits: np.ndarray) -> None:
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
            self.logits.append(logits)
            return
        index = int(self._rng.integers(0, self.seen))
        if index < self.capacity:
            self.items[index] = item
            self.logits[index] = logits


class DERLearner(MultiClassLearner):
    """Shared always-training trunk, rehearsing answers or functions."""

    def __init__(
        self,
        config: ContinualConfig | None = None,
        *,
        classes: int = 10,
        mode: Mode = "hard",
        probe: Probe = "real",
        alpha: float = 0.5,
        beta: float = 0.5,
    ) -> None:
        if mode not in ("hard", "logits", "der++"):
            raise ValueError(f"unknown mode: {mode}")
        if probe not in ("real", "noise", "shuffled"):
            raise ValueError(f"unknown probe: {probe}")
        self.mode = mode
        self.probe = probe
        self.alpha = float(alpha)
        self.beta = float(beta)
        super().__init__(config, classes=classes)
        self._fresh: dict[str, int] = {}
        self._targets: dict[str, np.ndarray | None] = {}

    def _ensure_head(self, skill: str) -> None:
        already = skill in self.heads
        super()._ensure_head(skill)
        if not already:
            self._replay[skill] = _LogitReservoir(self.config.replay_capacity, self._rng)

    # ------------------------------------------------------------------
    def _head_signal(self, skill: str, h: np.ndarray, y: np.ndarray, total: int):
        """Cross-entropy on new experience, the chosen rehearsal loss on the rest.

        Rows are ordered fresh-first, so one index splits them. New experience
        always trains against its real label -- only *rehearsal* changes, which
        keeps the comparison to a single variable.
        """
        z = self._logits(skill, h)
        err = np.zeros_like(z)
        n = min(self._fresh.get(skill, len(y)), len(y))
        stored = self._targets.get(skill)
        if stored is not None and len(stored) != len(y) - n:
            # The learner also calls this path outside observe() -- importance
            # accumulation passes the whole buffer at once. The split recorded
            # by the last observe does not describe that batch, so fall back to
            # plain cross-entropy rather than silently mismatching rows.
            n, stored = len(y), None

        p_new = self._softmax(z[:n])
        onehot = np.zeros_like(p_new)
        onehot[np.arange(n), y[:n].astype(int)] = 1.0
        err[:n] = p_new - onehot

        if n < len(y):
            if self.mode == "hard" or stored is None:
                p_old = self._softmax(z[n:])
                hot = np.zeros_like(p_old)
                hot[np.arange(len(p_old)), y[n:].astype(int)] = 1.0
                err[n:] = p_old - hot
            else:
                # Squared error on logits: pull the function back to what it was.
                err[n:] = self.alpha * (z[n:] - stored)
                if self.mode == "der++":
                    p_old = self._softmax(z[n:])
                    hot = np.zeros_like(p_old)
                    hot[np.arange(len(p_old)), y[n:].astype(int)] = 1.0
                    err[n:] += self.beta * (p_old - hot)

        err /= total
        return err.T @ h, err.sum(axis=0), err @ self.heads[skill]

    # ------------------------------------------------------------------
    def _replay_with_targets(self, skill: str, count: int):
        buffer = self._replay[skill]
        if not buffer.items or count <= 0:
            return None
        picks = self._rng.choice(len(buffer.items), size=min(count, len(buffer.items)), replace=False)
        chosen = [buffer.items[int(i)] for i in picks]
        return (
            np.asarray([c.inputs for c in chosen], dtype=float),
            np.asarray([c.target for c in chosen], dtype=float),
            np.stack([buffer.logits[int(i)] for i in picks]),
        )

    def _probe_inputs(self, x: np.ndarray) -> np.ndarray:
        """What gets stored as the input half of a rehearsal pair.

        If rehearsal works by transmitting the *function* rather than the data,
        the stored inputs are only probe points -- places where the old and new
        networks are compared -- and need not be real examples at all. That
        would matter well beyond this benchmark: a system that can preserve its
        own capability without retaining any real user data has a very
        different deployment story from one that cannot.

        ``noise`` tests the strong version (inputs carrying no data whatsoever)
        and ``shuffled`` the weak one (pixel statistics preserved, structure
        destroyed).
        """
        if self.probe == "real":
            return x
        if self.probe == "noise":
            return self._rng.uniform(x.min(), x.max(), size=x.shape)
        shuffled = x.copy()
        for row in shuffled:
            self._rng.shuffle(row)
        return shuffled

    def observe(self, skill: str, batch: Sequence[Sample]) -> None:
        if not batch:
            raise ValueError("batch must not be empty")
        self._ensure_head(skill)

        fresh_x = np.asarray([c.inputs for c in batch], dtype=float)
        fresh_y = np.asarray([c.target for c in batch], dtype=float)
        # Logits recorded at insertion time, which is what makes them a record
        # of the function as it stood when the example was current.
        stored_x = self._probe_inputs(fresh_x)
        snapshot = self._logits(skill, self._forward(stored_x)[-1])
        stored = [
            Sample(row, int(item.target)) for row, item in zip(stored_x, batch)
        ] if self.probe != "real" else list(batch)
        for item, row in zip(stored, snapshot):
            self._replay[skill].add(item, row.copy())

        replayed = self._replay_with_targets(skill, self.config.replay_per_step)
        if replayed is None:
            self._fresh[skill] = len(batch)
            self._targets[skill] = None
            self._step({skill: (fresh_x, fresh_y)})
            return

        past_x, past_y, past_z = replayed
        self._fresh[skill] = len(batch)
        self._targets[skill] = past_z
        self._step(
            {skill: (np.concatenate([fresh_x, past_x]), np.concatenate([fresh_y, past_y]))}
        )
