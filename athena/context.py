"""Discrete context inference: "which world am I in?"

The continuous hierarchy in :mod:`athena.core` is good at predicting *within*
one set of dynamics and bad at the moment those dynamics change, because a
single set of weights can only hold one story at a time. Gradient descent on a
regime switch does the only thing it can: it slowly overwrites what it knew.

So the continuous model keeps a *bank* of transition operators and this gate
decides which of them is speaking. The mechanism that makes that work is less
obvious than it looks, and it took a few failed versions to find:

*   Inference over the gate is exact rather than gradient-based. Each
    operator's prediction error is a log-likelihood, so the posterior over
    operators is one softmax away, and evidence accumulates across timesteps
    through a standard filter. Recognising a regime therefore takes a few
    timesteps, not a training run.

*   At a change point, **learning stops**. This is the part that everything
    else depends on. If the incumbent expert keeps adapting while the model is
    still deciding what it is looking at, it simply relearns the new regime in
    place -- and then there is nothing left to recognise the old one with. The
    memory is destroyed by the very adaptation that makes the model look good
    in the moment.

*   Only after a probation window, with the evidence in, does the gate commit:
    switch to an expert that already explains the data, or recruit a fresh one
    if none does.

Novelty cannot be judged on a single timestep. When a regime changes, *every*
expert looks wrong, including the one holding the incoming regime, because the
continuous state beneath it still carries the outgoing regime's phase. "A world
I have never seen" and "a world I know, caught mid-turn" are the same picture
until you wait.
"""

from __future__ import annotations

from enum import Enum

import numpy as np


class GateState(str, Enum):
    STEADY = "steady"
    PROBATION = "probation"
    COMMITTED = "committed"


class ContextGate:
    """A Bayesian filter over which set of dynamics is currently active."""

    def __init__(
        self,
        n_experts: int,
        temperature: float = 0.05,
        lr: float = 0.02,
        stay: float = 0.98,
        floor: float = 1e-3,
        change_factor: float = 6.0,
        probation: int = 40,
        commit_steps: int = 60,
        fresh_claims: float = 25.0,
        seed: int = 0,
    ) -> None:
        if n_experts < 1:
            raise ValueError("need at least one expert")
        self.k = n_experts
        self.temperature = float(temperature)
        self.lr = float(lr)
        self.floor = float(floor)
        self.change_factor = float(change_factor)
        self.probation = int(probation)
        self.commit_steps = int(commit_steps)
        self.fresh_claims = float(fresh_claims)

        self.belief = np.full(n_experts, 1.0 / n_experts)
        self.prev_belief = self.belief.copy()

        # Column-stochastic transition matrix over contexts: A[k, j] is
        # P(context k now | context j before). Initialised to "regimes are
        # sticky", which is true of most worlds and a much better starting
        # guess than "anything can follow anything".
        off = (1.0 - stay) / max(n_experts - 1, 1)
        self.A = np.full((n_experts, n_experts), off)
        np.fill_diagonal(self.A, stay if n_experts > 1 else 1.0)

        # Raw posterior mass is the wrong measure of commitment: early on the
        # belief is near-uniform, so every expert would accrue mass without any
        # of them having claimed anything. Only a decisive posterior counts.
        self.usage = np.zeros(n_experts)
        self.claimed = np.zeros(n_experts)

        self.state = GateState.STEADY
        self.baseline: float | None = None
        self.allocations: list[int] = []
        self.switches: list[int] = []

        self._left = 0
        self._locked = -1
        self._evidence = np.zeros(n_experts)
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    @property
    def learning_enabled(self) -> bool:
        """False while the gate is still deciding what it is looking at."""
        return self.state is not GateState.PROBATION

    @property
    def entropy(self) -> float:
        b = self.belief
        return float(-(b * np.log(b + 1e-12)).sum())

    @property
    def active(self) -> int:
        return int(self.belief.argmax())

    def prior(self) -> np.ndarray:
        """Predicted context distribution before this timestep's evidence."""
        p = self.A @ self.belief
        return p / (p.sum() + 1e-12)

    # ------------------------------------------------------------------
    def _one_hot(self, k: int) -> np.ndarray:
        post = np.full(self.k, self.floor)
        post[k] = 1.0
        return post / post.sum()

    def _softmax_posterior(self, expert_error: np.ndarray) -> np.ndarray:
        logp = np.log(self.prior() + 1e-12) - expert_error / (2.0 * self.temperature)
        logp -= logp.max()
        post = np.exp(logp)
        post /= post.sum() + 1e-12
        # A floor keeps every expert marginally alive. An expert whose
        # posterior reaches exactly zero can never be selected again, however
        # well it would fit, because it stops accumulating evidence.
        post = np.maximum(post, self.floor)
        return post / post.sum()

    def infer(self, expert_error: np.ndarray) -> np.ndarray:
        """Posterior over experts given each one's precision-weighted error."""
        expert_error = np.asarray(expert_error, dtype=float)
        best = float(expert_error.min())

        if self.state is GateState.PROBATION:
            self._evidence -= expert_error / (2.0 * self.temperature)
            self._left -= 1
            if self._left <= 0:
                return self._resolve()
            # Still predict as well as we can while deciding.
            ev = self._evidence - self._evidence.max()
            post = np.maximum(np.exp(ev), self.floor)
            return post / post.sum()

        if self.state is GateState.COMMITTED:
            self._left -= 1
            if self._left <= 0:
                self.state = GateState.STEADY
            # A freshly recruited expert is protected while it learns. Without
            # this it loses the very next comparison to whichever expert is
            # best trained overall, and recruitment accomplishes nothing.
            return self._one_hot(self._locked)

        if self.k > 1 and self.baseline is not None and best > self.change_factor * self.baseline:
            # Something changed. Stop learning and start gathering evidence.
            self.state = GateState.PROBATION
            self._left = self.probation
            self._evidence = np.zeros(self.k)
            return self.belief.copy()

        return self._softmax_posterior(expert_error)

    def _resolve(self) -> np.ndarray:
        """Probation is over: switch to a known expert, or recruit a new one."""
        best_k = int(self._evidence.argmax())
        # Convert accumulated log-evidence back into a mean per-step error.
        mean_error = -2.0 * self.temperature * float(self._evidence[best_k]) / self.probation
        explained = self.baseline is None or mean_error <= self.change_factor * self.baseline
        spare = self.claimed.min() < self.fresh_claims

        if explained or not spare:
            self.state = GateState.STEADY
            self.switches.append(best_k)
            return self._one_hot(best_k)

        target = int(np.lexsort((self.usage, self.claimed))[0])
        self.state = GateState.COMMITTED
        self._left = self.commit_steps
        self._locked = target
        self.allocations.append(target)
        return self._one_hot(target)

    # ------------------------------------------------------------------
    def commit(self, posterior: np.ndarray, best_error: float | None = None) -> None:
        """Accept a posterior as this timestep's belief and learn from it."""
        self.prev_belief = self.belief
        self.belief = posterior
        self.usage += posterior
        if float(posterior.max()) > 0.5:
            self.claimed[int(posterior.argmax())] += 1.0

        if best_error is not None and self.state is GateState.STEADY:
            # Slow, and frozen while the gate is unsettled, so that a genuine
            # change stands out against it instead of being absorbed into it.
            self.baseline = (
                best_error
                if self.baseline is None
                else self.baseline + 0.01 * (best_error - self.baseline)
            )

        if self.k > 1 and self.state is GateState.STEADY:
            self.A += self.lr * np.outer(self.belief, self.prev_belief)
            self.A = np.maximum(self.A, 1e-4)
            self.A /= self.A.sum(axis=0, keepdims=True)
