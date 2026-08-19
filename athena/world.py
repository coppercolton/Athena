"""Streams to predict.

These exist so the demos have something to be surprised by. The interesting
ones are non-stationary: a model that only ever sees one regime can be tuned
into looking good, while a model that has to survive the rug being pulled has
to actually adapt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np


@dataclass
class Regime:
    """One stable set of dynamics: a bank of sinusoids per channel."""

    name: str
    freqs: Sequence[float]
    phases: Sequence[float] | None = None
    amps: Sequence[float] | None = None

    def __post_init__(self) -> None:
        n = len(self.freqs)
        self.phases = list(self.phases) if self.phases is not None else [0.0] * n
        self.amps = list(self.amps) if self.amps is not None else [1.0] * n

    @property
    def n_channels(self) -> int:
        return len(self.freqs)

    def at(self, t: float) -> np.ndarray:
        f = np.asarray(self.freqs, dtype=float)
        p = np.asarray(self.phases, dtype=float)
        a = np.asarray(self.amps, dtype=float)
        return a * np.sin(f * t + p)


@dataclass
class SwitchingWorld:
    """Cycles through regimes, switching every ``dwell`` steps.

    The clock keeps running across switches rather than restarting, so the
    signal is genuinely discontinuous at the boundary -- exactly the situation
    where a model that averages over its whole history does badly and a model
    that notices volatility does well.
    """

    regimes: Sequence[Regime]
    dwell: int = 600
    noise: float | Sequence[float] = 0.0
    seed: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.regimes:
            raise ValueError("need at least one regime")
        widths = {r.n_channels for r in self.regimes}
        if len(widths) != 1:
            raise ValueError("all regimes must have the same number of channels")
        self._rng = np.random.default_rng(self.seed)

    @property
    def n_channels(self) -> int:
        return self.regimes[0].n_channels

    def regime_index(self, t: int) -> int:
        return (t // self.dwell) % len(self.regimes)

    def regime_at(self, t: int) -> Regime:
        return self.regimes[self.regime_index(t)]

    def switch_points(self, n_steps: int) -> list[int]:
        return [t for t in range(1, n_steps) if self.regime_index(t) != self.regime_index(t - 1)]

    def at(self, t: int) -> np.ndarray:
        obs = self.regime_at(t).at(t)
        sigma = np.asarray(self.noise, dtype=float)
        if np.any(sigma > 0):
            obs = obs + self._rng.normal(0.0, 1.0, size=obs.shape) * sigma
        return obs

    def stream(self, n_steps: int) -> Iterator[np.ndarray]:
        for t in range(n_steps):
            yield self.at(t)


def smooth_world(n_channels: int = 4, seed: int = 0, **kwargs) -> SwitchingWorld:
    """A single stationary regime -- the easy case, for baseline comparisons."""
    rng = np.random.default_rng(seed)
    freqs = rng.uniform(0.03, 0.17, size=n_channels)
    phases = rng.uniform(0.0, 2 * np.pi, size=n_channels)
    return SwitchingWorld([Regime("steady", freqs, phases)], dwell=10**9, seed=seed, **kwargs)


def shifting_world(n_channels: int = 4, n_regimes: int = 3, dwell: int = 600, seed: int = 0):
    """Several unrelated regimes in rotation."""
    rng = np.random.default_rng(seed)
    regimes = []
    for k in range(n_regimes):
        freqs = rng.uniform(0.03, 0.25, size=n_channels)
        phases = rng.uniform(0.0, 2 * np.pi, size=n_channels)
        amps = rng.uniform(0.6, 1.0, size=n_channels)
        regimes.append(Regime(f"regime-{k}", freqs, phases, amps))
    return SwitchingWorld(regimes, dwell=dwell, seed=seed)


# ----------------------------------------------------------------------
# baselines
# ----------------------------------------------------------------------
def persistence_mse(observations: Sequence[np.ndarray]) -> list[float]:
    """"Tomorrow looks like today" -- a genuinely strong baseline on smooth data."""
    out = [float(np.mean(observations[0] ** 2))]
    for prev, cur in zip(observations, observations[1:]):
        out.append(float(np.mean((cur - prev) ** 2)))
    return out


def linear_mse(observations: Sequence[np.ndarray]) -> list[float]:
    """"Whatever just happened, keeps happening" -- constant-velocity extrapolation.

    The baseline that matters once the model is given velocity as an input,
    because linear extrapolation is then available to it for free. Beating
    persistence proves nothing; beating this means the model has actually
    learned something about the signal's structure.
    """
    out = [float(np.mean(observations[0] ** 2))]
    if len(observations) > 1:
        out.append(float(np.mean((observations[1] - observations[0]) ** 2)))
    for a, b, c in zip(observations, observations[1:], observations[2:]):
        out.append(float(np.mean((c - (2 * b - a)) ** 2)))
    return out


def zero_mse(observations: Sequence[np.ndarray]) -> list[float]:
    """"Nothing ever happens" -- the do-nothing floor."""
    return [float(np.mean(o * o)) for o in observations]
