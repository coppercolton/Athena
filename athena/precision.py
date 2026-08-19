"""Precision and volatility estimation.

In predictive processing, a prediction error is not worth the same everywhere.
An error on a channel that is normally reliable is *informative*; the same error
on a channel that is always noisy is not. Precision is the inverse variance of
a channel's error history, and it gates how strongly that error is allowed to
revise beliefs and weights.

Volatility is the second-order version of the same question: not "how noisy is
this channel?" but "has the world itself changed?". When errors get suddenly
and persistently larger than their own recent history, the right response is
not to distrust the sensor but to learn faster.
"""

from __future__ import annotations

import numpy as np


class Precision:
    """Per-unit inverse-variance estimate, tracked online.

    Maintains an exponential moving average of squared prediction error for
    each unit and exposes its (clipped) reciprocal as a precision vector.
    """

    def __init__(
        self,
        size: int,
        rate: float = 0.02,
        floor: float = 1e-2,
        ceiling: float = 1e3,
        init_var: float = 1.0,
    ) -> None:
        self.var = np.full(size, float(init_var))
        self.rate = float(rate)
        self.floor = float(floor)
        self.ceiling = float(ceiling)

    @property
    def value(self) -> np.ndarray:
        """Precision vector, pi = 1 / variance, clipped to a sane range."""
        return np.clip(1.0 / (self.var + 1e-8), self.floor, self.ceiling)

    def update(self, err: np.ndarray) -> None:
        self.var += self.rate * (err * err - self.var)
        np.clip(self.var, 1e-6, 1e6, out=self.var)

    @property
    def relative(self) -> np.ndarray:
        """Precision renormalised to mean 1.

        The *ratios* between units are the meaningful part -- "trust channel 3
        more than channel 7". The absolute magnitude is not, and letting it
        through is actively harmful: precision rises as the model improves, so
        a raw-precision update rule silently multiplies its own learning rate
        by a growing number until it oscillates. Normalising removes that
        feedback loop while keeping the weighting.
        """
        v = self.value
        return v / (float(v.mean()) + 1e-12)

    @property
    def mean_precision(self) -> float:
        return float(self.value.mean())

    def weighted(self, err: np.ndarray) -> np.ndarray:
        """Precision-weighted error, the quantity that actually propagates."""
        return self.relative * err


class VolatilityTracker:
    """Detects regime change by comparing fast and slow surprise averages.

    A single scalar per model. When the fast average of surprise runs well
    above the slow average, the world has probably moved and the model should
    temporarily raise its learning rate rather than average the change away.
    This is the knob that biology appears to implement with neuromodulators.
    """

    def __init__(
        self,
        fast_rate: float = 0.15,
        slow_rate: float = 0.005,
        max_gain: float = 8.0,
    ) -> None:
        self.fast = 0.0
        self.slow = 0.0
        self.fast_rate = float(fast_rate)
        self.slow_rate = float(slow_rate)
        self.max_gain = float(max_gain)
        self._seen = 0

    def update(self, surprise: float) -> None:
        surprise = float(surprise)
        if self._seen == 0:
            self.fast = self.slow = surprise
        else:
            self.fast += self.fast_rate * (surprise - self.fast)
            self.slow += self.slow_rate * (surprise - self.slow)
        self._seen += 1

    @property
    def gain(self) -> float:
        """Learning-rate multiplier in [1, max_gain]."""
        if self._seen < 2 or self.slow <= 1e-12:
            return 1.0
        return float(np.clip(self.fast / self.slow, 1.0, self.max_gain))
