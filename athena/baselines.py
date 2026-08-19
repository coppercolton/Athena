"""Strong online baselines for honest continual-prediction experiments.

If Athena cannot beat a small model matched to the structure of a stream, the
right conclusion is that the benchmark is solved -- not that the baseline
should be hidden. These predictors obey the same prequential contract as
Athena: predict first, reveal one observation, then update exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BaselineReport:
    """One prediction made before an online baseline saw the observation."""

    prediction: np.ndarray
    observation: np.ndarray
    mse: float


class OnlineRLS:
    """Per-channel recursive least-squares autoregression.

    RLS is an intentionally demanding baseline for smooth numeric streams. It
    learns an order-``p`` linear recurrence online with no replay buffer and
    can adapt to drift through a forgetting factor below one.

    Parameters
    ----------
    n_channels:
        Width of each observation.
    order:
        Number of past values used per channel. ``order=2`` can represent a
        noiseless sinusoid exactly.
    forgetting:
        Exponential retention in ``(0, 1]``. One is stationary RLS; values such
        as ``0.995`` adapt more quickly when dynamics drift.
    ridge:
        Initial regularisation. Smaller values express a broader prior.
    """

    def __init__(
        self,
        n_channels: int,
        order: int = 2,
        forgetting: float = 1.0,
        ridge: float = 1e-3,
    ) -> None:
        if n_channels < 1:
            raise ValueError("n_channels must be >= 1")
        if order < 1:
            raise ValueError("order must be >= 1")
        if not 0.0 < forgetting <= 1.0:
            raise ValueError("forgetting must be in (0, 1]")
        if ridge <= 0.0:
            raise ValueError("ridge must be > 0")

        self.n_channels = int(n_channels)
        self.order = int(order)
        self.forgetting = float(forgetting)
        self.ridge = float(ridge)
        self.theta = np.zeros((self.n_channels, self.order + 1))
        initial_covariance = np.eye(self.order + 1) / self.ridge
        self.covariance = np.repeat(
            initial_covariance[None, :, :], self.n_channels, axis=0
        )
        self.history: list[np.ndarray] = []
        self.step_count = 0

    def _features_from(self, history: list[np.ndarray], channel: int) -> np.ndarray:
        lagged = [history[-lag][channel] for lag in range(1, self.order + 1)]
        return np.asarray([*lagged, 1.0], dtype=float)

    def _features(self, channel: int) -> np.ndarray:
        return self._features_from(self.history, channel)

    def predict(self, horizon: int = 1) -> np.ndarray:
        """Predict one or more steps ahead without changing the model."""
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        history = [observation.copy() for observation in self.history]
        prediction = np.zeros(self.n_channels)
        for _ in range(horizon):
            if not history:
                prediction = np.zeros(self.n_channels)
            elif len(history) < self.order:
                prediction = history[-1].copy()
            else:
                prediction = np.asarray(
                    [
                        self.theta[c] @ self._features_from(history, c)
                        for c in range(self.n_channels)
                    ]
                )
            history.append(prediction.copy())
            if len(history) > self.order:
                history.pop(0)
        return prediction

    def observe(
        self,
        observation: np.ndarray,
        learn: bool = True,
        responsibility: float = 1.0,
    ) -> BaselineReport:
        """Score one unseen observation and optionally update the recurrence.

        ``responsibility`` is a fractional sample weight used when several RLS
        memories compete under a context gate. Ordinary baseline use leaves it
        at one.
        """
        observation = np.asarray(observation, dtype=float).reshape(-1)
        if observation.shape != (self.n_channels,):
            raise ValueError(
                f"observation has {observation.size} dims, "
                f"model expects {self.n_channels}"
            )
        if responsibility < 0.0:
            raise ValueError("responsibility must be >= 0")

        prediction = self.predict()
        error = observation - prediction
        if learn and responsibility > 0.0 and len(self.history) >= self.order:
            for channel in range(self.n_channels):
                feature = self._features(channel)
                projected = self.covariance[channel] @ feature
                gain = projected / (
                    self.forgetting / responsibility + float(feature @ projected)
                )
                self.theta[channel] += gain * error[channel]
                self.covariance[channel] = (
                    self.covariance[channel]
                    - np.outer(gain, feature) @ self.covariance[channel]
                ) / self.forgetting

        self.history.append(observation.copy())
        if len(self.history) > self.order:
            self.history.pop(0)
        self.step_count += 1
        return BaselineReport(
            prediction=prediction,
            observation=observation.copy(),
            mse=float(np.mean(error * error)),
        )

    def run(self, stream, learn: bool = True) -> list[BaselineReport]:
        """Consume a stream under the same predict-then-learn protocol."""
        return [self.observe(observation, learn=learn) for observation in stream]
