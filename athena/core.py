"""Athena: a hierarchical predictive-processing model.

The whole system is one loop:

    1.  predict   -- roll beliefs forward in time and generate top-down to the
                     senses, producing an expectation of the *next* observation
                     before that observation arrives.
    2.  compare   -- when the observation arrives, take the difference.
    3.  settle    -- let the latent beliefs relax until they explain the
                     observation, using precision-weighted error as the force.
    4.  learn     -- nudge the weights in the direction the settled errors
                     point, and update the precision estimates.

Nothing here is trained offline on a dataset. There is no separate training
phase, no backpropagation through time, and no replay buffer. Every update is
local to a pair of adjacent levels and happens on the same timestep as the
observation that caused it, which is what lets the model keep improving for as
long as it is running.

Notation
--------
``x[i]``   belief state of level i (level 0 is sensory, clamped to observation)
``W[i]``   generative weights, level i -> level i-1 ("what I expect to see
           below, given what I believe up here")
``M[i]``   transition weights, level i at t-1 -> level i at t ("what I expect
           to believe next, given what I believe now")

Two errors are computed at every level: a *spatial* error against the top-down
prediction from the level above, and a *temporal* error against the level's own
forward prediction from the previous timestep. Free energy is the sum of both,
each weighted by its own precision.

The top level plays a special role. Its states are read through a softmax as a
context code -- a distribution over "which world am I in" -- and that code
gates which transition operator the levels below it use. Inference over the
context is fast, so a returning regime is *recognised* within a few timesteps,
while learning the operators themselves stays slow. That split is the
difference between a model that adapts and a model that remembers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .context import ContextGate
from .precision import Precision, VolatilityTracker


def act_grad(x: np.ndarray) -> np.ndarray:
    t = np.tanh(x)
    return 1.0 - t * t


@dataclass
class Config:
    """Hyperparameters. ``sizes[0]`` must match the observation dimension."""

    sizes: Sequence[int]

    # How many orders of motion the sensory level represents: 1 for raw values
    # only, 2 to add velocity, 3 to add acceleration. See Athena._embed.
    orders: int = 2

    # inference (perception): fast, no weight changes
    inference_steps: int = 16
    inference_rate: float = 0.12
    state_clip: float = 8.0

    # learning: slow, local, always on
    lr: float = 3e-1
    lr_transition: float = 1.0
    weight_decay: float = 1e-5
    weight_clip: float = 1.5
    spectral_clip: float = 2.5
    evidence_rate: float = 0.002
    lr_floor: float = 0.3

    # precision and volatility
    precision_rate: float = 0.02
    precision_floor: float = 1e-2
    precision_ceiling: float = 1e3
    top_prior_precision: float = 0.05
    surprise_gain_max: float = 8.0

    # hierarchy and context
    tau: float = 2.2
    experts: int = 6
    context_temp: float = 0.05
    context_lr: float = 0.02

    seed: int = 0

    def __post_init__(self) -> None:
        if len(self.sizes) < 2:
            raise ValueError("need at least a sensory level and one latent level")
        if any(n < 1 for n in self.sizes):
            raise ValueError("all level sizes must be >= 1")
        if self.orders < 1:
            raise ValueError("orders must be >= 1")


@dataclass
class StepReport:
    """What happened on one timestep, for logging and for the demos."""

    step: int
    prediction: np.ndarray
    observation: np.ndarray
    mse: float
    surprise: float
    free_energy: float
    gain: float
    context: np.ndarray
    level_error: list[float] = field(default_factory=list)


class Athena:
    """A hierarchy of levels that continually predicts its own input.

    Typical use::

        model = Athena(Config(sizes=[4, 24, 8]))
        for obs in stream:
            guess = model.predict()      # before seeing obs
            report = model.observe(obs)  # after seeing obs
    """

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        rng = self.rng

        sizes = list(config.sizes)
        # The sensory level is widened to hold generalised coordinates: the
        # observation together with its recent motion. Everything above it is
        # unchanged.
        self.n_obs = sizes[0]
        self.orders = int(config.orders)
        sizes[0] = self.n_obs * self.orders
        self.sizes = sizes
        self.n_levels = len(sizes)
        self.top = self.n_levels - 1

        self.n_experts = max(1, int(config.experts))
        self.gate = ContextGate(
            self.n_experts,
            temperature=config.context_temp,
            lr=config.context_lr,
            seed=config.seed,
        )

        self.x = [np.zeros(n) for n in sizes]
        self.x_prev = [np.zeros(n) for n in sizes]

        # W[i] maps level i down to level i-1. W[0] is unused (nothing below).
        # Generative weights are banked per expert too. Gating only the
        # transition operator is not enough: a regime differs not just in how
        # its state evolves but in how that state shows up in the senses, and
        # a shared generative map forces every regime through one compromise.
        self.W: list[np.ndarray | None] = [None]
        self.b: list[np.ndarray | None] = [None]
        for i in range(1, self.n_levels):
            scale = 1.0 / np.sqrt(sizes[i])
            self.W.append(
                rng.normal(0.0, scale, size=(self.n_experts, sizes[i - 1], sizes[i]))
            )
            self.b.append(np.zeros((self.n_experts, sizes[i - 1])))

        # Transition banks. Identity is "expect no change", which is the right
        # thing to believe before seeing any data; the small perturbation is
        # there to break symmetry, because identical experts receiving
        # proportional updates would stay identical forever.
        self.M: list[np.ndarray] = []
        for i in range(self.n_levels):
            bank = np.stack([np.eye(sizes[i]) for _ in range(self.n_experts)])
            bank += rng.normal(0.0, 0.05, size=bank.shape)
            self.M.append(bank)

        self.pi_spatial = [
            Precision(n, config.precision_rate, config.precision_floor, config.precision_ceiling)
            for n in sizes
        ]
        self.pi_temporal = [
            Precision(n, config.precision_rate, config.precision_floor, config.precision_ceiling)
            for n in sizes
        ]
        # Separate precisions for the two *forecast* pathways. The spatial and
        # temporal precisions above are estimated after settling, when the
        # level above has already seen the observation -- they measure how well
        # the model reconstructs, not how well it predicts. Weighting a
        # forecast by a reconstruction score systematically over-trusts the
        # top-down path, so the fusion gets its own estimates, scored on the
        # same one-step-ahead predictions it is actually combining.
        self.pi_fuse_top = [
            Precision(n, config.precision_rate, config.precision_floor, config.precision_ceiling)
            for n in sizes
        ]
        self.pi_fuse_time = [
            Precision(n, config.precision_rate, config.precision_floor, config.precision_ceiling)
            for n in sizes
        ]

        self.volatility = VolatilityTracker(max_gain=config.surprise_gain_max)
        # Accumulated evidence about the parameters; see _learning_scale.
        self.evidence = 0.0
        self.step_count = 0

        self._err_spatial = [np.zeros(n) for n in sizes]
        self._err_temporal = [np.zeros(n) for n in sizes]
        self._a_prev = [np.zeros(n) for n in sizes]
        self._history: list[np.ndarray] = []
        self._pred_top = [np.zeros(n) for n in sizes]
        self._pred_time = [np.zeros(n) for n in sizes]

    # ------------------------------------------------------------------
    # timescales, gains, context
    # ------------------------------------------------------------------
    def _act(self, level: int, x: np.ndarray) -> np.ndarray:
        """Level 0 is linear; every level above it is squashed.

        The sensory level holds physical quantities -- a reading and its rate
        of change -- and squashing those through a tanh destroys the very
        relationship the model is trying to exploit: with a compressed state,
        even "keep moving at the current speed" is no longer expressible as a
        linear step. Latent levels are a different matter. There the state is a
        code the model invents, bounding it is what keeps the hierarchy stable,
        and nothing is lost by doing so.
        """
        return x if level == 0 else np.tanh(x)

    def _act_grad(self, level: int, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x) if level == 0 else act_grad(x)

    def _rate(self, level: int, base: float) -> float:
        """Higher levels move more slowly.

        This one line is what makes the hierarchy represent anything
        interesting: level 1 can chase the signal, but level 3 physically
        cannot, so the only states it can settle into are ones that stay useful
        across many timesteps -- context, regime, slow structure.
        """
        return base / (self.cfg.tau**level)

    def _learning_scale(self, gain: float) -> float:
        """How large a step the weights should take right now.

        A model that runs forever cannot keep a fixed learning rate: a constant
        step size means constant gradient noise, so the parameters random-walk
        around the solution and predictions decay back toward mediocre. The fix
        is the same idea as everything else here -- track confidence. Each
        observation adds evidence about the parameters and shrinks the step;
        each surprise discounts that evidence and re-opens learning.

        The floor matters as much as the decay. Evidence is never allowed to
        drive the rate to zero, because a model that has stopped learning
        cannot notice that it should start again.
        """
        self.evidence = self.evidence / max(gain, 1.0) + 1.0
        return max(self.cfg.lr_floor, 1.0 / (1.0 + self.cfg.evidence_rate * self.evidence))

    @property
    def channel_precision(self) -> np.ndarray:
        """How much the model trusts each raw input channel.

        This reads the *forecast* precision rather than the reconstruction
        precision, and the distinction is the whole measurement. A latent level
        that has already seen the observation can fit noise as happily as
        signal, so reconstruction error says almost nothing about how reliable
        a channel is. Nobody can forecast noise, so forecast error says
        everything: on a stream where half the channels are buried in noise,
        this separates them by two orders of magnitude while the reconstruction
        precision barely moves.
        """
        return self.pi_temporal[0].value[: self.n_obs].copy()

    @property
    def context(self) -> np.ndarray:
        """Current belief about which set of dynamics is active."""
        return self.gate.belief.copy()

    def _generative(self, level: int, ctx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Generative weights and bias mapping ``level`` down, under ``ctx``."""
        if self.n_experts == 1:
            return self.W[level][0], self.b[level][0]
        return (
            np.einsum("k,kab->ab", ctx, self.W[level]),
            ctx @ self.b[level],
        )

    def _transition(self, level: int, ctx: np.ndarray) -> np.ndarray:
        """The transition operator in force at ``level`` under context ``ctx``."""
        bank = self.M[level]
        if bank.shape[0] == 1:
            return bank[0]
        return np.einsum("k,kab->ab", ctx, bank)

    def _stream_weights(self, level: int) -> tuple[float, float]:
        """How much a level should trust context from above vs. its own momentum.

        Returns two weights summing to 2 (so their average is 1). This keeps
        the relative reliability of the two error streams while preventing the
        absolute precision scale from leaking into step sizes.
        """
        ms = self.pi_spatial[level].mean_precision
        mt = self.pi_temporal[level].mean_precision
        total = ms + mt + 1e-12
        return 2.0 * ms / total, 2.0 * mt / total

    def _eps_spatial(self, level: int) -> np.ndarray:
        if level == self.top:
            return self.cfg.top_prior_precision * self._err_spatial[level]
        ws, _ = self._stream_weights(level)
        return ws * self.pi_spatial[level].weighted(self._err_spatial[level])

    def _eps_temporal(self, level: int) -> np.ndarray:
        _, wt = self._stream_weights(level)
        return wt * self.pi_temporal[level].weighted(self._err_temporal[level])

    # ------------------------------------------------------------------
    # prediction
    # ------------------------------------------------------------------
    def _fuse(self, i: int, top_down: np.ndarray, temporal: np.ndarray) -> np.ndarray:
        """Combine the two priors by their precisions.

        A level gets two independent opinions about what it should be next: one
        from above (context) and one from its own past (momentum). The correct
        way to merge two Gaussian opinions is to average them weighted by
        confidence, which is exactly what precision is.
        """
        ps = self.pi_fuse_top[i].value
        pt = self.pi_fuse_time[i].value
        return (ps * top_down + pt * temporal) / (ps + pt)

    def _rollout(
        self, states: list[np.ndarray], ctx: np.ndarray, record: bool = False
    ) -> list[np.ndarray]:
        """One step of the generative model: roll time forward, then generate down."""
        ops = [self._transition(i, ctx) for i in range(self.n_levels)]
        nxt: list[np.ndarray] = [np.zeros(n) for n in self.sizes]
        # The top has no context above it, so its forward roll stands alone.
        nxt[self.top] = ops[self.top] @ self._act(self.top, states[self.top])
        for i in range(self.top - 1, -1, -1):
            temporal = ops[i] @ self._act(i, states[i])
            W, b = self._generative(i + 1, ctx)
            top_down = W @ self._act(i + 1, nxt[i + 1]) + b
            nxt[i] = self._fuse(i, top_down, temporal)
            if record:
                self._pred_top[i] = top_down
                self._pred_time[i] = temporal
        if record:
            self._pred_time[self.top] = nxt[self.top].copy()
            self._pred_top[self.top] = nxt[self.top].copy()
        return [np.clip(s, -self.cfg.state_clip, self.cfg.state_clip) for s in nxt]

    def _embed(self, obs: np.ndarray) -> np.ndarray:
        """Generalised coordinates: the observation and its motion.

        A level whose state is a function of the current observation alone
        cannot predict a moving signal, because position at time t simply does
        not determine position at time t+1 -- you need velocity, and velocity
        is not in the picture. The model can in principle discover a velocity
        code in its latents, but nothing in a one-step local learning rule
        pushes it to: the latent is warm-started at its own prediction and
        barely moved, so its temporal error is near zero by construction and
        the transition weights receive almost no signal.

        Handing the sensory level explicit orders of motion removes the
        problem instead of hoping the model routes around it, and it is what
        the free-energy literature does for exactly this reason. The
        derivatives are computed from the model's own past inputs, so no
        information from the future is involved.
        """
        self._history.append(np.asarray(obs, dtype=float))
        if len(self._history) > self.orders:
            self._history.pop(0)
        parts = [self._history[-1]]
        series = list(self._history)
        for _ in range(self.orders - 1):
            if len(series) < 2:
                parts.append(np.zeros(self.n_obs))
                continue
            series = [b - a for a, b in zip(series, series[1:])]
            parts.append(series[-1])
        return np.concatenate(parts)

    def predict(self, horizon: int = 1, context: np.ndarray | None = None) -> np.ndarray:
        """Predict the observation ``horizon`` steps ahead, before seeing it.

        With ``horizon > 1`` the model runs on its own predictions and consumes
        no observations, which is the same machinery as imagination.
        """
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        ctx = self.gate.prior() if context is None else context
        states = [x.copy() for x in self.x]
        for step in range(horizon):
            states = self._rollout(states, ctx, record=(step == 0))
        return states[0][: self.n_obs].copy()

    def _expert_errors(self, embedded: np.ndarray) -> np.ndarray:
        """How badly each expert alone would have predicted this observation.

        This is the likelihood term the gate needs. Each expert is scored by
        running the whole generative model with that expert forced, which asks
        the question that actually matters -- "would this set of dynamics have
        produced what I just saw?" -- rather than a proxy on some latent.
        """
        errors = np.zeros(self.n_experts)
        pi = self.pi_spatial[0].relative
        for k in range(self.n_experts):
            one_hot = np.zeros(self.n_experts)
            one_hot[k] = 1.0
            pred = self._rollout(self.x_prev, one_hot)[0]
            d = embedded - pred
            errors[k] = float(np.sum(pi * d * d))
        return errors

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    def _refresh_errors(self, temporal_prior: list[np.ndarray], ctx: np.ndarray) -> None:
        for i in range(self.top):
            W, b = self._generative(i + 1, ctx)
            self._err_spatial[i] = self.x[i] - (W @ self._act(i + 1, self.x[i + 1]) + b)
        # The top level is pinned toward zero by a weak prior; without it
        # nothing stops the highest beliefs from drifting off to infinity.
        self._err_spatial[self.top] = self.x[self.top]
        for i in range(self.n_levels):
            self._err_temporal[i] = self.x[i] - temporal_prior[i]

    def _settle(self, embedded: np.ndarray, ctx: np.ndarray) -> None:
        """Relax the latent states until they explain the observation.

        Gradient descent on free energy with respect to the beliefs, with the
        sensory level pinned to reality. Weights are untouched here --
        perception first, learning after.
        """
        cfg = self.cfg
        self.x[0] = embedded.copy()
        a_prev = self._a_prev
        temporal_prior = [self._transition(i, ctx) @ a_prev[i] for i in range(self.n_levels)]

        for _ in range(cfg.inference_steps):
            self._refresh_errors(temporal_prior, ctx)
            # Gradients are computed for every level before any level moves, so
            # that one settling step is a step of the whole system rather than
            # a sweep in which upper levels react to already-updated lower ones.
            grads = []
            for i in range(1, self.n_levels):
                eps_s = self._eps_spatial(i)
                eps_t = self._eps_temporal(i)
                eps_below = self._eps_spatial(i - 1)
                # Pulled by its own unexplained error, pushed by the error it
                # is responsible for at the level below.
                W, _ = self._generative(i, ctx)
                grads.append(
                    eps_s + eps_t - self._act_grad(i, self.x[i]) * (W.T @ eps_below)
                )

            for i in range(1, self.n_levels):
                self.x[i] -= self._rate(i, cfg.inference_rate) * grads[i - 1]
                np.clip(self.x[i], -cfg.state_clip, cfg.state_clip, out=self.x[i])

        # Errors at the settled point are what learning consumes.
        self._refresh_errors(temporal_prior, ctx)

    # ------------------------------------------------------------------
    # learning
    # ------------------------------------------------------------------
    def _learn(self, scale: float) -> None:
        """Local Hebbian updates: presynaptic activity times postsynaptic error."""
        cfg = self.cfg
        resp = self.gate.belief
        for i in range(1, self.n_levels):
            eps_below = self._eps_spatial(i - 1)
            a = self._act(i, self.x[i])
            lr = self._rate(i - 1, cfg.lr) * scale / (float(a @ a) + 1.0)
            update = np.outer(eps_below, a)
            if self.n_experts == 1:
                self.W[i][0] += lr * update
                self.b[i][0] += lr * eps_below
            else:
                self.W[i] += lr * resp[:, None, None] * update
                self.b[i] += lr * resp[:, None] * eps_below
            self.W[i] *= 1.0 - cfg.weight_decay
            for k in range(self.n_experts):
                self._project_columns(self.W[i][k])

        for i in range(self.n_levels):
            eps_t = self._eps_temporal(i)
            a = self._a_prev[i]
            # Normalised step: dividing by the presynaptic energy makes the
            # effective rate independent of how large the level's activity
            # happens to be, which is what lets one learning rate work at every
            # level instead of sitting just under the divergence threshold of
            # whichever level is loudest.
            lr = self._rate(i, cfg.lr_transition) * scale / (float(a @ a) + 1.0)
            update = np.outer(eps_t, a)
            if self.M[i].shape[0] == 1:
                self.M[i][0] += lr * update
            else:
                # Responsibility-weighted: an expert only learns in proportion
                # to how much it was being listened to. This is what keeps
                # regimes in separate operators instead of averaging them into
                # one blurry compromise.
                self.M[i] += lr * resp[:, None, None] * update
            self.M[i] *= 1.0 - cfg.weight_decay
            self._stabilise(i)

    def _project_columns(self, W: np.ndarray) -> None:
        """Bound the norm of each generative weight column, in place.

        Without this the hierarchy has a free scale: a level can shrink its own
        activity toward zero while the weights below grow to compensate, and
        the pair drifts along that ridge indefinitely. Predictions stay briefly
        correct and then fall apart, because a vanishing state carries no
        information while a growing weight matrix amplifies whatever noise is
        left. Fixing the weight scale forces the *states* to carry the signal.
        """
        limit = self.cfg.weight_clip
        norms = np.linalg.norm(W, axis=0)
        over = norms > limit
        if np.any(over):
            W[:, over] *= limit / norms[over]

    def _stabilise(self, i: int) -> None:
        """Keep each transition operator non-explosive.

        A learned recurrence whose spectral radius exceeds one runs away given
        enough timesteps, and this model is meant to run indefinitely.

        The bound is on the matrix norm rather than the spectral radius, and
        the difference matters in both directions. Bounding the radius is the
        textbook stability condition, but it permits large transient
        amplification in a non-normal operator, and with the weights moving
        every step that transient is what actually blows up. Bounding the norm
        is stricter and steadier -- but it has to be set with room to spare:
        "carry the current velocity forward" is the block matrix [[I, I], [0,
        I]], whose norm is about 1.6, so a limit near 1 silently forbids the
        single most useful operator the model could learn. Nothing diverges;
        the model just quietly never gets good.
        """
        limit = self.cfg.spectral_clip
        for k in range(self.M[i].shape[0]):
            norm = float(np.linalg.norm(self.M[i][k], 2))
            if norm > limit:
                self.M[i][k] *= limit / norm

    # ------------------------------------------------------------------
    # the loop
    # ------------------------------------------------------------------
    def observe(self, obs: np.ndarray, learn: bool = True) -> StepReport:
        """Take in one observation: predict, compare, settle, learn."""
        obs = np.asarray(obs, dtype=float).reshape(-1)
        if obs.shape[0] != self.n_obs:
            raise ValueError(
                f"observation has {obs.shape[0]} dims, model expects {self.n_obs}"
            )

        # Predict before looking, and score against the raw observation so the
        # number is comparable with any other forecaster.
        prediction = self.predict()
        err = obs - prediction
        mse = float(np.mean(err * err))
        embedded = self._embed(obs)
        # Raw (un-normalised) precision here on purpose: surprise should scale
        # with confidence, so a model that was sure and wrong reports a large
        # number. Only the *update rules* use normalised precision.
        surprise = float(np.sum(self.pi_spatial[0].value[: self.n_obs] * err * err))

        self.x_prev = [x.copy() for x in self.x]
        self._a_prev = [self._act(i, x) for i, x in enumerate(self.x_prev)]

        # Ask which set of dynamics just generated this observation, and let
        # the answer choose the operator the rest of the step runs on.
        expert_error = self._expert_errors(embedded)
        posterior = self.gate.infer(expert_error)
        self.gate.commit(posterior, best_error=float(expert_error.min()))
        ctx = self.gate.belief

        # Warm-start each latent at its own forward prediction, so settling is
        # a correction to a guess rather than a search from nothing.
        for i in range(1, self.n_levels):
            self.x[i] = self._transition(i, ctx) @ self._a_prev[i]

        self._settle(embedded, ctx)

        self.volatility.update(surprise)
        gain = self.volatility.gain
        scale = self._learning_scale(gain)
        # The gate withholds learning while it is deciding whether this is a
        # world it already knows. Adapting through that window is what destroys
        # the memory of the regime being left behind.
        if learn and self.gate.learning_enabled:
            self._learn(scale)

        for i in range(self.n_levels):
            self.pi_spatial[i].update(self._err_spatial[i])
            self.pi_temporal[i].update(self._err_temporal[i])
            # Score each forecast pathway against where the state actually
            # ended up, which is the comparison the fusion needs.
            self.pi_fuse_top[i].update(self.x[i] - self._pred_top[i])
            self.pi_fuse_time[i].update(self.x[i] - self._pred_time[i])

        free_energy = 0.0
        for i in range(self.n_levels):
            es, et = self._err_spatial[i], self._err_temporal[i]
            free_energy += 0.5 * float(np.sum(self.pi_spatial[i].value * es * es))
            free_energy += 0.5 * float(np.sum(self.pi_temporal[i].value * et * et))

        self.step_count += 1
        return StepReport(
            step=self.step_count,
            prediction=prediction,
            observation=obs.copy(),
            mse=mse,
            surprise=surprise,
            free_energy=free_energy,
            gain=gain,
            context=self.gate.belief.copy(),
            level_error=[float(np.linalg.norm(e)) for e in self._err_spatial],
        )

    def run(self, stream, learn: bool = True) -> list[StepReport]:
        """Consume an iterable of observations, returning one report each."""
        return [self.observe(obs, learn=learn) for obs in stream]
