"""Tests for Athena.

Run with pytest, or directly: ``python3 tests/test_athena.py``.

The interesting assertions here are the behavioural ones. A predictive model
is easy to write and easy to fool yourself about -- most of the bugs found
while building this one produced code that ran fine, converged nicely on a
signal simple enough to be predicted by accident, and did nothing on a real
one. So these tests check against baselines rather than against zero.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena import (  # noqa: E402
    Athena,
    Config,
    Regime,
    SwitchingWorld,
    linear_mse,
    persistence_mse,
    shifting_world,
)


def _stationary(n_channels: int = 4, seed: int = 3) -> SwitchingWorld:
    rng = np.random.default_rng(seed)
    return SwitchingWorld(
        [
            Regime(
                "steady",
                rng.uniform(0.03, 0.17, n_channels),
                rng.uniform(0.0, 2 * np.pi, n_channels),
            )
        ],
        dwell=10**9,
    )


# ----------------------------------------------------------------------
# construction and contracts
# ----------------------------------------------------------------------
def test_rejects_bad_config():
    for bad in (dict(sizes=[4]), dict(sizes=[4, 0, 3]), dict(sizes=[4, 8], orders=0)):
        try:
            Config(**bad)
        except ValueError:
            continue
        raise AssertionError(f"Config accepted invalid spec: {bad}")


def test_rejects_wrong_width_observation():
    model = Athena(Config(sizes=[4, 12, 6]))
    try:
        model.observe(np.zeros(5))
    except ValueError:
        return
    raise AssertionError("accepted an observation of the wrong width")


def test_prediction_precedes_observation():
    """predict() must not depend on the observation it is predicting."""
    model = Athena(Config(sizes=[3, 12, 6], seed=0))
    world = _stationary(3)
    for t in range(50):
        model.observe(world.at(t))
    first = model.predict()
    second = model.predict()
    assert np.allclose(first, second), "predict() is not side-effect free"
    assert first.shape == (3,)
    assert model.predict(horizon=5).shape == (3,)


def test_deterministic_given_seed():
    world = _stationary(3)
    obs = [world.at(t) for t in range(200)]
    runs = []
    for _ in range(2):
        model = Athena(Config(sizes=[3, 12, 6], seed=7))
        runs.append([r.mse for r in model.run(obs)])
    assert np.allclose(runs[0], runs[1]), "same seed gave different results"


# ----------------------------------------------------------------------
# the actual claim: it gets better by predicting
# ----------------------------------------------------------------------
def test_error_falls_over_time():
    world = _stationary()
    model = Athena(Config(sizes=[4, 24, 12], seed=1))
    mse = np.array([r.mse for r in model.run(world.stream(4000))])
    early, late = mse[:500].mean(), mse[-500:].mean()
    assert late < early / 5, f"error barely improved: {early:.5f} -> {late:.5f}"


def test_beats_trivial_baselines():
    """Beating persistence is necessary; beating linear extrapolation is the point.

    The model is handed velocity as an input, so constant-velocity
    extrapolation is available to it for free. Only beating *that* shows it
    has learned something about the signal rather than about smoothness.
    """
    world = _stationary()
    obs = [world.at(t) for t in range(6000)]
    model = Athena(Config(sizes=[4, 24, 12], seed=1))
    mse = np.array([r.mse for r in model.run(obs)])
    tail = slice(-1000, None)
    assert mse[tail].mean() < np.mean(persistence_mse(obs)[tail]) / 5
    assert mse[tail].mean() < np.mean(linear_mse(obs)[tail])


def test_learning_is_stable_over_a_long_run():
    """A model meant to run forever must not decay once it has converged."""
    world = _stationary()
    model = Athena(Config(sizes=[4, 24, 12], seed=2))
    mse = np.array([r.mse for r in model.run(world.stream(8000))])
    assert np.all(np.isfinite(mse)), "training produced non-finite errors"
    mid, end = mse[3000:4000].mean(), mse[-1000:].mean()
    assert end <= mid * 2.0, f"error regressed late in the run: {mid:.6f} -> {end:.6f}"


def test_multi_step_prediction_degrades_gracefully():
    world = _stationary()
    model = Athena(Config(sizes=[4, 24, 12], seed=1))
    model.run(world.stream(3000))
    errors = []
    for h in (1, 2, 5):
        err = [
            float(np.mean((world.at(3000 + t + h - 1) - model.predict(horizon=h)) ** 2))
            for t in range(1)
        ]
        errors.append(err[0])
    assert all(np.isfinite(errors))
    assert errors[0] <= errors[-1] + 1e-6, "one-step prediction was worse than five-step"


# ----------------------------------------------------------------------
# precision and volatility
# ----------------------------------------------------------------------
def test_precision_separates_clean_from_noisy_channels():
    """Noisy channels must end up trusted far less than clean ones."""
    world = SwitchingWorld(
        [Regime("mixed", [0.08, 0.11, 0.05, 0.13])],
        dwell=10**9,
        noise=[0.0, 0.0, 0.4, 0.4],
        seed=0,
    )
    model = Athena(Config(sizes=[4, 24, 12], seed=1))
    model.run(world.stream(3000))
    pi = model.channel_precision
    assert min(pi[0], pi[1]) > 10 * max(pi[2], pi[3]), f"precision did not separate: {pi}"


def test_surprise_spikes_and_learning_reopens_on_a_regime_change():
    world = shifting_world(n_channels=4, n_regimes=2, dwell=800, seed=5)
    model = Athena(Config(sizes=[4, 24, 12], seed=1))
    reports = model.run(world.stream(2400))
    gains = np.array([r.gain for r in reports])
    quiet = gains[600:790].max()
    at_switch = gains[800:860].max()
    assert at_switch > quiet, f"volatility did not react: {quiet:.2f} -> {at_switch:.2f}"
    assert at_switch > 1.5, "surprise did not raise the learning rate at all"


def test_context_belief_is_a_distribution():
    world = shifting_world(n_channels=4, n_regimes=3, dwell=400, seed=2)
    model = Athena(Config(sizes=[4, 16, 8], seed=1, experts=4))
    for report in model.run(world.stream(1500)):
        assert report.context.shape == (4,)
        assert abs(float(report.context.sum()) - 1.0) < 1e-9
        assert np.all(report.context >= 0.0)


def test_single_expert_model_still_runs():
    world = _stationary(3)
    model = Athena(Config(sizes=[3, 12, 6], seed=1, experts=1))
    reports = model.run(world.stream(500))
    assert len(reports) == 500
    assert np.all(np.isfinite([r.mse for r in reports]))


# ----------------------------------------------------------------------
if __name__ == "__main__":
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_") and callable(o)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {exc}")
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
