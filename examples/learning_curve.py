"""Does it actually get better by predicting?

Runs Athena on a stationary multi-frequency signal and compares it against the
two baselines that matter. The comparison is the point: a predictive model on
a smooth signal will look impressive against nothing at all, so the bar here is
constant-velocity extrapolation, which is a prediction the model gets for free
from its own inputs.

    python3 examples/learning_curve.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena import Athena, Config, Regime, SwitchingWorld, linear_mse, persistence_mse
from athena.plot import chart

STEPS = 8000
CHANNELS = 4


def main() -> None:
    rng = np.random.default_rng(3)
    world = SwitchingWorld(
        [
            Regime(
                "steady",
                rng.uniform(0.03, 0.17, CHANNELS),
                rng.uniform(0.0, 2 * np.pi, CHANNELS),
            )
        ],
        dwell=10**9,
    )
    observations = [world.at(t) for t in range(STEPS)]

    model = Athena(Config(sizes=[CHANNELS, 24, 12], seed=1))
    reports = model.run(observations)
    mse = np.array([r.mse for r in reports])

    persistence = np.array(persistence_mse(observations))
    linear = np.array(linear_mse(observations))

    print(__doc__.strip().splitlines()[0])
    print()
    print(
        chart(
            {
                "athena": mse,
                "persistence": persistence,
                "linear": linear,
            },
            log=True,
            title=f"prediction error over {STEPS} steps (log scale)",
        )
    )

    tail = slice(-1000, None)
    print()
    print("mean squared error over the last 1000 steps")
    print(f"  persistence  (x_t)            {persistence[tail].mean():.3e}")
    print(f"  linear       (x_t + dx_t)     {linear[tail].mean():.3e}")
    print(f"  athena                        {mse[tail].mean():.3e}")
    factor = persistence[tail].mean() / max(mse[tail].mean(), 1e-12)
    print(f"\n  {factor:.0f}x better than persistence, "
          f"{linear[tail].mean() / max(mse[tail].mean(), 1e-12):.1f}x better than linear.")

    print(f"\n  context experts recruited: {model.gate.allocations or 'none'} "
          "(a world that never changes needs none)")


if __name__ == "__main__":
    main()
