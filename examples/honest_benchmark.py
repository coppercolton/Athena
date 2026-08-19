"""Compare Athena with strong baselines under a frozen holdout protocol.

Every model predicts each point before seeing it. After ``--freeze-after``,
long-term learning stops for Athena and RLS while their dynamic histories keep
advancing. The final window therefore measures retained predictive knowledge,
not improvement occurring inside the evaluation window.

    python3 examples/honest_benchmark.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena import (  # noqa: E402
    Athena,
    Config,
    OnlineRLS,
    Regime,
    SwitchingWorld,
    linear_mse,
    persistence_mse,
)


def make_world(channels: int, steps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    world = SwitchingWorld(
        [
            Regime(
                "steady",
                rng.uniform(0.03, 0.17, channels),
                rng.uniform(0.0, 2 * np.pi, channels),
            )
        ],
        dwell=10**9,
    )
    return np.asarray([world.at(t) for t in range(steps)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--freeze-after", type=int, default=5000)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()
    if not 2 <= args.freeze_after < args.steps:
        parser.error("--freeze-after must be between 2 and --steps")

    observations = make_world(args.channels, args.steps, args.seed)
    athena = Athena(Config(sizes=[args.channels, 24, 12], seed=1))
    rls = OnlineRLS(args.channels, order=2)
    athena_mse = []
    athena_nll = []
    rls_mse = []

    for t, observation in enumerate(observations):
        learn = t < args.freeze_after
        athena_report = athena.observe(observation, learn=learn)
        rls_report = rls.observe(observation, learn=learn)
        athena_mse.append(athena_report.mse)
        athena_nll.append(athena_report.nll)
        rls_mse.append(rls_report.mse)

    scores = {
        "persistence": np.asarray(persistence_mse(observations)),
        "constant velocity": np.asarray(linear_mse(observations)),
        "online RLS(2)": np.asarray(rls_mse),
        "Athena": np.asarray(athena_mse),
    }
    holdout = slice(args.freeze_after, args.steps)
    print(f"trained on steps 0..{args.freeze_after - 1}")
    print(f"frozen evaluation on steps {args.freeze_after}..{args.steps - 1}\n")
    print("model                       frozen MSE")
    print("--------------------------------------")
    for name, values in scores.items():
        print(f"{name:<26} {values[holdout].mean():.12e}")
    print(f"\nAthena frozen NLL/channel   {np.mean(athena_nll[holdout]):.6f}")


if __name__ == "__main__":
    main()
