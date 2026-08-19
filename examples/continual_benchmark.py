"""Measure adaptation and memory when previously seen dynamics return.

The world rotates through three unrelated regimes. This compares Athena's
banked hierarchy against one overwrite-prone hierarchy and two online RLS
baselines: a stationary memory and a deliberately forgetful adaptive model.

    python3 examples/continual_benchmark.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena import Athena, Config, OnlineRLS, shifting_world  # noqa: E402


def profile(errors: np.ndarray, dwell: int, last_dwells: int = 6) -> np.ndarray:
    dwells = np.asarray(
        [errors[start : start + dwell] for start in range(0, len(errors), dwell)]
    )
    return dwells[-last_dwells:].mean(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--dwell", type=int, default=500)
    parser.add_argument("--channels", type=int, default=4)
    args = parser.parse_args()
    if args.steps % args.dwell:
        parser.error("--steps must be divisible by --dwell")

    world = shifting_world(
        n_channels=args.channels,
        n_regimes=3,
        dwell=args.dwell,
        seed=3,
    )
    observations = [world.at(t) for t in range(args.steps)]
    models = {
        "Athena banked": Athena(
            Config(sizes=[args.channels, 24, 12], seed=1, experts=6)
        ),
        "Athena single": Athena(
            Config(sizes=[args.channels, 24, 12], seed=1, experts=1)
        ),
        "RLS stationary": OnlineRLS(args.channels, order=2, forgetting=1.0),
        "RLS adaptive": OnlineRLS(args.channels, order=2, forgetting=0.995),
    }
    errors = {
        name: np.asarray([report.mse for report in model.run(observations)])
        for name, model in models.items()
    }

    print("model                       after switch     settled")
    print("----------------------------------------------------")
    for name, values in errors.items():
        curve = profile(values, args.dwell)
        switch_end = min(100, args.dwell)
        settled_start = min(200, args.dwell // 2)
        print(
            f"{name:<26} {curve[:switch_end].mean():>12.3e} "
            f"{curve[settled_start:].mean():>12.3e}"
        )

    banked = models["Athena banked"]
    print(f"\nexperts recruited: {banked.gate.allocations or 'none'}")
    print(f"context usage: {banked.gate.claimed.astype(int)}")


if __name__ == "__main__":
    main()
