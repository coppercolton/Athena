"""Does learning a skill make the next skill cheaper?

This is the measurement that separates "gets smarter as it learns" from "keeps
a tidy filing cabinet of skills". Both look identical on a per-skill accuracy
report. They differ on one number: what learning skill 1 and 2 does to the cost
of skill 3.

The task family is a curriculum. Each task reuses the feature the previous one
had to discover, which is the situation the goal describes -- learn a
sub-skill, then a problem that needs it:

    t1  x0*x1 > 0                    the base feature
    t2  x0*x1 + x2 > 0               reuses it
    t3  x0*x1 + x2*x3 > 0            reuses it and extends
    t4  x4 > 0                       unrelated, to detect negative transfer

Two registries see identical data, seeds, and training budgets. One isolates
every skill, which is what Athena's protected experts do today. The other keeps
every earlier skill frozen but lets a new skill read their features.

    python3 examples/transfer_benchmark.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena.transfer import Example, ProgressiveRegistry, TransferConfig

INPUT_DIM = 6
TASKS = (
    ("t1-base", lambda x: x[0] * x[1] > 0),
    ("t2-reuse", lambda x: x[0] * x[1] + x[2] > 0),
    ("t3-extend", lambda x: x[0] * x[1] + x[2] * x[3] > 0),
    ("t4-unrelated", lambda x: x[4] > 0),
)
SEEDS = 12
TRAIN_SIZES = (24, 48, 96)
VALIDATION = 512


def cases(fn, count: int, seed: int) -> list[Example]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        x = rng.uniform(-1.0, 1.0, INPUT_DIM)
        out.append(Example(tuple(x), int(bool(fn(x)))))
    return out


def run(lateral: bool, n_train: int, seed: int) -> tuple[list[float], float]:
    config = TransferConfig(input_dim=INPUT_DIM, seed=seed)
    registry = ProgressiveRegistry(config, lateral=lateral)
    accuracies = []
    probe = None
    first_before = None
    for index, (name, fn) in enumerate(TASKS):
        report = registry.learn(
            name,
            cases(fn, n_train, seed * 1000 + index),
            cases(fn, VALIDATION, seed * 1000 + 500 + index),
        )
        accuracies.append(report.accuracy)
        if index == 0 and report.promoted:
            probe = cases(fn, VALIDATION, seed * 1000 + 900)
            first_before = registry.accuracy(name, probe)
    forgetting = 0.0
    if probe is not None and first_before is not None:
        forgetting = first_before - registry.accuracy(TASKS[0][0], probe)
    return accuracies, forgetting


def main() -> None:
    print(__doc__.strip().splitlines()[0])
    print(f"\n{SEEDS} seeds, mean held-out accuracy per task\n")

    for n_train in TRAIN_SIZES:
        iso = np.array([run(False, n_train, s)[0] for s in range(SEEDS)])
        lat_runs = [run(True, n_train, s) for s in range(SEEDS)]
        lat = np.array([r[0] for r in lat_runs])
        forgetting = np.array([r[1] for r in lat_runs])

        print(f"training examples per skill: {n_train}")
        print(f"  {'task':<14} {'isolated':>10} {'lateral':>10} {'gain':>9}")
        for index, (name, _) in enumerate(TASKS):
            a, b = iso[:, index].mean(), lat[:, index].mean()
            flag = "" if index == 0 else ("  <-- transfer" if b - a > 0.01 else "")
            print(f"  {name:<14} {a:>10.4f} {b:>10.4f} {b - a:>+9.4f}{flag}")
        later = slice(1, 3)
        print(
            f"  {'reuse tasks':<14} {iso[:, later].mean():>10.4f} "
            f"{lat[:, later].mean():>10.4f} {lat[:, later].mean() - iso[:, later].mean():>+9.4f}"
        )
        print(f"  forgetting of skill 1 after 3 more: {forgetting.max():+.6f} (worst seed)\n")


if __name__ == "__main__":
    main()
