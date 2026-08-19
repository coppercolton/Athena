"""What letting the network keep training actually buys, and what it costs.

One architecture, one codebase, one set of seeds and data. The only thing that
changes between the two rows is whether the shared trunk is allowed to keep
learning after the first skill is in place.

That control matters more than it sounds. Comparing this module against
``athena.plasticity`` would compare two different networks with different
depths, initialisations and hyperparameters, and the result would say more
about the implementations than about the mechanism. So the frozen baseline here
is this same class with ``freeze_trunk=True``: identical layers, identical
initialisation, only the heads still train. That is the protected-expert design
reproduced exactly, with nothing else varying.

Two numbers:

    forward transfer    does skill N cost less because skills 1..N-1 exist?
    backward transfer   does skill 1 change while later skills are learned?

A frozen trunk pins backward transfer to zero by construction -- the weights
under skill 1 never move, so it can neither improve nor decay. Letting the
trunk train unpins it in both directions at once. The table shows the price.

    python3 examples/never_stop_learning.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena.continual import ContinualConfig, ContinualLearner, Experience

DIM = 6
SEEDS = 10
TRAIN = 96
STEPS = 600

TASKS = (
    ("t1-base", lambda x: x[0] * x[1] > 0),
    ("t2-reuse", lambda x: x[0] * x[1] + x[2] > 0),
    ("t3-extend", lambda x: x[0] * x[1] + x[2] * x[3] > 0),
    ("t4-compose", lambda x: x[0] * x[1] + x[2] * x[3] + x[4] > 0),
)

SETTINGS = (
    ("frozen trunk (control)", dict(freeze_trunk=True)),
    ("always training", dict()),
    ("always, tighter retention", dict(retention_tolerance=0.02, checkpoint_every=100)),
)


def cases(fn, count: int, seed: int) -> list[Experience]:
    rng = np.random.default_rng(seed)
    return [Experience(tuple(v), int(bool(fn(v)))) for v in rng.uniform(-1.0, 1.0, (count, DIM))]


def run(**kwargs):
    new_skill, first_skill, rollbacks = [], [], 0
    for seed in range(SEEDS):
        brain = ContinualLearner(ContinualConfig(input_dim=DIM, seed=seed, **kwargs))
        probe = cases(TASKS[0][1], 512, 9000 + seed)
        news, firsts = [], []
        for index, (name, fn) in enumerate(TASKS):
            report = brain.teach(
                name,
                cases(fn, TRAIN, seed * 100 + index),
                cases(fn, 256, seed * 100 + 50 + index),
                steps=STEPS,
            )
            news.append(report.accuracy)
            firsts.append(brain.accuracy(TASKS[0][0], probe))
        new_skill.append(news)
        first_skill.append(firsts)
        rollbacks += brain.rollbacks
    return np.array(new_skill), np.array(first_skill), rollbacks


def main() -> None:
    print(__doc__.strip().splitlines()[0])
    print(f"\n{SEEDS} seeds, {TRAIN} examples per skill, identical data and initialisation\n")

    results = {label: run(**kwargs) for label, kwargs in SETTINGS}

    print("new-skill held-out accuracy (forward transfer)")
    print(f"  {'task':<13}" + "".join(f"{label:>28}" for label, _ in SETTINGS))
    for index, (name, _) in enumerate(TASKS):
        row = f"  {name:<13}"
        for label, _ in SETTINGS:
            row += f"{results[label][0][:, index].mean():>28.4f}"
        print(row)
    row = f"  {'tasks 2-4':<13}"
    for label, _ in SETTINGS:
        row += f"{results[label][0][:, 1:].mean():>28.4f}"
    print(row + "   <-- the headline")

    print("\nskill 1 over the deployment (backward transfer)")
    for label, _ in SETTINGS:
        firsts = results[label][1]
        delta = firsts[:, -1] - firsts[:, 0]
        print(
            f"  {label:<28} start {firsts[:, 0].mean():.4f} -> end {firsts[:, -1].mean():.4f}"
            f"   change {delta.mean():+.4f}  (worst seed {delta.min():+.4f})"
        )

    print("\nrollbacks triggered")
    for label, _ in SETTINGS:
        print(f"  {label:<28} {results[label][2]}")

    frozen = results[SETTINGS[0][0]]
    live = results[SETTINGS[1][0]]
    gain = live[0][:, 1:].mean() - frozen[0][:, 1:].mean()
    cost = (live[1][:, -1] - live[1][:, 0]).mean()
    print(
        f"\n  Letting the trunk keep training: {gain:+.3f} on new skills, "
        f"{cost:+.3f} on the oldest one."
    )


if __name__ == "__main__":
    main()
