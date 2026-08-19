"""Does learning more make it better overall?

This is the scaling intuition: bigger models trained on more see broad
improvement and new capabilities, without being taught each one. The question
is whether a system learning from a stream can show the same shape, and if not,
what is actually missing.

The answer turned out to depend on one property of the *tasks*, not of the
architecture. Everything below runs the same trunk, the same capacity, the same
learning rule. Only the relationship between the tasks changes:

    related     every task is a different combination of the same few
                underlying factors, so one representation serves them all
    unrelated   each task is its own thing, sharing nothing

Real domains look like the first. Benchmarks built from independently sampled
tasks look like the second, and they cannot show transfer no matter what
architecture is underneath -- there is nothing to transfer.

    python3 examples/does_more_help.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena.continual import (
    ContinualConfig,
    ContinualLearner,
    related_tasks,
    task_cases,
    unrelated_tasks,
)

DIM = 8
SEEDS = 3
PROBE = 512
PER_TASK = 512
STEPS = 500


def deployment(family, n_tasks: int, hidden: tuple[int, ...], seed: int):
    """Learn n_tasks in sequence; return every task's final accuracy and skill 1's curve."""
    matrices = family(n_tasks, 2000 + seed, dim=DIM)
    brain = ContinualLearner(
        ContinualConfig(input_dim=DIM, hidden=hidden, seed=seed, replay_capacity=512)
    )
    probes = [task_cases(m, PROBE, 7000 + seed * 100 + i) for i, m in enumerate(matrices)]
    first_curve = []
    for index, matrix in enumerate(matrices):
        brain.teach(
            f"t{index}",
            task_cases(matrix, PER_TASK, seed * 100 + index),
            task_cases(matrix, 256, seed * 100 + 60 + index),
            steps=STEPS,
        )
        first_curve.append(brain.accuracy("t0", probes[0]))
    finals = [brain.accuracy(f"t{i}", probes[i]) for i in range(n_tasks)]
    return float(np.mean(finals)), first_curve


def main() -> None:
    print(__doc__.strip().splitlines()[0])

    print("\n1. average accuracy over ALL tasks learned, fixed capacity (48, 32)\n")
    print(f"  {'tasks':>6}{'related':>12}{'unrelated':>12}")
    for n in (2, 4, 8, 12):
        rel = np.mean([deployment(related_tasks, n, (48, 32), s)[0] for s in range(SEEDS)])
        unr = np.mean([deployment(unrelated_tasks, n, (48, 32), s)[0] for s in range(SEEDS)])
        print(f"  {n:>6}{rel:>12.4f}{unr:>12.4f}")
    print("\n  Related: more tasks, better at all of them. Unrelated: interference.")

    print("\n2. skill 1, never retrained, as later skills arrive\n")
    for label, family in (("related", related_tasks), ("unrelated", unrelated_tasks)):
        curves = [deployment(family, 12, (48, 32), s)[1] for s in range(SEEDS)]
        curve = np.array(curves).mean(axis=0)
        points = " ".join(f"{curve[i]:.3f}" for i in (0, 1, 3, 7, 11))
        print(f"  {label:<10} after 1,2,4,8,12 skills:  {points}   net {curve[-1] - curve[0]:+.4f}")
    print("\n  Backward transfer is real, and it is a property of the task family.")

    print("\n3. capacity sets the level, 8 related tasks\n")
    print(f"  {'hidden':>12}{'avg over all':>14}")
    for hidden in ((16, 12), (32, 24), (64, 48)):
        score = np.mean([deployment(related_tasks, 8, hidden, s)[0] for s in range(SEEDS)])
        print(f"  {str(hidden):>12}{score:>14.4f}")


if __name__ == "__main__":
    main()
