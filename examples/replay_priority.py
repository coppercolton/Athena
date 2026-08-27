"""Testing a hypothesis about what a replay buffer should keep.

Hypothesis
----------
Replay should be prioritised by loss *in excess of expected loss*, not by raw
loss. Raw loss cannot separate "I was confident and wrong" -- worth rehearsing
-- from "this example is inherently unpredictable" -- not worth rehearsing.
Under label noise the highest-loss examples are overwhelmingly the mislabelled
ones, so a surprise-ranked buffer should fill with precisely the examples that
must not be rehearsed, and should fall below a uniform buffer. Subtracting an
online estimate of expected loss should restore it.

Predictions, stated before running:

    noise    uniform     surprise              reducible
    0%       baseline    at or above uniform   at or above uniform
    10%      baseline    BELOW uniform         at or above uniform
    30%      baseline    WELL below uniform    at or above uniform

The crux is the surprise column at 10-30%. If it does not fall below uniform,
the hypothesis loses its motivation regardless of how `reducible` does.

The mechanism check matters as much as the outcome: the fraction of the
retained buffer that is actually mislabelled. If `surprise` fails without its
buffer filling with noise, it failed for some other reason and the explanation
is wrong even if the prediction was right.

    python3 examples/replay_priority.py --data <dir with MNIST idx.gz files>
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena.continual import ContinualConfig, Sample
from athena.priority import PrioritisedLearner
from examples.permuted_mnist import load_mnist

TASKS = 10
CLASSES = 10
HIDDEN = (256, 128)
STEPS = 400
BATCH = 64
TRAIN_PER_TASK = 8_000
TEST_PER_TASK = 2_000
BUFFER = 1_000
POLICIES = ("uniform", "surprise", "reducible", "uncertain")
NOISE_LEVELS = (0.0, 0.10, 0.30)


def build(x, y, xt, yt, seed: int, noise: float):
    """Permuted-MNIST with label noise injected into the training stream only.

    Test labels stay clean: the question is whether the learner survives
    corrupted supervision, not whether it reproduces it.
    """
    rng = np.random.default_rng(seed)
    train_idx = rng.choice(len(x), TRAIN_PER_TASK, replace=False)
    test_idx = rng.choice(len(xt), TEST_PER_TASK, replace=False)
    xa, ya, xb, yb = x[train_idx], y[train_idx].copy(), xt[test_idx], yt[test_idx]

    flip = rng.random(len(ya)) < noise
    corrupted_positions = np.flatnonzero(flip)
    ya[flip] = (ya[flip] + rng.integers(1, CLASSES, size=flip.sum())) % CLASSES

    tasks, corrupted_ids = [], set()
    for index in range(TASKS):
        perm = np.arange(784) if index == 0 else rng.permutation(784)
        train = [Sample(row, int(label)) for row, label in zip(xa[:, perm], ya)]
        for position in corrupted_positions:
            corrupted_ids.add(id(train[position]))
        tasks.append((train, [Sample(row, int(label)) for row, label in zip(xb[:, perm], yb)]))
    return tasks, corrupted_ids


def run(tasks, corrupted_ids, seed: int, policy: str):
    brain = PrioritisedLearner(
        ContinualConfig(
            input_dim=784,
            hidden=HIDDEN,
            seed=seed,
            replay_capacity=BUFFER,
            replay_per_step=64,
            consolidation=0.0,
            retention_tolerance=1.0,
        ),
        classes=CLASSES,
        policy=policy,
    )
    for train, test in tasks:
        brain.teach("shared", train, test[:500], steps=STEPS, batch_size=BATCH)
    final = np.array([brain.accuracy("shared", test) for _, test in tasks])
    return final.mean(), final[0], brain.buffer_label_noise_rate("shared", corrupted_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()

    x, y, xt, yt = load_mnist(args.data)
    print(
        f"Permuted-MNIST, single-head, {TASKS} tasks, MLP{HIDDEN}, buffer {BUFFER}, "
        f"{args.seeds} seed(s)\n"
    )
    print(f"  {'noise':>6}  {'policy':<11}{'avg acc':>10}{'first task':>13}{'buffer noise':>15}")

    for noise in NOISE_LEVELS:
        row = {}
        for policy in POLICIES:
            accs, firsts, dirt = [], [], []
            for seed in range(args.seeds):
                tasks, corrupted = build(x, y, xt, yt, seed, noise)
                a, f, d = run(tasks, corrupted, seed, policy)
                accs.append(a)
                firsts.append(f)
                dirt.append(d)
            row[policy] = float(np.mean(accs))
            print(
                f"  {noise:>6.0%}  {policy:<11}{np.mean(accs):>10.4f}"
                f"{np.mean(firsts):>13.4f}{np.mean(dirt):>15.1%}",
                flush=True,
            )
        base = row["uniform"]
        deltas = "   ".join(
            f"{name} {row[name] - base:+.4f}" for name in POLICIES if name != "uniform"
        )
        print(f"          {'vs uniform:':<11}{deltas}\n", flush=True)


if __name__ == "__main__":
    main()
