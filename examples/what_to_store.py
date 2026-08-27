"""Round two: if you cannot choose better slots, put more in each one.

The previous experiment ruled out the obvious lever. Every rule for deciding
*which* examples to keep -- surprise, reducible loss, predictive entropy --
lost to uniform reservoir sampling, at every noise level including none,
because ranking concentrates the buffer on whatever is hard now and evicts
everything older. Coverage, not informativeness, is the scarce resource.

That leaves one axis: what each retained slot carries. A hard label is a weak
constraint on the network; the logits it computed when the example was current
are a much tighter one, encoding the whole similarity structure rather than
just the answer. Rehearsing logits asks the network to still compute what it
used to compute. That is Dark Experience Replay.

Predictions, stated before running:

    1. der++ > logits > hard, at every buffer size.
    2. The gap is LARGER at small buffers, because when coverage is scarcest,
       information per slot matters most. This is the discriminating
       prediction -- if the ordering holds but the gap does not widen as the
       buffer shrinks, the "information per slot" explanation is wrong even
       though the ranking came out right.
    3. Old-task retention improves more than average accuracy, because a
       functional constraint is aimed at preserving old computation.

Buffer contents, sampling, capacity, optimiser, seeds and data are identical
across conditions. Only the rehearsal loss differs.

    python3 examples/what_to_store.py --data <dir with MNIST idx.gz files>
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena.continual import ContinualConfig, Sample
from athena.der import DERLearner
from examples.permuted_mnist import load_mnist

TASKS = 10
CLASSES = 10
HIDDEN = (256, 128)
STEPS = 400
BATCH = 64
TRAIN_PER_TASK = 8_000
TEST_PER_TASK = 2_000
MODES = ("hard", "logits", "der++")
# Tuned on a held-out probe: the logit term's gradient is unbounded where the
# cross-entropy term's is not, so this weight is a scale correction, not a
# free parameter. At 1.0 the rehearsal term swamps new learning and DER++
# loses most of its advantage.
ALPHA = 0.1
BUFFERS = (200, 1_000)


def build(x, y, xt, yt, seed: int):
    rng = np.random.default_rng(seed)
    train_idx = rng.choice(len(x), TRAIN_PER_TASK, replace=False)
    test_idx = rng.choice(len(xt), TEST_PER_TASK, replace=False)
    xa, ya, xb, yb = x[train_idx], y[train_idx], xt[test_idx], yt[test_idx]
    tasks = []
    for index in range(TASKS):
        perm = np.arange(784) if index == 0 else rng.permutation(784)
        tasks.append(
            (
                [Sample(row, int(label)) for row, label in zip(xa[:, perm], ya)],
                [Sample(row, int(label)) for row, label in zip(xb[:, perm], yb)],
            )
        )
    return tasks


def run(tasks, seed: int, mode: str, buffer: int):
    brain = DERLearner(
        ContinualConfig(
            input_dim=784,
            hidden=HIDDEN,
            seed=seed,
            replay_capacity=buffer,
            replay_per_step=64,
            consolidation=0.0,
            retention_tolerance=1.0,
        ),
        classes=CLASSES,
        mode=mode,
        alpha=ALPHA,
    )
    for train, test in tasks:
        brain.teach("shared", train, test[:500], steps=STEPS, batch_size=BATCH)
    final = np.array([brain.accuracy("shared", test) for _, test in tasks])
    return final.mean(), final[:5].mean(), final[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()

    x, y, xt, yt = load_mnist(args.data)
    print(
        f"Permuted-MNIST, single-head, {TASKS} tasks, MLP{HIDDEN}, "
        f"{args.seeds} seed(s). Only the rehearsal loss differs.\n"
    )
    print(f"  {'buffer':>7}  {'mode':<9}{'avg acc':>10}{'oldest 5':>11}{'first task':>13}")

    for buffer in BUFFERS:
        scores = {}
        for mode in MODES:
            runs = []
            for seed in range(args.seeds):
                runs.append(run(build(x, y, xt, yt, seed), seed, mode, buffer))
            avg, old5, first = (float(np.mean([r[i] for r in runs])) for i in range(3))
            scores[mode] = avg
            print(f"  {buffer:>7}  {mode:<9}{avg:>10.4f}{old5:>11.4f}{first:>13.4f}", flush=True)
        base = scores["hard"]
        gains = "   ".join(f"{m} {scores[m] - base:+.4f}" for m in MODES if m != "hard")
        print(f"  {'':>7}  {'vs hard:':<9}{gains}\n", flush=True)


if __name__ == "__main__":
    main()
