"""How many timescales does continual learning need?

Every method that works in this literature keeps a slower copy of itself and is
pulled toward it. EWC anchors weights; DER++ anchors the function to a frozen
snapshot of its own outputs; self-distillation makes the previous model the
teacher; SuRe pairs a fast and a slow adapter by EMA; Nested Learning
generalises to a continuum of modules at different rates.

One anchor, two anchors, a continuum. The principle is asserted everywhere and
isolated nowhere, because each paper proposes a whole architecture and the
number of timescales varies alongside everything else. Here everything else is
fixed and only the anchor set changes.

The control that makes this a test of timescales rather than of strength: the
total anchoring weight is constant and split equally among active anchors.
Three anchors pull no harder in total than one. Without that, more anchors
would win for a reason having nothing to do with time.

Predictions, before running:

    - If multiple timescales genuinely matter, two anchors beat one at equal
      total weight, and three are at least as good as two.
    - If the benefit is really just "anchor to something older", every
      anchored condition ties and only the unanchored one differs.

The second outcome would say the field's continuum framing is decoration on a
one-bit fact, which is worth knowing either way.

    python3 examples/how_many_timescales.py --data <dir with MNIST idx.gz>
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena.continual import ContinualConfig
from athena.timescales import TimescaleLearner
from examples.what_to_store import BATCH, CLASSES, HIDDEN, STEPS, TASKS, build
from examples.permuted_mnist import load_mnist

ALPHA = 0.1
BUFFER = 200

CONDITIONS = (
    ("none (hard labels)", ()),
    ("1: snapshot", ("snapshot",)),
    ("1: slow ema", ("slow",)),
    ("1: fast ema", ("fast",)),
    ("2: snapshot+slow", ("snapshot", "slow")),
    ("3: snapshot+slow+fast", ("snapshot", "slow", "fast")),
)


def run(tasks, seed: int, anchors):
    brain = TimescaleLearner(
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
        anchors=anchors,
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
    sets = [build(x, y, xt, yt, s) for s in range(args.seeds)]
    print(
        f"Permuted-MNIST, single-head, {TASKS} tasks, buffer {BUFFER}, "
        f"{args.seeds} seed(s).\nTotal anchor weight fixed at {ALPHA}, split equally "
        "among active anchors.\n"
    )
    print(f"  {'anchors':<24}{'avg acc':>10}{'oldest 5':>11}{'first task':>13}")

    scores = {}
    for label, anchors in CONDITIONS:
        runs = [run(sets[s], s, anchors) for s in range(args.seeds)]
        avg, old5, first = (float(np.mean([r[i] for r in runs])) for i in range(3))
        scores[label] = avg
        print(f"  {label:<24}{avg:>10.4f}{old5:>11.4f}{first:>13.4f}", flush=True)

    base = scores["1: snapshot"]
    print("\n  against the single best anchor (snapshot):")
    for label in scores:
        if label != "1: snapshot":
            print(f"    {label:<24}{scores[label] - base:+.4f}")


if __name__ == "__main__":
    main()
