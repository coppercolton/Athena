"""Round three: how little does a memory need to contain?

Two findings set this up. Choosing *which* examples to keep cannot beat picking
at random, because coverage is the scarce resource. But changing *what each
slot holds* -- the network's own output distribution instead of a hard label --
was worth +0.080, and the gain doubled as the buffer shrank.

That second fact is the interesting one. If the advantage grows as storage
shrinks, then rehearsal is not really about retaining data. It is about
retaining the *function*: the stored inputs are probe points where the old and
new networks get compared, and the stored outputs are what the old network said
there. Which raises a question with consequences well beyond this benchmark.

Frontier systems cannot keep their users' data indefinitely, and in deployment
there are usually no labels at all. A rehearsal scheme whose targets are
self-generated needs no labels by construction. If its *inputs* also need not
be real, then a system can preserve its own capability while storing nothing
that came from anyone -- which is a different deployment proposition entirely.

Two sweeps:

    1. Compression. How does each rehearsal loss degrade as the buffer shrinks
       from 2000 slots to 50? If the function view is right, storing outputs
       should degrade far more gracefully than storing labels.

    2. Realism. At fixed buffer, replace the stored inputs with uniform noise,
       or with real images whose pixels have been shuffled -- statistics kept,
       structure destroyed. The stored outputs are the old network's response
       to whatever is stored, so the rehearsal signal stays self-consistent.

Predictions, before running:

    - der++ degrades more gracefully than hard labels; the gap widens
      monotonically as the buffer shrinks.
    - `shuffled` retains a substantial fraction of the benefit; `noise`
      retains less but more than zero. If both collapse to the hard-label
      baseline, rehearsal genuinely needs real data and the function view is
      wrong at this scale.

    python3 examples/how_little.py --data <dir with MNIST idx.gz files>
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena.continual import ContinualConfig
from athena.der import DERLearner
from examples.what_to_store import CLASSES, HIDDEN, STEPS, BATCH, TASKS, build
from examples.permuted_mnist import load_mnist

ALPHA = 0.1
SIZES = (50, 200, 1000, 2000)


def run(tasks, seed: int, mode: str, buffer: int, probe: str = "real"):
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
        probe=probe,
        alpha=ALPHA,
    )
    for train, test in tasks:
        brain.teach("shared", train, test[:500], steps=STEPS, batch_size=BATCH)
    final = np.array([brain.accuracy("shared", test) for _, test in tasks])
    return final.mean(), final[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()

    x, y, xt, yt = load_mnist(args.data)
    sets = [build(x, y, xt, yt, s) for s in range(args.seeds)]

    def average(mode, buffer, probe="real"):
        runs = [run(sets[s], s, mode, buffer, probe) for s in range(args.seeds)]
        return float(np.mean([r[0] for r in runs])), float(np.mean([r[1] for r in runs]))

    print(f"Permuted-MNIST, single-head, {TASKS} tasks, {args.seeds} seed(s)\n")
    print("1. How gracefully does each rehearsal loss survive a shrinking buffer?\n")
    print(f"  {'slots':>7}{'hard':>10}{'der++':>10}{'gap':>10}")
    for buffer in SIZES:
        hard, _ = average("hard", buffer)
        der, _ = average("der++", buffer)
        print(f"  {buffer:>7}{hard:>10.4f}{der:>10.4f}{der - hard:>+10.4f}", flush=True)

    print("\n2. Do the stored inputs have to be real data?\n")
    print(f"  {'slots':>7}  {'stored inputs':<20}{'avg acc':>10}{'first task':>13}")
    for buffer in (200, 1000):
        hard, _ = average("hard", buffer)
        print(f"  {buffer:>7}  {'(hard-label floor)':<20}{hard:>10.4f}{'':>13}", flush=True)
        results = {p: average("der++", buffer, p) for p in ("real", "shuffled", "noise")}
        ceiling = results["real"][0] - hard
        for probe, (avg, first) in results.items():
            share = (avg - hard) / max(1e-9, ceiling)
            print(
                f"  {buffer:>7}  {probe:<20}{avg:>10.4f}{first:>13.4f}"
                f"   keeps {share:>5.0%} of the benefit",
                flush=True,
            )
        print(flush=True)


if __name__ == "__main__":
    main()
