"""Permuted-MNIST: the continual learner on a benchmark it did not choose.

Everything Athena had been measured on was built for it, which can only show
that it behaves as designed on data selected to show it behaving as designed.
Permuted-MNIST is the standard continual-learning benchmark and the one the
elastic-weight-consolidation paper used: N tasks, each a fixed random shuffle
of the 784 pixels, learned strictly in sequence, every task re-tested at the
end. It is normally run with a plain MLP, so this implementation is directly
comparable to published work rather than handicapped by its backbone.

All conditions share one architecture, budget, seed and task order. Only the
continual-learning machinery differs:

    finetune    no replay, no consolidation -- the lower bound, and the
                demonstration of catastrophic forgetting
    ewc         consolidation only
    replay      replay only
    athena      replay + consolidation + rollback (the shipped default)
    joint       every task trained together -- the upper bound

    python3 examples/permuted_mnist.py --data <dir>
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena.continual import ContinualConfig, MultiClassLearner, Sample

TASKS = 10
CLASSES = 10
HIDDEN = (256, 128)
STEPS = 500
BATCH = 64
TRAIN_PER_TASK = 10_000
TEST_PER_TASK = 2_000


def _read(path: str, header: int, shape):
    with gzip.open(path, "rb") as handle:
        raw = np.frombuffer(handle.read(), dtype=np.uint8, offset=header)
    return raw.reshape(shape)


def load_mnist(path: str):
    x = _read(os.path.join(path, "train-images-idx3-ubyte.gz"), 16, (-1, 784)).astype(np.float32)
    y = _read(os.path.join(path, "train-labels-idx1-ubyte.gz"), 8, (-1,)).astype(int)
    xt = _read(os.path.join(path, "t10k-images-idx3-ubyte.gz"), 16, (-1, 784)).astype(np.float32)
    yt = _read(os.path.join(path, "t10k-labels-idx1-ubyte.gz"), 8, (-1,)).astype(int)
    return x / 255.0, y, xt / 255.0, yt


def build_tasks(x, y, xt, yt, seed: int):
    rng = np.random.default_rng(seed)
    train_idx = rng.choice(len(x), TRAIN_PER_TASK, replace=False)
    test_idx = rng.choice(len(xt), TEST_PER_TASK, replace=False)
    xa, ya, xb, yb = x[train_idx], y[train_idx], xt[test_idx], yt[test_idx]
    tasks = []
    for index in range(TASKS):
        # Task 0 is the identity permutation, as in the original protocol.
        perm = np.arange(784) if index == 0 else rng.permutation(784)
        tasks.append(
            (
                [Sample(row, int(label)) for row, label in zip(xa[:, perm], ya)],
                [Sample(row, int(label)) for row, label in zip(xb[:, perm], yb)],
            )
        )
    return tasks


CONDITIONS = {
    "finetune": dict(replay_per_step=0, consolidation=0.0, retention_tolerance=1.0),
    "ewc": dict(replay_per_step=0, consolidation=200.0, retention_tolerance=1.0),
    "replay": dict(replay_per_step=64, consolidation=0.0, retention_tolerance=1.0),
    "athena": dict(replay_per_step=64, consolidation=200.0),
}


def learner(seed: int, **kwargs):
    return MultiClassLearner(
        ContinualConfig(
            input_dim=784, hidden=HIDDEN, seed=seed, replay_capacity=200, **kwargs
        ),
        classes=CLASSES,
    )


def run_sequential(tasks, seed: int, single_head: bool = False, **kwargs):
    """Sequential arrival, optionally with every task sharing one output head.

    The multi-head setting gives each task its own output layer and the task id
    at test time. It is the mild regime: tasks barely collide, so there is
    little forgetting for any mechanism to prevent. The single-head setting is
    the one the elastic-weight-consolidation results were reported on, and it
    is where forgetting actually bites.
    """
    brain = learner(seed, **kwargs)
    peak = np.zeros(TASKS)
    for index, (train, test) in enumerate(tasks):
        name = "shared" if single_head else f"task{index}"
        brain.teach(name, train, test[:500], steps=STEPS, batch_size=BATCH)
        peak[index] = brain.accuracy(name, test)
    names = ["shared"] * TASKS if single_head else [f"task{i}" for i in range(TASKS)]
    final = np.array([brain.accuracy(names[i], tasks[i][1]) for i in range(TASKS)])
    return final, float(np.mean(peak - final)), brain.rollbacks


def run_joint(tasks, seed: int, single_head: bool = False):
    brain = learner(seed)
    names = ["shared"] * TASKS if single_head else [f"task{i}" for i in range(TASKS)]
    for name in names:
        brain._ensure_head(name)
    rng = np.random.default_rng(seed)
    for _ in range(STEPS * TASKS):
        index = int(rng.integers(0, TASKS))
        train = tasks[index][0]
        picks = rng.choice(len(train), size=BATCH, replace=False)
        brain.observe(names[index], [train[int(j)] for j in picks])
    final = np.array([brain.accuracy(names[i], tasks[i][1]) for i in range(TASKS)])
    return final, 0.0, 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--single-head",
        action="store_true",
        help="domain-incremental: one head for every task, no task id at test time",
    )
    args = parser.parse_args()

    x, y, xt, yt = load_mnist(args.data)
    tasks = build_tasks(x, y, xt, yt, args.seed)
    mode = "single-head (domain-incremental)" if args.single_head else "multi-head (task-incremental)"
    print(
        f"Permuted-MNIST: {TASKS} tasks, MLP{HIDDEN}, {TRAIN_PER_TASK} train/task, "
        f"{mode}, chance = {1 / CLASSES:.0%}\n"
    )
    print(f"  {'condition':<12}{'avg accuracy':>14}{'forgetting':>13}{'first task':>13}{'rollbacks':>11}")
    for label in (*CONDITIONS, "joint"):
        if label == "joint":
            final, forgetting, rollbacks = run_joint(tasks, args.seed, args.single_head)
        else:
            final, forgetting, rollbacks = run_sequential(
                tasks, args.seed, args.single_head, **CONDITIONS[label]
            )
        print(
            f"  {label:<12}{final.mean():>14.4f}{forgetting:>13.4f}"
            f"{final[0]:>13.4f}{rollbacks:>11}"
        )


if __name__ == "__main__":
    main()
