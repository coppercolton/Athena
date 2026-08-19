"""Split-CIFAR-100: how the continual learner does on a benchmark it did not choose.

Everything Athena has been measured on so far was built for it. That cannot
tell you whether the mechanisms are any good, only that they behave as designed
on data selected to show them behaving as designed. Split-CIFAR-100 is the
standard task-incremental continual-learning benchmark: 100 classes cut into 20
disjoint tasks of 5 classes each, learned strictly in sequence, then every task
re-tested at the end.

The conditions below share one architecture, one budget, one seed and one data
order. Only the continual-learning machinery differs:

    finetune    no replay, no consolidation -- the lower bound, and the
                demonstration of catastrophic forgetting
    ewc         consolidation only
    replay      replay only
    athena      replay + consolidation + rollback (the shipped default)
    joint       every task trained together -- the upper bound no sequential
                learner can beat

Two caveats, stated up front because they decide how the numbers should be
read. The backbone is a plain NumPy MLP over raw pixels, where published
results use convolutional networks, so the absolute accuracies are far below
the literature and are not comparable to it. What *is* comparable is the
spread between the conditions, since they differ in exactly one thing.

    python3 examples/split_cifar100.py --data <dir> [--seeds 1]
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import tarfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena.continual import ContinualConfig, MultiClassLearner, Sample

TASKS = 20
CLASSES_PER_TASK = 5
HIDDEN = (256, 128)
STEPS = 300
BATCH = 64


def load_cifar100(path: str):
    archive = os.path.join(path, "cifar-100-python.tar.gz")
    root = os.path.join(path, "cifar-100-python")
    if not os.path.isdir(root):
        with tarfile.open(archive) as tar:
            tar.extractall(path)

    def read(name):
        with open(os.path.join(root, name), "rb") as handle:
            raw = pickle.load(handle, encoding="bytes")
        return raw[b"data"].astype(np.float32), np.asarray(raw[b"fine_labels"], dtype=int)

    x_train, y_train = read("train")
    x_test, y_test = read("test")
    # Standardise with training statistics only.
    x_train /= 255.0
    x_test /= 255.0
    mean, std = x_train.mean(axis=0), x_train.std(axis=0) + 1e-6
    return (x_train - mean) / std, y_train, (x_test - mean) / std, y_test


def split(x, y, x_test, y_test, seed: int):
    """Cut 100 classes into 20 disjoint 5-class tasks, in a shuffled order."""
    order = np.random.default_rng(seed).permutation(100)
    tasks = []
    for index in range(TASKS):
        labels = order[index * CLASSES_PER_TASK : (index + 1) * CLASSES_PER_TASK]
        remap = {int(label): position for position, label in enumerate(labels)}
        train_mask = np.isin(y, labels)
        test_mask = np.isin(y_test, labels)
        tasks.append(
            (
                [Sample(row, remap[int(label)])
                 for row, label in zip(x[train_mask], y[train_mask])],
                [Sample(row, remap[int(label)])
                 for row, label in zip(x_test[test_mask], y_test[test_mask])],
            )
        )
    return tasks


CONDITIONS = {
    "finetune": dict(replay_per_step=0, consolidation=0.0, retention_tolerance=1.0),
    "ewc": dict(replay_per_step=0, consolidation=200.0, retention_tolerance=1.0),
    "replay": dict(replay_per_step=64, consolidation=0.0, retention_tolerance=1.0),
    "athena": dict(replay_per_step=64, consolidation=200.0),
}


def evaluate(brain, tasks) -> list[float]:
    return [brain.accuracy(f"task{i}", test) for i, (_, test) in enumerate(tasks)]


def run_sequential(tasks, seed: int, **kwargs):
    brain = MultiClassLearner(
        ContinualConfig(
            input_dim=3072, hidden=HIDDEN, seed=seed, replay_capacity=100, **kwargs
        ),
        classes=CLASSES_PER_TASK,
    )
    peak = np.zeros(TASKS)
    for index, (train, test) in enumerate(tasks):
        brain.teach(f"task{index}", train, test[:200], steps=STEPS, batch_size=BATCH)
        peak[index] = brain.accuracy(f"task{index}", test)
    final = np.array(evaluate(brain, tasks))
    return final, float(np.mean(peak - final)), brain.rollbacks


def run_joint(tasks, seed: int):
    brain = MultiClassLearner(
        ContinualConfig(input_dim=3072, hidden=HIDDEN, seed=seed, replay_capacity=100),
        classes=CLASSES_PER_TASK,
    )
    for index in range(TASKS):
        brain._ensure_head(f"task{index}")
    rng = np.random.default_rng(seed)
    for _ in range(STEPS * TASKS):
        index = int(rng.integers(0, TASKS))
        train = tasks[index][0]
        picks = rng.choice(len(train), size=BATCH, replace=False)
        brain.observe(f"task{index}", [train[int(j)] for j in picks])
    return np.array(evaluate(brain, tasks)), 0.0, 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--seeds", type=int, default=1)
    args = parser.parse_args()

    x, y, x_test, y_test = load_cifar100(args.data)
    print(f"Split-CIFAR-100: {TASKS} tasks x {CLASSES_PER_TASK} classes, "
          f"MLP{HIDDEN} over raw pixels, chance = {1 / CLASSES_PER_TASK:.0%}\n")
    print(f"  {'condition':<12}{'avg accuracy':>14}{'forgetting':>13}{'rollbacks':>11}")

    for label in (*CONDITIONS, "joint"):
        accuracies, forgetting, rollbacks = [], [], 0
        for seed in range(args.seeds):
            tasks = split(x, y, x_test, y_test, seed)
            if label == "joint":
                final, forget, rolled = run_joint(tasks, seed)
            else:
                final, forget, rolled = run_sequential(tasks, seed, **CONDITIONS[label])
            accuracies.append(final.mean())
            forgetting.append(forget)
            rollbacks += rolled
        print(
            f"  {label:<12}{np.mean(accuracies):>14.4f}"
            f"{np.mean(forgetting):>13.4f}{rollbacks:>11}"
        )


if __name__ == "__main__":
    main()
