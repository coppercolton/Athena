"""How expensive is it to learn one new rule, and does prior learning help?

The benchmark this project has used until now could not answer that question.
Permuted-MNIST tasks share no structure, so there is nothing to be taught and
nothing to carry over. Here every rule is a small logical statement over the
same attributes -- "large and red and not round" -- so a system that has
understood nine of them has a real head start on the tenth, if it is the kind
of system that can have one.

Three things are measured, on the same tenth rule:

    cost        accuracy after N examples, N from 4 to 256, learned cold
    compounding the same rule learned *after* nine others, versus cold. This
                is the claim Athena exists to make: experience should make
                the next thing cheaper.
    retention   the first rule, re-tested after all ten have been learned

Two ways of acquiring it. ``gradient`` is how everything in this repository
learns: many steps, knowledge left implicit in the weights. ``episodic`` stores
the examples on sight and answers by similarity: acquisition is instant, and
generalisation reaches exactly as far as similarity does.

Neither is being taught. A person given one sentence -- "bishops move
diagonally" -- needs no drill and no stored cases, and can apply it immediately
to a board they have never seen. The distance between these two columns is a
measure of what is missing.

    python3 examples/being_taught.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena.taught import (
    EpisodicLearner,
    GradientLearner,
    make_composed_rules,
    make_rules,
    rule_examples,
)

FEATURES = 12
RULES = 10
SEEDS = 5
STEPS = 300
TEST = 2000
COUNTS = (4, 16, 64, 256)


FAMILY = {"independent": make_rules, "composed": make_composed_rules}


def cold(n: int, seed: int, family="independent"):
    """Learn only the target rule, with no prior experience."""
    rules = FAMILY[family](RULES, FEATURES, 100 + seed)
    target = rules[-1]
    train = rule_examples(target, n, FEATURES, 900 + seed)
    test = rule_examples(target, TEST, FEATURES, 5000 + seed)

    g = GradientLearner(FEATURES, seed)
    g.teach("r9", train, STEPS)
    e = EpisodicLearner()
    e.teach("r9", train)
    return g.accuracy("r9", test), e.accuracy("r9", test)


def after_nine(n: int, seed: int, family="independent"):
    """Learn nine other rules first, then the same target rule."""
    rules = FAMILY[family](RULES, FEATURES, 100 + seed)
    target = rules[-1]
    test = rule_examples(target, TEST, FEATURES, 5000 + seed)
    first_rule_test = rule_examples(rules[0], TEST, FEATURES, 6000 + seed)

    g = GradientLearner(FEATURES, seed)
    e = EpisodicLearner()
    for index, rule in enumerate(rules[:-1]):
        examples = rule_examples(rule, 256, FEATURES, seed * 50 + index)
        g.teach(f"r{index}", examples, STEPS)
        e.teach(f"r{index}", examples)

    first_before = g.accuracy("r0", first_rule_test)
    train = rule_examples(target, n, FEATURES, 900 + seed)
    g.teach("r9", train, STEPS)
    e.teach("r9", train)
    return (
        g.accuracy("r9", test),
        e.accuracy("r9", test),
        first_before,
        g.accuracy("r0", first_rule_test),
        e.accuracy("r0", first_rule_test),
    )


def main() -> None:
    print(__doc__.strip().splitlines()[0])
    print(f"\n{FEATURES} attributes, 3-literal rules, {SEEDS} seeds, {TEST} held-out cases\n")

    print("cost of one rule, learned cold")
    print(f"  {'examples':>9}{'gradient':>11}{'episodic':>11}")
    for n in COUNTS:
        runs = [cold(n, s) for s in range(SEEDS)]
        g = float(np.mean([r[0] for r in runs]))
        e = float(np.mean([r[1] for r in runs]))
        print(f"  {n:>9}{g:>11.4f}{e:>11.4f}", flush=True)

    for family in ("independent", "composed"):
        label = (
            "rules sharing only a vocabulary"
            if family == "independent"
            else "rules sharing actual sub-expressions"
        )
        print(f"\nthe same rule after nine others -- {label}")
        print(f"  {'examples':>9}{'cold':>9}{'after 9':>10}{'gain':>10}{'episodic':>11}")
        retention = []
        for n in COUNTS:
            runs = [after_nine(n, s, family) for s in range(SEEDS)]
            g = float(np.mean([r[0] for r in runs]))
            e = float(np.mean([r[1] for r in runs]))
            gc = float(np.mean([r[0] for r in [cold(n, s, family) for s in range(SEEDS)]]))
            retention.append(
                (float(np.mean([r[2] for r in runs])), float(np.mean([r[3] for r in runs])),
                 float(np.mean([r[4] for r in runs])))
            )
            print(f"  {n:>9}{gc:>9.4f}{g:>10.4f}{g - gc:>+10.4f}{e:>11.4f}", flush=True)

    before, after_g, after_e = retention[-1]
    print("\nretention of the first rule, after all ten")
    print(f"  gradient  {before:.4f} -> {after_g:.4f}   ({after_g - before:+.4f})")
    print(f"  episodic  stored verbatim -> {after_e:.4f}   (cannot forget)")


if __name__ == "__main__":
    main()
