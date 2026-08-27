"""Does keeping the pieces explicitly produce the compounding we want?

Round five found compounding in a gradient network, but faintly: nine rules of
prior experience made the tenth worth +0.035 at sixteen examples, and only when
the rules genuinely shared sub-expressions. The diagnosis was that a
distributed network learns features, and features interfere -- nothing in it
says "this exact piece was useful, keep it whole."

Three learners, identical rules, identical examples, identical seeds:

``gradient``   shared trunk, per-rule head, replay. Knowledge implicit in
               weights.
``episodic``   stores examples verbatim, answers by similarity. Instant
               acquisition, no forgetting, no abstraction.
``library``    searches conjunctions, but keeps the sub-expressions of every
               rule it learns and prefers hypotheses that reuse them. The
               search space never changes; only the *cost* of a hypothesis
               does, so what is being measured is reuse and nothing else.

Each learns the tenth rule twice: cold, and after nine others. The difference
is what experience was worth.

    python3 examples/library_vs_gradient.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena.library import LibraryLearner
from athena.taught import (
    EpisodicLearner,
    GradientLearner,
    make_composed_rules,
    rule_examples,
)

FEATURES = 12
RULES = 10
SEEDS = 8
STEPS = 300
TEST = 2000
PRIOR_EXAMPLES = 256
COUNTS = (4, 8, 16, 64)


def trial(n: int, seed: int):
    rules = make_composed_rules(RULES, FEATURES, 100 + seed)
    target = rules[-1]
    train = rule_examples(target, n, FEATURES, 900 + seed)
    test = rule_examples(target, TEST, FEATURES, 5000 + seed)

    cold, warm = {}, {}

    g = GradientLearner(FEATURES, seed)
    g.teach("r9", train, STEPS)
    cold["gradient"] = g.accuracy("r9", test)
    e = EpisodicLearner()
    e.teach("r9", train)
    cold["episodic"] = e.accuracy("r9", test)
    lib = LibraryLearner(features=FEATURES)
    lib.teach("r9", train)
    cold["library"] = lib.accuracy("r9", test)

    g2 = GradientLearner(FEATURES, seed)
    e2 = EpisodicLearner()
    lib2 = LibraryLearner(features=FEATURES)
    for index, rule in enumerate(rules[:-1]):
        prior = rule_examples(rule, PRIOR_EXAMPLES, FEATURES, seed * 50 + index)
        g2.teach(f"r{index}", prior, STEPS)
        e2.teach(f"r{index}", prior)
        lib2.teach(f"r{index}", prior)
    g2.teach("r9", train, STEPS)
    e2.teach("r9", train)
    lib2.teach("r9", train)
    warm["gradient"] = g2.accuracy("r9", test)
    warm["episodic"] = e2.accuracy("r9", test)
    warm["library"] = lib2.accuracy("r9", test)
    return cold, warm, len(lib2.library)


def main() -> None:
    print(__doc__.strip().splitlines()[0])
    print(f"\n{FEATURES} attributes, rules sharing sub-expressions, {SEEDS} seeds\n")

    for n in COUNTS:
        trials = [trial(n, s) for s in range(SEEDS)]
        print(f"  {n} examples of the new rule")
        print(f"    {'learner':<11}{'cold':>9}{'after 9':>10}{'gain':>10}")
        for who in ("gradient", "episodic", "library"):
            cold = float(np.mean([t[0][who] for t in trials]))
            warm = float(np.mean([t[1][who] for t in trials]))
            print(f"    {who:<11}{cold:>9.4f}{warm:>10.4f}{warm - cold:>+10.4f}")
        print(flush=True)

    print(f"  library size after nine rules: {trials[0][2]} pieces")


if __name__ == "__main__":
    main()
