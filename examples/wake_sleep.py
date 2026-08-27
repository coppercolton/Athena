"""Does imagining problems make the library better?

The library learner keeps the pieces of every rule it learns and prefers later
hypotheses that reuse them, which produced four to six times the compounding of
a gradient network. It only ever learns from real examples.

Two offline operations are added here, both borrowed from what sleep is thought
to do, and measured separately so it is clear which one carries the result:

``consolidate``
    Compression. Drop every piece that appears in only one learned rule, and
    rescore what remains by how many distinct rules contain it. Nothing new is
    acquired; the day's coincidences are separated from its concepts.

``dream``
    Imagination. Invent rules by composing library pieces, generate examples
    for them, solve them as if they were new, and keep whatever the solutions
    used. No real data is consumed.

Dreaming has an obvious failure mode and the experiment exists to expose it: a
system that imagines only what it already believes can amplify its own
coincidences into convictions. If the gain is real it should hold where data is
scarcest and not reverse when data is plentiful.

    python3 examples/wake_sleep.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena.library import LibraryLearner
from athena.taught import make_composed_rules, rule_examples

FEATURES = 12
RULES = 10
SEEDS = 12
TEST = 2000
PRIOR = 256
DREAMS = 40
COUNTS = (4, 8, 16, 64)
MODES = ("wake only", "+consolidate", "+consolidate +dream")


def build(mode: str, rules, seed: int) -> LibraryLearner:
    learner = LibraryLearner(features=FEATURES)
    for index, rule in enumerate(rules[:-1]):
        learner.teach(f"r{index}", rule_examples(rule, PRIOR, FEATURES, seed * 50 + index))
    if mode != "wake only":
        learner.consolidate()
    if mode == "+consolidate +dream":
        learner.dream(DREAMS, np.random.default_rng(seed))
    return learner


def main() -> None:
    print(__doc__.strip().splitlines()[0])
    print(f"\n{FEATURES} attributes, {SEEDS} seeds, {DREAMS} imagined rules per run\n")
    print(f"  {'examples':>9}{'cold':>8}" + "".join(f"{m:>22}" for m in MODES))

    for n in COUNTS:
        cold, gains = [], {m: [] for m in MODES}
        sizes = {m: [] for m in MODES}
        for seed in range(SEEDS):
            rules = make_composed_rules(RULES, FEATURES, 100 + seed)
            train = rule_examples(rules[-1], n, FEATURES, 900 + seed)
            test = rule_examples(rules[-1], TEST, FEATURES, 5000 + seed)

            base = LibraryLearner(features=FEATURES)
            base.teach("r9", train)
            reference = base.accuracy("r9", test)
            cold.append(reference)

            for mode in MODES:
                learner = build(mode, rules, seed)
                sizes[mode].append(len(learner.library))
                learner.teach("r9", train)
                gains[mode].append(learner.accuracy("r9", test) - reference)

        row = f"  {n:>9}{np.mean(cold):>8.4f}"
        for mode in MODES:
            row += f"{np.mean(gains[mode]):>+22.4f}"
        print(row, flush=True)

    print("\n  library size after nine rules:")
    for mode in MODES:
        print(f"    {mode:<22}{np.mean(sizes[mode]):>6.1f} pieces")


if __name__ == "__main__":
    main()
