"""What does discovered structure actually buy a learner?

Rounds eight to twelve built a symbolic layer -- binding, analogy, and finally
role discovery from raw co-occurrence -- and every result in them is
*intrinsic*. They ask whether the true roles were recovered, or whether
transfer worked given roles already known to be right. Not one of them shows
that discovered structure makes any learner better at anything.

That is the load-bearing untested claim of this repository, and it is the
founding one: that watching the world should make later learning cheaper.

This tests two ways it might, and only one of them is real.

**The first is that structure compresses the search.** Knowing which features
are alternatives for one role means two of them can never both hold, so those
conjunctions are contradictions and can be struck from the hypothesis space. A
smaller space is harder to fit by luck, so the same learner should need fewer
examples. Four conditions, identical learner, identical examples, identical
scoring and tie-break -- only the candidate set differs:

    raw           every conjunction. No structure known.
    discovered    roles recovered by watching unlabelled situations.
    given         the true roles, handed over. The upper bound.
    shuffled      a random grouping of the same shape. The control: if this
                  helps too, the win is having *any* prior, not the right one.

**It buys nothing.** All four are identical to three decimals. The reason is
worth more than the hypothesis was: a contradiction predicts "always false",
which scores at chance on balanced data and can never win. The impossible
hypotheses were eliminating themselves already, and pruning them removes
competitors that were never competing.

**The second is that structure changes what is expressible.** Some rules do
not mention any particular filler at all -- they say two roles *agree*. Whether
the stalk is the same colour above and below the ring is a fact about role
identity, and mushrooms have exactly that pair. Without roles this is not a
long hypothesis, it is not a hypothesis: a conjunction over features cannot say
"whatever the first one is, the second matches", and covering it by enumeration
needs one disjunct per filler, in a space that explodes.

That is where discovered structure pays, and the same shuffled control decides
whether it is the roles doing the work or merely the extra predicate.

    python3 examples/paying_rent.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena.continual import Sample
from athena.joints import discover_slots
from athena.library import LibraryLearner
from athena.taught import Rule

SIZES = (4, 3, 4, 3, 2)  # fillers per role
FEATURES = sum(SIZES)
WATCH = 1500  # unlabelled situations, free
RULES = 40
SEEDS = 6
COUNTS = (4, 6, 8, 12, 16, 24, 32)
TEST = 400

ROLES = []
_next = 0
for _size in SIZES:
    ROLES.append(tuple(range(_next, _next + _size)))
    _next += _size


def situations(rng, count: int) -> np.ndarray:
    """One filler per role, as a binary vector. No role structure is marked."""
    out = np.zeros((count, FEATURES))
    for role in ROLES:
        picks = rng.integers(0, len(role), size=count)
        out[np.arange(count), np.asarray(role)[picks]] = 1.0
    return out


def watch(rng) -> tuple[tuple[int, ...], ...]:
    """Recover the roles from unlabelled situations alone."""
    data = situations(rng, WATCH)
    episodes = [[f"f{i}" for i in np.flatnonzero(row)] for row in data]
    groups = discover_slots(episodes)
    return tuple(tuple(sorted(int(t[1:]) for t in g)) for g in groups)


def shuffled_roles(rng) -> tuple[tuple[int, ...], ...]:
    """A grouping with the same shape as the truth and none of its content."""
    order = list(rng.permutation(FEATURES))
    out, at = [], 0
    for size in SIZES:
        out.append(tuple(sorted(int(i) for i in order[at : at + size])))
        at += size
    return tuple(out)


def make_rule(rng) -> Rule:
    """A conjunction over two or three distinct roles -- always satisfiable."""
    arity = int(rng.integers(2, 4))
    chosen = rng.choice(len(ROLES), size=arity, replace=False)
    literals = tuple(
        sorted((int(ROLES[r][int(rng.integers(0, len(ROLES[r])))]), True) for r in chosen)
    )
    return Rule(literals)


def balanced(rng, rule: Rule, count: int) -> list[Sample]:
    """Half satisfying, half not. Otherwise a rule true 2% of the time is
    learned perfectly by answering "no" and the measurement means nothing."""
    want = count // 2
    positives = np.zeros((want, FEATURES))
    for row in range(want):
        one = situations(rng, 1)[0]
        for index, _ in rule.literals:  # force the rule's fillers, randomise the rest
            role = next(r for r in ROLES if index in r)
            one[list(role)] = 0.0
            one[index] = 1.0
        positives[row] = one

    negatives, tries = [], 0
    while len(negatives) < count - want and tries < 10_000:
        tries += 1
        one = situations(rng, 1)
        if not rule.holds(one)[0]:
            negatives.append(one[0])
    rows = np.vstack([positives, np.asarray(negatives)])
    labels = [1] * want + [0] * len(negatives)
    order = rng.permutation(len(rows))
    return [Sample(inputs=rows[i], target=labels[i]) for i in order]


# ---------------------------------------------------------------------------
# Second question: rules that mention no filler at all.


def agreement_rule(rng) -> tuple[int, int]:
    """Two roles of equal size that have to agree. Mentions no filler."""
    pairs = [
        (a, b)
        for a in range(len(ROLES))
        for b in range(a + 1, len(ROLES))
        if len(ROLES[a]) == len(ROLES[b])
    ]
    return pairs[int(rng.integers(0, len(pairs)))]


def agrees(x: np.ndarray, groups, a: int, b: int) -> np.ndarray:
    """Does the filler chosen in role ``a`` sit at the same position as in ``b``?

    Defined by position within the group, which is what makes it a statement
    about roles rather than about fillers -- and what makes it meaningless when
    the groups are wrong, which is the control.
    """
    ga, gb = list(groups[a]), list(groups[b])
    width = min(len(ga), len(gb))
    left = x[:, ga[:width]]
    right = x[:, gb[:width]]
    hit_l = left.max(axis=1) > 0.5
    hit_r = right.max(axis=1) > 0.5
    return hit_l & hit_r & (left.argmax(axis=1) == right.argmax(axis=1))


def agreement_examples(rng, pair, count: int) -> list[Sample]:
    a, b = pair
    rows, labels = [], []
    for _ in range(count):
        one = situations(rng, 1)[0]
        want = len(rows) % 2 == 0
        pos_a = int(np.argmax(one[list(ROLES[a])]))
        target = pos_a if want else (pos_a + 1 + int(rng.integers(0, len(ROLES[b]) - 1))) % len(ROLES[b])
        one[list(ROLES[b])] = 0.0
        one[ROLES[b][target]] = 1.0
        rows.append(one)
        labels.append(int(want))
    order = rng.permutation(len(rows))
    return [Sample(inputs=rows[i], target=labels[i]) for i in order]


def best_hypothesis(x, y, groups, compiled) -> tuple[str, object]:
    """Exhaustive search over conjunctions, plus agreement predicates if roles
    are known. Same fit-then-shortest rule as the conjunction learner."""
    candidates, index, polarity, used = compiled
    holds = (((x[:, index] > 0.5) == polarity) | ~used).all(axis=2)
    fit = (holds == y[:, None].astype(bool)).mean(axis=0)
    size = used.sum(axis=1)
    order = np.lexsort((size, -fit))
    best = (float(fit[order[0]]), 1 + int(size[order[0]]), ("conj", candidates[order[0]]))

    if groups is not None:
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                if len(groups[a]) != len(groups[b]):
                    continue
                got = agrees(x, groups, a, b)
                score = float((got == y.astype(bool)).mean())
                if (score, -1) > (best[0], -best[1]):
                    best = (score, 1, ("agree", (a, b)))
    return best[2]


def apply_hypothesis(kind, payload, x, groups) -> np.ndarray:
    if kind == "conj":
        return Rule(payload).holds(x)
    return agrees(x, groups, *payload)


def agreement_section() -> None:
    print("\n\nRules that mention no filler: two roles must agree.")
    print("A conjunction over features cannot express this at any length.\n")

    conditions = ("raw", "discovered", "given", "shuffled")
    scores = {c: {n: [] for n in COUNTS} for c in conditions}

    for seed in range(SEEDS):
        rng = np.random.default_rng(100 + seed)
        found = watch(rng)
        groups = {
            "raw": None,
            "discovered": found,
            "given": tuple(ROLES),
            "shuffled": shuffled_roles(rng),
        }
        compiled = {
            n: LibraryLearner(features=FEATURES, groups=g)._compiled()
            for n, g in groups.items()
        }
        for _ in range(RULES):
            pair = agreement_rule(rng)
            cases = agreement_examples(rng, pair, TEST)
            cx = np.asarray([c.inputs for c in cases], dtype=float)
            cy = np.asarray([c.target for c in cases], dtype=int)
            for count in COUNTS:
                examples = agreement_examples(rng, pair, count)
                x = np.asarray([e.inputs for e in examples], dtype=float)
                y = np.asarray([e.target for e in examples], dtype=int)
                for name, g in groups.items():
                    kind, payload = best_hypothesis(x, y, g, compiled[name])
                    got = apply_hypothesis(kind, payload, cx, g)
                    scores[name][count].append(float((got.astype(int) == cy).mean()))

    print(f"  {'labelled examples':<24}" + "".join(f"{c:>12}" for c in conditions))
    for count in COUNTS:
        row = "".join(f"{np.mean(scores[c][count]):>12.3f}" for c in conditions)
        print(f"  {count:<24}{row}")


def main() -> None:
    print(__doc__.strip().splitlines()[0])
    print(f"\n{len(SIZES)} roles over {FEATURES} features, {RULES} novel rules, {SEEDS} seeds.")
    print(f"Roles discovered from {WATCH} unlabelled situations -- no labels, no slot count.\n")

    conditions = ("raw", "discovered", "given", "shuffled")
    scores = {c: {n: [] for n in COUNTS} for c in conditions}
    spaces, recovered = {}, []

    for seed in range(SEEDS):
        rng = np.random.default_rng(seed)
        found = watch(rng)
        recovered.append(set(found) == set(ROLES))
        groups = {
            "raw": None,
            "discovered": found,
            "given": tuple(ROLES),
            "shuffled": shuffled_roles(rng),
        }
        for name, g in groups.items():
            spaces[name] = len(LibraryLearner(features=FEATURES, groups=g)._from_scratch())

        for _ in range(RULES):
            rule = make_rule(rng)
            cases = balanced(rng, rule, TEST)
            for count in COUNTS:
                examples = balanced(rng, rule, count)
                for name, g in groups.items():
                    learner = LibraryLearner(features=FEATURES, groups=g)
                    learner.teach("r", examples)
                    scores[name][count].append(learner.accuracy("r", cases))

    print(f"  roles recovered exactly: {sum(recovered)}/{SEEDS} seeds\n")
    print(f"  {'hypotheses considered':<24}" + "".join(f"{c:>12}" for c in conditions))
    print(f"  {'':<24}" + "".join(f"{spaces[c]:>12}" for c in conditions))

    print(f"\n  {'labelled examples':<24}" + "".join(f"{c:>12}" for c in conditions))
    for count in COUNTS:
        row = "".join(f"{np.mean(scores[c][count]):>12.3f}" for c in conditions)
        print(f"  {count:<24}{row}")

    print(f"\n  {'examples to reach 0.90':<24}", end="")
    for c in conditions:
        hit = next((n for n in COUNTS if np.mean(scores[c][n]) >= 0.90), None)
        print(f"{(hit if hit else '>' + str(COUNTS[-1])):>12}", end="")
    print()

    agreement_section()


if __name__ == "__main__":
    main()
