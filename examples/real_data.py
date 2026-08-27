"""Does role discovery survive data nobody generated for it?

Every result in this repository so far, including round ten's, was measured on
symbols this repository produced. That is the standard way to fool yourself:
a generator and the algorithm that reads it share assumptions the world does
not have to honour -- balanced slots, clean exclusion, no rare values, no
attribute quietly determined by another.

So this runs the same ``discover_slots`` on three categorical datasets from the
UCI repository, published decades before any of this and by people with no
interest in it. Each row is stripped of its column structure and handed over as
an unordered bag of opaque tokens: ``t3_n`` is just an identifier, and the
algorithm never parses it. Two values from the same attribute look exactly like
two values from different attributes. The true column of each token is used
only to score the answer afterwards.

This is the honest version of the scene-graph test. A real extractor reading
"a brown cap above white gills" emits two facts that happen to share a colour
and belong to different roles, and nothing marks which is which -- exactly the
situation these bags reproduce.

Three numbers are reported:

    found     how many roles were recovered, against how many attributes exist
    purity    fraction of values placed in a group of otherwise-correct values
    exact     attributes recovered as a complete group, no value missing or added

The second half measures the debris the first half made necessary. A real
extractor invents items that belong to no role at all, and the criterion that
rejects them decides whether rare *genuine* values survive. Two candidate
criteria are compared under a rising extraction error rate.

    python3 examples/real_data.py
"""

from __future__ import annotations

import os
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena.joints import COVERAGE, EXCLUSION, _greedy_slots, cooccurrence, discover_slots

BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases"
CACHE = os.environ.get("ATHENA_DATA", os.path.join(os.path.dirname(__file__), "..", "data"))

# name -> (url suffix, attribute columns, first attribute column)
SETS = {
    "mushroom": (f"{BASE}/mushroom/agaricus-lepiota.data", 22, 1),  # column 0 is the class
    "car": (f"{BASE}/car/car.data", 6, 0),
    "nursery": (f"{BASE}/nursery/nursery.data", 8, 0),
}


def fetch(name: str) -> str:
    url, _, _ = SETS[name]
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{name}.csv")
    if not os.path.exists(path):
        with urllib.request.urlopen(url, timeout=60) as response:
            open(path, "wb").write(response.read())
    return path


def load(name: str):
    """Rows as unordered bags of opaque tokens. Column structure withheld."""
    _, width, offset = SETS[name]
    episodes, truth = [], {}
    for line in open(fetch(name)):
        fields = line.strip().split(",")[offset : offset + width]
        if len(fields) != width:
            continue
        bag = []
        for column, value in enumerate(fields):
            token = f"t{column}_{value}"
            truth[token] = column
            bag.append(token)
        episodes.append(bag)
    return episodes, truth


def score(groups: list[list[str]], truth: dict[str, int]) -> tuple[float, int, int]:
    attributes: dict[int, set[str]] = {}
    for token, column in truth.items():
        attributes.setdefault(column, set()).add(token)

    exact, correct, total = set(), 0, 0
    for group in groups:
        labels = [truth[t] for t in group if t in truth]
        if not labels:
            continue
        winner = max(set(labels), key=labels.count)
        correct += labels.count(winner)
        total += len(labels)
        if set(group) == attributes[winner]:
            exact.add(winner)
    return correct / max(total, 1), len(exact), len(attributes)


# ---------------------------------------------------------------------------
# The debris question, on symbols we do control so the error rate is known.

SLOTS = [
    ["water", "oil", "steam", "slurry"],
    ["pipe", "tank", "hose"],
    ["pressure", "heat", "sediment", "surge"],
    ["valve", "burst"],
]
TRUTH = {word: s for s, slot in enumerate(SLOTS) for word in slot}


def corrupt(rng, episodes: int, rate: float, pool: int = 25) -> list[list[str]]:
    """Situations, some of which the extractor got wrong."""
    out = []
    for _ in range(episodes):
        picks = [int(rng.integers(0, len(SLOTS[s]))) for s in range(3)]
        picks.append(picks[2] % len(SLOTS[3]))  # relief follows the buildup
        bag = [SLOTS[s][i] for s, i in enumerate(picks)]
        if rng.random() < rate:
            if rng.random() < 0.5:
                bag.append(f"stray{int(rng.integers(0, pool))}")  # invented item
            else:
                bag.append(SLOTS[int(rng.integers(0, 4))][0])  # hallucinated pair
        out.append(bag)
    return out


def by_item_floor(episodes: list[list[str]], floor: float = 0.05) -> list[list[str]]:
    """The rejected alternative: discard any item seen in under ``floor`` of situations.

    It holds the role count down under noise, which is why it was adopted, and
    it takes every rare genuine value with it, which is why it was dropped.
    """
    counts, items = cooccurrence(episodes)
    total = len(episodes)
    occurrences = np.diag(counts).copy()
    common = occurrences >= floor * total
    if common.sum() < 2:
        common = np.ones(len(items), dtype=bool)
    keep = np.flatnonzero(common)
    groups = _greedy_slots(counts[np.ix_(keep, keep)], occurrences[keep], total)
    return [sorted(items[keep[i]] for i in group) for group in groups]


def grouping_purity(groups: list[list[str]]) -> float:
    correct = total = 0
    for group in groups:
        labels = [TRUTH[i] for i in group if i in TRUTH]
        if not labels:
            continue
        winner = max(set(labels), key=labels.count)
        correct += labels.count(winner)
        total += len(labels)
    return correct / max(total, 1)


def optional(rng, count: int, present: float) -> list[list[str]]:
    """Situations where a role may simply be absent, as in a scene graph.

    A table guarantees one value per attribute in every row -- which is exactly
    the assumption this method needs, and exactly what a parsed scene does not
    give you. Not every object has a stated colour.
    """
    out = []
    for _ in range(count):
        bag = [
            SLOTS[s][int(rng.integers(0, len(SLOTS[s])))]
            for s in range(4)
            if rng.random() < present
        ]
        out.append(bag or [SLOTS[0][0]])
    return out


def optional_section() -> None:
    from athena import joints

    print("\n\nWhen a role is optional, as it is in a parsed scene.")
    print("The UCI tables above always fill every attribute; scenes do not.\n")
    truth = {frozenset(s) for s in SLOTS}
    thresholds = (0.5, 0.2)
    print(f"  {'role present in':>16}" + "".join(f"{f'coverage {t:g}':>15}" for t in thresholds))
    original = joints.COVERAGE
    try:
        for p in (0.8, 0.6, 0.5, 0.4, 0.3, 0.2):
            row = ""
            for t in thresholds:
                joints.COVERAGE = t
                exact = [
                    len({frozenset(g) for g in discover_slots(optional(np.random.default_rng(s), 1500, p))} & truth)
                    for s in range(6)
                ]
                row += f"{f'{np.mean(exact):.1f}/4':>15}"
            print(f"  {p:>15.0%}{row}")
    finally:
        joints.COVERAGE = original
    print("\n  Lowering the threshold buys one notch, to roles present in 40% of")
    print("  situations. Below about 30% nothing recovers it: a role that fills a")
    print("  third of the situations has stopped partitioning them, so a criterion")
    print("  built on partitioning can no longer see it. That is the boundary.")


def main() -> None:
    print(__doc__.strip().splitlines()[0])
    print("\nThree UCI categorical datasets, column structure withheld.")
    print("Every value is an opaque token; the algorithm is told nothing else.\n")
    print(f"  {'dataset':<10}{'rows':>8}{'values':>8}{'attributes':>12}{'found':>7}{'purity':>9}{'exact':>8}")
    for name in SETS:
        episodes, truth = load(name)
        groups = discover_slots(episodes)
        purity, exact, attributes = score(groups, truth)
        print(
            f"  {name:<10}{len(episodes):>8}{len(truth):>8}{attributes:>12}"
            f"{len(groups):>7}{purity:>9.3f}{f'{exact}/{attributes}':>8}"
        )

    print("\nWhat rejects an extractor's debris, and what it costs.")
    print(f"Four true roles. Exclusion below {EXCLUSION:g} of independence;")
    print(f"a group is kept if its members account for {COVERAGE:g} of the situations.\n")
    print(f"  {'error rate':>11}  {'coverage: roles':>16}{'purity':>8}   {'item floor: roles':>18}{'purity':>8}")
    for rate in (0.0, 0.02, 0.05, 0.10, 0.20, 0.35):
        results = {"coverage": [[], []], "floor": [[], []]}
        for seed in range(8):
            rng = np.random.default_rng(seed)
            episodes = corrupt(rng, 800, rate)
            for key, fn in (("coverage", discover_slots), ("floor", by_item_floor)):
                groups = fn(episodes)
                results[key][0].append(len(groups))
                results[key][1].append(grouping_purity(groups))
        a, b = results["coverage"], results["floor"]
        print(
            f"  {rate:>11.2f}  {np.mean(a[0]):>16.2f}{np.mean(a[1]):>8.3f}"
            f"   {np.mean(b[0]):>18.2f}{np.mean(b[1]):>8.3f}"
        )

    print("\nBoth criteria reject debris. Only one keeps rare genuine values:")
    episodes, truth = load("mushroom")
    for label, groups in (("coverage", discover_slots(episodes)), ("item floor", by_item_floor(episodes))):
        purity, exact, attributes = score(groups, truth)
        print(f"  mushroom, {label:<11} purity {purity:.3f}   exact {exact}/{attributes}")

    optional_section()


if __name__ == "__main__":
    main()
