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

from athena.joints import ALPHA, EXCLUSION, cooccurrence, discover_slots, log_tail

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
    counts = counts[np.ix_(keep, keep)]
    occurrences = occurrences[keep]
    names = [items[i] for i in keep]

    expected = np.outer(occurrences, occurrences) / max(total, 1)
    exclusive = counts < EXCLUSION * expected
    np.fill_diagonal(exclusive, True)
    order = list(np.argsort(-occurrences))
    unassigned, slots = set(order), []
    while unassigned:
        seed = next(i for i in order if i in unassigned)
        group, mass = [seed], occurrences[seed]
        for candidate in order:
            if candidate in group or candidate not in unassigned:
                continue
            if not all(exclusive[candidate, m] for m in group):
                continue
            if mass + occurrences[candidate] > total * 1.35:
                continue
            group.append(candidate)
            mass += occurrences[candidate]
        unassigned -= set(group)
        slots.append(group)
    return [sorted(names[i] for i in g) for g in slots]


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


def mixed(rng, count: int, present: float) -> list[list[str]]:
    """Two roles in every situation; two more only sometimes.

    A table guarantees one value per attribute in every row -- exactly the
    assumption this method needs, and exactly what a parsed scene does not
    give you. Every object has a category; few have a stated material. The
    *mixed* case is the one that matters, and the one a uniform test hides:
    when every role is equally rare a coverage rule rejects all of them and
    falls back to keeping everything, which looks like success.
    """
    out = []
    for _ in range(count):
        picks = [int(rng.integers(0, len(SLOTS[s]))) for s in range(3)]
        picks.append(picks[2] % len(SLOTS[3]))
        roles = [0, 1] + ([2, 3] if rng.random() < present else [])
        out.append([SLOTS[s][picks[s]] for s in roles])
    return out


def optional_section() -> None:
    print("\n\nWhen only some roles are present, as in a parsed scene.")
    print("Roles 0 and 1 fill every situation; roles 2 and 3 appear at the given rate.\n")
    truth = {frozenset(s) for s in SLOTS}
    counts = (1500, 3000, 6000, 12000)
    print(f"  {'rare roles':>11}" + "".join(f"{n:>9}" for n in counts))
    for present in (0.6, 0.4, 0.3, 0.2, 0.1):
        row = ""
        for n in counts:
            exact = [
                len({frozenset(g) for g in discover_slots(mixed(np.random.default_rng(s), n, present))} & truth)
                for s in range(5)
            ]
            row += f"{np.mean(exact):>9.1f}"
        print(f"  {present:>10.0%}{row}")
    print("\n  Roles recovered exactly, out of 4. What matters is the direction along")
    print("  a row: a rare role is recoverable, it just needs the situations that")
    print("  contain it. An absolute coverage rule made this a wall instead --")
    print("  2/4 at 40% presence, and still 2/4 at twelve thousand situations.")


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
    print(f"a group is kept if some pair in it excludes at p < {ALPHA:g}.\n")
    print(f"  {'error rate':>11}  {'provable: roles':>16}{'purity':>8}   {'item floor: roles':>18}{'purity':>8}")
    for rate in (0.0, 0.02, 0.05, 0.10, 0.20, 0.35):
        results = {"provable": [[], []], "floor": [[], []]}
        for seed in range(8):
            rng = np.random.default_rng(seed)
            episodes = corrupt(rng, 800, rate)
            for key, fn in (("provable", discover_slots), ("floor", by_item_floor)):
                groups = fn(episodes)
                results[key][0].append(len(groups))
                results[key][1].append(grouping_purity(groups))
        a, b = results["provable"], results["floor"]
        print(
            f"  {rate:>11.2f}  {np.mean(a[0]):>16.2f}{np.mean(a[1]):>8.3f}"
            f"   {np.mean(b[0]):>18.2f}{np.mean(b[1]):>8.3f}"
        )

    print("\nBoth criteria reject debris. Only one keeps rare genuine values:")
    episodes, truth = load("mushroom")
    for label, groups in (("provable", discover_slots(episodes)), ("item floor", by_item_floor(episodes))):
        purity, exact, attributes = score(groups, truth)
        print(f"  mushroom, {label:<11} purity {purity:.3f}   exact {exact}/{attributes}")

    optional_section()


if __name__ == "__main__":
    main()
