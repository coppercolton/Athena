"""Tests for role discovery.

Run with pytest, or directly: ``python3 tests/test_joints.py``.

Discovering the vocabulary is the one place in this repository where nothing
is supplied, so it is also the easiest place to write a test that passes for
the wrong reason. A grouping can be perfectly *pure* and still useless if it
returns every item as its own singleton, and it can find exactly the right
*number* of groups while shuffling the members. These check both halves, plus
the two failure modes that earlier versions of the criterion actually had.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import athena.joints as joints
from athena.joints import align, cooccurrence, discover_slots, slot_signature

SLOTS = [
    ["water", "oil", "steam", "slurry"],
    ["pipe", "tank", "hose"],
    ["pressure", "heat", "sediment", "surge"],
    ["valve", "burst"],
]


def episodes(rng, count: int, rate: float = 0.0, dependent: bool = True):
    """Situations as unordered bags, optionally with a dependency and extractor noise."""
    out = []
    for _ in range(count):
        picks = [int(rng.integers(0, len(SLOTS[s]))) for s in range(3)]
        if dependent:
            picks.append(picks[2] % len(SLOTS[3]))  # slot 3 follows slot 2
        else:
            picks.append(int(rng.integers(0, len(SLOTS[3]))))
        bag = [SLOTS[s][i] for s, i in enumerate(picks)]
        if rng.random() < rate:
            if rng.random() < 0.5:
                bag.append(f"stray{int(rng.integers(0, 25))}")
            else:
                bag.append(SLOTS[int(rng.integers(0, 4))][0])
        out.append(bag)
    return out


def as_sets(groups):
    return {frozenset(g) for g in groups}


TRUE = {frozenset(s) for s in SLOTS}


def test_recovers_the_exact_slots():
    found = discover_slots(episodes(np.random.default_rng(0), 800))
    assert as_sets(found) == TRUE, found


def test_a_dependency_does_not_fuse_or_shatter_slots():
    """Slot 3 is determined by slot 2, so cross-slot pairs also never co-occur.

    Exclusion alone fuses the two; distributional equivalence alone shatters
    slot 3. Only the counting rule separates them.
    """
    found = discover_slots(episodes(np.random.default_rng(1), 800, dependent=True))
    assert len(found) == 4, found
    assert as_sets(found) == TRUE, found


def test_survives_extraction_noise():
    """Exclusion is a rate, not an exact zero: one bad pair must not sever a slot."""
    for rate in (0.05, 0.20, 0.35):
        for seed in range(4):
            found = discover_slots(episodes(np.random.default_rng(seed), 800, rate=rate))
            assert len(found) == 4, (rate, seed, found)
            assert as_sets(found) == TRUE, (rate, seed, found)


def test_debris_is_rejected_by_coverage_not_by_rarity():
    """A rare genuine alternative is kept; a group that explains nothing is not.

    Slot 0 gets a fifth filler chosen in 1% of situations -- well under any
    sensible frequency floor, and still a real alternative for that slot.
    """
    rng = np.random.default_rng(3)
    data = episodes(rng, 2000, rate=0.10)
    for bag in data:
        if rng.random() < 0.01:
            bag[bag.index(next(i for i in bag if i in SLOTS[0]))] = "brine"
    found = discover_slots(data)
    assert len(found) == 4, found
    home = next(g for g in found if "water" in g)
    assert "brine" in home, found
    assert not any(i.startswith("stray") for g in found for i in g), found


def test_grouping_is_a_partition():
    """Every item lands in exactly one group -- no duplicates, nothing dropped."""
    data = episodes(np.random.default_rng(4), 800)
    found = discover_slots(data)
    placed = [i for g in found for i in g]
    assert len(placed) == len(set(placed)), placed
    assert set(placed) == {i for bag in data for i in bag}


def test_signature_mentions_no_filler_and_matches_across_domains():
    """Two domains sharing no vocabulary must still align by relational shape."""
    rng = np.random.default_rng(5)
    data = episodes(rng, 1200)
    renamed = [[f"{i}_b" for i in bag] for bag in data]

    a, b = discover_slots(data), discover_slots(renamed)
    sig_a = slot_signature(data, a)
    sig_b = slot_signature(renamed, b)
    order = align(sig_a, sig_b)

    assert sorted(order) == list(range(len(a))), order
    for source, target in enumerate(order):
        assert {f"{i}_b" for i in a[source]} == set(b[target]), (a[source], b[target])


def test_no_structure_yields_no_confident_roles():
    """Independent items fill no slot between them; nothing should be invented."""
    rng = np.random.default_rng(6)
    vocab = [f"x{i}" for i in range(12)]
    data = [[w for w in vocab if rng.random() < 0.5] or [vocab[0]] for _ in range(800)]
    found = discover_slots(data)
    assert as_sets(found) != TRUE
    assert all(len(g) <= 2 for g in found), found


def test_cooccurrence_is_exact_at_any_block_size():
    """Blocking bounds memory and must not touch the result -- including the
    duplicate items a noisy extractor produces, which are counted with
    multiplicity."""
    data = episodes(np.random.default_rng(7), 600, rate=0.35)
    reference = np.zeros((0, 0))
    items: list[str] = []
    original = joints.BLOCK
    try:
        for block in (4096, 128, 7, 1):
            joints.BLOCK = block
            counts, items = cooccurrence(data)
            if reference.size == 0:
                reference = np.zeros((len(items), len(items)))
                where = {item: i for i, item in enumerate(items)}
                for episode in data:
                    ids = [where[i] for i in episode]
                    for a in ids:
                        for b in ids:
                            reference[a, b] += 1
            assert np.array_equal(counts, reference), block
    finally:
        joints.BLOCK = original


def test_scales_to_many_roles():
    """Roles are close to free: the cost is in pairs of items, not in slots."""
    rng = np.random.default_rng(8)
    roles, fillers = 40, 4
    truth = {frozenset(f"r{r}_v{v}" for v in range(fillers)) for r in range(roles)}
    picks = rng.integers(0, fillers, size=(2000, roles))
    data = [[f"r{r}_v{picks[e, r]}" for r in range(roles)] for e in range(2000)]
    found = {frozenset(g) for g in discover_slots(data)}
    assert found == truth, len(found & truth)


def test_wide_roles_need_more_situations_not_a_different_method():
    """A 16-way attribute is recoverable; it just needs the data to see the
    pairs. This pins the failure as statistical rather than structural."""
    def run(count):
        rng = np.random.default_rng(9)
        truth = {frozenset(f"r{r}_v{v}" for v in range(16)) for r in range(6)}
        picks = rng.integers(0, 16, size=(count, 6))
        data = [[f"r{r}_v{picks[e, r]}" for r in range(6)] for e in range(count)]
        return len({frozenset(g) for g in discover_slots(data)} & truth)

    assert run(200) < 6
    assert run(4000) == 6


if __name__ == "__main__":
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_") and callable(o)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
