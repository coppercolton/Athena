"""Tests for the library learner and its hypothesis space.

Run with pytest, or directly: ``python3 tests/test_library.py``.

Two things here are easy to get wrong in ways that still run. The candidate
space is memoised and compiled to arrays, so a bug there silently changes what
every learner can reach; and the role constraint must shrink the space *without*
removing anything satisfiable, or a win could be an artefact of having quietly
excluded the competition.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena.continual import Sample
from athena.library import LibraryLearner
from athena.taught import Rule

GROUPS = ((0, 1, 2), (3, 4, 5), (6, 7))
FEATURES = 8


def situations(rng, count):
    x = np.zeros((count, FEATURES))
    for group in GROUPS:
        picks = rng.integers(0, len(group), size=count)
        x[np.arange(count), np.asarray(group)[picks]] = 1.0
    return x


def taught(rule, rng, count, groups=None):
    x = situations(rng, count)
    y = rule.holds(x).astype(int)
    learner = LibraryLearner(features=FEATURES, groups=groups)
    learner.teach("r", [Sample(inputs=row, target=int(t)) for row, t in zip(x, y)])
    return learner


def test_role_constraint_shrinks_the_space():
    raw = LibraryLearner(features=FEATURES)._from_scratch()
    grouped = LibraryLearner(features=FEATURES, groups=GROUPS)._from_scratch()
    assert len(grouped) < len(raw), (len(grouped), len(raw))


def test_role_constraint_removes_only_contradictions():
    """Everything dropped must be unsatisfiable in this world, or the smaller
    space is a hint about the answer rather than a statement about the world."""
    rng = np.random.default_rng(0)
    x = situations(rng, 4000)
    raw = set(LibraryLearner(features=FEATURES)._from_scratch())
    grouped = set(LibraryLearner(features=FEATURES, groups=GROUPS)._from_scratch())
    assert grouped <= raw
    for literals in raw - grouped:
        assert not Rule(literals).holds(x).any(), literals


def test_every_satisfiable_conjunction_survives():
    grouped = set(LibraryLearner(features=FEATURES, groups=GROUPS)._from_scratch())
    rng = np.random.default_rng(1)
    x = situations(rng, 4000)
    for literals in LibraryLearner(features=FEATURES)._from_scratch():
        if Rule(literals).holds(x).any():
            assert literals in grouped, literals


def test_compiled_space_matches_the_candidate_list():
    learner = LibraryLearner(features=FEATURES, groups=GROUPS)
    candidates, index, polarity, used = learner._compiled()
    assert len(candidates) == len(index)
    rng = np.random.default_rng(2)
    x = situations(rng, 200)
    holds = (((x[:, index] > 0.5) == polarity) | ~used).all(axis=2)
    for position in rng.integers(0, len(candidates), size=40):
        expected = Rule(candidates[position]).holds(x)
        assert np.array_equal(holds[:, position], expected), candidates[position]


def test_learns_a_conjunction_it_can_express():
    rule = Rule(((0, True), (3, True)))
    for groups in (None, GROUPS):
        learner = taught(rule, np.random.default_rng(3), 200, groups)
        x = situations(np.random.default_rng(9), 500)
        cases = [Sample(inputs=r, target=int(t)) for r, t in zip(x, rule.holds(x))]
        assert learner.accuracy("r", cases) == 1.0, groups


def test_constraint_does_not_change_what_is_learned():
    """The pruned space must reach the same answer -- round thirteen's null
    result depends on this being true rather than on the two spaces differing."""
    rng = np.random.default_rng(4)
    for _ in range(10):
        rule = Rule(((int(rng.integers(0, 3)), True), (int(rng.integers(3, 6)), True)))
        seed = int(rng.integers(0, 10_000))
        a = taught(rule, np.random.default_rng(seed), 120, None)
        b = taught(rule, np.random.default_rng(seed), 120, GROUPS)
        x = situations(np.random.default_rng(seed + 1), 400)
        cases = [Sample(inputs=r, target=int(t)) for r, t in zip(x, rule.holds(x))]
        assert a.accuracy("r", cases) == b.accuracy("r", cases)


def test_memoised_space_is_not_shared_across_group_settings():
    raw = LibraryLearner(features=FEATURES)._from_scratch()
    grouped = LibraryLearner(features=FEATURES, groups=GROUPS)._from_scratch()
    again = LibraryLearner(features=FEATURES)._from_scratch()
    assert len(again) == len(raw) != len(grouped)


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
