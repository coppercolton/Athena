"""Tests for the lifelong agent loop.

Run with pytest, or directly: ``python3 tests/test_agent.py``.

The loop is the easiest place in the repository to write a test that passes
for the wrong reason, because every step has a degenerate version that looks
fine: a verdict of "known" that was never checked, a transfer that fit by
chance, a retention score over hypotheses that were never tested, a control
that changes two things. These pin each step to the behaviour it must have.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena.agent import LifelongAgent, Problem
from athena.hypotheses import HypothesisSpace, experiment
from athena.worlds import Agreement, Conjunction, Domain, Oracle, balanced, random_agreement, random_conjunction

ROLES_A = (("a", "b", "c", "d"), ("e", "f", "g"), ("h", "i", "j", "k"), ("l", "m"),
           ("n", "o", "p", "q"), ("r", "s", "t"), ("u", "v", "w", "x"), ("y", "z", "zz"))
ROLES_B = tuple(tuple(f + "2" for f in role) for role in ROLES_A)
A = Domain("A", ROLES_A, dependency=(2, 3))
B = Domain("B", ROLES_B, dependency=(2, 3))


def carry(rule, src, dst):
    if isinstance(rule, Agreement):
        return rule
    lits = []
    for f, w in rule.literals:
        r = next(i for i, g in enumerate(src.groups) if f in g)
        lits.append((int(dst.groups[r][src.groups[r].index(f)]), w))
    return Conjunction(tuple(sorted(lits)))


def fresh_agent(seed=0, **kw):
    rng = np.random.default_rng(seed)
    agent = LifelongAgent(seed, **kw)
    for d in (A, B):
        agent.watch(d, d.situations(rng, 1500))
    return agent, rng


def held_out(rng, domain, rule, n=40):
    tb, ty = balanced(rng, domain, rule, n)
    return domain.encode(tb), ty


def test_world_never_force_fills_and_fails_loudly():
    rng = np.random.default_rng(0)
    bags = A.situations(rng, 500, present=0.3)
    assert any(len(b) == 0 for b in bags)              # empty situations are allowed to exist
    tiny = Domain("tiny", (("a", "b"), ("c", "d")))
    try:
        balanced(rng, tiny, Conjunction(((0, True), (2, True))), 40)
    except ValueError:
        return
    raise AssertionError("a domain too small for the request must raise, not spin or duplicate")


def test_watching_recovers_the_roles():
    agent, _ = fresh_agent()
    assert set(agent.groups["A"]) == set(A.groups)


def test_experiment_pins_a_rule_the_agent_can_then_apply():
    rng = np.random.default_rng(1)
    space = HypothesisSpace(len(A.vocabulary), A.groups)
    for rule in (random_conjunction(rng, A, 2), random_agreement(rng, A)):
        h, x, y, used = experiment(space, A, Oracle(A, rule), rng, budget=30)
        assert h is not None and used <= 30
        tx, ty = held_out(rng, A, rule)
        assert (space.hypothesis(h).holds(tx).astype(int) == ty).mean() == 1.0


def test_novel_problem_is_learned_and_retained():
    agent, rng = fresh_agent()
    rule = random_conjunction(rng, A, 2)
    rec = agent.encounter(Problem(A, rule, "experiment", 40), Oracle(A, rule), test=held_out(rng, A, rule))
    assert rec.verdict == "novel" and rec.accuracy == 1.0
    assert len(agent.skills) == 1


def test_known_problem_costs_less_than_learning_it():
    agent, rng = fresh_agent()
    rule = random_conjunction(rng, A, 2)
    first = agent.encounter(Problem(A, rule, "experiment", 40), Oracle(A, rule), test=held_out(rng, A, rule))
    again = agent.encounter(Problem(A, rule, "experiment", 40), Oracle(A, rule), test=held_out(rng, A, rule))
    assert again.verdict == "known"
    assert again.accuracy == 1.0
    assert again.cost < first.cost


def test_a_shape_transfers_across_domains_that_share_no_fillers():
    agent, rng = fresh_agent()
    rule = random_conjunction(rng, A, 2)
    agent.encounter(Problem(A, rule, "experiment", 40), Oracle(A, rule), test=held_out(rng, A, rule))
    moved = carry(rule, A, B)
    scratch, rng2 = fresh_agent(seed=0)
    cold = scratch.encounter(Problem(B, moved, "experiment", 40), Oracle(B, moved), test=held_out(rng2, B, moved))
    warm = agent.encounter(Problem(B, moved, "experiment", 40), Oracle(B, moved), test=held_out(rng, B, moved))
    assert warm.verdict == "transfer" and warm.accuracy == 1.0
    assert cold.verdict == "novel"
    assert warm.cost < cold.cost


def test_transfer_verdict_is_not_given_to_a_genuinely_new_rule():
    """A carried shape must not be accepted just because something fit a few probes."""
    agent, rng = fresh_agent()
    rule = random_conjunction(rng, A, 2)
    agent.encounter(Problem(A, rule, "experiment", 40), Oracle(A, rule), test=held_out(rng, A, rule))
    for _ in range(4):
        other = random_conjunction(rng, B, 3)
        rec = agent.encounter(Problem(B, other, "experiment", 60), Oracle(B, other), test=held_out(rng, B, other))
        assert rec.verdict in ("novel", "failed"), rec
        if rec.verdict == "novel":
            assert rec.accuracy == 1.0


def test_verification_is_what_stops_false_transfer():
    """Same situation, verification off: the wrong candidate can get through."""
    hits = 0
    for seed in range(4):
        agent, rng = fresh_agent(seed=seed, verify=False)
        rule = random_conjunction(rng, A, 2)
        agent.encounter(Problem(A, rule, "experiment", 40), Oracle(A, rule), test=held_out(rng, A, rule))
        other = random_conjunction(rng, B, 3)
        rec = agent.encounter(Problem(B, other, "experiment", 40), Oracle(B, other), test=held_out(rng, B, other))
        hits += rec.verdict == "transfer" and rec.accuracy < 1.0
    assert hits > 0, "with verification off some wrong transfer should slip through; if not, the test above proves nothing"


def test_instruction_learns_from_examples_alone():
    agent, rng = fresh_agent()
    rule = random_conjunction(rng, A, 2)
    tx, ty = held_out(rng, A, rule)
    eb, ey = balanced(rng, A, rule, 40, exclude=set())
    rec = agent.encounter(Problem(A, rule, "instruction", 40), Oracle(A, rule), examples=(A.encode(eb), ey), test=(tx, ty))
    assert rec.verdict == "novel" and rec.accuracy >= 0.9


def test_backward_transfer_is_exactly_zero():
    agent, rng = fresh_agent()
    seen = []
    for _ in range(4):
        rule = random_conjunction(rng, A, 2)
        rec = agent.encounter(Problem(A, rule, "experiment", 40), Oracle(A, rule), test=held_out(rng, A, rule))
        seen.append(rec.accuracy)
    now = [(s.hypothesis.holds(s.test_x).astype(int) == s.test_y).mean() for s in agent.skills]
    assert np.allclose(now, seen)


def test_agent_never_draws_a_held_out_situation():
    """Every situation the agent asks about, verifies on, or pools from must be
    outside the set it will be scored on. A reviewer found the earlier version
    could query test situations through the oracle; this pins the fix."""
    agent, rng = fresh_agent()
    rule = random_conjunction(rng, A, 2)
    tb, ty = balanced(rng, A, rule, 40)
    exclude = frozenset(tuple(sorted(b)) for b in tb)
    seen = []
    class Spy:
        def __init__(self, inner): self.inner = inner
        def label(self, bags):
            seen.extend(tuple(sorted(b)) for b in bags)
            return self.inner.label(bags)
    for _ in range(3):
        agent.encounter(Problem(A, random_conjunction(rng, A, 2), "experiment", 60, exclude), Spy(Oracle(A, rule)), test=(A.encode(tb), ty))
    assert seen, "the spy saw no queries; the test proves nothing"
    assert not (set(seen) & exclude), "a held-out situation was queried"


def test_amnesiac_control_never_recognises_anything():
    agent, rng = fresh_agent()
    rule = random_conjunction(rng, A, 2)
    for _ in range(2):
        rec = agent.encounter(Problem(A, rule, "experiment", 40), Oracle(A, rule), test=held_out(rng, A, rule))
        agent.forget()
    assert rec.verdict == "novel" and not agent.skills


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
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
