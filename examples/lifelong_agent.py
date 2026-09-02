"""Does the loop get better at learning, or only bigger?

One agent meets a stream of problems across several domains that share no
fillers. Each problem is a rule it has never seen, delivered either as
labelled examples or as an oracle it may question. It watches each domain
first (unlabelled, free), discovers the roles, and then must: say whether
anything it retains explains the problem; learn it if not; solve it on
situations it has never seen; keep it; and keep every earlier skill intact.

The claim under test is compounding: the Nth problem should cost fewer labels
or questions than the first, because the store carries shapes across domains
and sub-expressions within them. The controls are what make that claim mean
something, and each changes exactly one thing:

    amnesiac      the store is wiped after every problem
    no-transfer   skills are kept but never carried across domains
    shuffled      roles replaced by a random grouping of the same shape
    reversed      the same problems in reverse order

The curriculum is fixed and identical across conditions, held-out sets are
distinct situations disjoint from everything the agent saw, and surprise is
reported beside competence, never instead of it.

    python3 examples/lifelong_agent.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena.agent import LifelongAgent, Problem
from athena.worlds import Agreement, Conjunction, Domain, Oracle, balanced, random_agreement, random_conjunction

WATCH = 1500
TEST = 40
BUDGET = 60
SEEDS = 5

VOCAB = {
    "plumbing": (("water", "oil", "steam", "slurry"), ("pipe", "tank", "hose"), ("pressure", "heat", "sediment", "surge"),
                 ("valve", "burst"), ("copper", "iron", "pvc", "lead"), ("dawn", "noon", "dusk"), ("wet", "dry", "icy", "hot"), ("rural", "urban", "coastal")),
    "negotiation": (("tension", "money", "time", "ego"), ("talks", "contract", "summit"), ("friction", "urgency", "grievance", "deadlock"),
                    ("concession", "walkout"), ("lawyer", "broker", "envoy", "rival"), ("open", "mid", "close"), ("calm", "tense", "hostile", "warm"), ("local", "national", "global")),
    "geology": (("magma", "gas", "brine", "ash"), ("chamber", "fault", "vent"), ("strain", "buoyancy", "sealing", "swarm"),
                ("eruption", "collapse"), ("basalt", "granite", "shale", "tuff"), ("early", "middle", "late"), ("cold", "warm", "molten", "frozen"), ("island", "rift", "plateau")),
}
DOMAINS = {n: Domain(n, r, dependency=(2, 3)) for n, r in VOCAB.items()}


def carry(rule, src: Domain, dst: Domain):
    """The same abstract rule in another domain, by role and position."""
    if isinstance(rule, Agreement):
        return Agreement(rule.role_a, rule.role_b)
    lits = []
    for f, w in rule.literals:
        r = next(i for i, g in enumerate(src.groups) if f in g)
        lits.append((int(dst.groups[r][src.groups[r].index(f)]), w))
    return Conjunction(tuple(sorted(lits)))


def extend(rng, d: Domain, base: Conjunction) -> Conjunction:
    """``base`` conjoined with one filler from a role it does not mention, and
    verified satisfiable -- a dependency can still forbid the pair."""
    from athena.worlds import satisfiable
    used = {next(i for i, g in enumerate(d.groups) if f in g) for f, _ in base.literals}
    free = [r for r in range(len(d.groups)) if r not in used]
    for _ in range(200):
        r = int(rng.choice(free))
        f = int(rng.choice(d.groups[r]))
        rule = Conjunction(tuple(sorted((*base.literals, (f, True)))))
        if satisfiable(rng, d, rule):
            return rule
    return random_conjunction(rng, d, 3)


def curriculum(rng) -> list[Problem]:
    """Rules that recur: within a domain as shared sub-expressions, across
    domains as shared shapes, plus fresh ones. Channels alternate."""
    names = list(DOMAINS)
    problems: list[tuple[str, object]] = []
    base = {n: random_conjunction(rng, DOMAINS[n], 2) for n in names}
    # Every round mixes a fresh two-literal rule, a carried shape, and a
    # three-literal extension, so position never encodes difficulty -- the
    # order confound the literature warns about, and the first version of
    # this curriculum had.
    for round_ in range(3):
        n0, n1, n2 = names[round_ % 3], names[(round_ + 1) % 3], names[(round_ + 2) % 3]
        problems.append((n0, base[n0] if round_ == 0 else random_conjunction(rng, DOMAINS[n0], 2)))
        problems.append((n1, carry(base[n0], DOMAINS[n0], DOMAINS[n1])))       # a shape seen elsewhere
        problems.append((n2, extend(rng, DOMAINS[n2], base[n2])))               # base concept plus one literal
        problems.append((n1, random_agreement(rng, DOMAINS[n1])))
    out = []
    for i, (n, rule) in enumerate(problems):
        out.append(Problem(DOMAINS[n], rule, "instruction" if i % 2 == 0 else "experiment", BUDGET))
    return out


def run(condition: str, seed: int) -> list:
    rng = np.random.default_rng(seed)
    agent = LifelongAgent(seed, transfer=condition != "no-transfer")
    watched = {n: DOMAINS[n].situations(rng, WATCH) for n in DOMAINS}
    for n, d in DOMAINS.items():
        if condition == "shuffled":
            perm = rng.permutation(len(d.vocabulary))
            groups, at = [], 0
            for g in d.groups:
                groups.append(tuple(sorted(int(i) for i in perm[at: at + len(g)]))); at += len(g)
            agent.watch(d, watched[n], tuple(groups))
        else:
            agent.watch(d, watched[n])
    problems = curriculum(np.random.default_rng(1000 + seed))
    if condition == "reversed":
        problems = problems[::-1]
    records = []
    for p in problems:
        tb, ty = balanced(rng, p.domain, p.rule, TEST)
        exclude = frozenset(tuple(sorted(b)) for b in tb)
        p = Problem(p.domain, p.rule, p.channel, p.budget, exclude)   # the agent may not draw these
        test = (p.domain.encode(tb), ty)
        oracle = Oracle(p.domain, p.rule)
        if p.channel == "instruction":
            eb, ey = balanced(rng, p.domain, p.rule, p.budget, exclude=exclude)
            rec = agent.encounter(p, oracle, examples=(p.domain.encode(eb), ey), test=test)
        else:
            rec = agent.encounter(p, oracle, test=test)
        records.append(rec)
        if condition == "amnesiac":
            agent.forget()
    # backward transfer, computed rather than assumed: each skill's accuracy on
    # its own held-out now, minus what it was when committed.
    final = {s.learned_at: float((s.hypothesis.holds(s.test_x).astype(int) == s.test_y).mean()) for s in agent.skills}
    for r in records:
        r.bwt = final.get(r.index, r.accuracy) - r.accuracy
    return records


def main() -> None:
    print(__doc__.strip().splitlines()[0])
    conditions = ("agent", "amnesiac", "no-transfer", "shuffled", "reversed")
    results = {c: [run(c, s) for s in range(SEEDS)] for c in conditions}
    n = len(results["agent"][0])
    print(f"\n{len(DOMAINS)} domains, {n} problems, {SEEDS} seeds. Cost = labels consumed or questions asked before a verified hypothesis.\n")

    print(f"  {'problem':>8}" + "".join(f"{c:>13}" for c in conditions) + "   verdicts (agent)")
    for i in range(n):
        row = ""
        for c in conditions:
            row += f"{np.mean([r[i].cost for r in results[c]]):>13.1f}"
        verdicts = "/".join(sorted({r[i].verdict for r in results['agent']}))
        print(f"  {i:>8}{row}   {verdicts}")

    paired = [np.mean([a[i].cost - b[i].cost for a, b in zip(results["agent"], results["amnesiac"])]) for i in range(n)]
    print(f"\n  paired cost, agent minus amnesiac, per problem: " + " ".join(f"{p:+.0f}" for p in paired))
    print(f"  total over the curriculum: {np.sum(paired):+.1f} labels/questions ({np.sum(paired) / np.sum([np.mean([r[i].cost for r in results['amnesiac']]) for i in range(n)]):+.1%})")
    per_seed = [sum(r.cost for r in a) - sum(r.cost for r in b) for a, b in zip(results["agent"], results["amnesiac"])]
    print(f"  per seed: {' '.join(f'{d:+d}' for d in per_seed)}   (agent minus amnesiac, total cost; negative is better)")

    print(f"\n  {'summary':<28}" + "".join(f"{c:>13}" for c in conditions))
    for label, fn in (
        ("total cost", lambda rs: np.sum([r.cost for r in rs])),
        ("mean cost, first third", lambda rs: np.mean([r.cost for r in rs[: n // 3]])),
        ("mean cost, last third", lambda rs: np.mean([r.cost for r in rs[-(n // 3):]])),
        ("held-out accuracy", lambda rs: np.mean([r.accuracy for r in rs])),
        ("solved by transfer", lambda rs: np.mean([r.verdict == "transfer" for r in rs])),
        ("solved as known", lambda rs: np.mean([r.verdict == "known" for r in rs])),
        ("failed", lambda rs: np.mean([r.verdict == "failed" for r in rs])),
        ("retention of earlier skills", lambda rs: np.mean([r.retention for r in rs])),
        ("backward transfer", lambda rs: np.mean([r.bwt for r in rs])),
        ("accuracy, instruction", lambda rs: np.mean([r.accuracy for r in rs if r.channel == "instruction"])),
        ("accuracy, experiment", lambda rs: np.mean([r.accuracy for r in rs if r.channel == "experiment"])),
        # The tracker's gain sits at 1.0 in quiet periods (checklist item 13), so
        # the informative quantity is the raw first-probe error of the best prior
        # candidate: high when nothing retained explains the problem, zero when
        # something does. Reported beside competence, never instead of it.
        ("surprise: first-probe error", lambda rs: np.mean([r.surprise for r in rs])),
        ("surprise on novel problems", lambda rs: np.mean([r.surprise for r in rs if r.verdict == "novel"] or [float("nan")])),
        ("surprise on recognised", lambda rs: np.mean([r.surprise for r in rs if r.verdict in ("known", "transfer")] or [float("nan")])),
    ):
        print(f"  {label:<28}" + "".join(f"{np.mean([fn(r) for r in results[c]]):>13.3f}" for c in conditions))


if __name__ == "__main__":
    main()
