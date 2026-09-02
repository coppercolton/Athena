"""Does the gain open up when learning from scratch gets expensive?

Round sixteen found the loop only 3.7% cheaper than forgetting everything,
and gave a structural reason: a carried shape costs about twenty questions to
be believed, learning a rule from scratch costs about twenty-five, so there is
almost nothing for reuse to save. That is a prediction, and this tests it.

The same agent, the same controls, the same curriculum shape. Only the rules
get longer: two, three, then four literals, and the agent's hypothesis space
grows to match. Every rule has exactly the arity under test and the agent's space matches it,
so the within-domain extension problems are left out -- transfer is what is
being tested here. Learning from scratch has to search a space that grows
combinatorially; checking a carried shape has to disambiguate a few hundred
fillers. If the reasoning is right, the paired saving against the amnesiac
should grow with the arity. If it is wrong, it will not.

Nothing here was tuned on the result. The shape cap is raised so that the
carried shape is never cut off by a size limit, which would penalise transfer
for a reason that has nothing to do with the claim.

    python3 examples/harder_worlds.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import examples.lifelong_agent as bench
from athena.hypotheses import HypothesisSpace

SEEDS = 3
ARITIES = (2, 3, 4)


def main() -> None:
    print(__doc__.strip().splitlines()[0])
    bench.SEEDS = SEEDS
    bench.SHAPE_CAP = 60_000
    bench.EXTEND = False   # every rule has exactly the arity under test; the space matches it
    d = next(iter(bench.DOMAINS.values()))
    print(f"\n{SEEDS} seeds per arity. Cost = labels or questions before a verified hypothesis.\n")
    print(f"  {'arity':>6}{'hypotheses':>12}{'amnesiac':>10}{'agent':>8}{'saving':>8}{'':>3}{'by seed':<22}{'transfer':>9}{'acc agent':>10}{'acc amnesiac':>13}{'seconds':>8}")
    for arity in ARITIES:
        bench.ARITY = arity
        bench.MAX_LITERALS = arity
        size = HypothesisSpace(len(d.vocabulary), d.groups, arity).size
        start = time.time()
        agent = [bench.run("agent", s) for s in range(SEEDS)]
        amnesiac = [bench.run("amnesiac", s) for s in range(SEEDS)]
        cost_a = np.mean([sum(r.cost for r in rs) for rs in agent])
        cost_m = np.mean([sum(r.cost for r in rs) for rs in amnesiac])
        by_seed = [sum(r.cost for r in a) - sum(r.cost for r in m) for a, m in zip(agent, amnesiac)]
        transfer = np.mean([r.verdict == "transfer" for rs in agent for r in rs])
        acc_a = np.mean([r.accuracy for rs in agent for r in rs])
        acc_m = np.mean([r.accuracy for rs in amnesiac for r in rs])
        print(f"  {arity:>6}{size:>12,}{cost_m:>10.1f}{cost_a:>8.1f}{cost_a - cost_m:>+8.1f}{'':>3}{' '.join(f'{v:+d}' for v in by_seed):<22}{transfer:>9.2f}{acc_a:>10.3f}{acc_m:>13.3f}{time.time() - start:>8.0f}", flush=True)


if __name__ == "__main__":
    main()
