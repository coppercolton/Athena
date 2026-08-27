"""Where does this stop working, and why?

Every result in rounds ten to thirteen was measured on four or five roles with
a handful of fillers each. That is small enough that "it works" carries almost
no information about whether it keeps working. So this maps the boundaries by
pushing each dimension until something breaks, and reports where.

Three dimensions matter, and they fail for different reasons:

    more roles      a situation with 64 attributes instead of 4. Every pair of
                    items has to be tested for exclusion, so the work grows
                    quadratically -- and so does the number of chances for a
                    pair to look exclusive by accident.
    more fillers    64 ways to fill a role instead of 4. Two alternatives for
                    one role are now each rare, and the evidence that they
                    exclude each other is correspondingly thinner.
    more situations what buys back the thinness above. The question is how
                    fast it has to grow.

The third is the one that decides whether this is usable, because roles and
fillers are set by the world and only the data is yours to choose.

    python3 examples/does_it_scale.py
"""

from __future__ import annotations

import os
import sys
import time
from math import comb

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena.joints import cooccurrence, discover_slots

SEEDS = 3


def world(rng, roles: int, fillers: int, episodes: int) -> tuple[list[list[str]], set]:
    """``roles`` attributes, ``fillers`` alternatives each, one chosen per role."""
    truth = {
        frozenset(f"r{r}_v{v}" for v in range(fillers)) for r in range(roles)
    }
    picks = rng.integers(0, fillers, size=(episodes, roles))
    bags = [
        [f"r{r}_v{picks[e, r]}" for r in range(roles)] for e in range(episodes)
    ]
    return bags, truth


def measure(roles: int, fillers: int, episodes: int) -> tuple[float, float]:
    """Fraction of roles recovered exactly, and seconds per run."""
    exact, elapsed = [], []
    for seed in range(SEEDS):
        rng = np.random.default_rng(seed)
        bags, truth = world(rng, roles, fillers, episodes)
        start = time.time()
        found = {frozenset(g) for g in discover_slots(bags)}
        elapsed.append(time.time() - start)
        exact.append(len(found & truth) / roles)
    return float(np.mean(exact)), float(np.mean(elapsed))


def main() -> None:
    print(__doc__.strip().splitlines()[0])

    print("\n\nMore roles, with data held fixed at 2000 situations, 4 fillers each.")
    print("Quadratically many pairs to test, and to get wrong.\n")
    print(f"  {'roles':>7}{'vocabulary':>12}{'recovered':>12}{'seconds':>10}")
    for roles in (4, 8, 16, 32, 64, 128):
        got, secs = measure(roles, 4, 2000)
        print(f"  {roles:>7}{roles * 4:>12}{got:>12.3f}{secs:>10.2f}", flush=True)

    print("\n\nMore fillers per role, 8 roles, 2000 situations.")
    print("Each alternative gets rarer, so the evidence for exclusion thins.\n")
    print(f"  {'fillers':>7}{'vocabulary':>12}{'recovered':>12}{'seconds':>10}")
    for fillers in (2, 4, 8, 16, 32, 64):
        got, secs = measure(8, fillers, 2000)
        print(f"  {fillers:>7}{fillers * 8:>12}{got:>12.3f}{secs:>10.2f}", flush=True)

    print("\n\nHow much data does a given world need? 8 roles.")
    print("The entry is the fraction of roles recovered exactly.\n")
    counts = (250, 500, 1000, 2000, 4000, 8000, 16000)
    print(f"  {'fillers':>7}" + "".join(f"{n:>8}" for n in counts))
    need = {}
    for fillers in (4, 8, 16, 32):
        row = ""
        for n in counts:
            got, _ = measure(8, fillers, n)
            row += f"{got:>8.2f}"
            if got == 1.0 and fillers not in need:
                need[fillers] = n
        print(f"  {fillers:>7}{row}", flush=True)

    print("\n  situations needed for exact recovery:")
    for fillers, n in sorted(need.items()):
        print(f"    {fillers:>3} fillers -> {n:>6}   ({n / fillers**2:.1f} x fillers squared)")

    print("\n\nWhat the discovered roles are then used for.")
    print("Rules over pairs of roles stay cheap; general conjunctions do not.\n")
    print(f"  {'roles':>7}{'agreement predicates':>22}{'conjunctions <=3 literals':>28}")
    for roles in (4, 16, 64, 256, 1024):
        vocab = roles * 4
        pairs = roles * (roles - 1) // 2
        conj = sum(comb(vocab, size) * (2**size) for size in (1, 2, 3))
        print(f"  {roles:>7}{pairs:>22,}{conj:>28,}")


if __name__ == "__main__":
    main()
