"""Can the vocabulary be learned instead of supplied?

Every mechanism in this repository composes flawlessly over a vocabulary
someone else wrote down, and none can produce one from what it observes. Round
nine made that concrete: analogical transfer between domains is exact with
shared roles and collapses to chance without them.

So this asks whether the roles can be *found*. A system watches situations go
by -- unordered bags of things that happened together, no labels, no slots, no
hint that "role" is a concept -- and must recover the slot structure, align it
across domains that share no fillers, and then do analogy over what it found.

The signals it is allowed to use are only statistical:

    within a domain   things in the same role never co-occur and keep
                      identical company
    across domains    a role is identified by its relational signature -- how
                      many ways it can be filled, and how strongly it moves
                      with the others -- which survives translation because it
                      mentions no filler

Four conditions, same algebra throughout, differing only in where the roles
came from:

    given          the true roles, handed over (the round-nine upper bound)
    discovered     roles recovered from co-occurrence, aligned by signature
    misaligned     roles correctly discovered, then aligned at random
    random         roles assigned at random (the floor)

If ``discovered`` approaches ``given``, the joints are learnable and the wall
this repository kept hitting is not where it appeared to be.

    python3 examples/finding_the_joints.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena.binding import Space
from athena.joints import align, discover_slots, slot_signature

DIMENSIONS = 20_000
EPISODES = 800
SEEDS = 12
SLOT_SIZES = (4, 3, 4, 2)  # the last is determined by the third

DOMAINS = {
    "plumbing": [["water", "oil", "steam", "slurry"], ["pipe", "tank", "hose"],
                 ["pressure", "heat", "sediment", "surge"], ["valve", "burst"]],
    "negotiation": [["tension", "money", "time", "ego"], ["talks", "contract", "summit"],
                    ["friction", "urgency", "grievance", "deadlock"], ["concession", "walkout"]],
    "geology": [["magma", "gas", "water_g", "ash"], ["chamber", "fault", "vent"],
                ["strain", "buoyancy", "sealing", "swarm"], ["eruption", "collapse"]],
    "market": [["demand", "credit", "inventory", "hype"], ["market", "auction", "exchange"],
               ["scarcity", "leverage", "backlog", "mania"], ["price_rise", "crash"]],
}


def generate(rng, domain: str, episodes: int):
    """Situations as unordered bags. The relief depends on what is building."""
    slots = DOMAINS[domain]
    out, picks = [], []
    for _ in range(episodes):
        chosen = [int(rng.integers(0, len(slots[s]))) for s in range(3)]
        chosen.append(chosen[2] % len(slots[3]))  # relief follows the buildup
        out.append([slots[s][i] for s, i in enumerate(chosen)])
        picks.append(chosen)
    return out, picks


def purity(found: list[list[str]], domain: str) -> float:
    """Fraction of items placed in a group containing only same-slot items."""
    truth = {item: s for s, slot in enumerate(DOMAINS[domain]) for item in slot}
    correct = total = 0
    for group in found:
        labels = [truth[i] for i in group if i in truth]
        if not labels:
            continue
        winner = max(set(labels), key=labels.count)
        correct += labels.count(winner)
        total += len(labels)
    return correct / max(total, 1)


def transfer(space, roles_a, roles_b, fillers_a, fillers_b, vocab) -> tuple[int, int]:
    """Build both situations, multiply them, and send each element across."""
    a = space.bundle(*[space.bind(space.atom(r), space.atom(f)) for r, f in zip(roles_a, fillers_a)])
    b = space.bundle(*[space.bind(space.atom(r), space.atom(f)) for r, f in zip(roles_b, fillers_b)])
    m = space.bind(a, b)
    hits = 0
    for element, answer in zip(fillers_a, fillers_b):
        got = space.nearest(space.bind(space.atom(element), m), vocab)[0]
        hits += got == answer
    return hits, len(fillers_a)


def main() -> None:
    print(__doc__.strip().splitlines()[0])
    print(f"\n{len(DOMAINS)} domains, {EPISODES} unlabelled situations each, {SEEDS} seeds.")
    print("No roles, no slot count, no supervision of any kind.\n")

    vocab = [w for slots in DOMAINS.values() for slot in slots for w in slot]
    names = list(DOMAINS)
    scores = {k: [0, 0] for k in ("given", "discovered", "misaligned", "random")}
    purities, slot_counts = [], []

    for seed in range(SEEDS):
        rng = np.random.default_rng(seed)
        space = Space(DIMENSIONS, seed=seed)

        found, signatures, picks_by_domain = {}, {}, {}
        for domain in names:
            episodes, picks = generate(rng, domain, EPISODES)
            groups = discover_slots(episodes)
            found[domain] = groups
            signatures[domain] = slot_signature(episodes, groups)
            picks_by_domain[domain] = picks
            purities.append(purity(groups, domain))
            slot_counts.append(len(groups))

        for source in names:
            for target in names:
                if source == target:
                    continue
                # Paired situations: same indices, so correspondence is known.
                indices = picks_by_domain[source][0]
                fa = [DOMAINS[source][s][i] for s, i in enumerate(indices)]
                fb = [DOMAINS[target][s][i] for s, i in enumerate(indices)]

                true_roles = [f"ROLE{s}" for s in range(4)]
                hit, tot = transfer(space, true_roles, true_roles, fa, fb, vocab)
                scores["given"][0] += hit; scores["given"][1] += tot

                # Discovered: name each found group, then align by signature.
                src_groups, tgt_groups = found[source], found[target]
                if len(src_groups) == len(tgt_groups) == len(SLOT_SIZES):
                    order = align(signatures[source], signatures[target])
                    where_s = {i: g for g, grp in enumerate(src_groups) for i in grp}
                    where_t = {i: g for g, grp in enumerate(tgt_groups) for i in grp}
                    ra = [f"D{where_s[f]}" for f in fa]
                    rb = [f"D{order[where_t[f]]}" if order[where_t[f]] >= 0 else "D9" for f in fb]
                    hit, tot = transfer(space, ra, rb, fa, fb, vocab)
                    scores["discovered"][0] += hit; scores["discovered"][1] += tot

                    shuffled = list(rng.permutation(4))
                    rb2 = [f"D{shuffled[where_t[f]]}" for f in fb]
                    hit, tot = transfer(space, ra, rb2, fa, fb, vocab)
                    scores["misaligned"][0] += hit; scores["misaligned"][1] += tot

                rr_a = [f"R{int(rng.integers(0, 4))}" for _ in fa]
                rr_b = [f"R{int(rng.integers(0, 4))}" for _ in fb]
                hit, tot = transfer(space, rr_a, rr_b, fa, fb, vocab)
                scores["random"][0] += hit; scores["random"][1] += tot

    print(f"  roles discovered per domain: {np.mean(slot_counts):.2f} (true value 4)")
    print(f"  grouping purity:             {np.mean(purities):.3f}\n")
    print(f"  {'roles come from':<22}{'transfer accuracy':>19}")
    for key in ("given", "discovered", "misaligned", "random"):
        hit, tot = scores[key]
        print(f"  {key:<22}{(hit / max(tot, 1)):>19.3f}")
    print(f"  {'chance':<22}{1 / len(vocab):>19.3f}")


if __name__ == "__main__":
    main()
