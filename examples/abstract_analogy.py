"""The real test: concepts with no surface in common.

Binding keeps "red cube" apart from "blue cube", but nothing there is abstract.
"Pressure" in a pipe and "pressure" in a negotiation share no colour, no shape,
no sensory feature whatsoever. What they share is the role something plays: a
quantity accumulating against a constraint until something gives.

So the real test of a compositional representation is analogy — carrying an
answer between situations with nothing in common on the surface.

Bound structures do this by multiplication alone. For two situations described
over the same roles, their product cancels the roles and leaves a mapping
between their fillers, so any element of one can be sent through it into the
other. The roles are never named, looked up, or even known.

And the same expression shows the boundary. The cancellation needs both
situations carved up by the same roles. When each domain describes itself in
its own vocabulary, nothing cancels and the mapping is noise -- which is
measured here beside the success, because it is the more important number.

    python3 examples/abstract_analogy.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena.analogy import carry_across, mapping, structure, unaligned_structure
from athena.binding import Space

DIMENSIONS = 20_000
SEEDS = 20

ROLES = ("MEDIUM", "VESSEL", "BUILDS", "RELIEF")
DOMAINS = {
    "plumbing": ("water", "pipe", "pressure", "valve"),
    "negotiation": ("tension", "talks", "friction", "concession"),
    "market": ("demand", "market", "scarcity", "price_rise"),
    "geology": ("magma", "chamber", "strain", "eruption"),
    "psychology": ("stress", "person", "anxiety", "outburst"),
}
VOCAB = [w for fillers in DOMAINS.values() for w in fillers]


def describe(space: Space, name: str, aligned: bool = True):
    fillers = dict(zip(ROLES, DOMAINS[name]))
    if aligned:
        return structure(space, fillers)
    return unaligned_structure(space, fillers, f"_{name}")


def transfer_accuracy(aligned: bool) -> float:
    """Across every ordered pair of domains, how often does an element land right?"""
    correct = total = 0
    for seed in range(SEEDS):
        space = Space(DIMENSIONS, seed=seed)
        for source in DOMAINS:
            for target in DOMAINS:
                if source == target:
                    continue
                translation = mapping(
                    space, describe(space, source, aligned), describe(space, target, aligned)
                )
                for index, element in enumerate(DOMAINS[source]):
                    answer = space.nearest(carry_across(space, element, translation), VOCAB)[0]
                    correct += answer == DOMAINS[target][index]
                    total += 1
    return correct / total


def main() -> None:
    print(__doc__.strip().splitlines()[0])
    print(f"\n{len(DOMAINS)} domains sharing one relational skeleton, no shared fillers.")
    print(f"{DIMENSIONS}-dimensional space, {SEEDS} seeds, every ordered pair of domains.\n")

    space = Space(DIMENSIONS, seed=0)
    print("Example transfers:")
    for target in ("negotiation", "geology", "psychology"):
        translation = mapping(space, describe(space, "plumbing"), describe(space, target))
        answers = [
            f"{element}->{space.nearest(carry_across(space, element, translation), VOCAB)[0]}"
            for element in DOMAINS["plumbing"]
        ]
        print(f"  plumbing -> {target:12s} " + "  ".join(answers))

    aligned = transfer_accuracy(True)
    unaligned = transfer_accuracy(False)
    chance = 1.0 / len(VOCAB)

    print(f"\n{'condition':<34}{'accuracy':>10}")
    print(f"  {'shared roles':<32}{aligned:>10.3f}")
    print(f"  {'each domain its own roles':<32}{unaligned:>10.3f}")
    print(f"  {'chance':<32}{chance:>10.3f}")
    print(
        "\nAnalogical transfer across domains with zero shared surface features is\n"
        "exact — when both are carved up the same way. Give each domain its own\n"
        "vocabulary of roles and it collapses to chance. Nothing about the\n"
        "situations changed; only who chose the joints."
    )


if __name__ == "__main__":
    main()
