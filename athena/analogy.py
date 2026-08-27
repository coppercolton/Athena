"""Abstract concepts: the same structure wearing different surfaces.

Binding solved the concrete case — "red cube" stays distinct from "blue cube"
because the pairing survives. But nothing abstract has been touched. "Pressure"
in a pipe and "pressure" in a negotiation share no colour, no shape, no sensory
atom at all. What they share is the *role* something plays: a quantity that
accumulates against a constraint until something gives.

So the real test of a compositional representation is not recognition, it is
**analogy** — recognising that two situations with nothing in common on the
surface have the same shape underneath, and carrying an answer from one to the
other.

There is a remarkable fact about bound representations here. Given two
structures over the same roles

    A = role1*a1 + role2*a2 + role3*a3
    B = role1*b1 + role2*b2 + role3*b3

their product ``A * B`` contains ``a1*b1 + a2*b2 + a3*b3`` plus noise, because
each role cancels against itself. That product is a **mapping between the two
domains**, and

    a2 * (A * B)  ≈  b2

carries any element across. The roles never have to be named, inspected, or
even known. Two structured situations produce, by multiplication alone, the
translation between them.

That is analogical transfer as arithmetic, and it is the closest thing here to
what abstraction is for.

The honest boundary is in the same expression. It works because the roles
cancel — which requires both situations to be *described in the same
vocabulary of roles*. Two domains carved up differently produce no cancellation
and no mapping, and that failure is measured alongside the success rather than
left out.
"""

from __future__ import annotations

import numpy as np

from .binding import Space


def structure(space: Space, pairs: dict[str, str]) -> np.ndarray:
    """A situation: what fills each role, bound and superposed."""
    return space.bundle(
        *[space.bind(space.atom(role), space.atom(filler)) for role, filler in pairs.items()]
    )


def mapping(space: Space, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The translation between two situations, obtained by multiplying them.

    No roles are named and none are looked up. The cancellation does the work.
    """
    return space.bind(source, target)


def carry_across(space: Space, element: str, translation: np.ndarray) -> np.ndarray:
    """Send one element of the source through the mapping into the target."""
    return Space.bind(space.atom(element), translation)


def unaligned_structure(space: Space, pairs: dict[str, str], suffix: str) -> np.ndarray:
    """The same situation described with a private vocabulary of roles.

    Every role name is given a suffix, so the two descriptions are equally
    informative but carve the situation up differently -- which is the ordinary
    case across real domains, and the case the mapping cannot handle.
    """
    return space.bundle(
        *[
            space.bind(space.atom(f"{role}{suffix}"), space.atom(filler))
            for role, filler in pairs.items()
        ]
    )
