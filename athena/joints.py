"""Finding the joints: recovering roles from observation instead of being told.

Every mechanism in this repository composes perfectly over a vocabulary someone
else supplied. Predictive coding was given its hierarchy, the library learner
was told rules are conjunctions, binding was handed its atoms, and analogy
needs its roles pre-aligned. The composition is solved. Producing the
vocabulary from raw experience is not.

This is an attempt at that, on the narrowest version of the problem where it
can still be checked. A system watches situations go by. Each situation is an
unordered bag of things that happened together -- no roles, no slots, no labels,
no indication that "roles" is even a concept. From co-occurrence alone it has
to work out that the world has slots and which things fill which.

Two statistical facts make it possible:

*   Things filling the **same** role never co-occur. A situation has one medium,
    one vessel; water and oil do not appear together, because whatever occupies
    a slot excludes the alternatives.
*   Things filling **different** roles always co-occur, and things filling the
    same role are interchangeable -- they keep the same company.

So "never appears with, but keeps identical company" is the signature of a role,
and it is visible in a co-occurrence matrix without anyone naming anything.

Aligning the discovered roles *across* domains is the harder half, and it needs
a second signal: relational structure. If in every domain the relief depends on
what is building, that dependency survives translation even though every filler
changes. Roles embedded in distinguishable relations can be matched; roles that
are statistically symmetric cannot be, and that limit is measured rather than
hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BLOCK = 4096
"""Situations counted at once. Bounds memory without changing the result."""


@dataclass
class Discovery:
    """What was recovered from watching, and how good it was."""

    slots: list[list[str]] = field(default_factory=list)
    order: list[int] = field(default_factory=list)
    purity: float = 0.0


def cooccurrence(episodes: list[list[str]]) -> tuple[np.ndarray, list[str]]:
    """How often each pair of items was seen together, and the vocabulary.

    Counted as an indicator matrix times its own transpose. The obvious nested
    loop is quadratic in items *per episode* and interpreted, which puts the
    ceiling on any measurement of how this scales at a few dozen roles -- far
    below where the method itself gives out.

    Multiplicity is kept rather than collapsed: an item an extractor reports
    twice in one situation counts twice, which is what the loop did. Counting
    set membership instead is defensible and is *not* the same statistic --
    it changes the measured noise tolerance -- so it is left as a separate
    question rather than smuggled in with an optimisation.

    Episodes are processed in blocks. Building one indicator matrix over the
    whole corpus would be 40GB at a million situations and a ten-thousand word
    vocabulary -- a memory wall the nested loop did not have, traded for the
    time wall it did. Blocking keeps both bounded; only the vocabulary-squared
    result matrix grows, and that is intrinsic to counting pairs.
    """
    items = sorted({item for episode in episodes for item in episode})
    index = {item: i for i, item in enumerate(items)}
    counts = np.zeros((len(items), len(items)))
    for start in range(0, len(episodes), BLOCK):
        block = episodes[start : start + BLOCK]
        present = np.zeros((len(block), len(items)), dtype=np.float32)
        for row, episode in enumerate(block):
            for item in episode:
                present[row, index[item]] += 1.0
        counts += present.T @ present
    return counts, items


EXCLUSION = 0.25
"""How far below independence two items must co-occur to be treated as exclusive."""

ALPHA = 0.01
"""Significance required to call an exclusion *proved*, before correction."""

ALWAYS = 0.9
"""Share of situations a group must fill to count as a role without proving one."""


def log_tail(observed: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Bound on log P(X as extreme as ``observed``) for X ~ Poisson(``expected``).

    The Chernoff form, which is conservative and needs no special functions,
    and which stays finite in log space where the probabilities underflow. The
    same expression bounds both tails, so one call decides whether a pair
    co-occurs provably *less* than chance predicts or provably more.
    """
    out = np.zeros(observed.shape)
    usable = expected > 0
    none = usable & (observed == 0)
    out[none] = -expected[none]
    some = usable & (observed > 0) & (observed != expected)
    k, mu = observed[some], expected[some]
    out[some] = -mu + k * (1.0 + np.log(mu) - np.log(k))
    return out


def discover_slots(episodes: list[list[str]]) -> list[list[str]]:
    """Group items into roles from co-occurrence alone.

    The exact criterion: **a role is a set of pairwise mutually exclusive items
    whose occurrence counts sum to the number of situations.** Exactly one
    thing fills a slot in every situation, so the alternatives for that slot
    partition the episodes between them. Nothing else in the data has that
    property.

    A first version of this used "never co-occur, and keeps the same company",
    which seems equivalent and is not. Where one slot *depends* on another --
    the relief following the buildup -- the fillers of a slot keep systematically
    different company, so distributional equivalence fails exactly where the
    interesting relational structure lives, and the slot gets shattered. Pure
    mutual exclusion has the opposite failure: a dependency also makes some
    *cross*-slot pairs never co-occur, which fuses the two slots together.
    Counting is what separates them, because a fused pair over-counts: its
    members sum to twice the episodes, not once.

    Exclusion has to be measured as a rate rather than an exact zero. Testing
    ``counts == 0`` assumes a perfect observer: one hallucinated co-occurrence
    in eight hundred situations permanently severs two items that belong
    together, and at a 2% extraction error rate the four true roles shatter
    into eight.

    **Two different tests, because grouping and debris-rejection want opposite
    things.** Grouping needs near-perfect *recall* on sibling pairs: joining a
    group requires agreement with every member, so one missed true edge
    fragments a role and the false edges then decide everything. The loose rate
    test has that recall by construction. Rejecting an extractor's invented
    items needs *precision*, and that is what significance is for -- a stray is
    independent of everything, so it never achieves a provable exclusion, while
    a genuine alternative always has one with some common sibling.

    Crucially the significance test is applied to *groups*, not items. Debris
    does not merely fail individually: strays exclude each other only by
    chance, so under the loose test they collect into one group, and that whole
    group contains no pair whose exclusion can be demonstrated. Discarding rare
    *items* up front instead is the frequency floor again in better clothes,
    and it costs every thin genuine value -- 46 of mushroom's 117. Nothing is
    discarded on its own account here, so no rare value pays for being rare. A
    group earns its place by containing one provable exclusion, or by filling
    essentially every situation, which is how a constant attribute like
    mushroom's veil type survives while debris does not.

    No labels, no slot count, no supervision.
    """
    counts, items = cooccurrence(episodes)
    total = len(episodes)
    occurrences = np.diag(counts).copy()
    expected = np.outer(occurrences, occurrences) / max(total, 1)

    exclusive = counts < EXCLUSION * expected
    np.fill_diagonal(exclusive, True)

    threshold = np.log(ALPHA) - 2.0 * np.log(max(len(items), 2))  # over every pair tested
    provable = (counts < expected) & (log_tail(counts, expected) < threshold)
    np.fill_diagonal(provable, False)

    order = list(np.argsort(-occurrences))
    unassigned = set(order)
    slots: list[list[int]] = []
    while unassigned:
        seed = next(i for i in order if i in unassigned)
        group = [seed]
        mass = occurrences[seed]
        for candidate in order:
            if candidate in group or candidate not in unassigned:
                continue
            if not all(exclusive[candidate, member] for member in group):
                continue
            if mass + occurrences[candidate] > total * 1.35:
                continue  # would over-fill the slot: these are not alternatives
            group.append(candidate)
            mass += occurrences[candidate]
        unassigned -= set(group)
        slots.append(group)

    kept = [
        group for group in slots
        if provable[np.ix_(group, group)].any()
        or occurrences[group].sum() >= ALWAYS * total
    ]
    return [sorted(items[i] for i in group) for group in (kept or slots)]


def slot_signature(episodes: list[list[str]], slots: list[list[str]]) -> np.ndarray:
    """A description of each role by the company it keeps, not by its name.

    Entropy says how many ways a slot can be filled; the mutual information
    with every other slot says how it sits in the relational structure. Both
    survive translation between domains, because neither mentions a filler.
    """
    where = {item: s for s, slot in enumerate(slots) for item in slot}
    picks = np.full((len(episodes), len(slots)), -1)
    for e, episode in enumerate(episodes):
        for item in episode:
            slot = where.get(item)
            if slot is not None:
                picks[e, slot] = slots[slot].index(item)

    def entropy(column: np.ndarray) -> float:
        values, counts = np.unique(column, return_counts=True)
        p = counts / counts.sum()
        return float(-(p * np.log(p + 1e-12)).sum())

    k = len(slots)
    signature = np.zeros((k, k + 1))
    for a in range(k):
        signature[a, 0] = entropy(picks[:, a])
        for b in range(k):
            if a == b:
                continue
            joint = picks[:, a] * 100 + picks[:, b]
            signature[a, b + 1] = entropy(picks[:, a]) + entropy(picks[:, b]) - entropy(joint)
    return signature


def align(source: np.ndarray, target: np.ndarray) -> list[int]:
    """Match roles between domains by their relational signature alone.

    Greedy on signature distance, using only the sorted profile of mutual
    information so that the matching does not depend on the arbitrary order
    slots were discovered in.
    """
    def profile(signature: np.ndarray) -> np.ndarray:
        return np.column_stack([signature[:, 0], np.sort(signature[:, 1:], axis=1)[:, ::-1]])

    a, b = profile(source), profile(target)
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    assignment = [-1] * len(a)
    taken: set[int] = set()
    for _ in range(len(a)):
        best = None
        for i in range(len(a)):
            if assignment[i] != -1:
                continue
            for j in range(len(b)):
                if j in taken:
                    continue
                if best is None or cost[i, j] < cost[best[0], best[1]]:
                    best = (i, j)
        if best is None:
            break
        assignment[best[0]] = best[1]
        taken.add(best[1])
    return assignment
