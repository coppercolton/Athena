"""What the agent is willing to believe, and how it narrows that down.

A hypothesis space is every rule the agent will entertain for a problem,
evaluable on a batch of situations at once: conjunctions over items,
constrained by discovered roles so that two fillers of one role are never
conjoined (round thirteen), plus agreement predicates between roles of equal
size where the roles' fillers are comparable.

Two ways of narrowing it, differing only in who chooses the examples:

*   **instruction** -- examples are handed over; keep the hypotheses that fit
    every one, prefer the shortest.
*   **experimentation** -- the agent chooses. It asks about the situation the
    surviving hypotheses disagree on most, so each answer eliminates about
    half of them. That is version-space / query-by-disagreement active
    learning, and it pins a two-literal rule in about ten questions.

The experimentation loop draws its candidate questions from situations that
satisfy at least one surviving hypothesis. Drawing them uniformly looks
equivalent and is not: a conjunction is false on most situations, so a
uniform pool rarely contains one the survivors disagree about, each question
eliminates almost nothing, and wrong survivors are accepted. That was measured
before it was understood (0.47 on held-out after twelve questions).

Acceptance is never on the grounds that "the pool did not split them". A
survivor is committed only after it also predicts a verification set it was
not selected on -- the round-seven rule, imagination proposes and observation
disposes, applied to hypotheses instead of dreams.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .library import LibraryLearner
from .worlds import Agreement, Conjunction, Domain


@dataclass(frozen=True)
class Hypothesis:
    """One rule the agent might believe, with the roles it was built over."""

    rule: object  # Conjunction | Agreement
    groups: tuple[tuple[int, ...], ...]

    def holds(self, x: np.ndarray) -> np.ndarray:
        if isinstance(self.rule, Conjunction):
            return self.rule.holds(x)
        return self.rule.holds_on(x, self.groups)

    @property
    def length(self) -> int:
        return len(self.rule.literals) if isinstance(self.rule, Conjunction) else 1


class HypothesisSpace:
    """Every hypothesis over a domain's discovered roles, scored as arrays."""

    def __init__(self, features: int, groups, max_literals: int = 3, agreements: bool = True):
        self.features = int(features)
        self.groups = tuple(tuple(int(i) for i in g) for g in groups) if groups is not None else None
        learner = LibraryLearner(features=self.features, groups=self.groups, max_literals=max_literals)
        self.candidates, self.index, self.polarity, self.used = learner._compiled()
        # A conjunction is satisfied when every one of its literals is. Encode
        # each hypothesis as a 0/1 column over the 2F possible literals
        # (feature true, feature false); then "literals satisfied" is one
        # matrix product with the batch's own literal vector, and the
        # prediction is that count reaching the hypothesis's length. The
        # obvious fancy-indexed version materialises rows x hypotheses x
        # literals as float64 and needs 22 GB at two million hypotheses.
        h = len(self.candidates)
        self.membership = np.zeros((2 * self.features, h), dtype=np.float32)
        for row, literals in enumerate(self.candidates):
            for feature, want in literals:
                self.membership[feature + (0 if want else self.features), row] = 1.0
        self.lengths = self.membership.sum(axis=0)
        self.agreements: list[tuple[int, int]] = []
        if agreements and self.groups is not None:
            for a in range(len(self.groups)):
                for b in range(a + 1, len(self.groups)):
                    if len(self.groups[a]) == len(self.groups[b]):
                        self.agreements.append((a, b))
        self.size = len(self.candidates) + len(self.agreements)

    # ------------------------------------------------------------------
    def predictions(self, x: np.ndarray, chunk: int = 64) -> np.ndarray:
        """Boolean matrix, one column per hypothesis."""
        bits = x > 0.5
        literal = np.concatenate([bits, ~bits], axis=1).astype(np.float32)   # (rows, 2F)
        conj = np.empty((len(x), len(self.candidates)), dtype=bool)
        for start in range(0, len(x), chunk):
            block = literal[start:start + chunk] @ self.membership            # literals satisfied
            conj[start:start + chunk] = block >= self.lengths - 0.5
        if not self.agreements:
            return conj
        cols = []
        for a, b in self.agreements:
            ga, gb = list(self.groups[a]), list(self.groups[b])
            w = min(len(ga), len(gb))
            left, right = x[:, ga[:w]], x[:, gb[:w]]
            cols.append((left.max(1) > 0.5) & (right.max(1) > 0.5) & (left.argmax(1) == right.argmax(1)))
        return np.column_stack([conj, np.column_stack(cols)])

    def hypothesis(self, h: int) -> Hypothesis:
        if h < len(self.candidates):
            return Hypothesis(Conjunction(self.candidates[h]), self.groups or ())
        return Hypothesis(Agreement(*self.agreements[h - len(self.candidates)]), self.groups or ())

    def length(self, h: int) -> int:
        return int(self.used[h].sum()) if h < len(self.candidates) else 1

    def consistent(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Mask of hypotheses that fit every labelled example exactly."""
        if len(x) == 0:
            return np.ones(self.size, dtype=bool)
        return (self.predictions(x) == y[:, None].astype(bool)).all(axis=0)

    def shortest(self, alive: np.ndarray) -> int | None:
        survivors = np.flatnonzero(alive)
        if len(survivors) == 0:
            return None
        return int(min(survivors, key=lambda h: (self.length(int(h)), int(h))))


# ----------------------------------------------------------------------------
def choose_query(preds: np.ndarray, alive: np.ndarray) -> int:
    """Index of the pool row that splits the surviving hypotheses most evenly."""
    p = preds[:, alive]
    ones = p.sum(axis=1)
    split = np.minimum(ones, alive.sum() - ones)
    return int(np.argmax(split))


def informative_pool(space: HypothesisSpace, alive: np.ndarray, domain: Domain, rng, size: int = 256, draws: int = 8) -> tuple[list, np.ndarray, np.ndarray]:
    """Situations that satisfy at least one survivor, with the survivors' votes.

    Draws several batches and keeps the rows some survivor predicts true, so
    that a question can actually divide the survivors. Falls back to whatever
    was drawn if nothing qualifies.
    """
    kept_bags, kept_x, kept_p = [], [], []
    for _ in range(draws):
        bags = domain.situations(rng, size)
        x = domain.encode(bags)
        p = space.predictions(x)
        hit = p[:, alive].any(axis=1)
        for i in np.flatnonzero(hit):
            kept_bags.append(bags[i]); kept_x.append(x[i]); kept_p.append(p[i])
        if len(kept_bags) >= size:
            break
    if not kept_bags:
        bags = domain.situations(rng, size)
        x = domain.encode(bags)
        return bags, x, space.predictions(x)
    return kept_bags, np.asarray(kept_x), np.asarray(kept_p)


def experiment(space: HypothesisSpace, domain: Domain, oracle, rng, budget: int,
               seed_x: np.ndarray | None = None, seed_y: np.ndarray | None = None):
    """Ask until one hypothesis remains or the budget is spent.

    Returns (index or None, x, y, questions_used). The index is the shortest
    survivor; the caller must still verify it on situations it was not
    selected on before believing it.
    """
    x = seed_x if seed_x is not None else np.zeros((0, space.features))
    y = seed_y if seed_y is not None else np.zeros(0, dtype=int)
    alive = space.consistent(x, y)
    used = 0
    while used < budget and alive.sum() > 1:
        bags, px, pp = informative_pool(space, alive, domain, rng)
        q = choose_query(pp, alive)
        votes = pp[q, alive]
        if votes.all() or (~votes).all():
            break  # nothing in reach divides them
        label = oracle.label([bags[q]])
        x = np.vstack([x, px[q:q + 1]])
        y = np.concatenate([y, label])
        alive = space.consistent(x, y)
        used += 1
    return space.shortest(alive), x, y, used
