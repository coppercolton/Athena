"""A library of reusable parts, and what it buys.

Round five found compounding in a gradient network -- learning nine rules made
the tenth cheaper -- but only barely (+0.035 at sixteen examples) and only when
the rules genuinely shared sub-expressions. A distributed network learns
*features*, and features interfere: nothing in it says "this exact piece was
useful before, keep it whole and reuse it."

This is the other option. Keep the pieces explicitly. When a rule is learned,
break it into sub-expressions and add them to a library. When the next rule
arrives, search compositions of library pieces before searching from scratch.
The library is the memory, and unlike a weight it does not get overwritten by
the next thing learned.

The mechanism is not subtle, which is the point. Finding a three-literal rule
over twelve attributes from nothing means choosing among 1,760 candidates. With
a library holding the right two-literal concept, it means choosing among about
sixty -- one library entry plus one more literal. From four examples that
difference is decisive, because a space of 1,760 hypotheses contains many that
fit four examples by luck and a space of sixty contains far fewer.

This is the idea behind library learning in program synthesis (DreamCoder and
its relatives), reduced to the smallest form that can be measured against the
gradient learner on identical data.

**What this comparison does and does not show.** The searcher below knows rules
are conjunctions; the network does not. So a win here is not evidence that
symbolic beats neural. It measures what the *right inductive bias plus explicit
reuse* is worth, which is the question -- and it is only interesting because
the network was given the same examples and the same nine rules of prior
experience.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .continual import Sample
from .taught import Rule

Literals = tuple[tuple[int, bool], ...]


@dataclass
class LibraryLearner:
    """Learns conjunctions, keeping useful sub-expressions for reuse."""

    features: int
    max_literals: int = 3
    library: dict[Literals, int] = field(default_factory=dict)
    learned: dict[str, Rule] = field(default_factory=dict)
    searched: dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def _from_library(self) -> list[Literals]:
        """Candidates built by extending a remembered piece with one literal.

        Ordered by how often each piece has proved useful, so a library that
        has seen a concept repeatedly reaches for it first.
        """
        out: list[Literals] = []
        for concept in sorted(self.library, key=lambda c: -self.library[c]):
            taken = {i for i, _ in concept}
            for index in range(self.features):
                if index in taken:
                    continue
                for polarity in (True, False):
                    out.append(tuple(sorted((*concept, (index, polarity)))))
        return out

    def _from_scratch(self) -> list[Literals]:
        out: list[Literals] = []
        for size in range(1, self.max_literals + 1):
            for picks in combinations(range(self.features), size):
                for mask in range(1 << size):
                    out.append(
                        tuple(
                            (int(i), bool(mask >> b & 1))
                            for b, i in enumerate(picks)
                        )
                    )
        return out

    @staticmethod
    def _score(literals: Literals, x: np.ndarray, y: np.ndarray) -> float:
        holds = np.ones(len(x), dtype=bool)
        for index, want in literals:
            holds &= (x[:, index] > 0.5) == want
        return float((holds.astype(int) == y).mean())

    # ------------------------------------------------------------------
    def _cost(self, literals: Literals) -> float:
        """Description length: a remembered piece costs less than its parts.

        This is where the library earns its keep, and the first version of this
        class got it wrong in an instructive way. Searching library candidates
        *first* changed both the reachable hypotheses and the order they were
        found in, so a win could not be attributed to reuse rather than to a
        different tie-break -- and with four examples, where dozens of
        hypotheses fit perfectly, the tie-break is the entire decision.

        Now the search space is identical whether the library is empty or full.
        Only the cost changes: a rule that reuses a known concept is cheaper to
        describe, so among hypotheses that fit the data equally well the
        reusing one is preferred. That is the whole mechanism, isolated.
        """
        raw = np.log(2.0 * self.features)
        best = len(literals) * raw
        total = sum(self.library.values())
        if total == 0:
            return float(best)
        # How often a piece has been used has to enter the cost, and the
        # version before this one left it out: membership alone made a
        # coincidental pair seen once as cheap as a concept seen nine times,
        # so every hypothesis got the same discount and the library did
        # nothing. Under a description-length prior a piece costs -log of its
        # frequency, so a genuinely recurring concept is dramatically cheaper
        # than a coincidence and the preference becomes sharp.
        for size in range(2, len(literals) + 1):
            for part in combinations(literals, size):
                key = tuple(sorted(part))
                count = self.library.get(key)
                if count:
                    alt = -np.log(count / total) + (len(literals) - size) * raw
                    best = min(best, alt)
        return float(best)

    def teach(self, name: str, examples: list[Sample]) -> None:
        x = np.asarray([e.inputs for e in examples], dtype=float)
        y = np.asarray([e.target for e in examples], dtype=int)

        candidates = self._from_scratch()
        self.searched[name] = len(candidates)
        best, best_key = None, None
        for literals in candidates:
            # Fit first, then the cheapest description among equal fits.
            key = (self._score(literals, x, y), -self._cost(literals))
            if best_key is None or key > best_key:
                best, best_key = literals, key
        if best is None:
            return
        self.learned[name] = Rule(best)
        # Every proper sub-expression of a rule that worked is a candidate
        # piece for the next one.
        for size in range(2, len(best)):
            for part in combinations(best, size):
                key = tuple(sorted(part))
                self.library[key] = self.library.get(key, 0) + 1

    # ------------------------------------------------------------------
    # sleep
    # ------------------------------------------------------------------
    def consolidate(self, min_reuse: int = 2) -> int:
        """Keep only the pieces that explain more than one thing.

        Learning adds every sub-expression of every rule, so the library fills
        with coincidences: a pair that happened to co-occur in one rule looks
        exactly like a concept that recurs throughout. Under a
        description-length prior an abstraction is only worth its own symbol if
        naming it shortens the description of the corpus, and a piece appearing
        in a single rule never does -- it costs one symbol to save one.

        So this drops anything that has not been reused, and rescores what
        survives by how many *distinct* rules contain it rather than by how
        many times enumeration happened to emit it. That is the offline
        compression pass, and it is the operation sleep is usually credited
        with: not acquiring anything new, but working out which of the day's
        pieces were worth keeping.

        Returns how many pieces were discarded.
        """
        counts: dict[Literals, int] = {}
        for rule in self.learned.values():
            for size in range(2, len(rule.literals)):
                for part in combinations(rule.literals, size):
                    key = tuple(sorted(part))
                    counts[key] = counts.get(key, 0) + 1
        kept = {k: c for k, c in counts.items() if c >= min_reuse}
        dropped = len(self.library) - len(kept)
        self.library = kept
        return max(dropped, 0)

    def dream(self, rounds: int, rng: np.random.Generator, examples: int = 64) -> int:
        """Invent problems from the library, solve them, and learn from that.

        Sampling a concept the library already believes in, inventing a rule
        around it and then rediscovering that rule proves nothing on its own --
        the answer was baked into the question. What it can do is find which
        *compositions* are learnable and reinforce the pieces that keep
        appearing in solutions, without spending real examples.

        The self-confirmation risk is real and is exactly what the experiment
        is for: a system that dreams only what it already believes should show
        no gain, or a loss as coincidences get amplified into convictions.

        Returns the number of imagined rules solved.
        """
        if not self.library:
            return 0
        concepts = list(self.library)
        solved = 0
        for index in range(rounds):
            concept = concepts[int(rng.integers(0, len(concepts)))]
            taken = {i for i, _ in concept}
            free = [i for i in range(self.features) if i not in taken]
            if not free:
                continue
            extra = (int(rng.choice(free)), bool(rng.random() < 0.5))
            invented = Rule(tuple(sorted((*concept, extra))))

            x = (rng.random((examples, self.features)) < 0.5).astype(float)
            y = invented.holds(x).astype(int)
            if y.mean() in (0.0, 1.0):
                continue  # nothing to learn from a rule nothing satisfies
            imagined = [Sample(row, int(label)) for row, label in zip(x, y)]
            self.teach(f"__dream{index}", imagined)
            solved += 1
        for key in [k for k in self.learned if k.startswith("__dream")]:
            del self.learned[key]
        return solved

    def verify(self) -> int:
        """Discard imagined pieces that never appear in anything actually observed.

        Dreaming without this accepts every invention, which is why forty
        imagined rules turned a three-piece library into fifty-three and made
        the learner worse wherever real evidence existed. The imagining was
        never the problem; keeping all of it was.

        What survives here is only what is *grounded*: a piece that occurs in
        some rule learned from real examples. Crucially the counts accumulated
        while dreaming are kept for those survivors, so imagination can still
        do the one thing it is genuinely able to do -- sharpen belief about
        concepts that are already attested -- while being unable to invent new
        ones out of its own confidence.

        Imagination proposes; observation disposes. Returns the number of
        imagined pieces discarded.
        """
        grounded: set[Literals] = set()
        for rule in self.learned.values():
            for size in range(2, len(rule.literals)):
                for part in combinations(rule.literals, size):
                    grounded.add(tuple(sorted(part)))
        before = len(self.library)
        self.library = {k: v for k, v in self.library.items() if k in grounded}
        return before - len(self.library)

    def accuracy(self, name: str, cases: list[Sample]) -> float:
        rule = self.learned.get(name)
        if rule is None:
            return 0.5
        x = np.asarray([c.inputs for c in cases], dtype=float)
        y = np.asarray([c.target for c in cases], dtype=int)
        return float((rule.holds(x).astype(int) == y).mean())
