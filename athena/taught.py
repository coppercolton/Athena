"""Being taught: acquiring a rule from a handful of examples, and keeping it.

Everything measured so far used Permuted-MNIST, where each task is an arbitrary
shuffle of the pixels. That benchmark has no compositional structure at all --
there is nothing a system could be *taught*, because there is nothing to learn
*about*. It can only be drilled. Our own earlier sweep showed what that costs:
with tasks that share structure, learning more made the model better at
everything (+0.062 on the oldest skill); with unrelated tasks, −0.042. Permuted
MNIST is the unrelated case by construction, so it cannot show the phenomenon
this project exists for.

This module builds a world where teaching is possible. Objects have attributes.
Rules are small logical statements over those attributes -- "large and red and
not round" -- so every rule is written in the same vocabulary as every other
one, and a system that has understood some of them has a genuine head start on
the next.

Two ways of acquiring a rule, which is the real subject:

``gradient``   the way every system in this repository learns: many small
               steps, thousands of examples, knowledge left implicit in the
               weights.
``episodic``   store the examples verbatim on first sight and answer by
               retrieving them. Acquisition is instant and needs no steps at
               all, in exchange for storage and no compression.

Neither is being taught, and that is the point of measuring them together. A
person told "bishops move diagonally" needs one sentence, no drill, and can use
it immediately in situations sharing none of its surface features. Gradient
descent gets the generalisation and pays thousands of examples for it;
retrieval gets the speed and generalises only as far as similarity reaches. The
size of the gap between them is the size of the problem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .continual import ContinualConfig, MultiClassLearner, Sample


@dataclass(frozen=True)
class Rule:
    """A conjunction of literals over binary attributes."""

    literals: tuple[tuple[int, bool], ...]

    def holds(self, x: np.ndarray) -> np.ndarray:
        out = np.ones(len(x), dtype=bool)
        for index, want in self.literals:
            out &= (x[:, index] > 0.5) == want
        return out

    def describe(self, names: tuple[str, ...]) -> str:
        return " and ".join(
            ("" if want else "not ") + names[i] for i, want in self.literals
        )


def make_rules(count: int, features: int, seed: int, arity: int = 3) -> list[Rule]:
    """Rules drawn from one shared vocabulary, so structure carries between them."""
    rng = np.random.default_rng(seed)
    rules = []
    while len(rules) < count:
        picks = rng.choice(features, size=arity, replace=False)
        literals = tuple((int(i), bool(rng.random() < 0.5)) for i in sorted(picks))
        if literals not in [r.literals for r in rules]:
            rules.append(Rule(literals))
    return rules


def make_composed_rules(count: int, features: int, seed: int, concepts: int = 3):
    """Rules that genuinely share sub-expressions, not merely a vocabulary.

    The distinction matters and the first version of this benchmark missed it.
    Drawing each rule as an independent triple of attributes gives rules that
    share an alphabet and nothing else -- two random 3-subsets of twelve
    attributes usually overlap in nothing at all, so there is no sub-structure
    to reuse and "compositional" is a claim about the notation rather than
    about the problem.

    Here a small library of base concepts is drawn once, and every rule is one
    of those concepts conjoined with one extra literal. A system that has
    learned several rules has genuinely met each base concept several times,
    so reuse is available to anything able to take it.
    """
    rng = np.random.default_rng(seed)
    base = []
    while len(base) < concepts:
        picks = rng.choice(features, size=2, replace=False)
        literals = tuple((int(i), bool(rng.random() < 0.5)) for i in sorted(picks))
        if literals not in base:
            base.append(literals)
    rules, used = [], set()
    while len(rules) < count:
        core = base[int(rng.integers(0, concepts))]
        taken = {i for i, _ in core}
        extra = int(rng.choice([i for i in range(features) if i not in taken]))
        literals = tuple(sorted((*core, (extra, bool(rng.random() < 0.5)))))
        if literals not in used:
            used.add(literals)
            rules.append(Rule(literals))
    return rules


def rule_examples(rule: Rule, count: int, features: int, seed: int) -> list[Sample]:
    """Balanced positive and negative cases, so accuracy is meaningful."""
    rng = np.random.default_rng(seed)
    wanted = count // 2
    pos: list[np.ndarray] = []
    neg: list[np.ndarray] = []
    while len(pos) < wanted or len(neg) < count - wanted:
        batch = (rng.random((512, features)) < 0.5).astype(float)
        hits = rule.holds(batch)
        for row, hit in zip(batch, hits):
            if hit and len(pos) < wanted:
                pos.append(row)
            elif not hit and len(neg) < count - wanted:
                neg.append(row)
    rows = pos + neg
    labels = [1] * len(pos) + [0] * len(neg)
    order = rng.permutation(len(rows))
    return [Sample(rows[i], labels[i]) for i in order]


class EpisodicLearner:
    """One-shot acquisition by storing examples and answering by similarity.

    No training, no steps, no forgetting -- a rule is available the instant its
    examples arrive. What it cannot do is generalise beyond the reach of
    similarity, which is exactly where a taught rule would still apply.
    """

    def __init__(self, k: int = 5) -> None:
        self.k = int(k)
        self.store: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def teach(self, name: str, examples: list[Sample]) -> None:
        x = np.asarray([e.inputs for e in examples], dtype=float)
        y = np.asarray([e.target for e in examples], dtype=int)
        if name in self.store:
            old_x, old_y = self.store[name]
            x, y = np.concatenate([old_x, x]), np.concatenate([old_y, y])
        self.store[name] = (x, y)

    def accuracy(self, name: str, cases: list[Sample]) -> float:
        x, y = self.store[name]
        q = np.asarray([c.inputs for c in cases], dtype=float)
        truth = np.asarray([c.target for c in cases], dtype=int)
        # Hamming similarity over binary attributes.
        agree = (q[:, None, :] > 0.5) == (x[None, :, :] > 0.5)
        score = agree.sum(axis=2)
        k = min(self.k, len(x))
        nearest = np.argpartition(-score, k - 1, axis=1)[:, :k]
        votes = y[nearest].mean(axis=1)
        return float(((votes >= 0.5).astype(int) == truth).mean())


class GradientLearner:
    """The shared always-training trunk, taught one rule at a time."""

    def __init__(self, features: int, seed: int, buffer: int = 2000) -> None:
        self.model = MultiClassLearner(
            ContinualConfig(
                input_dim=features,
                hidden=(64, 32),
                seed=seed,
                replay_capacity=buffer,
                replay_per_step=64,
                consolidation=0.0,
                retention_tolerance=1.0,
            ),
            classes=2,
        )

    def teach(self, name: str, examples: list[Sample], steps: int) -> None:
        self.model.teach(name, examples, examples[: max(2, len(examples) // 2)], steps=steps, batch_size=min(32, len(examples)))

    def accuracy(self, name: str, cases: list[Sample]) -> float:
        return self.model.accuracy(name, cases)
