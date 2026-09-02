"""Symbolic worlds for the lifelong agent: roles, fillers, rules, an oracle.

A domain is a set of roles, each filled by exactly one of its fillers in every
situation (optionally with one role determined by another). A problem is a
rule over a domain -- a conjunction of fillers, or the agreement of two roles
-- and a channel through which it can be learned: labelled examples handed
over, or an oracle the agent may question.

Every constraint here was paid for earlier in the repository:

*   never force-fill an empty situation (round fifteen's generator artefact
    manufactured a limit that did not exist);
*   balanced labels, or a rule true 2% of the time is learned perfectly by
    answering "no" (round thirteen);
*   examples are distinct situations and held-out sets are disjoint from
    training, so a domain must be large enough to allow it -- and the code
    fails loudly when it is not, instead of spinning or quietly duplicating;
*   a rule must be satisfiable under the domain's own distribution. Under a
    dependency, ``heat and valve`` can be impossible, and rejection sampling
    for its positives never returns.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Domain:
    name: str
    roles: tuple[tuple[str, ...], ...]          # fillers per role
    dependency: tuple[int, int] | None = None   # (source_role, dependent_role): dependent = source % len

    @property
    def vocabulary(self) -> tuple[str, ...]:
        return tuple(f for role in self.roles for f in role)

    @property
    def index(self) -> dict[str, int]:
        return {f: i for i, f in enumerate(self.vocabulary)}

    @property
    def size(self) -> int:
        """Distinct situations this domain can produce."""
        n = 1
        for r, role in enumerate(self.roles):
            if self.dependency is None or r != self.dependency[1]:
                n *= len(role)
        return n

    @property
    def groups(self) -> tuple[tuple[int, ...], ...]:
        """True role structure as index groups (the *given* condition)."""
        out, at = [], 0
        for role in self.roles:
            out.append(tuple(range(at, at + len(role))))
            at += len(role)
        return tuple(out)

    def situations(self, rng: np.random.Generator, count: int, present: float = 1.0) -> list[list[str]]:
        """Unordered bags. A role absent with probability 1-present. No force-fill."""
        out = []
        for _ in range(count):
            picks = [int(rng.integers(0, len(r))) for r in self.roles]
            if self.dependency is not None:
                s, d = self.dependency
                picks[d] = picks[s] % len(self.roles[d])
            bag = [self.roles[r][p] for r, p in enumerate(picks) if rng.random() < present]
            out.append(bag)
        return out

    def encode(self, bags: list[list[str]]) -> np.ndarray:
        idx = self.index
        x = np.zeros((len(bags), len(idx)))
        for row, bag in enumerate(bags):
            for item in bag:
                x[row, idx[item]] = 1.0
        return x


# ----------------------------------------------------------------------------
# Rule families. Each is a predicate over encoded situations.

def _holds(rule, x, domain):
    return rule.holds(x) if isinstance(rule, Conjunction) else rule.holds_on(x, domain.groups)


@dataclass(frozen=True)
class Conjunction:
    literals: tuple[tuple[int, bool], ...]

    def holds(self, x: np.ndarray) -> np.ndarray:
        out = np.ones(len(x), dtype=bool)
        for i, want in self.literals:
            out &= (x[:, i] > 0.5) == want
        return out

    def describe(self, domain: Domain) -> str:
        v = domain.vocabulary
        return " and ".join(("" if w else "not ") + v[i] for i, w in self.literals)


@dataclass(frozen=True)
class Agreement:
    """Two roles must have the same filler *position*. Mentions no filler."""
    role_a: int
    role_b: int

    def holds_on(self, x: np.ndarray, groups) -> np.ndarray:
        ga, gb = list(groups[self.role_a]), list(groups[self.role_b])
        w = min(len(ga), len(gb))
        left, right = x[:, ga[:w]], x[:, gb[:w]]
        return (left.max(1) > 0.5) & (right.max(1) > 0.5) & (left.argmax(1) == right.argmax(1))

    def describe(self, domain: Domain) -> str:
        return f"role{self.role_a} agrees with role{self.role_b}"


def satisfiable(rng, domain: Domain, rule, sample: int = 2048) -> bool:
    """Both labels occur under the domain's own distribution. A dependency can
    make an innocent-looking conjunction impossible -- ``heat and valve`` when
    heat forces burst -- and sampling for its positives then never returns."""
    y = _holds(rule, domain.encode(domain.situations(rng, sample)), domain)
    return 0.02 < y.mean() < 0.98


def random_conjunction(rng, domain: Domain, arity: int, positive_only: bool = True) -> Conjunction:
    """A conjunction over distinct roles, verified satisfiable in this domain."""
    g = domain.groups
    for _ in range(1000):
        roles = rng.choice(len(domain.roles), size=arity, replace=False)
        lits = []
        for r in roles:
            i = int(g[r][int(rng.integers(0, len(g[r])))])
            lits.append((i, True if positive_only else bool(rng.random() < 0.7)))
        rule = Conjunction(tuple(sorted(lits)))
        if satisfiable(rng, domain, rule):
            return rule
    raise ValueError(f"no satisfiable conjunction of arity {arity} in {domain.name}")


def random_agreement(rng, domain: Domain) -> Agreement | None:
    pairs = [(a, b) for a in range(len(domain.roles)) for b in range(a + 1, len(domain.roles))
             if len(domain.roles[a]) == len(domain.roles[b])]
    if not pairs:
        return None
    a, b = pairs[int(rng.integers(0, len(pairs)))]
    return Agreement(a, b)


# ----------------------------------------------------------------------------
# Labelled examples, balanced, disjoint held-out.

def balanced(rng, domain: Domain, rule, count: int, exclude: set[tuple[str, ...]] = frozenset(),
             tries: int = 400) -> tuple[list[list[str]], np.ndarray]:
    """Half satisfying, half not, all DISTINCT situations, none in ``exclude``.

    Rejection sampling only -- no forcing of fillers, so nothing about the
    distribution is distorted. Distinctness matters: a domain with 48 possible
    situations and a 200-example test set is the whole world four times over,
    and accuracy on duplicates measures nothing. Fails loudly if the domain is
    too small for the request, which is a design error and should read as one."""
    want_pos = count // 2
    want_neg = count - want_pos
    pos, neg, seen = [], [], set(exclude)
    for _ in range(tries):
        if len(pos) >= want_pos and len(neg) >= want_neg:
            break
        batch = domain.situations(rng, 256)
        labels = _holds(rule, domain.encode(batch), domain)
        for bag, hit in zip(batch, labels):
            key = tuple(sorted(bag))
            if key in seen:
                continue
            if hit and len(pos) < want_pos:
                pos.append(bag); seen.add(key)
            elif not hit and len(neg) < want_neg:
                neg.append(bag); seen.add(key)
    if len(pos) < want_pos or len(neg) < want_neg:
        raise ValueError(f"could not balance {count} examples for {rule} in {domain.name}: "
                         f"{len(pos)} positive, {len(neg)} negative after {tries * 256} draws")
    bags = pos + neg
    y = np.array([1] * len(pos) + [0] * len(neg))
    order = rng.permutation(len(bags))
    return [bags[i] for i in order], y[order]


class Oracle:
    """Answers label queries for experimentation. Counts them."""
    def __init__(self, domain: Domain, rule):
        self.domain, self.rule, self.queries = domain, rule, 0

    def label(self, bags: list[list[str]]) -> np.ndarray:
        self.queries += len(bags)
        return _holds(self.rule, self.domain.encode(bags), self.domain).astype(int)


