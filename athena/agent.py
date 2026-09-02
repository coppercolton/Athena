"""The loop: encounter, find the gap, learn, solve, keep, carry across, check.

Fifteen rounds produced validated pieces -- a continual learner, a library
learner, binding, analogy, role discovery -- and never assembled them into the
thing the project was for. This is that assembly, on the narrowest problems
where every step can still be checked:

    encounter    a problem in some domain, through one of two channels
    identify     whether something already retained explains it: a skill from
                 this domain, or the *shape* of a skill from another domain
                 carried across the discovered roles (rounds ten, thirteen)
    learn        by instruction, from examples handed over; or by
                 experimentation, asking about the situations the surviving
                 hypotheses disagree on (athena.hypotheses)
    solve        on situations it has never seen
    retain       the hypothesis, and the sub-expressions it shares with
                 earlier ones (round six's library)
    check        every earlier skill, every time

Three decisions were forced by measurement rather than taste.

**Diagnosis is a test, not a lookup.** With a few probes and a few hundred
carried-across candidates, something always fits by chance, and a candidate
that is a sub-conjunction of the truth agrees with it on half of all random
probe sets. So candidates are eliminated by the questions they disagree on,
and a survivor is believed only after predicting situations it was not
selected on. Where the agent cannot ask, a verdict is provisional and is
overturned by the first prediction error.

**Transfer carries a shape, not a rule.** Roles align across domains by
their relational signature, but nothing aligns the *fillers*: nothing says
water is to plumbing what money is to negotiation. What survives is which
roles a rule mentions and how, and the target domain's own examples must
fix the rest. That restricts a search of twenty thousand hypotheses to a
few hundred, which is worth about half the labels a new domain would cost.

**Forgetting is impossible here by construction.** Skills are symbols and
are never overwritten, so a stored skill's accuracy never changes after it is
committed: backward transfer is exactly zero, and retention equals accuracy
at commitment. Both are reported because a benchmark that does not report
them is hiding something, not because zero is a result. The forgetting that
is real -- in the neural trunk -- was measured in rounds one to five and is
not re-litigated by this loop.

The surprise the founding idea asks for is here as prediction error on the
first probes, precision-weighted over time (athena.precision). It is reported
alongside held-out competence and never instead of it, because a rising
surprise curve with flat competence is an agent fixated on noise, not one
learning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product

import numpy as np

from .hypotheses import Hypothesis, HypothesisSpace, choose_query
from .joints import discover_slots
from .precision import VolatilityTracker
from .worlds import Agreement, Conjunction, Domain


@dataclass(frozen=True)
class Problem:
    """Something to be learned: a rule over a domain, reachable one of two ways."""

    domain: Domain
    rule: object
    channel: str  # "instruction" | "experiment"
    budget: int   # labels handed over, or questions allowed


@dataclass
class Skill:
    domain: str
    hypothesis: Hypothesis
    test_x: np.ndarray
    test_y: np.ndarray
    learned_at: int


@dataclass
class Record:
    index: int
    domain: str
    channel: str
    verdict: str          # known | transfer | novel | failed
    cost: int             # labels consumed or questions asked before commitment
    candidates: int       # retained/carried hypotheses considered
    accuracy: float       # on held-out situations
    surprise: float       # prediction error of the best prior candidate on first probes
    gain: float           # VolatilityTracker gain after this problem
    retention: float      # mean accuracy of every earlier skill on its own held-out


class LifelongAgent:
    """One agent, one growing store, many domains."""

    def __init__(self, seed: int = 0, *, max_literals: int = 3, verify_size: int = 12,
                 chunk: int = 4, transfer: bool = True, retain: bool = True,
                 verify: bool = True, shape_cap: int = 3000) -> None:
        self.rng = np.random.default_rng(seed)
        self.max_literals = max_literals
        self.verify_size = verify_size
        self.chunk = chunk
        self.use_transfer = transfer
        self.use_retain = retain
        self.use_verify = verify
        self.shape_cap = shape_cap
        self.groups: dict[str, tuple[tuple[int, ...], ...]] = {}
        self.spaces: dict[str, HypothesisSpace] = {}
        self.skills: list[Skill] = []
        self.library: dict[tuple, int] = {}
        self.surprise = VolatilityTracker()
        self.records: list[Record] = []
        self._last_candidates = 0

    # ------------------------------------------------------------------
    # watching: the vocabulary comes from co-occurrence, never from the domain
    def watch(self, domain: Domain, bags: list[list[str]], groups=None) -> None:
        if groups is None:
            index = domain.index
            slots = discover_slots(bags)
            groups = tuple(tuple(sorted(index[i] for i in s if i in index)) for s in slots)
        self.groups[domain.name] = groups
        self.spaces[domain.name] = HypothesisSpace(len(domain.vocabulary), groups, self.max_literals)

    # ------------------------------------------------------------------
    # what might already explain this?
    def _shapes(self, skill: Skill, target: tuple[tuple[int, ...], ...]) -> list[Hypothesis]:
        """Every target hypothesis with the same roles-and-relation as a source skill.

        A source role maps to every target role of the same size, because the
        relational signature cannot tell two independent same-sized roles apart.
        Fillers are enumerated, because nothing aligns them.
        """
        source = skill.hypothesis
        rule = source.rule
        out: list[Hypothesis] = []
        if isinstance(rule, Agreement):
            sa, sb = len(source.groups[rule.role_a]), len(source.groups[rule.role_b])
            for p in range(len(target)):
                for q in range(len(target)):
                    if p != q and len(target[p]) == sa and len(target[q]) == sb and p < q:
                        out.append(Hypothesis(Agreement(p, q), target))
            return out
        wants = []
        for f, want in rule.literals:
            r = next((i for i, g in enumerate(source.groups) if f in g), None)
            if r is None:
                return []
            wants.append((len(source.groups[r]), want))
        choices = [[t for t in range(len(target)) if len(target[t]) == size] for size, _ in wants]
        for roles in product(*choices):
            if len(set(roles)) != len(roles):
                continue
            for fillers in product(*[target[t] for t in roles]):
                lits = tuple(sorted((int(f), w) for f, (_, w) in zip(fillers, wants)))
                out.append(Hypothesis(Conjunction(lits), target))
                if len(out) >= self.shape_cap:
                    return out
        return out

    def _candidates(self, domain: Domain) -> list[tuple[str, Hypothesis]]:
        target = self.groups[domain.name]
        out: list[tuple[str, Hypothesis]] = []
        for s in self.skills:
            if s.domain == domain.name:
                out.append(("known", s.hypothesis))
            elif self.use_transfer:
                out.extend(("transfer", h) for h in self._shapes(s, target))
        if not out:
            return out
        # A carried shape can be impossible in the target -- a dependency may
        # forbid the pair -- and an impossible hypothesis is never eliminated
        # by a question, because it never disagrees with anything by being true.
        probe = domain.encode(domain.situations(self.rng, 512))
        votes = self._votes([h for _, h in out], probe)
        return [c for c, ok in zip(out, votes.any(axis=0)) if ok]

    # ------------------------------------------------------------------
    @staticmethod
    def _votes(hyps: list[Hypothesis], x: np.ndarray) -> np.ndarray:
        if not hyps:
            return np.zeros((len(x), 0), dtype=bool)
        return np.column_stack([h.holds(x) for h in hyps])

    def _pool(self, domain: Domain, hyps: list[Hypothesis], alive: np.ndarray, size: int = 256):
        """Situations some survivor predicts true, so a question can divide them."""
        keep_b, keep_x, keep_v = [], [], []
        for _ in range(8):
            bags = domain.situations(self.rng, size)
            x = domain.encode(bags)
            v = self._votes(hyps, x)
            hit = v[:, alive].any(axis=1) if alive.any() else np.zeros(len(x), dtype=bool)
            for i in np.flatnonzero(hit):
                keep_b.append(bags[i]); keep_x.append(x[i]); keep_v.append(v[i])
            if len(keep_b) >= size:
                break
        if not keep_b:
            bags = domain.situations(self.rng, size)
            x = domain.encode(bags)
            return bags, x, self._votes(hyps, x)
        return keep_b, np.asarray(keep_x), np.asarray(keep_v)

    def _verify(self, domain: Domain, hyp: Hypothesis, oracle, x_seen: np.ndarray, y_seen: np.ndarray):
        """Predict situations the hypothesis was not selected on.

        Half of them are situations it predicts true, so a near-miss cannot
        pass by being false everywhere; a hypothesis that cannot be exhibited
        cannot be verified at all. Checked in blocks, stopping at the first
        miss: a wrong candidate is usually caught in the first block, so the
        price of being wrong is a few labels rather than the full set, while
        acceptance still requires the full set. Returns (passed, error, x, y);
        every label paid for is returned so that learning can reuse it.
        """
        if not self.use_verify:
            return True, 0.0, x_seen, y_seen
        block = max(2, self.verify_size // 3)
        confirmed = 0
        seen = {tuple(r) for r in x_seen.round().astype(int).tolist()}
        while confirmed < self.verify_size:
            want = min(block, self.verify_size - confirmed) // 2 or 1
            pos, neg = [], []
            for _ in range(16):
                bags = domain.situations(self.rng, 256)
                x = domain.encode(bags)
                p = hyp.holds(x)
                for i in range(len(x)):
                    key = tuple(x[i].astype(int).tolist())
                    if key in seen:
                        continue
                    if p[i] and len(pos) < want:
                        pos.append(bags[i]); seen.add(key)
                    elif not p[i] and len(neg) < want:
                        neg.append(bags[i]); seen.add(key)
                if len(pos) >= want and len(neg) >= want:
                    break
            if len(pos) < want or len(neg) < want:
                return False, 1.0, x_seen, y_seen
            bags = pos + neg
            x = domain.encode(bags)
            y = oracle.label(bags)
            x_seen, y_seen = np.vstack([x_seen, x]), np.concatenate([y_seen, y])
            err = float((hyp.holds(x).astype(int) != y).mean())
            if err > 0.0:
                return False, err, x_seen, y_seen
            confirmed += len(bags)
        return True, 0.0, x_seen, y_seen

    # ------------------------------------------------------------------
    def _diagnose_active(self, domain: Domain, oracle, budget: int):
        """Eliminate retained and carried candidates by the questions they disagree on."""
        cands = self._candidates(domain)
        self._last_candidates = len(cands)
        hyps = [h for _, h in cands]
        x = np.zeros((0, len(domain.vocabulary)))
        y = np.zeros(0, dtype=int)
        if not hyps:
            return "novel", None, x, y, 0, 1.0
        alive = np.ones(len(hyps), dtype=bool)
        used = 0
        while used < budget and alive.sum() > 1:
            bags, px, pv = self._pool(domain, hyps, alive)
            q = choose_query(pv, alive)
            votes = pv[q, alive]
            if votes.all() or (~votes).all():
                break
            label = oracle.label([bags[q]])
            x = np.vstack([x, px[q:q + 1]]); y = np.concatenate([y, label])
            alive &= (pv[q] == bool(label[0]))
            used += 1
        first_err = 1.0
        if len(x):
            first_err = float(min((hyps[i].holds(x).astype(int) != y).mean() for i in range(len(hyps))))
        if not alive.any():
            return "novel", None, x, y, used, first_err
        i = int(min(np.flatnonzero(alive), key=lambda k: hyps[k].length))
        before = len(x)
        ok, err, x, y = self._verify(domain, hyps[i], oracle, x, y)
        used += len(x) - before
        if ok:
            return cands[i][0], hyps[i], x, y, used, first_err
        return "novel", None, x, y, used, max(first_err, err)

    def _learn_active(self, domain: Domain, oracle, budget: int, x: np.ndarray, y: np.ndarray, used: int):
        """Experiment over the full space, seeded with every label already paid for.

        A survivor that cannot be exhibited -- never true under the domain's
        distribution -- is refused by verification without any label being
        spent, and must then be struck out explicitly, or the loop asks about
        it forever.
        """
        space = self.spaces[domain.name]
        dead = np.zeros(space.size, dtype=bool)
        while used < budget:
            alive = space.consistent(x, y) & ~dead
            while used < budget and alive.sum() > 1:
                bags = domain.situations(self.rng, 256)
                px = domain.encode(bags)
                pv = space.predictions(px)
                hit = pv[:, alive].any(axis=1)
                if hit.any():
                    bags = [bags[i] for i in np.flatnonzero(hit)]; px = px[hit]; pv = pv[hit]
                q = choose_query(pv, alive)
                votes = pv[q, alive]
                if votes.all() or (~votes).all():
                    break
                label = oracle.label([bags[q]])
                x = np.vstack([x, px[q:q + 1]]); y = np.concatenate([y, label])
                alive = space.consistent(x, y) & ~dead
                used += 1
            h = space.shortest(alive)
            if h is None:
                return None, used
            hyp = space.hypothesis(h)
            before = len(x)
            ok, err, x, y = self._verify(domain, hyp, oracle, x, y)
            used += len(x) - before
            if ok:
                return hyp, used
            if len(x) == before:
                dead[h] = True  # unverifiable: never true, so no question can remove it
        return None, used

    # ------------------------------------------------------------------
    def _diagnose_passive(self, domain: Domain, x_all: np.ndarray, y_all: np.ndarray):
        """Examples arrive in chunks; a candidate is believed once it survives
        the chunk after the one it was selected on."""
        cands = self._candidates(domain)
        hyps = [h for _, h in cands]
        space = self.spaces[domain.name]
        n = len(x_all)
        first_err = 1.0
        if hyps:
            v = self._votes(hyps, x_all[: self.chunk])
            first_err = float((v != y_all[: self.chunk, None].astype(bool)).mean(axis=0).min())
        used = 0
        alive_c = np.ones(len(hyps), dtype=bool)
        alive_s = None
        pending = None   # (kind, hyp, labels_confirmed) awaiting confirmation
        while used < n:
            xs, ys = x_all[used: used + self.chunk], y_all[used: used + self.chunk]
            if pending is not None:
                kind, hyp, confirmed = pending
                if not self.use_verify:
                    return kind, hyp, used, len(cands), first_err
                if (hyp.holds(xs).astype(int) == ys).all():
                    confirmed += len(xs)
                    if confirmed >= self.verify_size:
                        return kind, hyp, used + len(xs), len(cands), first_err
                    pending = (kind, hyp, confirmed)
                    used += len(xs)
                    continue
                pending = None
            used += len(xs)
            x, y = x_all[:used], y_all[:used]
            if alive_c.any():
                alive_c &= (self._votes(hyps, x) == y[:, None].astype(bool)).all(axis=0)
            if alive_c.any():
                i = int(min(np.flatnonzero(alive_c), key=lambda k: hyps[k].length))
                pending = (cands[i][0], hyps[i], 0)
                continue
            alive_s = space.consistent(x, y)
            h = self._shortest_with_library(space, alive_s)
            if h is not None:
                pending = ("novel", space.hypothesis(h), 0)
        if pending is not None and not self.use_verify:
            return pending[0], pending[1], used, len(cands), first_err
        return "failed", None, used, len(cands), first_err

    def _shortest_with_library(self, space: HypothesisSpace, alive: np.ndarray):
        """Round six's tie-break: among equally short survivors, prefer reused pieces."""
        survivors = np.flatnonzero(alive)
        if len(survivors) == 0:
            return None
        def key(h):
            h = int(h)
            reuse = 0
            if h < len(space.candidates):
                lits = space.candidates[h]
                for size in range(2, len(lits) + 1):
                    for part in combinations(lits, size):
                        reuse += self.library.get(tuple(sorted(part)), 0)
            return (space.length(h), -reuse, h)
        return int(min(survivors, key=key))

    # ------------------------------------------------------------------
    def encounter(self, problem: Problem, oracle, examples=None, test=None) -> Record:
        """One full cycle. ``examples`` (x, y) for instruction; ``test`` (x, y) held out."""
        domain = problem.domain
        index = len(self.records)
        if problem.channel == "experiment":
            verdict, hyp, x, y, used, first_err = self._diagnose_active(domain, oracle, problem.budget)
            if hyp is None:
                hyp, used = self._learn_active(domain, oracle, problem.budget, x, y, used)
                verdict = "novel" if hyp is not None else "failed"
            cands = self._last_candidates
        else:
            x_all, y_all = examples
            verdict, hyp, used, cands, first_err = self._diagnose_passive(domain, x_all, y_all)

        tx, ty = test
        acc = float((hyp.holds(tx).astype(int) == ty).mean()) if hyp is not None else 0.5
        self.surprise.update(first_err)
        if hyp is not None and self.use_retain:
            self.skills.append(Skill(domain.name, hyp, tx, ty, index))
            if isinstance(hyp.rule, Conjunction):
                lits = hyp.rule.literals
                for size in range(2, len(lits)):
                    for part in combinations(lits, size):
                        k = tuple(sorted(part))
                        self.library[k] = self.library.get(k, 0) + 1
        rec = Record(index, domain.name, problem.channel, verdict, int(used), int(cands),
                     acc, float(first_err), float(self.surprise.gain), self.retention())
        self.records.append(rec)
        return rec

    def retention(self) -> float:
        """Every earlier skill on its own held-out situations. 1.0 by construction."""
        if not self.skills:
            return 1.0
        return float(np.mean([(s.hypothesis.holds(s.test_x).astype(int) == s.test_y).mean() for s in self.skills]))

    def forget(self) -> None:
        """The amnesiac control: no store survives a problem."""
        self.skills.clear()
        self.library.clear()
