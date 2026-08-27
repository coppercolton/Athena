"""What a replay buffer should choose to keep.

Replay does essentially all the work in this repository's continual learner:
on Permuted-MNIST it recovers 34 points where consolidation recovers 18 and
the two together add nothing over replay alone. So the question worth asking
is not what else to bolt on, but what replay should be storing.

Three retention policies, one variable:

``uniform``
    Reservoir sampling. Every example seen has an equal chance of being in the
    buffer, whatever it cost to predict.

``surprise``
    Keep the examples with the highest loss. This is the rule SuRe (2025) uses
    for continual LLM learning, and the intuition is sound -- an example you
    found easy has little left to teach you.

``reducible``
    Keep the examples whose loss is highest *relative to what the model
    expected to pay on them*, the expectation being a linear head on the shared
    trunk trained to predict the model's own loss. The hypothesis as first
    stated.

``uncertain``
    Keep the examples the model is least certain about, by the entropy of its
    own predictive distribution -- ignoring the label entirely.

The fourth policy exists because the third has a flaw visible before running
anything. Random label noise is not a function of the input: the same image can
arrive correctly labelled or not. A head reading only the trunk therefore
cannot predict that a label was flipped, so its estimate of expected loss
misses exactly the cases the hypothesis was about, and ``reducible`` degenerates
toward ``surprise``. Entropy sidesteps this by never looking at the label. A
mislabelled example is one the model is *confident* about and simply disagrees
with, so it scores low and is not retained; a genuinely ambiguous example
scores high. If the aleatoric-versus-epistemic distinction is what matters,
this is the policy that should show it.

The difference between the last two is the whole point. Raw loss cannot
separate "I was confident and I was wrong", which is worth rehearsing, from
"this example is inherently unpredictable", which is not. Under label noise the
highest-loss examples are overwhelmingly the mislabelled ones, so a
surprise-ranked buffer fills with exactly the examples that should never be
rehearsed -- a known failure of high-loss prioritisation.

The established fix is to subtract the *irreducible* loss, estimated by a
separate model trained on held-out data. A deployed continual learner cannot do
that: there is no held-out set and no second training run. But predictive
processing computes the same quantity for free. Precision is an online estimate
of the error a model expects to be unable to remove, so loss in excess of
predicted loss is reducible loss, available at every step without a holdout
model.

Here that estimate is a linear head on the shared trunk, trained to predict the
model's own loss on each example. An example it correctly expects to find hard
scores near zero and is not retained. An example it expected to find easy and
did not scores high, and is kept.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

from .continual import ContinualConfig, MultiClassLearner, Sample, _Reservoir

Policy = Literal["uniform", "surprise", "reducible", "uncertain"]


class _PriorityBuffer:
    """Bounded store that evicts the lowest-scoring example.

    Sampling *from* the buffer stays uniform in every policy; only what earns a
    place differs. Otherwise two things change at once and the comparison says
    nothing.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.items: list[Sample] = []
        self.scores = np.full(self.capacity, -np.inf)
        self.seen = 0

    def add(self, item: Sample, score: float) -> None:
        self.seen += 1
        if len(self.items) < self.capacity:
            self.scores[len(self.items)] = score
            self.items.append(item)
            return
        weakest = int(self.scores.argmin())
        if score > self.scores[weakest]:
            self.items[weakest] = item
            self.scores[weakest] = score


class PrioritisedLearner(MultiClassLearner):
    """The shared always-training trunk, with a choice of replay retention rule."""

    def __init__(
        self,
        config: ContinualConfig | None = None,
        *,
        classes: int = 10,
        policy: Policy = "uniform",
    ) -> None:
        if policy not in ("uniform", "surprise", "reducible", "uncertain"):
            raise ValueError(f"unknown policy: {policy}")
        self.policy = policy
        super().__init__(config, classes=classes)
        width = self.config.hidden[-1]
        # Expected-loss head: the online stand-in for a holdout model. Reads the
        # shared trunk, predicts the loss the model will pay on this example.
        self.expect_w = np.zeros(width)
        self.expect_b = 0.0
        self.expect_lr = 0.05
        self._priority_log: list[float] = []

    # ------------------------------------------------------------------
    def _ensure_head(self, skill: str) -> None:
        already = skill in self.heads
        super()._ensure_head(skill)
        if not already and self.policy != "uniform":
            self._replay[skill] = _PriorityBuffer(self.config.replay_capacity)

    def _per_example_loss(self, x: np.ndarray, y: np.ndarray, skill: str) -> tuple[np.ndarray, np.ndarray]:
        h = self._forward(x)[-1]
        p = self._softmax(self._logits(skill, h))
        picked = p[np.arange(len(y)), y.astype(int)]
        return -np.log(np.clip(picked, 1e-12, 1.0)), h

    def _expected_loss(self, h: np.ndarray) -> np.ndarray:
        return h @ self.expect_w + self.expect_b

    def _fit_expected_loss(self, h: np.ndarray, loss: np.ndarray) -> None:
        """Regress observed loss on trunk features.

        Deliberately linear and slow. The head must learn what is *typically*
        hard about a region of input space, not memorise individual examples --
        a predictor strong enough to fit each example exactly would drive every
        reducible score to zero and the policy would degenerate to uniform.
        """
        error = self._expected_loss(h) - loss
        n = max(len(loss), 1)
        self.expect_w -= self.expect_lr * (h.T @ error) / n
        self.expect_b -= self.expect_lr * float(error.mean())

    def _score(self, x: np.ndarray, y: np.ndarray, skill: str) -> np.ndarray:
        loss, h = self._per_example_loss(x, y, skill)
        if self.policy == "surprise":
            score = loss
        elif self.policy == "uncertain":
            p = self._softmax(self._logits(skill, h))
            score = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=1)
        else:
            score = loss - self._expected_loss(h)
        self._fit_expected_loss(h, loss)
        return score

    # ------------------------------------------------------------------
    def observe(self, skill: str, batch: Sequence[Sample]) -> None:
        if not batch:
            raise ValueError("batch must not be empty")
        self._ensure_head(skill)

        if self.policy == "uniform":
            for item in batch:
                self._replay[skill].add(item)
        else:
            x = np.asarray([c.inputs for c in batch], dtype=float)
            y = np.asarray([c.target for c in batch], dtype=float)
            scores = self._score(x, y, skill)
            self._priority_log.extend(float(s) for s in scores)
            for item, score in zip(batch, scores):
                self._replay[skill].add(item, float(score))

        batches = self._replay_batch()
        fresh_x = np.asarray([c.inputs for c in batch], dtype=float)
        fresh_y = np.asarray([c.target for c in batch], dtype=float)
        if skill in batches:
            past_x, past_y = batches[skill]
            batches[skill] = (
                np.concatenate([fresh_x, past_x]),
                np.concatenate([fresh_y, past_y]),
            )
        else:
            batches[skill] = (fresh_x, fresh_y)
        self._step(batches)

    def buffer_label_noise_rate(self, skill: str, corrupted: set[int]) -> float:
        """What fraction of the retained buffer is mislabelled.

        The mechanism check. Accuracy says whether a policy won; this says
        whether it won for the reason claimed. A prediction that survives its
        outcome test but fails its mechanism test was right by accident.
        """
        items = self._replay[skill].items
        if not items:
            return 0.0
        return sum(1 for item in items if id(item) in corrupted) / len(items)
