"""Durable, bounded memory for an AI that keeps learning after deployment.

Two kinds of memory are deliberately separated:

``EpisodicMemory``
    Stores what happened in individual interactions.  Similar episodes can be
    retrieved and shown to a foundation model, but an episode is not treated
    as truth merely because it was observed once.

``BeliefStore``
    Accumulates source-labelled evidence for explicit facts.  A fact becomes
    consolidated knowledge only after independent support clears a confidence
    threshold.  Contradictory evidence remains visible instead of being
    overwritten by the most recent statement.

The implementation uses a deterministic hashing encoder so it has no model or
network dependency.  Production deployments can replace retrieval with a
learned embedding service while keeping the same evidence contracts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import re
from typing import Iterable

import numpy as np


_TOKEN = re.compile(r"[a-z0-9]+(?:['_-][a-z0-9]+)?")


class HashingEncoder:
    """Turn text into a stable, fixed-width vector without a vocabulary.

    Index zero is an intercept.  The remaining dimensions use signed feature
    hashing, which keeps the state bounded even as entirely new words arrive.
    """

    def __init__(self, dimension: int = 64) -> None:
        if dimension < 8:
            raise ValueError("dimension must be >= 8")
        self.dimension = int(dimension)

    @staticmethod
    def tokens(text: str) -> tuple[str, ...]:
        return tuple(_TOKEN.findall(text.lower()))

    def encode(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=float)
        vector[0] = 1.0
        for token in self.tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            index = 1 + int.from_bytes(digest[:8], "little") % (self.dimension - 1)
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector[1:]))
        if norm > 0.0:
            vector[1:] /= norm
        return vector

    @staticmethod
    def similarity(left: np.ndarray, right: np.ndarray) -> float:
        """Cosine-like lexical similarity, excluding the regression intercept."""
        return max(0.0, float(left[1:] @ right[1:]))


@dataclass(frozen=True)
class Episode:
    """One decision and the outcome that arrived after it."""

    sequence: int
    decision_id: str
    situation: str
    context_key: str
    action: str
    response: str
    predicted_reward: float
    reward: float
    reliability: float
    observation: str = ""

    @property
    def prediction_error(self) -> float:
        return self.reward - self.predicted_reward

    @property
    def priority(self) -> float:
        """Retention value: trustworthy, surprising outcomes matter most."""
        return self.reliability * (0.25 + abs(self.prediction_error))


class EpisodicMemory:
    """Finite-capacity experience memory with deterministic retrieval."""

    def __init__(self, encoder: HashingEncoder, capacity: int = 1000) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.encoder = encoder
        self.capacity = int(capacity)
        self.episodes: list[Episode] = []

    def __len__(self) -> int:
        return len(self.episodes)

    def add(self, episode: Episode) -> None:
        self.episodes.append(episode)
        if len(self.episodes) > self.capacity:
            # Evict the least informative episode, preferring the oldest when
            # priorities tie. Memory stays bounded without discarding rare,
            # high-surprise events merely because they are old.
            victim = min(
                range(len(self.episodes)),
                key=lambda i: (self.episodes[i].priority, self.episodes[i].sequence),
            )
            self.episodes.pop(victim)

    def retrieve(
        self,
        situation: str,
        context_key: str,
        limit: int = 5,
    ) -> tuple[Episode, ...]:
        if limit < 0:
            raise ValueError("limit must be >= 0")
        if limit == 0 or not self.episodes:
            return ()
        query = self.encoder.encode(f"{context_key} {situation}")
        scored: list[tuple[float, int, Episode]] = []
        for episode in self.episodes:
            encoded = self.encoder.encode(f"{episode.context_key} {episode.situation}")
            similarity = self.encoder.similarity(query, encoded)
            context_bonus = 0.35 if episode.context_key == context_key else 0.0
            score = similarity + context_bonus + 0.05 * episode.priority
            scored.append((score, episode.sequence, episode))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return tuple(item[2] for item in scored[:limit])

    def to_state(self) -> list[dict[str, object]]:
        return [asdict(episode) for episode in self.episodes]

    @classmethod
    def from_state(
        cls,
        encoder: HashingEncoder,
        capacity: int,
        state: Iterable[dict[str, object]],
    ) -> "EpisodicMemory":
        memory = cls(encoder, capacity)
        memory.episodes = [Episode(**item) for item in state]
        return memory


@dataclass(frozen=True)
class Belief:
    """Current evidence-weighted belief about one explicit fact."""

    key: str
    value: str
    confidence: float
    support: float
    total_evidence: float
    alternatives: tuple[tuple[str, float], ...]
    consolidated: bool


class BeliefStore:
    """Evidence-gated semantic knowledge with contradiction tracking."""

    def __init__(
        self,
        encoder: HashingEncoder,
        minimum_support: float = 3.0,
        consolidation_threshold: float = 0.70,
    ) -> None:
        if minimum_support <= 0.0:
            raise ValueError("minimum_support must be > 0")
        if not 0.5 < consolidation_threshold < 1.0:
            raise ValueError("consolidation_threshold must be in (0.5, 1)")
        self.encoder = encoder
        self.minimum_support = float(minimum_support)
        self.consolidation_threshold = float(consolidation_threshold)
        self._evidence: dict[str, dict[str, float]] = {}
        self._sources: dict[str, dict[str, str]] = {}

    def observe(
        self,
        key: str,
        value: str,
        *,
        source: str,
        reliability: float = 1.0,
    ) -> Belief:
        key = key.strip()
        value = value.strip()
        source = source.strip()
        if not key or not value or not source:
            raise ValueError("key, value, and source must be non-empty")
        if not 0.0 < reliability <= 1.0:
            raise ValueError("reliability must be in (0, 1]")

        prior = self._sources.setdefault(key, {}).get(source)
        if prior is not None:
            if prior != value:
                raise ValueError(
                    f"source {source!r} already supplied a different value for {key!r}"
                )
            return self.belief(key)

        self._sources[key][source] = value
        values = self._evidence.setdefault(key, {})
        values[value] = values.get(value, 0.0) + float(reliability)
        return self.belief(key)

    def belief(self, key: str) -> Belief:
        if key not in self._evidence or not self._evidence[key]:
            raise KeyError(key)
        ranked = sorted(
            self._evidence[key].items(),
            key=lambda item: (item[1], item[0]),
            reverse=True,
        )
        value, support = ranked[0]
        total = float(sum(weight for _, weight in ranked))
        # Two units of uncommitted prior mass prevent one observation from
        # becoming "100% confident" just because no alternative was recorded.
        absolute_evidence = support / (support + 2.0)
        agreement = support / max(total, 1e-12)
        confidence = float(absolute_evidence * agreement)
        return Belief(
            key=key,
            value=value,
            confidence=confidence,
            support=float(support),
            total_evidence=total,
            alternatives=tuple((name, float(weight)) for name, weight in ranked[1:]),
            consolidated=(
                support >= self.minimum_support
                and confidence >= self.consolidation_threshold
            ),
        )

    def relevant(
        self,
        query: str,
        limit: int = 5,
        *,
        consolidated_only: bool = True,
    ) -> tuple[Belief, ...]:
        if limit < 0:
            raise ValueError("limit must be >= 0")
        if limit == 0:
            return ()
        encoded_query = self.encoder.encode(query)
        scored: list[tuple[float, Belief]] = []
        for key in self._evidence:
            belief = self.belief(key)
            if consolidated_only and not belief.consolidated:
                continue
            encoded_fact = self.encoder.encode(f"{key} {belief.value}")
            similarity = self.encoder.similarity(encoded_query, encoded_fact)
            if similarity <= 0.0:
                continue
            scored.append((similarity + 0.1 * belief.confidence, belief))
        scored.sort(key=lambda item: (item[0], item[1].key), reverse=True)
        return tuple(belief for _, belief in scored[:limit])

    def all(self) -> tuple[Belief, ...]:
        return tuple(self.belief(key) for key in sorted(self._evidence))

    def to_state(self) -> dict[str, object]:
        return {
            "minimum_support": self.minimum_support,
            "consolidation_threshold": self.consolidation_threshold,
            "evidence": self._evidence,
            "sources": self._sources,
        }

    @classmethod
    def from_state(
        cls,
        encoder: HashingEncoder,
        state: dict[str, object],
    ) -> "BeliefStore":
        store = cls(
            encoder,
            minimum_support=float(state["minimum_support"]),
            consolidation_threshold=float(state["consolidation_threshold"]),
        )
        store._evidence = {
            str(key): {str(value): float(weight) for value, weight in values.items()}
            for key, values in state["evidence"].items()
        }
        store._sources = {
            str(key): {str(source): str(value) for source, value in sources.items()}
            for key, sources in state["sources"].items()
        }
        return store
