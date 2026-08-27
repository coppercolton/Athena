"""Holding a scene together: why "red cube, blue cylinder" is hard.

Vision-language models reliably confuse a red cube beside a blue cylinder with
a blue cube beside a red cylinder. They have all four concepts and lose track
of which goes with which. The failure looks like a training problem and is not
-- it is arithmetic.

A scene encoded as a bag of features is a *sum*:

    red + cube + blue + cylinder

Addition commutes, so the swapped scene produces the identical vector. No
amount of data recovers a distinction the representation cannot express. This
is the binding problem, and it is why compositional reasoning fails while
concept recognition succeeds.

The fix is to bind before bundling, with an operator that keeps track of what
was paired with what:

    (red * cube) + (blue * cylinder)

where ``*`` is elementwise multiplication over bipolar vectors. Now the swapped
scene is a genuinely different vector, and the difference is recoverable.

The properties that make this work are all consequences of high dimension:

*   **Near-orthogonality is free.** Two random bipolar vectors in 10,000
    dimensions have cosine similarity near zero. Orthogonality does not have to
    be learned or enforced -- there is simply so much room that random
    directions miss each other. This is the same condition recent work derives
    as *necessary* for compositional generalization: concepts as linear
    directions, orthogonal across concepts.
*   **Binding is invertible.** Elementwise multiplication over {-1, +1} is its
    own inverse, so ``(a * b) * b = a``. What was bound can be asked back out.
*   **Binding preserves distance while destroying similarity to its parts.**
    ``a * b`` is near-orthogonal to both ``a`` and ``b``, so a bound pair does
    not get confused with either half.
*   **Bundling superposes.** A sum stays similar to each of its terms, so one
    fixed-width vector holds many facts at once and each can be queried.

This is Smolensky's tensor-product binding made practical: Plate's holographic
reduced representations, Kanerva's hyperdimensional computing, the family
usually called Vector Symbolic Architectures. It dates to the 1990s. It is not
new and it is not what modern multimodal models do.
"""

from __future__ import annotations

import numpy as np


class Space:
    """A high-dimensional space where concepts are near-orthogonal by default."""

    def __init__(self, dimensions: int = 10_000, seed: int = 0) -> None:
        self.dimensions = int(dimensions)
        self._rng = np.random.default_rng(seed)
        self._atoms: dict[str, np.ndarray] = {}

    def atom(self, name: str) -> np.ndarray:
        """A fresh random direction for a concept, or the one already assigned."""
        if name not in self._atoms:
            self._atoms[name] = self._rng.choice([-1.0, 1.0], size=self.dimensions)
        return self._atoms[name]

    # ------------------------------------------------------------------
    @staticmethod
    def bind(*vectors: np.ndarray) -> np.ndarray:
        """Pair things together, reversibly. Elementwise product, self-inverse."""
        out = vectors[0].copy()
        for v in vectors[1:]:
            out = out * v
        return out

    @staticmethod
    def bundle(*vectors: np.ndarray) -> np.ndarray:
        """Superpose things into one vector, keeping similarity to each part.

        The sum is left real-valued rather than thresholded back to +/-1.
        Thresholding looks harmless and is not: with an even number of terms
        the sum is zero in about 37% of dimensions, every one of those ties
        breaks the same way, and the result is a constant vector added to every
        structure the system builds. That shared component then dominates any
        product of two structures, which silently destroys exactly the
        cancellation that analogical mapping depends on. Cosine similarity
        ignores magnitude, so nothing is gained by thresholding anyway.
        """
        return np.sum(vectors, axis=0)

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    def unbind(self, composite: np.ndarray, key: np.ndarray) -> np.ndarray:
        """Ask a bound pair what was stored under this key."""
        return self.bind(composite, key)

    def nearest(self, query: np.ndarray, names: list[str]) -> tuple[str, float]:
        scores = [(n, self.similarity(query, self.atom(n))) for n in names]
        return max(scores, key=lambda kv: kv[1])


def bag_scene(space: Space, objects: list[tuple[str, str]]) -> np.ndarray:
    """How a bag-of-features model sees a scene: attributes, unpaired."""
    parts = []
    for colour, shape in objects:
        parts.extend([space.atom(colour), space.atom(shape)])
    return space.bundle(*parts)


def bound_scene(space: Space, objects: list[tuple[str, str]]) -> np.ndarray:
    """The same scene with each object's attributes bound before superposing."""
    return space.bundle(
        *[space.bind(space.atom(c), space.atom(s)) for c, s in objects]
    )
