"""The failure that stops multimodal models composing, and the arithmetic behind it.

Vision-language models confuse "a red cube and a blue cylinder" with "a blue
cube and a red cylinder". They hold all four concepts and lose the pairing.

This is not a training failure. A scene encoded as a bag of features is a sum,
addition commutes, and the two scenes are therefore *the same vector* -- their
cosine similarity is exactly 1.0. No quantity of data can separate what the
representation cannot express.

Binding before bundling fixes it. Three things are measured here:

    1. the swap test -- can the representation tell the two scenes apart
    2. the query task -- "is there a red cube?", where the hard negatives are
       scenes containing a red *something* and a *something* cube, but no red
       cube
    3. capacity -- how many objects one fixed-width vector holds before
       superposition saturates and the answers dissolve

The third is where honesty lives. Binding is not free, and the cost is a
ceiling on how much a single vector can carry.

    python3 examples/binding_problem.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena.binding import Space, bag_scene, bound_scene

COLOURS = ["red", "blue", "green", "yellow", "purple", "orange"]
SHAPES = ["cube", "cylinder", "sphere", "cone", "prism", "torus"]
DIMENSIONS = 10_000
TRIALS = 400


def scene_with_hard_negative(rng, n_objects: int):
    """A scene, a pair that is in it, and a pair whose halves are both in it.

    The second is the whole difficulty. Both attributes of the negative query
    are present in the scene -- just never on the same object -- so anything
    that pools attributes will say yes.
    """
    colours = list(rng.choice(COLOURS, size=n_objects, replace=False))
    shapes = list(rng.choice(SHAPES, size=n_objects, replace=False))
    objects = list(zip(colours, shapes))
    positive = objects[0]
    negative = (colours[0], shapes[1])  # right colour, wrong object's shape
    return objects, positive, negative


def run(n_objects: int, trials: int = TRIALS, seed: int = 0):
    rng = np.random.default_rng(seed)
    space = Space(DIMENSIONS, seed=seed)
    scores = {"bag": [], "bound": []}
    identical = 0

    for _ in range(trials):
        objects, positive, negative = scene_with_hard_negative(rng, n_objects)
        bag = bag_scene(space, objects)
        bound = bound_scene(space, objects)

        swapped = list(objects)
        swapped[0], swapped[1] = (objects[0][0], objects[1][1]), (objects[1][0], objects[0][1])
        if space.similarity(bag, bag_scene(space, swapped)) > 0.999:
            identical += 1

        # A bag model must ask about attributes separately; a bound model can
        # ask about the pair itself.
        bag_pos = min(
            space.similarity(bag, space.atom(positive[0])),
            space.similarity(bag, space.atom(positive[1])),
        )
        bag_neg = min(
            space.similarity(bag, space.atom(negative[0])),
            space.similarity(bag, space.atom(negative[1])),
        )
        bound_pos = space.similarity(bound, space.bind(*[space.atom(a) for a in positive]))
        bound_neg = space.similarity(bound, space.bind(*[space.atom(a) for a in negative]))

        scores["bag"].append(bag_pos > bag_neg)
        scores["bound"].append(bound_pos > bound_neg)

    return (
        float(np.mean(scores["bag"])),
        float(np.mean(scores["bound"])),
        identical / trials,
    )


def main() -> None:
    print(__doc__.strip().splitlines()[0])
    print(f"\n{DIMENSIONS}-dimensional space, {TRIALS} random scenes per row\n")
    print("Can the representation tell a real pair from a swapped one?")
    print(f"  {'objects':>8}{'bag':>10}{'bound':>10}{'bag scenes identical':>24}")
    for n in (2, 3, 4, 5, 6):
        bag, bound, identical = run(n)
        print(f"  {n:>8}{bag:>10.3f}{bound:>10.3f}{identical:>24.0%}", flush=True)

    print("\n  chance is 0.500. A bag of features cannot exceed it by construction:")
    print("  the two scenes are the same vector, so there is nothing to measure.")

    print("\nCapacity: one vector, many bound facts, queried back out")
    space = Space(DIMENSIONS, seed=1)
    roles = ["colour", "shape", "taste", "affords", "smell", "size", "texture", "origin"]
    fillers = ["red", "round", "sweet", "throwable", "fresh", "small", "waxy", "orchard"]
    print(f"  {'facts held':>11}{'recovered correctly':>22}")
    for k in (2, 4, 6, 8):
        apple = space.bundle(
            *[space.bind(space.atom(r), space.atom(f)) for r, f in zip(roles[:k], fillers[:k])]
        )
        hits = sum(
            space.nearest(space.unbind(apple, space.atom(role)), fillers)[0] == filler
            for role, filler in zip(roles[:k], fillers[:k])
        )
        print(f"  {k:>11}{f'{hits}/{k}':>22}", flush=True)


if __name__ == "__main__":
    main()
