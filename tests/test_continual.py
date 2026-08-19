"""Tests for the always-training trunk.

Every assertion here is about what happens to *other* skills while one skill
learns. A network that trains forever is trivial to write; one that trains
forever without destroying what it already knew is the actual problem, and it
cannot be checked one skill at a time.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena.continual import (  # noqa: E402
    ContinualConfig,
    ContinualLearner,
    Experience,
    SharedPlasticity,
    stream,
)

DIM = 6
BASE = ("base", lambda x: x[0] * x[1] > 0)
REUSE = ("reuse", lambda x: x[0] * x[1] + x[2] > 0)
EXTEND = ("extend", lambda x: x[0] * x[1] + x[2] * x[3] > 0)


def cases(fn, count: int, seed: int) -> list[Experience]:
    rng = np.random.default_rng(seed)
    return [Experience(tuple(v), int(bool(fn(v)))) for v in rng.uniform(-1.0, 1.0, (count, DIM))]


def _learner(**kwargs) -> ContinualLearner:
    return ContinualLearner(ContinualConfig(input_dim=DIM, seed=0, **kwargs))


def test_rejects_bad_config():
    for bad in (
        dict(input_dim=0),
        dict(hidden=()),
        dict(hidden=(4, 0)),
        dict(learning_rate=0.0),
        dict(momentum=1.0),
        dict(replay_capacity=0),
        dict(consolidation=-1.0),
        dict(retention_tolerance=2.0),
    ):
        try:
            ContinualConfig(**bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid config: {bad}")


def test_rejects_malformed_experience():
    for bad in (dict(inputs=(), target=1), dict(inputs=(0.1,), target=5)):
        try:
            Experience(**bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid experience: {bad}")


def test_rejects_wrong_width_experience():
    brain = _learner()
    try:
        brain.observe("s", [Experience((0.1, 0.2), 1)])
    except ValueError:
        return
    raise AssertionError("accepted an experience of the wrong width")


def test_training_actually_changes_the_trunk():
    brain = _learner()
    before = brain.checksum()
    brain.teach(BASE[0], cases(BASE[1], 64, 1), cases(BASE[1], 128, 2), steps=50)
    assert brain.checksum() != before, "the trunk never moved"


def test_frozen_trunk_control_really_freezes():
    """The control must isolate one variable, or every comparison using it lies."""
    brain = _learner(freeze_trunk=True)
    brain.teach(BASE[0], cases(BASE[1], 64, 1), cases(BASE[1], 128, 2), steps=50)
    after_first = brain.checksum()
    brain.teach(REUSE[0], cases(REUSE[1], 64, 3), cases(REUSE[1], 128, 4), steps=200)
    assert brain.checksum() == after_first, "trunk changed while frozen"


def test_learning_never_needs_a_separate_mode():
    brain = _learner()
    brain.teach(BASE[0], cases(BASE[1], 64, 1), cases(BASE[1], 128, 2), steps=50)
    steps_before = brain.steps
    fed = stream(brain, BASE[0], cases(BASE[1], 40, 7))
    assert fed == 40
    assert brain.steps == steps_before + 40, "streamed experience did not train"


def test_replay_buffer_stays_bounded():
    brain = _learner(replay_capacity=32)
    stream(brain, BASE[0], cases(BASE[1], 400, 11))
    assert len(brain._replay[BASE[0]].items) == 32


def test_a_live_trunk_beats_a_frozen_one_on_later_skills():
    """The headline claim, held to the same architecture, data, and seeds.

    Averaged over the later skills of a sequence rather than measured on one
    chosen pair. The per-task advantage varies a lot -- on this curriculum it
    ranges from about +0.02 to +0.25 -- so a single pair can be picked to show
    almost any answer, including no effect at all.
    """
    sequence = (BASE, REUSE, EXTEND)
    gains = []
    for seed in range(4):
        data = [
            (cases(fn, 96, seed * 50 + i), cases(fn, 256, seed * 50 + 25 + i))
            for i, (_, fn) in enumerate(sequence)
        ]
        later = []
        for freeze in (True, False):
            brain = ContinualLearner(
                ContinualConfig(input_dim=DIM, seed=seed, freeze_trunk=freeze)
            )
            scores = [
                brain.teach(name, *data[i], steps=500).accuracy
                for i, (name, _) in enumerate(sequence)
            ]
            later.append(float(np.mean(scores[1:])))
        gains.append(later[1] - later[0])
    mean_gain = float(np.mean(gains))
    assert mean_gain > 0.03, f"a live trunk gave no advantage: {mean_gain:+.4f}"


def test_old_skills_are_not_destroyed_by_new_ones():
    brain = _learner()
    probe = cases(BASE[1], 512, 900)
    brain.teach(BASE[0], cases(BASE[1], 96, 1), cases(BASE[1], 256, 2), steps=400)
    before = brain.accuracy(BASE[0], probe)
    for index, (name, fn) in enumerate((REUSE, EXTEND)):
        brain.teach(name, cases(fn, 96, 10 + index), cases(fn, 256, 20 + index), steps=400)
    after = brain.accuracy(BASE[0], probe)
    assert after > before - 0.10, f"catastrophic forgetting: {before:.4f} -> {after:.4f}"


def test_rollback_restores_the_checkpoint_when_retention_collapses():
    brain = _learner()
    probe = cases(BASE[1], 256, 900)
    brain.teach(BASE[0], cases(BASE[1], 96, 1), cases(BASE[1], 256, 2), steps=300)
    brain.set_probe(BASE[0], probe)
    brain.consolidate()
    good = brain.checksum()
    # Demand a standard the model cannot meet, so a rollback must fire.
    assert brain.check_retention({BASE[0]: 1.5}) is True
    assert brain.checksum() == good, "rollback did not restore the checkpoint"
    assert brain.rollbacks == 1


def test_shared_plasticity_matches_the_registry_interface():
    registry = SharedPlasticity(ContinualConfig(input_dim=DIM, seed=0), steps=200)
    report = registry.learn(BASE[0], cases(BASE[1], 64, 1), cases(BASE[1], 128, 2))
    assert 0.0 <= report.accuracy <= 1.0
    assert len(registry) == 1
    assert registry.predict(BASE[0], (0.5,) * DIM) in (0, 1)
    assert 0.0 <= registry.evaluate(BASE[0], cases(BASE[1], 64, 3)) <= 1.0


if __name__ == "__main__":
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_") and callable(o)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {exc}")
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
