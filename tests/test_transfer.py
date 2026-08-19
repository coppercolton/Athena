"""Tests for skill-to-skill transfer.

The important assertion here is not that a skill can be learned -- the existing
registries already do that. It is that learning one skill changes what it costs
to learn the next, while leaving the earlier skill bit-for-bit untouched. Those
two properties pull against each other, so both are tested together; either one
alone is easy and meaningless.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena.transfer import (  # noqa: E402
    Example,
    ProgressiveRegistry,
    TransferConfig,
)

DIM = 6
BASE = ("base", lambda x: x[0] * x[1] > 0)
EXTEND = ("extend", lambda x: x[0] * x[1] + x[2] * x[3] > 0)


def cases(fn, count: int, seed: int) -> list[Example]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        x = rng.uniform(-1.0, 1.0, DIM)
        out.append(Example(tuple(x), int(bool(fn(x)))))
    return out


def _registry(lateral: bool, seed: int = 0) -> ProgressiveRegistry:
    return ProgressiveRegistry(TransferConfig(input_dim=DIM, seed=seed), lateral=lateral)


def test_rejects_bad_config():
    for bad in (dict(input_dim=0), dict(hidden_dim=0), dict(epochs=0), dict(max_lateral_sources=-1)):
        try:
            TransferConfig(**bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid config: {bad}")


def test_rejects_bad_example():
    for bad in (dict(inputs=(), target=1), dict(inputs=(0.1,), target=2)):
        try:
            Example(**bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid example: {bad}")


def test_relearning_a_skill_is_refused():
    reg = _registry(True)
    reg.learn(BASE[0], cases(BASE[1], 64, 1), cases(BASE[1], 128, 2))
    try:
        reg.learn(BASE[0], cases(BASE[1], 64, 3), cases(BASE[1], 128, 4))
    except ValueError:
        return
    raise AssertionError("silently overwrote an existing skill")


def test_earlier_skills_are_never_written():
    """Retention must be structural, not a promotion gate catching damage late."""
    reg = _registry(True)
    reg.learn(BASE[0], cases(BASE[1], 96, 1), cases(BASE[1], 256, 2))
    before = reg._experts[BASE[0]].checksum()
    probe = cases(BASE[1], 256, 3)
    accuracy_before = reg.accuracy(BASE[0], probe)

    for index in range(3):
        name = f"later-{index}"
        fn = EXTEND[1] if index else (lambda x: x[4] > 0)
        reg.learn(name, cases(fn, 96, 10 + index), cases(fn, 256, 20 + index))

    assert reg._experts[BASE[0]].checksum() == before, "earlier skill weights changed"
    assert reg.accuracy(BASE[0], probe) == accuracy_before, "earlier skill behaviour changed"


def test_isolated_registry_has_exactly_zero_transfer():
    """Codifies the current protected-expert behaviour, so a regression is visible.

    With every skill isolated, learning other skills first must change the
    outcome by exactly nothing -- not approximately nothing.
    """
    together = _registry(False)
    together.learn(BASE[0], cases(BASE[1], 64, 1), cases(BASE[1], 256, 2))
    with_history = together.learn(EXTEND[0], cases(EXTEND[1], 64, 3), cases(EXTEND[1], 256, 4))

    alone = _registry(False)
    without_history = alone.learn(EXTEND[0], cases(EXTEND[1], 64, 3), cases(EXTEND[1], 256, 4))

    assert with_history.accuracy == without_history.accuracy


def test_lateral_registry_transfers_on_a_curriculum():
    """A skill that reuses an earlier skill's feature must get measurably cheaper."""
    gains = []
    for seed in range(6):
        train, validate = cases(EXTEND[1], 96, 100 + seed), cases(EXTEND[1], 512, 200 + seed)

        lateral = _registry(True, seed)
        lateral.learn(BASE[0], cases(BASE[1], 96, 300 + seed), cases(BASE[1], 256, 400 + seed))
        with_prior = lateral.learn(EXTEND[0], train, validate)

        isolated = _registry(False, seed)
        isolated.learn(BASE[0], cases(BASE[1], 96, 300 + seed), cases(BASE[1], 256, 400 + seed))
        without_prior = isolated.learn(EXTEND[0], train, validate)

        gains.append(with_prior.accuracy - without_prior.accuracy)

    mean_gain = float(np.mean(gains))
    assert mean_gain > 0.02, f"no useful transfer: mean gain {mean_gain:+.4f}"


def test_a_skill_reads_only_skills_that_preceded_it():
    reg = _registry(True)
    reg.learn(BASE[0], cases(BASE[1], 64, 1), cases(BASE[1], 256, 2))
    reg.learn(EXTEND[0], cases(EXTEND[1], 64, 3), cases(EXTEND[1], 256, 4))
    assert reg._source_of[BASE[0]] == ()
    assert reg._source_of[EXTEND[0]] == (BASE[0],)


def test_lateral_sources_are_capped():
    config = TransferConfig(input_dim=DIM, max_lateral_sources=2, seed=0)
    reg = ProgressiveRegistry(config, lateral=True)
    for index in range(5):
        fn = (lambda x, i=index: x[i % DIM] > 0)
        reg.learn(f"s{index}", cases(fn, 64, 500 + index), cases(fn, 256, 600 + index))
    assert len(reg._source_of["s4"]) <= 2


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
