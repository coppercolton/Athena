"""Behavioral tests for learned raw-perception representations."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena.representations import (  # noqa: E402
    GroundedExample,
    GroundedRepresentationSystem,
    RawObservation,
    RepresentationConfig,
    make_visual_cases,
    make_visual_observations,
)


def _learn_representation(
    system: GroundedRepresentationSystem | None = None,
) -> GroundedRepresentationSystem:
    system = system or GroundedRepresentationSystem()
    report = system.learn_representation(
        make_visual_observations(512, 101),
        make_visual_observations(128, 102),
    )
    assert report.promoted
    return system


def _learn_right_operator(
    system: GroundedRepresentationSystem | None = None,
) -> GroundedRepresentationSystem:
    system = _learn_representation(system)
    report = system.learn_operator(
        "right-of",
        make_visual_cases("horizontal_order", 128, 103),
        make_visual_cases("horizontal_order", 128, 104),
    )
    assert report.promoted
    return system


def _flipped(examples):
    return tuple(
        GroundedExample(item.observation, 1 - item.target) for item in examples
    )


def test_raw_pixels_learn_a_noncollapsed_compressed_representation():
    system = GroundedRepresentationSystem()
    report = system.learn_representation(
        make_visual_observations(512, 101),
        make_visual_observations(128, 102),
    )

    assert report.promoted and report.recruited
    assert report.candidate_loss < report.before_loss * 0.15
    assert report.candidate_loss <= system.config.maximum_reconstruction_loss
    assert report.latent_variance >= system.config.minimum_latent_variance
    assert report.weight_delta > 1.0
    assert report.checksum_before != report.checksum_after
    state = system.representation()
    assert state.sensor_dim == 128
    assert state.latent_dim == 16
    assert state.latent_dim < state.sensor_dim


def test_learned_representation_beats_same_head_on_untrained_encoder():
    system = GroundedRepresentationSystem()
    system.learn_representation(
        make_visual_observations(1024, 1),
        make_visual_observations(256, 2),
    )
    report = system.learn_operator(
        "right-of",
        make_visual_cases("horizontal_order", 128, 10),
        make_visual_cases("horizontal_order", 256, 11),
    )

    assert report.promoted
    assert report.candidate_accuracy >= 0.90
    assert report.candidate_accuracy > report.untrained_representation_accuracy


def test_two_operators_reuse_one_frozen_representation_and_transfer():
    system = GroundedRepresentationSystem()
    system.learn_representation(
        make_visual_observations(1024, 1),
        make_visual_observations(256, 2),
    )
    representation_checksum = system.representation().checksum
    reports = []
    for name, rule, seed in (
        ("right-of", "horizontal_order", 10),
        ("above", "vertical_order", 20),
    ):
        report = system.learn_operator(
            name,
            make_visual_cases(rule, 128, seed),
            make_visual_cases(rule, 256, seed + 1),
        )
        reports.append(report)
        assert report.promoted
        assert system.evaluate(
            name,
            make_visual_cases(
                rule,
                512,
                seed + 2,
                noise=0.06,
                brightness=0.70,
            ),
        ) >= 0.90

    assert len(system) == 2
    assert system.representation().checksum == representation_checksum
    assert all(item.representation_checksum == representation_checksum for item in reports)


def test_protected_refinement_changes_encoder_without_forgetting_operator():
    system = _learn_right_operator()
    representation_before = system.representation()
    operator_before = system.get_operator("right-of")

    report = system.learn_representation(
        make_visual_observations(256, 107),
        make_visual_observations(128, 108),
        epochs=300,
    )

    assert report.promoted and not report.recruited
    assert report.version == 2
    assert report.checksum_after != representation_before.checksum
    assert report.minimum_operator_accuracy >= (
        operator_before.validation_accuracy
        - system.config.operator_regression_tolerance
    )
    updated = system.get_operator("right-of")
    assert updated.version == 2
    assert updated.representation_version == 2


def test_unverifiable_representation_update_rolls_back_every_weight():
    system = _learn_right_operator()
    representation_before = system.representation()
    operator_before = system.get_operator("right-of")
    bad_validation = tuple(
        RawObservation(tuple([value] * system.config.sensor_dim))
        for value in (0.35, 0.65)
        for _ in range(64)
    )

    report = system.learn_representation(
        make_visual_observations(256, 120),
        bad_validation,
        epochs=200,
    )

    assert not report.promoted and report.rolled_back
    assert report.checksum_after == report.checksum_before
    assert system.representation() == representation_before
    assert system.get_operator("right-of") == operator_before


def test_contradictory_operator_update_is_rejected_without_forgetting():
    system = _learn_right_operator()
    operator_before = system.get_operator("right-of")
    representation_before = system.representation()

    report = system.learn_operator(
        "right-of",
        _flipped(make_visual_cases("horizontal_order", 128, 109)),
        _flipped(make_visual_cases("horizontal_order", 128, 110)),
    )

    assert not report.promoted and report.rolled_back
    assert report.replay_examples == operator_before.replay_examples
    assert system.get_operator("right-of") == operator_before
    assert system.representation() == representation_before


def test_representation_and_operators_survive_restart_exactly():
    system = _learn_right_operator()
    probe = make_visual_cases("horizontal_order", 32, 130)
    probabilities_before = tuple(
        system.predict_probability("right-of", item.observation) for item in probe
    )
    latent_before = system.encode(probe[0].observation)
    representation_before = system.representation()
    operator_before = system.get_operator("right-of")

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "representations.npz"
        system.save(checkpoint)
        restored = GroundedRepresentationSystem.load(checkpoint)

    probabilities_after = tuple(
        restored.predict_probability("right-of", item.observation) for item in probe
    )
    assert probabilities_after == probabilities_before
    assert restored.encode(probe[0].observation) == latent_before
    assert restored.representation() == representation_before
    assert restored.get_operator("right-of") == operator_before


def test_representation_learning_is_deterministic():
    first = _learn_right_operator()
    second = _learn_right_operator()

    assert first.representation() == second.representation()
    assert first.get_operator("right-of") == second.get_operator("right-of")


def test_invalid_or_underverified_representation_experience_is_rejected():
    for kwargs in (
        {"sensor_dim": 3},
        {"latent_dim": 128},
        {"head_hidden_dim": 1},
        {"representation_epochs": 0},
        {"operator_validation_threshold": 0.5},
    ):
        try:
            RepresentationConfig(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid representation config accepted: {kwargs}")

    system = GroundedRepresentationSystem()
    try:
        system.learn_representation(
            make_visual_observations(64, 140),
            make_visual_observations(8, 141),
        )
    except ValueError as exc:
        assert "held-out" in str(exc)
    else:
        raise AssertionError("underverified representation was accepted")

    try:
        RawObservation((0.0, 1.2))
    except ValueError:
        return
    raise AssertionError("out-of-range raw observation was accepted")


if __name__ == "__main__":
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failed = 0
    for name, function in tests:
        try:
            function()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {exc}")
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
