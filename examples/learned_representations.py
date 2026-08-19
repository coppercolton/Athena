"""Watch Athena learn a latent visual state and reuse it for new reasoning."""

from __future__ import annotations

from athena.representations import (
    GroundedRepresentationSystem,
    RawObservation,
    make_visual_cases,
    make_visual_observations,
)


system = GroundedRepresentationSystem()
representation = system.learn_representation(
    make_visual_observations(1024, 1),
    make_visual_observations(256, 2),
)
print("Athena v0.8 learned representations")
print(f"raw sensor width:          {system.config.sensor_dim} pixels")
print(f"learned latent width:      {system.config.latent_dim} values")
print(f"reconstruction loss:       {representation.before_loss:.4f} -> {representation.candidate_loss:.4f}")
print(f"representation weights:   {representation.checksum_before} -> {representation.checksum_after}")
print(f"promoted representation:  {representation.promoted}")

checksum = system.representation().checksum
for name, rule, seed in (
    ("right-of", "horizontal_order", 10),
    ("above", "vertical_order", 20),
):
    report = system.learn_operator(
        name,
        make_visual_cases(rule, 128, seed),
        make_visual_cases(rule, 256, seed + 1),
    )
    transfer = make_visual_cases(
        rule,
        512,
        seed + 2,
        noise=0.06,
        brightness=0.70,
    )
    print(f"\n{name}")
    print(f"  held-out accuracy:       {report.candidate_accuracy:.1%}")
    print(f"  untrained encoder:       {report.untrained_representation_accuracy:.1%}")
    if report.promoted:
        print(f"  shifted-sensor transfer: {system.evaluate(name, transfer):.1%}")
    print(f"  representation reused:  {system.representation().checksum == checksum}")

bad_validation = tuple(
    RawObservation(tuple([value] * system.config.sensor_dim))
    for value in (0.35, 0.65)
    for _ in range(64)
)
rejected = system.learn_representation(
    make_visual_observations(256, 50),
    bad_validation,
    epochs=200,
)
print("\nunverifiable representation update")
print(f"  promoted:                {rejected.promoted}")
print(f"  rolled back:             {rejected.rolled_back}")
print(f"  retained checksum exact: {system.representation().checksum == checksum}")
