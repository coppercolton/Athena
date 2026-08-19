"""Multi-seed learned-representation, reuse, transfer, and rollback benchmark."""

from __future__ import annotations

import argparse

from athena.representations import (
    GroundedRepresentationSystem,
    RawObservation,
    make_visual_cases,
    make_visual_observations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be >= 1")

    representations = 0
    operators = 0
    transfers = 0
    learned_representation_wins = 0
    frozen_during_reuse = 0
    safe_rollbacks = 0
    transfer_accuracies: list[float] = []
    advantages: list[float] = []

    for seed in range(args.seeds):
        base = seed * 10_000
        system = GroundedRepresentationSystem()
        representation = system.learn_representation(
            make_visual_observations(1024, base + 1),
            make_visual_observations(256, base + 2),
        )
        representations += int(representation.promoted)
        if not representation.promoted:
            continue
        representation_checksum = system.representation().checksum

        for index, (name, rule) in enumerate(
            (("right-of", "horizontal_order"), ("above", "vertical_order"))
        ):
            operator = system.learn_operator(
                name,
                make_visual_cases(rule, 128, base + 10 + index * 10),
                make_visual_cases(rule, 256, base + 11 + index * 10),
            )
            operators += int(operator.promoted)
            advantage = (
                operator.candidate_accuracy
                - operator.untrained_representation_accuracy
            )
            advantages.append(advantage)
            learned_representation_wins += int(advantage > 0.0)
            if operator.promoted:
                transfer_accuracy = system.evaluate(
                    name,
                    make_visual_cases(
                        rule,
                        512,
                        base + 12 + index * 10,
                        noise=0.06,
                        brightness=0.70,
                    ),
                )
                transfer_accuracies.append(transfer_accuracy)
                transfers += int(transfer_accuracy >= 0.90)

        frozen_during_reuse += int(
            system.representation().checksum == representation_checksum
        )
        bad_validation = tuple(
            RawObservation(tuple([value] * system.config.sensor_dim))
            for value in (0.35, 0.65)
            for _ in range(64)
        )
        rejected = system.learn_representation(
            make_visual_observations(256, base + 90),
            bad_validation,
            epochs=200,
        )
        safe_rollbacks += int(
            rejected.rolled_back
            and system.representation().checksum == representation_checksum
        )

    total_operators = args.seeds * 2
    mean_transfer = (
        sum(transfer_accuracies) / len(transfer_accuracies)
        if transfer_accuracies
        else 0.0
    )
    mean_advantage = sum(advantages) / len(advantages) if advantages else 0.0
    print("Athena v0.8 learned-representation benchmark")
    print(f"representations promoted: {representations}/{args.seeds}")
    print(f"reasoning operators promoted: {operators}/{total_operators}")
    print(f"shifted-sensor transfers: {transfers}/{total_operators}")
    print(f"mean transfer accuracy: {mean_transfer:.3%}")
    print(
        "learned encoder beat untrained encoder: "
        f"{learned_representation_wins}/{total_operators}"
    )
    print(f"mean held-out representation advantage: {mean_advantage:+.3%}")
    print(f"shared encoder frozen during head reuse: {frozen_during_reuse}/{args.seeds}")
    print(f"unverifiable updates safely rolled back: {safe_rollbacks}/{args.seeds}")


if __name__ == "__main__":
    main()
