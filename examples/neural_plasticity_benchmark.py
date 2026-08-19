"""Multi-seed acquisition, transfer, retention, and rollback benchmark."""

from __future__ import annotations

import argparse

from athena import NeuralExample, ProtectedPlasticity, make_reasoning_cases


def flipped(cases):
    return tuple(NeuralExample(item.inputs, 1 - item.target) for item in cases)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be >= 1")

    acquisitions = 0
    transfers = 0
    retained = 0
    protected_rollbacks = 0
    accuracies = []
    for seed in range(args.seeds):
        base = seed * 10_000
        system = ProtectedPlasticity()
        balance = system.learn(
            "balance",
            make_reasoning_cases("relative_balance", 128, base + 1),
            make_reasoning_cases("relative_balance", 96, base + 2),
        )
        acquisitions += int(balance.promoted)
        if balance.promoted:
            accuracy = system.evaluate(
                "balance",
                make_reasoning_cases("relative_balance", 256, base + 3, scale=5.0),
            )
            accuracies.append(accuracy)
            transfers += int(accuracy >= 0.95)
            checksum = system.get("balance").checksum
        else:
            checksum = ""

        nonlinear = system.learn(
            "same-sign",
            make_reasoning_cases("same_sign", 128, base + 4),
            make_reasoning_cases("same_sign", 96, base + 5),
        )
        acquisitions += int(nonlinear.promoted)
        if nonlinear.promoted:
            accuracy = system.evaluate(
                "same-sign",
                make_reasoning_cases("same_sign", 256, base + 6, scale=1.5),
            )
            accuracies.append(accuracy)
            transfers += int(accuracy >= 0.95)
        retained += int(bool(checksum) and system.get("balance").checksum == checksum)

        if checksum:
            rejected = system.learn(
                "balance",
                flipped(make_reasoning_cases("relative_balance", 128, base + 7)),
                flipped(make_reasoning_cases("relative_balance", 96, base + 8)),
            )
            protected_rollbacks += int(
                rejected.rolled_back and system.get("balance").checksum == checksum
            )

    total_operators = args.seeds * 2
    mean_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    print("Athena v0.7 neural plasticity benchmark")
    print(f"neural operators promoted: {acquisitions}/{total_operators}")
    print(f"unseen-distribution transfers: {transfers}/{total_operators}")
    print(f"mean transfer accuracy: {mean_accuracy:.3%}")
    print(f"first experts unchanged after recruitment: {retained}/{args.seeds}")
    print(f"contradictory updates safely rolled back: {protected_rollbacks}/{args.seeds}")


if __name__ == "__main__":
    main()
