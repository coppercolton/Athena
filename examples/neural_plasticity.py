"""Watch verified experience change Athena's retained neural parameters."""

from __future__ import annotations

import tempfile
from pathlib import Path

from athena import NeuralExample, ProtectedPlasticity, make_reasoning_cases


def flipped(cases):
    return tuple(NeuralExample(item.inputs, 1 - item.target) for item in cases)


system = ProtectedPlasticity()
training = make_reasoning_cases("relative_balance", 128, 1)
validation = make_reasoning_cases("relative_balance", 96, 2)
transfer = make_reasoning_cases("relative_balance", 256, 3, scale=5.0)

report = system.learn("compare-aggregates", training, validation)
print("Athena v0.7 protected neural plasticity")
print(f"before held-out accuracy: {report.before_accuracy:.1%}")
print(f"after held-out accuracy:  {report.candidate_accuracy:.1%}")
print(f"neural weight delta:      {report.weight_delta:.3f}")
print(f"parameters:               {report.checksum_before} -> {report.checksum_after}")
print(f"promoted:                 {report.promoted} ({report.reason})")
print(f"new-magnitude transfer:   {system.evaluate('compare-aggregates', transfer):.1%}")

first_checksum = system.get("compare-aggregates").checksum
second = system.learn(
    "same-side-relation",
    make_reasoning_cases("same_sign", 128, 4),
    make_reasoning_cases("same_sign", 96, 5),
)
print(f"second neural expert:     {second.promoted}")
print(
    "first expert retained:  "
    f"{system.get('compare-aggregates').checksum == first_checksum}"
)

with tempfile.TemporaryDirectory() as directory:
    checkpoint = Path(directory) / "plasticity.npz"
    system.save(checkpoint)
    restored = ProtectedPlasticity.load(checkpoint)
print(f"restart checksum exact:   {restored.get('compare-aggregates').checksum == first_checksum}")

rejected = restored.learn(
    "compare-aggregates",
    flipped(make_reasoning_cases("relative_balance", 128, 6)),
    flipped(make_reasoning_cases("relative_balance", 96, 7)),
)
print(f"contradictory update:     promoted={rejected.promoted}, rollback={rejected.rolled_back}")
print(
    "retained after rollback: "
    f"{restored.get('compare-aggregates').checksum == first_checksum}"
)
