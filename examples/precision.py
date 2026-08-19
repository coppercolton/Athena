"""Learning which of your senses to believe.

Two of the four channels carry a clean signal; the other two are the same
signal buried in noise. Nobody tells the model which is which. Precision --
the inverse variance of each channel's own error history -- separates them on
its own, and the noisy channels stop being allowed to drag the beliefs around.

    python3 examples/precision.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena import Athena, Config, Regime, SwitchingWorld
from athena.plot import chart

STEPS = 4000
NOISE = [0.0, 0.0, 0.35, 0.35]


def main() -> None:
    world = SwitchingWorld(
        [Regime("mixed", [0.08, 0.11, 0.05, 0.13], [0.0, 1.1, 2.2, 3.3])],
        dwell=10**9,
        noise=NOISE,
        seed=0,
    )
    model = Athena(Config(sizes=[4, 24, 12], seed=1))

    trace: list[list[float]] = [[] for _ in range(4)]
    for t in range(STEPS):
        model.observe(world.at(t))
        pi = model.channel_precision
        for c in range(4):
            trace[c].append(float(pi[c]))

    print(__doc__.strip().splitlines()[0])
    print()
    print(
        chart(
            {f"ch{c} ({'clean' if NOISE[c] == 0 else 'noisy'})": trace[c] for c in range(4)},
            log=True,
            height=12,
            title="learned precision per channel (log scale, higher = more trusted)",
        )
    )
    print("\nfinal precision:")
    for c in range(4):
        kind = "clean" if NOISE[c] == 0 else f"noisy (sigma={NOISE[c]})"
        print(f"  channel {c} {kind:<18} {trace[c][-1]:10.2f}")

    clean = min(trace[0][-1], trace[1][-1])
    noisy = max(trace[2][-1], trace[3][-1])
    print(f"\n  clean channels are trusted {clean / max(noisy, 1e-9):.0f}x more than noisy ones,")
    print("  and nothing in the setup told the model which was which.")


if __name__ == "__main__":
    main()
