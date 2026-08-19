"""What happens when the world changes underneath it.

The signal switches between unrelated regimes without warning. Watch three
things: prediction error spikes at each switch, the volatility signal notices
and re-opens learning, and the error comes back down without anyone resetting
anything.

Takes a couple of minutes: it runs two models over 12,000 observations, because
the banked model spends its first several regime cycles working out how many
worlds there are and only pulls ahead afterwards.

    python3 examples/regime_shift.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena import Athena, Config, persistence_mse, shifting_world
from athena.plot import chart

STEPS = 12000
DWELL = 1000
CHANNELS = 4


def main() -> None:
    world = shifting_world(n_channels=CHANNELS, n_regimes=3, dwell=DWELL, seed=3)
    observations = [world.at(t) for t in range(STEPS)]
    switches = world.switch_points(STEPS)

    # One model holds a single set of dynamics and must overwrite it at every
    # change; the other holds a bank and can put the old one down and pick it
    # back up. The gap between them is the whole point of the context gate.
    single = Athena(Config(sizes=[CHANNELS, 24, 12], seed=1, experts=1))
    banked = Athena(Config(sizes=[CHANNELS, 24, 12], seed=1, experts=6))
    lone = np.array([r.mse for r in single.run(observations)])
    reports = banked.run(observations)
    mse = np.array([r.mse for r in reports])
    gain = np.array([r.gain for r in reports])
    model = banked

    print(__doc__.strip().splitlines()[0])
    print()
    print(
        chart(
            {
                "banked (experts=6)": mse,
                "single (experts=1)": lone,
                "persistence": np.array(persistence_mse(observations)),
            },
            log=True,
            marks=switches,
            title=f"prediction error, world changes every {DWELL} steps (log scale)",
        )
    )
    print()
    print(
        chart(
            {"learning gain": gain},
            height=8,
            marks=switches,
            title="surprise-driven learning rate multiplier",
        )
    )

    print("\nerror by position within a regime, averaged over the last 4 dwells:")
    print("  steps after switch     single      banked")
    prof_b = np.array([mse[s : s + DWELL] for s in range(0, STEPS, DWELL)])[-4:].mean(0)
    prof_s = np.array([lone[s : s + DWELL] for s in range(0, STEPS, DWELL)])[-4:].mean(0)
    for start in (0, 200, 400, 800):
        end = start + 100
        print(f"  {start:>6}-{end:<12} {prof_s[start:end].mean():>10.2e}  {prof_b[start:end].mean():>10.2e}")

    print("\ncontext gate:")
    print(f"  experts recruited: {model.gate.allocations or 'none'}")
    print(f"  final belief:      {np.round(model.gate.belief, 3)}")
    print(f"  steps each expert was dominant: {model.gate.claimed.astype(int)}")


if __name__ == "__main__":
    main()
