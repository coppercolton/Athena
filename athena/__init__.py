"""Athena -- an AI that learns by predicting what it is about to see."""

from .baselines import BaselineReport, OnlineRLS
from .core import Athena, Config, StepReport
from .precision import Precision, VolatilityTracker
from .world import (
    Regime,
    SwitchingWorld,
    linear_mse,
    persistence_mse,
    shifting_world,
    smooth_world,
    zero_mse,
)

__all__ = [
    "Athena",
    "Config",
    "StepReport",
    "BaselineReport",
    "OnlineRLS",
    "Precision",
    "VolatilityTracker",
    "Regime",
    "SwitchingWorld",
    "shifting_world",
    "smooth_world",
    "persistence_mse",
    "linear_mse",
    "zero_mse",
]
__version__ = "0.2.0"
