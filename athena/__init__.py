"""Athena -- an AI that learns by predicting what it is about to see."""

from .core import Athena, Config, StepReport
from .precision import Precision, VolatilityTracker
from .transfer import Example, ProgressiveRegistry, TransferConfig, TransferReport
from .world import Regime, SwitchingWorld, linear_mse, persistence_mse, shifting_world, smooth_world, zero_mse

__all__ = [
    "Athena",
    "Config",
    "StepReport",
    "Precision",
    "ProgressiveRegistry",
    "TransferConfig",
    "TransferReport",
    "Example",
    "VolatilityTracker",
    "Regime",
    "SwitchingWorld",
    "shifting_world",
    "smooth_world",
    "persistence_mse",
    "linear_mse",
    "zero_mse",
]
__version__ = "0.1.0"
