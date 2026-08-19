"""Athena -- an AI that learns by predicting what it is about to see."""

from .agent import (
    AgentConfig,
    AthenaAgent,
    CallableFoundation,
    Candidate,
    Decision,
    FoundationModel,
    OutcomeReport,
    ScoredCandidate,
    StrategyKnowledge,
)
from .baselines import BaselineReport, OnlineRLS
from .core import Athena, Config, StepReport
from .memory import Belief, BeliefStore, Episode, EpisodicMemory, HashingEncoder
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
    "AthenaAgent",
    "AgentConfig",
    "FoundationModel",
    "CallableFoundation",
    "Candidate",
    "ScoredCandidate",
    "Decision",
    "OutcomeReport",
    "StrategyKnowledge",
    "HashingEncoder",
    "Episode",
    "EpisodicMemory",
    "Belief",
    "BeliefStore",
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
__version__ = "0.3.0"
