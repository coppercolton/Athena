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
from .foundation import (
    DemoFoundation,
    FoundationError,
    FoundationRefusal,
    OpenAIResponsesFoundation,
)
from .memory import Belief, BeliefStore, Episode, EpisodicMemory, HashingEncoder
from .precision import Precision, VolatilityTracker
from .skills import (
    ConsolidationReport,
    Example,
    Experiment,
    KnowledgeGap,
    LearningReport,
    NovelTaskLearner,
    Program,
    ProgramCatalog,
    Skill,
    SkillRegistry,
    VerificationReport,
)
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
    "DemoFoundation",
    "OpenAIResponsesFoundation",
    "FoundationError",
    "FoundationRefusal",
    "HashingEncoder",
    "Episode",
    "EpisodicMemory",
    "Belief",
    "BeliefStore",
    "Program",
    "ProgramCatalog",
    "Example",
    "Experiment",
    "KnowledgeGap",
    "VerificationReport",
    "Skill",
    "ConsolidationReport",
    "LearningReport",
    "SkillRegistry",
    "NovelTaskLearner",
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
__version__ = "0.5.0"
