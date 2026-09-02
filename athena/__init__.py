"""Athena — an experiment in whether experience can make learning cheaper.

Every module here has a measurement behind it in the README. The subsystems
that did not — nine versions of protected experts, agent scaffolding, skill
registries, tool learning, and a browser playground — were removed once the
benchmarks put their combined contribution at +0.001. They remain on the
``agent/athena-v9-live-apprenticeship`` branch and in this branch's history;
nothing was lost, it just stopped being presented as though it worked.

What survives, and what each part is worth:

``core``       hierarchical predictive coding: predict, compare, settle, learn.
               200x better than persistence on stationary streams, though the
               classical RLS head inside it does most of that work.
``continual``  a shared trunk that never stops training, with replay,
               consolidation and rollback. Replay is what matters: +34 points
               on Permuted-MNIST where consolidation adds +0.001.
``der``        rehearse the function, not the answer. +0.080 at fixed buffer,
               the single largest gain measured here.
``library``    keep the pieces of what you learn and reuse them. Compounds 4-6x
               better than a gradient network, with sleep-style consolidation
               (7x compression, free) and imagination that works only when
               paired with a rejection step.
``taught``     a world where teaching is possible, because Permuted-MNIST is
               not one.
``transfer``   progressive lateral connections: forward transfer with exact
               retention, superseded by the shared trunk.
``priority``   replay retention policies. All of them lose to uniform
               sampling; kept because that result is easy to rediscover.
``timescales`` one anchor is worth +0.080, a second is worth nothing.
``plasticity`` per-skill protected experts, kept as the isolation baseline.
"""

from .baselines import BaselineReport, OnlineRLS
from .context import ContextGate, GateState
from .continual import (
    ContinualConfig,
    ContinualLearner,
    Experience,
    LearningReport,
    MultiClassLearner,
    Sample,
    SharedPlasticity,
    related_tasks,
    task_cases,
    unrelated_tasks,
)
from .core import Athena, Config, StepReport
from .der import DERLearner
from .library import LibraryLearner
from .precision import Precision, VolatilityTracker
from .priority import PrioritisedLearner
from .taught import EpisodicLearner, GradientLearner, Rule, make_composed_rules, make_rules, rule_examples
from .timescales import TimescaleLearner
from .transfer import Example, ProgressiveRegistry, TransferConfig, TransferReport
from .agent import LifelongAgent, Problem, Record, Skill
from .hypotheses import Hypothesis, HypothesisSpace, experiment
from .worlds import Agreement, Conjunction, Domain, Oracle, balanced, random_agreement, random_conjunction
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
    "LifelongAgent",
    "Problem",
    "Record",
    "Skill",
    "Hypothesis",
    "HypothesisSpace",
    "experiment",
    "Domain",
    "Conjunction",
    "Agreement",
    "Oracle",
    "balanced",
    "random_conjunction",
    "random_agreement",
    "Athena",
    "Config",
    "StepReport",
    "Precision",
    "VolatilityTracker",
    "ContextGate",
    "GateState",
    "OnlineRLS",
    "BaselineReport",
    "ContinualLearner",
    "ContinualConfig",
    "MultiClassLearner",
    "SharedPlasticity",
    "Experience",
    "Sample",
    "LearningReport",
    "related_tasks",
    "unrelated_tasks",
    "task_cases",
    "DERLearner",
    "PrioritisedLearner",
    "TimescaleLearner",
    "LibraryLearner",
    "EpisodicLearner",
    "GradientLearner",
    "Rule",
    "make_rules",
    "make_composed_rules",
    "rule_examples",
    "ProgressiveRegistry",
    "TransferConfig",
    "TransferReport",
    "Example",
    "Regime",
    "SwitchingWorld",
    "shifting_world",
    "smooth_world",
    "persistence_mse",
    "linear_mse",
    "zero_mse",
]
__version__ = "0.10.0"
