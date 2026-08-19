"""Behavioral tests for Athena's deployment-learning agent.

Run directly with ``python3 tests/test_agent.py``.  The tests use a tiny
deterministic foundation stub so they measure Athena's experience loop, not an
external API or a language model's changing output.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena.agent import (  # noqa: E402
    AgentConfig,
    AthenaAgent,
    Candidate,
)


class FoundationStub:
    """Broad prior: email is normally good, but deployment can prove otherwise."""

    def __init__(self) -> None:
        self.last_memories = ()
        self.last_facts = ()
        self.last_strategies = ()

    def propose(self, situation, *, memories, facts, strategies, n):
        self.last_memories = tuple(memories)
        self.last_facts = tuple(facts)
        self.last_strategies = tuple(strategies)
        return (
            Candidate("email", "Send a thoughtful email", prior=0.75),
            Candidate("text", "Send a short text", prior=0.25),
        )[:n]


def _agent(foundation=None, **overrides):
    values = {
        "feature_dim": 32,
        "prior_strength": 2.0,
        "exploration_bonus": 0.25,
        "minimum_strategy_evidence": 6.0,
    }
    values.update(overrides)
    return AthenaAgent(foundation=foundation, config=AgentConfig(**values))


def test_policy_improves_from_consequences_after_deployment():
    foundation = FoundationStub()
    agent = _agent(foundation)
    actions, rewards, errors = [], [], []
    for _ in range(30):
        decision = agent.decide(
            "An urgent lead asks whether they can tour tonight",
            context_key="urgent-lead",
        )
        reward = 1.0 if decision.selected.action == "text" else 0.0
        report = agent.learn(
            decision.id,
            reward,
            observation="The lead replied quickly" if reward else "No reply",
        )
        actions.append(decision.selected.action)
        rewards.append(reward)
        errors.append(report.squared_error)

    assert actions[0] == "email", "agent ignored its pretrained prior"
    assert actions[-10:].count("text") >= 8, "experience did not change behavior"
    assert np.mean(rewards[-10:]) > np.mean(rewards[:5])
    assert np.mean(errors[-10:]) < np.mean(errors[:5])


def test_contexts_learn_opposite_strategies_without_overwriting():
    agent = _agent()
    text = [Candidate("text", "Send a text", prior=0.5)]
    email = [Candidate("email", "Send an email", prior=0.5)]

    # Controlled exploration supplies evidence for both actions. Text works
    # for urgent leads; email works for routine follow-ups.
    for _ in range(10):
        for context, situation, candidates, reward in (
            ("urgent", "lead needs an answer tonight", text, 1.0),
            ("urgent", "lead needs an answer tonight", email, 0.0),
            ("routine", "customer asks for a monthly recap", text, 0.0),
            ("routine", "customer asks for a monthly recap", email, 1.0),
        ):
            decision = agent.decide(
                situation,
                context_key=context,
                candidates=candidates,
            )
            agent.learn(decision.id, reward)

    urgent = agent.decide(
        "lead needs an answer tonight",
        context_key="urgent",
        candidates=[*text, *email],
    )
    routine = agent.decide(
        "customer asks for a monthly recap",
        context_key="routine",
        candidates=[*text, *email],
    )
    assert urgent.selected.action == "text"
    assert routine.selected.action == "email"
    assert agent.strategy("urgent", "text").status == "preferred"
    assert agent.strategy("urgent", "email").status == "avoid"


def test_strategy_can_reverse_when_the_same_world_changes():
    """Continual learning must stay plastic after it has accumulated evidence."""
    agent = _agent(forgetting=0.90, minimum_strategy_evidence=4.0)
    text = [Candidate("text", "Send a text", prior=0.5)]
    email = [Candidate("email", "Send an email", prior=0.5)]

    for _ in range(15):
        for candidates, reward in ((text, 1.0), (email, 0.0)):
            decision = agent.decide(
                "A customer is waiting",
                context_key="channel",
                candidates=candidates,
            )
            agent.learn(decision.id, reward)
    assert agent.strategy("channel", "text").status == "preferred"
    assert agent.strategy("channel", "email").status == "avoid"

    # The environment itself changes. Decayed evidence plus recursive updates
    # must eventually reverse the old policy rather than fossilising it.
    for _ in range(35):
        for candidates, reward in ((text, 0.0), (email, 1.0)):
            decision = agent.decide(
                "A customer is waiting",
                context_key="channel",
                candidates=candidates,
            )
            agent.learn(decision.id, reward)
    assert agent.strategy("channel", "text").status == "avoid"
    assert agent.strategy("channel", "email").status == "preferred"

    choice = agent.decide(
        "A customer is waiting",
        context_key="channel",
        candidates=[*text, *email],
    )
    assert choice.selected.action == "email"


def test_experiences_and_consolidated_knowledge_reach_foundation():
    foundation = FoundationStub()
    agent = _agent(foundation)
    first = agent.decide("A downtown buyer asks for a showing", context_key="buyer")
    agent.learn(first.id, 0.0, observation="The buyer did not respond")

    belief = None
    for index in range(5):
        belief = agent.learn_fact(
            "downtown office closes",
            "6 PM",
            source=f"calendar-{index}",
        )
    assert belief is not None and belief.consolidated

    agent.decide("When does the downtown office close?", context_key="buyer")
    assert foundation.last_memories
    assert foundation.last_memories[0].decision_id == first.id
    assert [fact.value for fact in foundation.last_facts] == ["6 PM"]


def test_contradictions_reduce_confidence_instead_of_overwriting():
    agent = _agent()
    for index in range(5):
        belief = agent.learn_fact(
            "office closes",
            "6 PM",
            source=f"source-a-{index}",
        )
    assert belief.consolidated
    confident = belief.confidence

    for index in range(2):
        belief = agent.learn_fact(
            "office closes",
            "7 PM",
            source=f"source-b-{index}",
        )
    assert belief.value == "6 PM"
    assert belief.alternatives == (("7 PM", 2.0),)
    assert belief.confidence < confident
    assert not belief.consolidated


def test_frozen_outcome_scores_without_changing_long_term_state():
    agent = _agent()
    candidate = [Candidate("call", "Call now", prior=0.8)]
    trained = agent.decide("A warm lead is waiting", candidates=candidate)
    agent.learn(trained.id, 1.0)
    theta = agent._models["call"].theta.copy()
    covariance = agent._models["call"].covariance.copy()
    episodes = len(agent.memory)
    evidence = agent.strategy("general", "call")

    frozen = agent.decide("A warm lead is waiting", candidates=candidate)
    report = agent.learn(frozen.id, 0.0, adapt=False)
    assert not report.adapted
    assert len(agent.memory) == episodes
    assert np.array_equal(theta, agent._models["call"].theta)
    assert np.array_equal(covariance, agent._models["call"].covariance)
    assert agent.strategy("general", "call") == evidence


def test_checkpoint_preserves_pending_predictions_and_next_learning_step():
    foundation = FoundationStub()
    agent = _agent(foundation)
    for _ in range(6):
        decision = agent.decide("An urgent lead is waiting", context_key="urgent")
        agent.learn(decision.id, float(decision.selected.action == "text"))
    for index in range(5):
        agent.learn_fact("team timezone", "America/Chicago", source=f"source-{index}")
    pending = agent.decide("An urgent lead is waiting", context_key="urgent")

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "agent.npz"
        assert agent.save(path) == path
        restored = AthenaAgent.load(path, foundation=FoundationStub())

    assert len(restored.memory) == len(agent.memory)
    assert restored.beliefs.belief("team timezone") == agent.beliefs.belief("team timezone")
    assert restored.strategy("urgent", "text") == agent.strategy("urgent", "text")
    original = agent.learn(pending.id, 1.0)
    resumed = restored.learn(pending.id, 1.0)
    assert original == resumed
    for action in agent._models:
        assert np.allclose(agent._models[action].theta, restored._models[action].theta)
        assert np.allclose(
            agent._models[action].covariance,
            restored._models[action].covariance,
        )


def test_each_prediction_can_be_resolved_only_once():
    agent = _agent()
    decision = agent.decide(
        "test",
        candidates=[Candidate("act", "Act", prior=0.5)],
    )
    agent.learn(decision.id, 1.0)
    try:
        agent.learn(decision.id, 1.0)
    except KeyError:
        return
    raise AssertionError("the same future outcome trained the agent twice")


if __name__ == "__main__":
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failed = 0
    for name, function in tests:
        try:
            function()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {exc}")
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
