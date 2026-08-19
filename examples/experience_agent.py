"""End-to-end deployment learning around a deterministic foundation prior.

The stub stands in for an LLM that has broad but static knowledge: it generally
prefers email.  Deployment experience reveals a local exception -- urgent leads
answer texts -- and Athena learns it without erasing the routine-email policy.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena import AgentConfig, AthenaAgent, Candidate  # noqa: E402


class PretrainedSalesModel:
    """A fixed pretrained prior with no ability to update its own weights."""

    def propose(self, situation, *, memories, facts, strategies, n):
        return (
            Candidate("email", "Send a detailed email", prior=0.75),
            Candidate("text", "Send a concise text", prior=0.25),
        )[:n]


def run_phase(agent, name, context, situation, successful_action, steps=25):
    rewards, errors, actions = [], [], []
    for _ in range(steps):
        decision = agent.decide(situation, context_key=context)
        reward = float(decision.selected.action == successful_action)
        report = agent.learn(
            decision.id,
            reward,
            observation=("Customer replied" if reward else "No reply"),
        )
        rewards.append(reward)
        errors.append(report.squared_error)
        actions.append(decision.selected.action)
    return {
        "phase": name,
        "early_reward": float(np.mean(rewards[:5])),
        "late_reward": float(np.mean(rewards[-10:])),
        "late_error": float(np.mean(errors[-10:])),
        "last_action": actions[-1],
    }


def main():
    agent = AthenaAgent(
        PretrainedSalesModel(),
        AgentConfig(
            feature_dim=32,
            prior_strength=2.0,
            exploration_bonus=0.25,
            minimum_strategy_evidence=6.0,
        ),
    )
    phases = [
        run_phase(
            agent,
            "new urgent context",
            "urgent-lead",
            "A lead wants to tour an apartment tonight",
            "text",
        ),
        run_phase(
            agent,
            "different routine context",
            "routine-follow-up",
            "A customer requests their monthly market recap",
            "email",
        ),
        run_phase(
            agent,
            "urgent context returns",
            "urgent-lead",
            "A lead wants to tour an apartment tonight",
            "text",
        ),
    ]

    print("phase                         early reward   late reward   late error   final")
    for row in phases:
        print(
            f"{row['phase']:<29} {row['early_reward']:>11.3f}   "
            f"{row['late_reward']:>11.3f}   {row['late_error']:>10.4f}   "
            f"{row['last_action']}"
        )
    print("\nConsolidated strategies:")
    for item in agent.knowledge():
        print(
            f"  {item.context_key:<18} {item.action:<6} "
            f"{item.status:<11} mean={item.mean_reward:.3f} "
            f"evidence={item.effective_samples:.1f}"
        )


if __name__ == "__main__":
    main()
