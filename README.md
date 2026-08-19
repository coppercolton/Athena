# Athena

Athena is an experimental architecture for intelligence that keeps learning
after deployment. It combines four complementary systems:

- a numeric continual world model that predicts the next observation and
  updates online; and
- a provider-neutral agent layer that gives a stable pretrained foundation
  model episodic memory, evidence-gated facts, and behavior learned from the
  consequences of its own decisions; and
- a verified procedural skill layer that identifies knowledge gaps, learns an
  executable rule through instruction or active experimentation, tests it on
  held-out cases, and retains it without rewriting earlier skills; and
- a permissioned tool-learning agent that explores unfamiliar operations,
  predicts results before acting, verifies real state, and compiles successful
  traces into reusable workflows that bind to differently named tools.

There is no hidden batch-training step in any learning loop. Predictions are
recorded before outcomes arrive, experience is the data, and feedback changes
future expectations without rewriting the foundation model after every event.

```text
predict -> observe -> measure surprise -> infer context -> update -> predict
   ^                                                               |
   +---------------------------------------------------------------+
```

The long-term goal is an intelligence that starts with broad pretrained
knowledge and develops through its own lifetime of experience. That does **not**
mean error must improve monotonically or that every environment is predictable.
It means its learning machinery remains plastic, bounded, inspectable,
testable, and resumable while observations and consequences keep arriving.

## Try Athena — v0.6

V0.6 exposes three different kinds of post-deployment learning in one local
browser interface:

- **Tool-workflow learning:** send Athena into an opaque virtual workspace. It
  inspects manuals, uses only policy-approved tools, snapshots reversible writes,
  verifies the goal, and tests the compiled workflow in worlds with new tool
  names and task values.
- **Capability learning:** give Athena an unrevealed black-box sequence world.
  It reports what it does not know, chooses experiments, induces a procedure,
  verifies it independently, stores it, and runs it on inputs not seen during
  discovery.
- **Experience learning:** give the agent a real-world situation. It proposes an
  action before the outcome exists, then updates contextual strategy evidence
  from the measured result.

The experience path is explicit rather than hidden behind a chat box:

1. Give Athena a problem and a context.
2. Athena retrieves related experiences and established knowledge.
3. A foundation backend proposes candidate actions.
4. Athena predicts success and chooses before seeing the outcome.
5. Report whether the action worked and what happened.
6. Watch its memories, contextual strategies, and confidence change.

Launch the offline learning demo from the repository:

```bash
python3 -m athena.playground
```

Then open [http://127.0.0.1:8765](http://127.0.0.1:8765). The installed package
also provides:

```bash
athena-playground
```

The demo backend, tool workspace, and symbolic skill world are deterministic
rather than disguised language models. They let you test Athena's
post-deployment learning immediately without credentials or network access.

### Connect broad foundation intelligence

Set an API key in the server environment to let the same agent use an OpenAI
foundation model. The browser never receives or stores the key.

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="gpt-5.6"
python3 -m athena.playground --foundation openai
```

The live adapter uses the Responses API with strict Structured Outputs for
candidate actions and strict function tools for unfamiliar-tool decisions. The
permission policy still runs locally after the model selects a call. You can
select another compatible model through `OPENAI_MODEL` or `--model`. See the
official [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
and [function calling](https://developers.openai.com/api/docs/guides/function-calling)
documentation.

Experience state is checkpointed after every prediction, outcome, and factual
update at `~/.athena/playground.npz` by default. Symbolic skills are stored in
`~/.athena/playground.skills.json`, and verified tool workflows in
`~/.athena/playground.tool-skills.json`. Use `--state PATH` for another identity
or experiment. Because live requests include the current task and retrieved
learning context, do not place sensitive information in the agent unless it is
appropriate to send to the configured model provider.

### What the playground can and cannot do

It can learn permissioned workflows across differently named virtual tools,
acquire procedures in its constrained symbolic language, reason through a
connected foundation model, learn which proposed actions work in different
contexts, remember qualitative outcomes, accept source-labelled facts, revise
strategies when the world changes, and resume experience and skill state after
restart.

It deliberately does **not** execute an arbitrary shell or connect the demo to
external accounts. V0.6 executes registered tools only inside `OpaqueKVWorld`;
the `ToolEnvironment` protocol is the boundary for future real integrations.
External writes are denied by default. Moving beyond the sandbox requires a
specific adapter, user authorization, task verifier, and rollback strategy.

## What v0.6 established

V0.6 turns a successful experience into a reusable tool procedure instead of
only an episode or action score. In each deployment, record-store operations are
assigned opaque names. Athena must inspect manuals to identify semantic
capabilities, predict each call's result, and accomplish a structured goal.

The lifecycle is:

1. state the knowledge gap: tool names and semantics are unknown;
2. inspect tool manuals to learn capability bindings;
3. snapshot before every reversible write;
4. execute one permission-checked call at a time;
5. compare predicted success with the observed result;
6. let the environment independently verify final state;
7. compile raw calls into parameterized semantic steps;
8. require success in two held-out worlds before consolidation; and
9. bind the retained procedure to differently named tools after restart.

A stored workflow refers to capabilities such as `write_value` and
`read_value`, not deployment-specific tool names such as `gaia` or `iris`.
Arguments learned from the first task become placeholders such as `$goal.key`
and `$goal.value`. That is what lets the procedure transfer instead of replaying
the training calls.

### Permission and verification boundary

- Registered tools declare `read`, `reversible_write`, or `external_write`.
- The default policy permits only reads and reversible writes.
- A reasoner cannot create a callable tool by naming one.
- Every reversible write receives a pre-call snapshot.
- A failed write is rolled back before another decision.
- Model-selected `finish_task` never determines success; the environment does.
- Skills with fewer than two independent validation worlds remain provisional.

The OpenAI adapter presents each available operation as a strict function tool,
disables parallel tool calls, and requests one action at a time. The API key
remains server-side and is excluded from prompts and checkpoints.

### Procedural multi-world benchmark

The benchmark covers store, update, and delete tasks across 30 random seeds.
Every training, validation, and transfer world independently renames all four
operations. Transfer tasks also use unseen keys and values.

| measure | result |
|---|---:|
| training worlds solved | **90 / 90** |
| held-out validation worlds | **180 / 180** |
| workflows consolidated | **90 / 90** |
| renamed-world transfers | **90 / 90** |
| mean acquisition decisions | **6.44** |
| new foundation decisions during retained transfer | **0** |

Run the transparent mission and full benchmark with:

```bash
python3 examples/tool_learning_agent.py
python3 examples/tool_agent_benchmark.py
```

## What v0.5 established

V0.5 distinguishes a durable capability from an episode or a success score. A
procedural `Skill` contains an executable program, acquisition source, version,
confidence, protected verification cases, and optional component skills.

The first learning environment is deliberately narrow: transformations over
token sequences inside an inspectable 11-primitive DSL. Athena begins with 78
canonical one- and two-step hypotheses. It selects the probe with the greatest
expected information gain, commits to the observation, eliminates contradicted
hypotheses, and repeats until one program remains. The candidate cannot enter
the registry until it passes inputs withheld from discovery.

This is real program induction, but it is **not** evidence of general reasoning
or a self-training foundation model. The finite task language makes the claim
falsifiable and establishes infrastructure that broader future learners need:

- an explicit epistemic state rather than invented certainty;
- active experimentation instead of passive feedback alone;
- independent verification rather than self-reported success;
- executable transfer rather than retrieval of a similar example;
- skill composition; and
- a regression gate that rejects a replacement which breaks protected behavior.

### Exhaustive constrained benchmark

Every canonical program is hidden from a fresh learner. Discovery uses only
letter probes; transfer uses numeric tokens not present during discovery. All 78
skills are then retained in one registry and rechecked after sequential learning.

| measure | result |
|---|---:|
| programs induced and verified | **78 / 78** |
| mean active experiments | **1.37** |
| maximum active experiments | **3** |
| held-out verification cases | **468 / 468** |
| novel-token transfer cases | **78 / 78** |
| first-to-last sequential retention | **78 / 78** |

Run the full benchmark with:

```bash
python3 examples/skill_benchmark.py
```

## What v0.4 established

V0.4 turned the experience architecture into a runnable local browser agent,
added the offline and live foundation adapters, and checkpointed every decision,
outcome, and factual update.

## What v0.3 established

V0.3 added `AthenaAgent`, the bridge between broad pretrained intelligence and
continual learning after deployment.

### Stable foundation, adaptive experience

A foundation model proposes candidate actions using its pretrained knowledge.
Athena retrieves relevant past episodes and consolidated knowledge, predicts
the reward of every candidate, and fuses the learned value with the
foundation's prior. The real consequence is then revealed exactly once and
updates a recursive contextual value model.

This separation is intentional. Fine-tuning a large model on every interaction
would let temporary noise, malicious feedback, and one unusual event corrupt
general knowledge. Athena learns quickly in its external world model and only
promotes repeated, source-labelled evidence into durable knowledge.

```python
from athena import AgentConfig, AthenaAgent, Candidate

class MyFoundationModel:
    def propose(self, situation, *, memories, facts, strategies, n):
        # Replace this body with any hosted or local foundation model adapter.
        return [
            Candidate("email", "Send a detailed email", prior=0.75),
            Candidate("text", "Send a concise text", prior=0.25),
        ]

agent = AthenaAgent(MyFoundationModel(), AgentConfig())

decision = agent.decide(
    "An urgent lead wants a showing tonight",
    context_key="urgent-lead",
)

# The consequence arrives after Athena has committed its prediction.
report = agent.learn(
    decision.id,
    reward=1.0,
    observation="The lead replied and booked",
    reliability=1.0,
)

# New facts remain provisional until independent evidence supports them.
agent.learn_fact(
    "downtown office closes",
    "6 PM",
    source="verified-calendar",
)

agent.save("experience.npz")
agent = AthenaAgent.load("experience.npz", foundation=MyFoundationModel())
```

### Three learning timescales

1. **Immediate adaptation:** recursive value models update after each measured
   consequence and alter the next decision in the same context. A small
   forgetting factor preserves a plasticity floor, so sustained new evidence
   can reverse an old strategy instead of leaving it permanently frozen.
2. **Episodic memory:** bounded memory retains trustworthy and surprising
   experiences; similar situations retrieve both successes and failures.
3. **Consolidated knowledge:** strategies need a confidence bound and minimum
   effective sample count; factual claims need repeated, source-labelled
   support. Contradictions reduce confidence instead of silently overwriting
   the old belief.

`adapt=False` resolves and scores a decision without changing any long-term
state, providing the same frozen-evaluation contract as the numeric model.

### Deployment-learning demonstration

The included deterministic demo gives the foundation model a fixed general
preference for email. In deployment, urgent leads actually respond to texts,
while routine follow-ups still work best by email.

| phase | first 5 reward | final 10 reward | final 10 squared error | final action |
|---|---:|---:|---:|---|
| new urgent context | 0.600 | **1.000** | 0.0090 | text |
| different routine context | **1.000** | **1.000** | 0.0021 | email |
| urgent context returns | **1.000** | **1.000** | 0.0018 | text |

Athena changes the pretrained behavior, preserves the opposite policy in a
different context, and recalls the learned exception immediately when its old
context returns. Run the reproducible demo with:

```bash
python3 examples/experience_agent.py
```

## Numeric world-model quick start

```python
from athena import Athena, Config

model = Athena(Config(sizes=[4, 24, 12]))

for observation in stream:
    prediction = model.predict()          # cannot see observation yet
    report = model.observe(observation)   # compare, infer, and learn once
    print(report.mse, report.nll, report.surprise)

model.save("athena.npz")
model = Athena.load("athena.npz")        # exact next prediction is preserved
```

For a frozen holdout, dynamic beliefs still follow the sequence while all
long-term learning stays fixed:

```python
for observation in holdout:
    report = model.observe(observation, learn=False)
```

## What v0.2 established

The original prototype showed that a predictive-coding hierarchy could improve
online on smooth synthetic signals. V0.2 makes the claim harder to fool and
adds a second timescale for retained dynamics.

### A recursive sensory-dynamics bank

A local linear recurrence should be remembered as a law, not continuously
repainted into a general neural hierarchy. Each context therefore owns a
recursive least-squares dynamics memory. It can retain simple local laws exactly
and update them online with fractional responsibility from the context gate.

The predictive-coding hierarchy runs beside it and learns nonlinear residuals,
slower structure, and context. Their forecasts are fused by measured forecast
precision. There is no fixed mixing weight.

### Honest evaluation contracts

- **Prequential scoring:** predict first, reveal one point, then update once.
- **Frozen evaluation:** `learn=False` cannot change weights, precision,
  volatility, evidence, context-transition statistics, or dynamics parameters.
- **Strong baselines:** online RLS is included because an AR(2) model solves a
  noiseless sinusoid almost exactly.
- **Calibrated scores:** next-observation NLL uses forecast precision, and the
  hierarchy's Gaussian energy includes the log-precision normalization term.
- **Behavioral tests:** returning-regime memory must beat a single model that
  overwrites itself; merely producing a valid context distribution is not
  enough.
- **Resumable learning:** checkpoints include fast beliefs, slow weights,
  uncertainty, context memory, histories, and random-generator state.

## Numeric world-model architecture

The numeric layer combines five mechanisms:

1. **Predictive-coding hierarchy.** Each level predicts the level below and its
   own next state. Latent beliefs settle against precision-weighted local errors
   before weights change.
2. **Recursive dynamics memories.** Each context has an online RLS recurrence
   for stable, locally linear sensory laws.
3. **Bayesian context gate.** A bank of experts prevents every new regime from
   overwriting the previous one. Learning pauses during change-point probation,
   then the gate recalls an existing expert or recruits unused capacity.
4. **Forecast calibration.** Independent precision estimates decide how much to
   trust the hierarchy and recursive dynamics on each channel.
5. **Volatility and evidence.** Persistent surprise reopens learning; accumulated
   evidence gradually reduces parameter noise without driving plasticity to
   zero.

Generalized coordinates at the sensory level include recent velocity. That is a
hand-designed inductive bias, not an emergent discovery, and the benchmarks say
so explicitly.

## Measured results

### Frozen stationary prediction

Four unrelated noiseless sinusoids. Models learn on steps 0–4,999, then all
long-term adaptation is frozen for steps 5,000–5,999.

| model | frozen MSE |
|---|---:|
| persistence | 4.802e-03 |
| constant velocity | 7.248e-05 |
| Athena v0.2 | **3.270e-09** |
| online RLS(2) | 1.715e-10 |

Athena retains the learned signal and is about 22,000x better than constant
velocity in the frozen window. It does not beat RLS on an exact AR(2) world and
should not: RLS is the smaller model matched perfectly to that generator. The
hierarchy has no extra structure to contribute there.

Run it with:

```bash
python3 examples/honest_benchmark.py
```

### Returning regimes

Three unrelated regimes rotate every 500 observations. Errors below are
averaged over the last six dwells while all models continue learning online.

| model | first 100 after switch | settled remainder |
|---|---:|---:|
| Athena, six context memories | **8.570e-03** | **4.139e-05** |
| Athena, one memory | 9.259e-03 | 9.450e-05 |
| stationary RLS | 8.863e-03 | 1.833e-04 |
| adaptive RLS, forgetting=0.995 | 1.020e-02 | 7.886e-05 |

Here the context bank adds value beyond the RLS component: it can retrieve a
previous law instead of compromising all regimes into one or forgetting the old
one to learn the new one.

```bash
python3 examples/continual_benchmark.py
```

These are deterministic synthetic experiments, not evidence of general
intelligence. They establish two narrower properties: retained prediction when
learning is frozen, and reduced interference when known dynamics return.

## The learning loop

For each observation Athena:

1. rolls every hierarchy level forward and generates a top-down forecast;
2. asks each recursive context memory for its forecast;
3. fuses predictions using reliability measured on prior forecasts;
4. scores the unseen observation with MSE, calibrated surprise, and NLL;
5. infers which context most likely generated it;
6. settles latent beliefs without changing weights;
7. updates only the responsible local weights and recursive memory; and
8. updates uncertainty and volatility after the prediction has been scored.

Multi-step `predict(horizon=n)` runs both models on their own predictions without
consuming observations.

## Scientific position

Predictive processing is an influential computational theory, not a settled
claim that every part of every brain works this way. The foundational visual
model is Rao & Ballard (1999); Friston's free-energy work extends the idea into a
broader account of inference and action. Predictive-coding networks with local
Hebbian updates can also approximate backpropagation under specific assumptions.

- [Rao & Ballard, 1999](https://pubmed.ncbi.nlm.nih.gov/10195184/)
- [Friston, 2010](https://www.nature.com/articles/nrn2787)
- [Whittington & Bogacz, 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5467749/)

Athena combines mechanisms from this literature with standard online system
identification. The current implementation is engineering research, not a claim
of novel neuroscience.

## What “learn forever” requires

Continuous updates alone are not enough. A credible forever learner needs:

- learnable signal and trustworthy feedback;
- a stability/plasticity mechanism so new learning does not erase old learning;
- calibrated uncertainty so noise is not mistaken for knowledge;
- held-out and prequential tests that the learner cannot update through;
- checkpoints, rollback, and versioned state;
- bounded updates so surprise cannot destabilize the system; and
- explicit resource limits or memory will grow forever even if capability does
  not.

Athena v0.6 implements early versions of these contracts across numeric streams,
outcome-scored agent decisions, constrained executable skills, and permissioned
virtual tool workflows, with a runnable interface and optional live foundation
adapter. It does not yet learn an open-ended representation space, improve the
neural foundation's weights, connect itself to arbitrary real-world accounts,
perform broad causal reasoning, form autonomous goals, or safely self-modify.
The repository does not bundle or train a foundation model.

## Next research milestones

1. Add opt-in adapters for real developer tools, starting with a disposable
   filesystem workspace and test runner, with explicit per-tool authorization.
2. Learn conditional, branching, and recovery procedures rather than only linear
   workflows.
3. Learn representations and reusable reasoning operators from instruction,
   demonstration, correction, and trial—not only select existing primitives.
4. Evaluate improvement, transfer, forgetting, and poisoned-feedback resistance
   on procedurally novel tasks over long deployments and multiple seeds.
5. Connect the agent layer to the numeric world model so imagined sensory
   consequences can inform candidate selection.
6. Add expandable neural adapters or expert modules only behind evaluation,
   regression, checkpoint, and rollback gates.
7. Add multimodal representations and offline reflection while keeping a
   protected evaluator responsible for promotion and rollback.

## Validation and layout

```bash
python3 tests/test_athena.py          # 16 numeric world-model tests
python3 tests/test_agent.py           # 8 deployment-learning tests
python3 tests/test_skills.py          # 7 skill acquisition/retention tests
python3 tests/test_tool_learning.py   # 12 tool, transfer, policy, rollback tests
python3 tests/test_playground.py      # 8 browser/API/foundation tests
python3 examples/tool_learning_agent.py
python3 examples/tool_agent_benchmark.py
python3 examples/novel_skill_learning.py
python3 examples/skill_benchmark.py
python3 examples/experience_agent.py
python3 examples/honest_benchmark.py
python3 examples/continual_benchmark.py
python3 examples/precision.py
```

| path | purpose |
|---|---|
| `athena/core.py` | hierarchy, recursive memory fusion, loop, checkpoints |
| `athena/context.py` | Bayesian regime inference and protected recruitment |
| `athena/agent.py` | foundation-model boundary, decisions, outcome learning |
| `athena/foundation.py` | offline demo and structured live model adapter |
| `athena/memory.py` | episodic retrieval and evidence-gated factual beliefs |
| `athena/skills.py` | knowledge gaps, active induction, verification, skill registry |
| `athena/tool_learning.py` | permission policy, unfamiliar tools, workflow compilation |
| `athena/playground.py` | local server, persistent API, and launch command |
| `athena/static/` | browser interface for tasks, outcomes, and memory |
| `athena/baselines.py` | online RLS and its prequential contract |
| `athena/precision.py` | uncertainty and volatility estimation |
| `athena/world.py` | deterministic synthetic worlds and simple baselines |
| `examples/` | reproducible stationary and continual benchmarks |
| `tests/test_athena.py` | numeric behavioral claims and persistence contracts |
| `tests/test_agent.py` | post-deployment adaptation, context, facts, checkpoints |
| `tests/test_skills.py` | induction, instruction, transfer, composition, regression |
| `tests/test_tool_learning.py` | tool calls, validation, transfer, persistence, safety |
| `tests/test_playground.py` | live adapter contract and end-to-end browser API |

Requires Python 3.10+ and NumPy:

```bash
python3 -m pip install -e .
```
