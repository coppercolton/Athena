# Athena

Athena is an experimental continual world model: it predicts the next thing it
will observe, receives reality, measures the error, and updates itself online.
There is no separate batch-training run. The observation stream *is* the data,
and every reported error is made before the corresponding observation arrives.

```text
predict -> observe -> measure surprise -> infer context -> update -> predict
   ^                                                               |
   +---------------------------------------------------------------+
```

The long-term goal is a model that can keep learning without periodically being
discarded and retrained. That does **not** mean error must improve monotonically
or that every environment is predictable. It means the machinery remains
plastic, calibrated, testable, and resumable for as long as observations keep
arriving.

## Quick start

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

## What is new in v0.2

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

## Architecture

Athena combines five mechanisms:

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

Athena v0.2 implements early versions of these contracts. It does not yet solve
representation learning, causal reasoning, goal formation, action selection,
or safe self-modification.

## Next research milestones

1. Evaluate on real, non-stationary sensor and forecasting streams with multiple
   seeds, noise levels, and ablations.
2. Make predictions action-conditioned: predict what follows from each possible
   action, act, then learn from the consequence.
3. Learn compact representations for images or audio instead of consuming only
   low-dimensional vectors.
4. Add episodic consolidation for rare surprises and finite-capacity eviction
   for obsolete contexts.
5. Separate a protected evaluator from the learner so recursive changes cannot
   redefine their own success criterion.

## Validation and layout

```bash
python3 tests/test_athena.py          # 16 behavioral tests
python3 examples/honest_benchmark.py
python3 examples/continual_benchmark.py
python3 examples/precision.py
```

| path | purpose |
|---|---|
| `athena/core.py` | hierarchy, recursive memory fusion, loop, checkpoints |
| `athena/context.py` | Bayesian regime inference and protected recruitment |
| `athena/baselines.py` | online RLS and its prequential contract |
| `athena/precision.py` | uncertainty and volatility estimation |
| `athena/world.py` | deterministic synthetic worlds and simple baselines |
| `examples/` | reproducible stationary and continual benchmarks |
| `tests/test_athena.py` | behavioral claims and persistence contracts |

Requires Python 3.10+ and NumPy:

```bash
python3 -m pip install -e .
```
