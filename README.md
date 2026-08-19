# Athena

An AI that learns the way the idea describes: it predicts what it is about to
see, looks, and corrects itself — forever, with no training phase.

```
        ┌─────────────┐   prediction   ┌──────────────┐
        │  the model  │ ─────────────► │  what is     │
        │  (beliefs)  │ ◄───────────── │  actually    │
        └─────────────┘  error signal  │  there       │
              ▲                        └──────────────┘
              └── every error, immediately, adjusts the beliefs
                  that produced it, and the weights behind them
```

Every timestep, Athena produces an expectation of the next observation *before*
that observation arrives. When it arrives, the difference between expectation
and reality is the only learning signal the system uses. There is no dataset,
no training run, no separate inference mode. It gets better for exactly as long
as you leave it running.

## Where the idea comes from

This is a real theory of how brains work, not a metaphor. It is called
**predictive processing** (or predictive coding), and the load-bearing papers
are Rao & Ballard (1999), which introduced hierarchical predictive coding in
visual cortex, and Karl Friston's free-energy work, which recast perception,
learning and action as one quantity being minimised. Andy Clark's *Surfing
Uncertainty* is the readable book-length version.

The idea has been picked up in machine learning too — world models, JEPA, and
essentially every self-supervised next-step predictor are relatives. So the
intuition is a good one, and it is not an unexplored one. What this repository
is: a small, complete, honest implementation you can read in an afternoon and
run in a terminal, with the measurements to say what it does and does not do.

## The loop

```python
from athena import Athena, Config

model = Athena(Config(sizes=[4, 24, 12]))   # 4 sensors, two latent levels

for observation in stream:
    guess  = model.predict()          # before looking
    report = model.observe(observation)  # look, compare, settle, learn
    print(report.mse, report.surprise, report.gain)
```

Four things happen per timestep:

1. **Predict.** Beliefs roll forward in time, then generate downward through
   the hierarchy to the senses. The bottom of that cascade is a prediction of
   the next observation.
2. **Compare.** The observation arrives. The difference is a prediction error.
3. **Settle.** The latent beliefs relax until they explain what actually
   happened. This is perception, and no weights change during it.
4. **Learn.** The settled errors nudge the weights. Every update is local to a
   pair of adjacent levels — no backpropagation through time, no replay buffer.

## What makes it keep improving instead of drifting

Four mechanisms, each of which was added because the model failed without it.

**Precision.** An error on a channel that is normally reliable means something;
the same error on a channel that is always noisy does not. Each unit tracks the
inverse variance of its own error history and errors are weighted by it. The
subtlety: precision is used for the *relative* weighting only, renormalised to
mean 1. Raw precision rises as the model improves, so feeding it into the
update rule multiplies the learning rate by a growing number until it
oscillates — a model that becomes confident becomes unstable.

**Volatility.** When errors run persistently larger than their own recent
history, the world has probably changed and the model should learn faster
rather than average the change away. A fast/slow surprise ratio drives a
learning-rate multiplier. `StepReport.gain` exposes it; it spikes at every
regime change.

**Evidence-scaled steps.** A model that runs forever cannot keep a fixed
learning rate: constant step size means constant gradient noise, so parameters
random-walk around the solution and predictions decay back toward mediocre.
Each observation accumulates evidence and shrinks the step; each surprise
discounts that evidence and re-opens learning. The floor matters as much as the
decay — a model that has stopped learning cannot notice it should start again.

**Generalized coordinates.** The sensory level holds each reading *and its rate
of change*. Position alone does not determine the next position; you need
velocity, and a one-step local learning rule gives the latents almost no
pressure to invent one. This is the standard move in the free-energy literature
and it is the single change that took the model from "worse than trivial" to
"much better than trivial".

## Results

Four channels of unrelated sinusoids, one observation at a time, measured as
mean squared error on the next observation. Two baselines, because a predictive
model on a smooth signal looks impressive against nothing at all:

* **persistence** — predict the last value. Strong on smooth data.
* **linear** — constant-velocity extrapolation. This is the honest bar: the
  model is *handed* velocity as an input, so this prediction is available to it
  for free. Beating persistence proves nothing. Beating this means it has
  learned something about the signal.

| steps | persistence | linear | Athena |
|------:|------------:|-------:|-------:|
| 0–2.5k | 4.8e-03 | 8.8e-05 | 4.4e-03 |
| 2.5k–5k | 4.8e-03 | 8.8e-05 | 1.2e-03 |
| 5k–7.5k | 4.8e-03 | 8.8e-05 | 5.4e-05 |
| 10k–12.5k | 4.8e-03 | 8.8e-05 | 1.2e-05 |
| 17.5k–20k | 4.8e-03 | 8.8e-05 | **6.5e-06** |

It starts no better than doing nothing, crosses persistence, crosses linear
extrapolation, and is still improving at 20,000 observations — 740x better than
persistence and 13x better than linear, with no sign of a floor. That last
column is the claim the whole idea rests on.

```
python3 examples/learning_curve.py    # the table above, with plots
python3 examples/regime_shift.py      # what happens when the world changes
python3 examples/precision.py         # learning which channels to believe
python3 tests/test_athena.py          # the behavioural tests
```

## When the world changes

A single set of weights can only hold one story. When the dynamics switch, it
does the only thing it can — slowly overwrite what it knew — and if the old
world ever comes back, it has to learn it again from nothing. Learning fast
makes this *worse*, not better: a quick learner thrashes.

So the model holds a *bank* of transition operators with a Bayesian gate
deciding which is currently active, recruiting a fresh one when nothing known
explains the input. Three regimes rotating every 1000 steps, error averaged over
the last four dwells, by position within a dwell:

| steps after a switch | `experts=1` | `experts=6` (default) |
|---------------------:|------------:|----------------------:|
| 0–100 | 2.2e-02 | 1.0e-02 |
| 200–300 | 3.8e-03 | **4.8e-06** |
| 400–500 | 1.2e-03 | 5.0e-06 |
| 800–900 | 6.3e-05 | 5.0e-06 |

The banked model is back under 1e-05 within 200 observations of every change
and stays there, because it is *recognising* a regime rather than relearning
it. The single-operator model spends most of each dwell climbing back, and
never gets as far down before the world moves again.

The bank is not free, and the cost is worth stating plainly: for the first
several regime cycles it is slightly *worse* than the single model, because it
is still working out how many worlds there are and each expert is learning from
only its share of the data. It pulls ahead after about six dwells and the gap
widens from there. Averaged over a whole run: 4.0e-03 for `experts=1` against
1.1e-03 for `experts=6`. On a world that never changes, the bank costs nothing
measurable and recruits nobody.

Two things worth recording, because they cost the most to find:

* Handing the model a **perfect oracle** telling it which regime is active did
  not help. If a mechanism does not beat its own oracle, the bottleneck is
  somewhere else, and no amount of tuning the mechanism will find it. That
  measurement is what redirected attention to the sensory representation, which
  is where the actual problem was.
* At the instant a regime changes, *every* expert looks wrong — including the
  one that holds the incoming regime, because the continuous state beneath it
  still carries the outgoing regime's phase. "A world I have never seen" and "a
  world I know, caught mid-turn" are the same picture until you wait. So the
  gate freezes learning during a probation window. Without that freeze the
  incumbent expert relearns the new regime in place, and the memory of the old
  one is destroyed by the very adaptation that makes the model look good in the
  moment.

## Getting smarter, versus keeping a filing cabinet

The goal for Athena is that it should *get better at learning* as it learns --
meet a novel problem, work out what it doesn't know, learn the skill, keep it,
and have the next problem be easier because of it. That last clause is the
whole claim, and it is the one that per-skill accuracy reports cannot see: a
system that genuinely compounds and a system that files each skill in a
separate drawer produce identical scorecards.

They differ on exactly one number. Measured on the v9 branch's own registries,
learning two skills first changes the third skill's held-out accuracy by:

```
horizontal_order  +0.000000
vertical_order    +0.000000
far_apart         +0.000000
```

Not approximately zero. Bit-for-bit identical, in both the representation path
and the neural-plasticity path. That is not a tuning problem, it is the
architecture working as designed: every skill gets a freshly seeded network,
and the shared representation is explicitly frozen while operators train
(`"""Learn a reusable operator while keeping the representation frozen."""`).
Forgetting measures 0.000000 too -- for the same reason. Isolation buys
perfect retention by making transfer impossible.

Stability and plasticity are not two features to build separately. They are the
two ends of one tradeoff, and freezing sits at one extreme.

`athena/transfer.py` takes the middle. Earlier experts stay bit-for-bit frozen,
so retention remains exactly perfect and every promotion and rollback guarantee
still holds -- but a *new* skill may read the internal features of the skills
already learned. Old skills cannot be damaged because their weights are never
written; new skills get cheaper because they start from what earlier skills
worked out. This is progressive networks (Rusu et al., 2016), which was
designed for this exact pair of requirements.

On a curriculum where each task reuses the previous task's feature, 12 seeds,
mean held-out accuracy:

| training examples | task | isolated | lateral | gain |
|---|---|---:|---:|---:|
| 48 | `x0*x1 + x2 > 0` | 0.845 | 0.850 | +0.006 |
| 48 | `x0*x1 + x2*x3 > 0` | 0.585 | 0.667 | **+0.082** |
| 96 | `x0*x1 + x2 > 0` | 0.861 | 0.883 | +0.023 |
| 96 | `x0*x1 + x2*x3 > 0` | 0.615 | 0.756 | **+0.141** |

Forgetting across every seed and every configuration: **+0.000000**.

Two honest costs. Transfer to a genuinely *unrelated* task is slightly negative
(about −0.01), because the new skill has more inputs to overfit; the gain is
not free, it is paid for by relatedness. And each expert's input grows with the
number of skills it may read, so lateral sources are capped -- which bounds the
cost but also bounds how far knowledge can compound. Neither is solved here.

```
python3 examples/transfer_benchmark.py
python3 tests/test_transfer.py
```

## What this is not

* **Not AGI, and not a language model.** It predicts low-dimensional continuous
  streams. Scaling this shape of model to rich sensory data is exactly the open
  research problem, not something this repository has solved.
* **Not novel research.** The mechanisms are from the literature cited above.
  What is here is a working, measured, readable implementation.
* **Not tested beyond synthetic signals.** Every number above comes from
  sinusoid mixtures. Real data is noisier, higher-dimensional, and less kind.
* **Not fast.** Pure NumPy, one timestep at a time, ~5 ms per step at these
  sizes. It is written to be read.

## Layout

| file | what it holds |
|------|---------------|
| `athena/core.py` | the hierarchy, the loop, inference and learning |
| `athena/precision.py` | inverse-variance weighting and volatility |
| `athena/context.py` | the discrete gate over regimes |
| `athena/transfer.py` | skills that make later skills cheaper |
| `athena/world.py` | signal generators and the baselines |
| `athena/plot.py` | terminal charts, so demos need nothing but NumPy |

Requires Python 3.10+ and NumPy. `pip install -r requirements.txt`.
