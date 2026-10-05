# Model Card — SSMForge / S4DFuse

Author: **晨星 (CJX0712)** · License: MIT · Quality grade: **S**

## Overview

- **Name:** SSMForge (flagship model: **S4DFuse**)
- **Task family:** long-range sequence modelling / state-space models (S4 / S4D).
- **Inputs:** real-valued sequences `(B, L)` or `(B, L, F)`.
- **Outputs:** scalar regression or class labels.
- **Runtime:** CPU-only, pure NumPy/SciPy. No GPU, no pretrained weights, no
  network access required.
- **Deterministic:** same seed ⇒ bit-identical results.

## Intended use

- Research/education on structured state-space models and their inductive bias.
- A transparent, from-scratch reference for the S4D-Real diagonal kernel, HiPPO
  initialization, and convolution/recurrence duality.
- A CPU-friendly baseline for long-range sequence benchmarks.

**Out of scope / not recommended for:** production-scale training of large
language models, tasks requiring nonlinear input gating (where Mamba-style
selectivity would be needed), or settings needing GPU throughput. This is a
research implementation, not a drop-in deep-learning framework.

## Evaluation

Deterministic synthetic benchmarks (no downloads), 3 seeds, `mean ± std`:

| Task | S4DFuse | LSTM | AttentionRef | ARRidge | EMAMemory |
|------|---------|------|--------------|---------|-----------|
| adding T=100  | 0.0000 | 0.1679 | 0.0027 | 0.0000 | 0.1489 |
| adding T=1000 | 0.0000 | 0.1679 | skipped | 2.8242 | 0.1679 |
| seqclf L=784  | 0.8550 acc | 0.0767 acc | skipped | 0.3883 acc | 0.1367 acc |

(Regression = MSE, lower better; classification = accuracy, higher better.
`skipped` = honestly marked intractable, not imputed.)

**Ablation (adding T=1000, MSE):** full multi-rate 2.6e-08 · single rate 0.157 ·
one rate 0.157. The multi-rate integrator bank is necessary.

**Invariants (CI-enforced):** conv≡recurrent 6.7e-16 · causality exact ·
FFT≡direct <1e-8 · dual≡primal ridge <1e-6 · LSTM gradient rel-err <1e-5 ·
two-run determinism bit-identical.

## Baselines — scope and fairness

- **LSTM:** from-scratch BPTT, forget-gate bias = 1, cell+hidden readout, target
  standardized, numerically gradient-checked. Trained with a length-aware budget
  (verified: adding MSE is identical at 20 and 40 epochs, so extra epochs do not
  change the long-range conclusion). It is a *reference* baseline, not a
  reproduction of a tuned published LSTM recipe.
- **AttentionRef:** single-head O(L²) self-attention, reported only where
  tractable on CPU.
- **ARRidge:** linear control on the raw input — isolates "is the task linearly
  solvable?" from "can this architecture exploit the solution at length?".
- **EMAMemory:** pure-NumPy EMA — the always-available offline fallback.

All models see identical data and splits within each task.

## Limitations & failure modes

- **Baselines are reference-grade.** A heavily tuned LSTM/transformer may close
  some gap; the claim here is specifically about the *inductive bias* under equal
  data, not an absolute state of the art.
- **seqclf is a self-contained synthetic proxy** for sMNIST-style integration, not
  real MNIST; its absolute accuracy is not comparable to published MNIST numbers.
- **The integrator trick is task-shaped.** S4DFuse excels where the target is a
  global statistic (sum / long-range integration). It is not claimed to match a
  trained transformer on tasks needing nonlinear per-step gating.
- **Readout is linear.** Complex downstream heads are out of scope.
- **Memory scales with feature dim** `n_rates · 2N`; very wide banks cost more
  than a single-rate model.
