# SSMForge — Architecture

Author: **晨星 (CJX0712)** · see [model_card.md](model_card.md) for metrics/limitations.

## 1. Design goals

1. **Isolate the SSM inductive bias.** Show *why* a structured state-space model
   solves long-range summation that a length-agnostic linear model and a
   transformer cannot, at O(L) cost.
2. **Determinism & zero setup.** No pretrained weights, no network, no GPU. Every
   number in the report is produced by a real run and is bit-reproducible.
3. **No silent failure.** A model that cannot run on a task is reported
   `skipped` with a reason, never as a fabricated score.

## 2. The model (S4DFuse)

### 2.1 Continuous SSM and ZOH discretization

A diagonal state-space model evolves

```
x'(t) = A x(t) + B u(t),   y(t) = C x(t)
```

with `A = Λ` diagonal, `B = 1` (real), and `C = 1` (absorbed into the downstream
linear readout, since a readout over the state is itself linear). HiPPO-LegS sets

```
Λ_n = -½(2n+1) + iπn,   n = 0 … N-1
```

Zero-order-hold gives the discrete system `x_t = Ā x_{t-1} + B̄ u_t` with

```
Ā = exp(Δ Λ),   B̄ = (Ā - 1) Λ^{-1}
```

and an exactly-compounding causal convolution kernel

```
k_t = Σ_n Ā_n^t B̄_n.
```

Because `A` is diagonal this is element-wise and exact — no eigen-decomposition,
no approximation. The real input sees only `Re(k)`; the real part of the complex
convolution is a valid real LTI system (the standard S4D-Real realization).

### 2.2 Per-channel, multi-rate features (the key design)

A first implementation that **summed** the `N` channels into one scalar kernel
per head failed: summing buries the slow integrator channel under fast, noisy
channels. The fix is to keep each channel's state separate. For a batch of inputs
and one rate `Δ`, the per-channel recurrence

```
x_n(t) = Ā_n x_n(t-1) + B̄_n u_t
```

is evaluated for all `n` at once (vectorized over batch and channel, looped over
`t`), and the real + imaginary parts of the pooled state form that rate's feature
block. The bank spans `n_rates` log-spaced `Δ` values, and the blocks are
**concatenated** (never summed) across rates. Feature dimension is
`n_rates · 2N`.

### 2.3 Why the integrator solves the adding problem

The adding problem's target is the **global sum** of a sparse spike train. The
slowest channel (`n = 0`, real part `−½`) with a very small `Δ` is a near-perfect
integrator: its state at the last step is `≈ B̄₀ · Σ_t u_t` (position-independent,
since `Ā₀ ≈ 1`). A linear readout on the last-step features therefore recovers the
sum directly. This is *why* the multi-rate bank (which contains that slow rate)
is the whole story — the ablation confirms removing it collapses performance to
the mean-predictor.

### 2.4 Readout

`RidgeHead` fits a closed-form ridge regression on the (standardized) features:

- features standardized using **train statistics only** (no leakage);
- a **bias column** is appended so the model can learn an intercept (without it a
  zero-mean design can only predict a zero-mean target);
- solved in whichever of the primal `(P,P)` or dual `(N,N)` form is cheaper, so
  it is stable whether features outnumber samples or not;
- handles regression, binary, and multiclass (one-vs-rest) via a **robust**
  target-type detector (see §4).

## 3. Module contracts (one-way, acyclic)

```
cli → pipeline → {data, ssm, training, eval} → core
```

- `core` — types, `ErrorCode` taxonomy, `Config` (`ENV_SSMFORGE_*` overrides +
  validation), `Protocol` interfaces, and the single `set_all(seed)` RNG entry.
- `data` — deterministic synthetic DGPs; no downloads.
- `ssm` — `kernel` (S4D math + convolution/recurrence duality), `s4dfuse`
  (flagship), `lstm`, `transformer`, `baselines` (ARRidge, EMAMemory).
- `training` — `RidgeHead` (dual ridge readout).
- `eval` — metrics (MSE↓ / accuracy↑) and the multi-seed benchmark runner.
- `pipeline` — `SSMPipeline.run / .benchmark / .ablation`.

## 4. Two bugs worth remembering (both enforced by tests now)

1. **Task-type misdetection.** Deciding "is this multiclass?" by
   `unique(y).size > 2` misreads a *continuous* regression target (whose values
   are all distinct) as an N-way classifier. Fixed by requiring the target to be
   integer-valued with a small, minority class set (`is_multiclass_target`).
2. **Missing intercept.** Standardizing features to zero mean but never fitting a
   bias leaves predictions offset by `mean(y)`. Fixed by an explicit bias column.

A third, subtler one: the conv/recurrence feature **must be concatenated across
rates**, not summed — summing averages the integrator channel away (§2.2).

## 5. Determinism

A single `set_all(seed)` pins Python, NumPy, and (best-effort) Torch RNGs. The
readout is closed-form, the SSM layer is fixed, and the benchmark uses fixed
`base_seed + i` seeds. Two demo runs produce bit-identical core metrics in
`benchmark.json` (verified in CI).
