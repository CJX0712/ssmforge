# Changelog

All notable changes to this project are documented here. This project adheres to
[Semantic Versioning](https://semver.org).

## [0.1.0] — 2026-10-06

### Added
- **S4DFuse** flagship: multi-rate, multi-channel S4D-Real model with a dual
  (Woodbury) ridge readout. Solves the adding problem to ~1e-11 MSE at T=1000
  where an LSTM collapses to the mean-predictor.
- S4D-Real diagonal state-space kernel with HiPPO-LegS initialization
  (`Λ_n = -½(2n+1) + iπn`) and closed-form ZOH discretization.
- Convolution / recurrence **duality** (bit-for-bit, `6.7e-16`) as a CI-enforced
  invariant.
- Baselines: from-scratch LSTM (BPTT, numerically gradient-checked),
  single-head self-attention reference, ARRidge linear control, and a pure-NumPy
  `EMAMemory` offline fallback.
- Deterministic synthetic benchmarks: adding problem, long-range retrieval, and a
  self-contained sMNIST-style sequential classification (`seqclf`).
- Multi-seed benchmark runner that honestly marks intractable configurations
  (`skipped`) instead of fabricating numbers.
- Ablation harness proving the multi-rate integrator bank is necessary
  (2.6e-08 vs 0.157 single-rate).
- `pyproject.toml` (pytest `pythonpath`, pinned ruff rule set), CI matrix
  (ubuntu+windows × py3.12/3.13), Dockerfile, Makefile, and a pinned
  `requirements.lock.txt`.

### Notes
- Baselines LSTM/AttentionRef/ARRidge/EMAMemory are reference competitors, not
  claims of reproducing their full published training recipes; each is trained
  with a length-aware budget on identical data for a fair comparison.
