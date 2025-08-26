# Robust vs Classic Multi-Fidelity GP — Simulation README

This script runs a Monte-Carlo study comparing a **classic multi-fidelity GP** against a **robustified objective** under controlled low-fidelity (LF) distortions. The experiment is fault-tolerant: failed optimizer runs are **skipped** via `try/catch`, and only successful runs are saved.

---

## Quick start

1. **Requirements**
   - MATLAB R2021a+ (recommended)
   - Optimization Toolbox (`fminunc`)
2. **Utilities on path** (update paths in the script to match your machine):
   - `Example\ClassicMF_utilities`
   - `Example\RobustMF_utilities`
3. **Run the script.**
     - `Simulations.m`

It will create:

- **`results_robust_experiment.mat`**
  - `metrics_tbl` — per-run error metrics & scenario info
  - `preds_tbl` — per-run predictions & raw outputs

---

## Overview of the experiment

For each distortion setting `(m, a)` and each Monte-Carlo seed:

1. **Simulate HF/LF data** and split into train/test: `simulate_data(seed, 0.8)`.
2. **Inject controlled distortion** into the **LF test subset** with `distort_sample(..., a, m, seed)`, then splice back so only test rows are distorted.
3. **Build `ModelInfo`** (designs, responses, kernel, combination rule, jitter).
4. **Fit Classic model** by minimizing `likelihood2Dsp` with `fminunc`.
5. **Fit Robust model** by minimizing `robustObjective2Dsp` with `fminunc`.
6. **Predict on HF test inputs** using `predict2Dsp`.
7. **Compute metrics:** MAE and RMSE for both methods.
8. **Append results** only if both fits succeed and `exitflag > 0`.

---

## Parameter grid & seeding

- Distortion level: `m_list = [2, 5, 10]`
- Distortion proportion: `a_list = [0.1, 0.3, 0.5]`
- Repeats per scenario: `n_runs_per_scenario = 100`

Deterministic seed per run:
```text
scenario_id = running index over (m,a) pairs starting at 1
seed        = 1000*scenario_id + r


