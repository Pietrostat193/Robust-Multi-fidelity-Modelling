# Multi-Fidelity GP (Classic & Robust) — Toy Example + Delta Sensitivity

This repository contains a compact MATLAB codebase for **multi-fidelity Gaussian Processes (MF-GP)** with a **robust Huber-loss variant**.  
It includes two runnable examples:

- **ToyExample** – a minimal end-to-end demo on a simple synthetic dataset.
- **Delta_sensitivity** – shows how predictions and errors change as you vary the Huber threshold **δ**.

All other folders provide the shared utilities used by the examples.

---


## Repository structure

├─ ClassicMF_utilities/     # Classic MF-GP functions (likelihood, prediction, kernels, etc.)
├─ RobustMF_utilities/      # Robust MF-GP pieces (Huber loss, robust objective, weights)
├─ Simulation_utilities/    # Data simulators, distortion functions, helpers for toy setups
├─ ToyExample/              # WORKING EXAMPLE: basic MF-GP on a toy dataset
├─ Delta_sensitivity/       # WORKING EXAMPLE: sweep δ and visualize sensitivity
└─ README.md                # This file

