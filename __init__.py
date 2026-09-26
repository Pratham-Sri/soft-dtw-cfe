"""
Soft-DTW Counterfactual Explanations for Time Series Classification
=====================================================================

Full implementation of:
  "Towards plausibility in time series counterfactual explanations"
  Kostrzewa, Galus, Zięba (2026) — arXiv:2603.08349v1

This package implements:
  - 1D CNN classifier with Optuna hyperparameter optimisation
  - Soft-DTW based counterfactual generation via gradient descent
  - Reference methods: Glacier (uniform), M-CELS
  - Evaluation metrics: Validity, L1, L2, DTW-kNN plausibility, Isolation Forest
  - Qualitative visualisation of counterfactuals

Quick start:
    from soft_dtw_cfe.run_experiments import run_dataset
    results = run_dataset("GunPoint", n_test=5, skip_optuna=True)

Or from the command line:
    python -m soft_dtw_cfe.run_experiments --skip_optuna --n_test 10
"""

__version__ = "1.0.0"
__author__ = "Implementation based on Kostrzewa, Galus, Zięba (2026)"
__paper__ = "arXiv:2603.08349"
