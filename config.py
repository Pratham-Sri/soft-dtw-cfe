"""
Configuration constants and hyperparameters from the paper.

Section 5.1 — Experiment Setup / Hyperparameters Tuning:
  λ = 1,  k = 10,  γ = 1  (defaults selected after grid search)
  Classifier: 80 epochs max, early stopping patience = 10
  Optuna: optimise dropout, learning rate, weight decay
"""

import torch
import os

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Paths ────────────────────────────────────────────────────────────────────
# Repo root is this file's directory (flat layout, not nested package).
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "datasets")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "saved")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "visualization", "figures")

for d in [DATA_DIR, RESULTS_DIR, MODELS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Dataset catalogue (Table 1 in paper) ─────────────────────────────────────
# Univariate datasets from UCR archive
UNIVARIATE_DATASETS = [
    "CBF",           # Note: may need re-download if corrupted (aeon cache issue)
    "TwoLeadECG",
    "GunPoint",
    "Earthquakes",
    "Coffee",
    "ItalyPowerDemand",
]

# Multivariate datasets from UEA archive
MULTIVARIATE_DATASETS = [
    "Cricket",       # Note: may need re-download if corrupted (aeon cache issue)
    "Epilepsy",
]

ALL_DATASETS = UNIVARIATE_DATASETS + MULTIVARIATE_DATASETS

# Datasets confirmed available without download issues
# (run_experiments.py handles per-dataset errors gracefully)
CONFIRMED_DATASETS = [
    "TwoLeadECG",
    "GunPoint",
    "Earthquakes",
    "Coffee",
    "ItalyPowerDemand",
    "Epilepsy",
]

# ── Classifier hyperparameters (Section 5.1) ─────────────────────────────────
CLASSIFIER_CONFIG = {
    "conv_channels": [32, 64, 128],     # 3 conv blocks
    "kernel_size": 3,
    "max_epochs": 80,
    "early_stopping_patience": 10,
    "optuna_n_trials": 30,              # number of Optuna trials
    "batch_size": 32,
}

# ── Counterfactual generation hyperparameters (Section 4 & 5.1) ──────────────
CFE_CONFIG = {
    "lambda_": 1.0,        # λ: balances validity+plausibility vs proximity+sparsity
    "k": 10,               # k: number of nearest target-class neighbours for DTW
    "gamma": 1.0,          # γ: soft-DTW smoothing parameter
    "tau": 0.5,            # τ: hinge-loss threshold for validity
    "lr": 0.01,            # learning rate for Adam
    "num_iterations": 100, # gradient descent iterations
}

# ── DTW-guided constrained deformation hyperparameters (novelty) ──────────
DTWCFE_CONFIG = {
    "M_a": 4,             # amplitude RBF count
    "M_t": 4,             # temporal RBF count
    "K": 3,               # prototypes per query
    "lambda_": 1.0,       # validity weight vs proximity
    "lambda_r": 1e-3,     # ||theta||^2 regularizer
    "m0": 0.0,            # desired logit margin (0 = just cross boundary)
    "sigma0": 0.5,        # CMA-ES initial step
    "maxfevals": None,    # None -> 100*D in optimizer
    "popsize": None,      # None -> cma default 4+3log(D)
    "seed": 0,
    "window": None,       # None = full DTW; auto-banded for T>500 in generator
    "width_factor": 1.0,  # RBF overlap
    "eps": 1e-3,          # velocity floor
}

# ── Evaluation hyperparameters (Section 5.1) ──────────────────────────────────
EVAL_CONFIG = {
    "dtw_k": 10,                     # k for DTW-kNN plausibility metric
    "isolation_forest_contamination": 0.1,  # IF contamination parameter
}

# ── Seed for reproducibility ─────────────────────────────────────────────────
SEED = 42
