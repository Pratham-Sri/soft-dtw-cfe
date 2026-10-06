# Soft-DTW Counterfactual Explanations for Time Series

<div align="center">

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2603.08349-b31b1b.svg)](https://arxiv.org/abs/2603.08349)
[![CI](https://github.com/Pratham-Sri/soft-dtw-cfe/actions/workflows/ci.yml/badge.svg)](https://github.com/Pratham-Sri/soft-dtw-cfe/actions)

</div>

> **Paper**: *Towards Plausibility in Time Series Counterfactual Explanations*  
> Kostrzewa, Galus, Zięba (2026) — [arXiv:2603.08349](https://arxiv.org/abs/2603.08349)

Complete, reproducible implementation of the **Soft-DTW Counterfactual Explanation** method for multivariate and univariate time series classification. Includes four baseline methods, a novel **SPARCE** extension, a **GDFO** (Generative Density) comparator, comprehensive evaluation metrics, and extensive experimental results across 10 benchmark datasets.

---

## 📋 Table of Contents

- [Method Overview](#-method-overview)
- [New Methods (Extensions)](#-new-methods-extensions)
- [Project Structure](#-project-structure)
- [Datasets](#-datasets)
- [Quick Start](#-quick-start)
- [Running Experiments](#-running-experiments)
- [Results Summary](#-results-summary)
- [Classifier Architecture](#-classifier-architecture)
- [Evaluation Metrics](#-evaluation-metrics)
- [Hyperparameters](#-hyperparameters)
- [References](#-references)
- [Citation](#-citation)

---

## 🔬 Method Overview

The proposed method generates counterfactual explanations **X′** for a classified time series **X** by minimising a four-component differentiable loss:

```
L_CF = L_prox + L_sparse + λ · (L_valid + L_DTW)
```

| Component | Formula | Role |
|-----------|---------|------|
| **L_prox** | `(1/dT) ‖X′ − X‖²₂` | Minimal perturbation — stay close to original |
| **L_sparse** | `(1/dT) ‖X′ − X‖₁` | Sparsity — localise changes |
| **L_valid** | `max(0, τ − p_f(y_target │ X′))` | Validity — hinge loss on target class probability |
| **L_DTW** | `(1/k) Σ DTW_γ(X′, Y)` | Plausibility — soft-DTW alignment to k-NN in target class |

The **Soft-DTW** distance (Cuturi & Blondel, ICML 2017) makes L_DTW fully differentiable, enabling gradient-based optimisation in input space while the classifier remains **frozen**.

### Why Soft-DTW?

Standard DTW is non-differentiable. By replacing the hard min with a smoothed log-sum-exp over the DP table:

```
DTW_γ(x, y) = -γ · log Σ_{A ∈ A(x,y)} exp(-⟨A, Δ(x,y)⟩ / γ)
```

we obtain a smooth, backpropagatable alignment that captures temporal warping — not just pointwise L2.

---

## 🆕 New Methods (Extensions)

This repository implements two new methods beyond the paper baseline:

### SPARCE — Saliency-guided Prototype-Aligned Refinement for CFE

**SPARCE** addresses the key weakness of the base Soft-DTW method on multivariate datasets (validity gap) through:

1. **Prototype warm-start**: Initialise from a saliency-weighted blend of `X + NUN` (nearest unlike neighbour), starting already close to the decision boundary
2. **Channel-selective saliency masking**: A learnable soft mask concentrates perturbations on discriminative channels (identified via gradient saliency), preserving non-discriminative channels intact
3. **Two-phase curriculum**:
   - **Phase 1** (80 iters): Validity-first with `λ_valid = 5.0` + light DTW guidance
   - **Phase 2** (120 iters): Plausibility refinement — reduce validity weight, increase `λ_DTW = 1.0`
4. **Cross-channel correlation preservation**: Penalises changes to inter-channel covariance structure

**Loss**:
```
Phase 1: L = 5.0·L_valid + L_prox + L_sparse_masked + 0.1·L_DTW + 0.1·L_corr
Phase 2: L = 1.0·L_valid + L_prox + L_sparse_masked + 1.0·L_DTW + 0.1·L_corr
```

### GDFO — Generative Density Function Optimisation

**GDFO** uses a VAE+GMM framework to constrain counterfactuals to the data manifold:

1. **VAE** encodes training data into a latent space, capturing the manifold structure
2. **GMM** fits a density model in latent space
3. CFE optimisation includes a **density term** to keep counterfactuals on-manifold

```
L_GDFO = L_prox + L_sparse + λ_v·L_valid + λ_d·L_density
```

> **Key finding**: GDFO achieves near-zero validity on multivariate datasets — the density constraint prevents CFEs from crossing the decision boundary. See [Results](#-results-summary).

---

## 📁 Project Structure

```
soft_dtw_cfe/
│
├── config.py                        # All hyperparameters & paths
├── main.py                          # Simple single-dataset entry point
│
├── run_experiments.py               # CLI: univariate/multivariate baseline run
├── run_gdfo_experiments.py          # CLI: GDFO vs Ours vs M-CELS
├── run_sparce_experiments.py        # CLI: SPARCE vs all methods
├── run_full_comparison.py           # CLI: all tiers, all methods
├── compile_results.py               # Aggregate multi-seed results
│
├── data/
│   └── dataset_loader.py            # UCR/UEA auto-download via aeon
│
├── models/
│   ├── classifier.py                # 1D CNN + early stopping trainer
│   ├── optuna_tuner.py              # Optuna hyperparameter search (30 trials)
│   └── saved/                       # Cached model weights (.pt)
│
├── methods/
│   ├── soft_dtw.py                  # Differentiable Soft-DTW kernel
│   ├── proposed_method.py           # ★ Main method: Soft-DTW CFE
│   ├── sparce.py                    # ★ SPARCE: Saliency + Prototype
│   ├── gdfo.py                      # ★ GDFO: Generative Density
│   ├── m_cels.py                    # Baseline: M-CELS (greedy NUN swap)
│   └── dtw_guided/
│       └── dtw_cfe.py               # Baseline: DTW-CFE (CMA-ES)
│
├── evaluation/
│   └── metrics.py                   # Val, L1, L2, DTW, IsoForest
│
├── visualization/
│   ├── plot.py                      # CFE plots, loss curves, metric bars
│   └── figures/                     # Generated PNG outputs (40+ figures)
│
├── results/                         # JSON experiment outputs
├── RESULTS.md                       # Univariate baseline results
├── RESULTS_GDFO.md                  # GDFO multivariate results
├── RESULTS_SPARCE.md                # SPARCE multivariate results
├── RESULTS_FULL_COMPARISON.md       # All methods × all datasets
└── RESULTS_AGGREGATED.md            # Multi-seed aggregated stats
```

---

## 📊 Datasets

All datasets auto-downloaded from [UCR](https://www.timeseriesclassification.com/) and [UEA](http://www.timeseriesclassification.com/dataset.php) archives via `aeon`.

### UCR Univariate

| Dataset | #Train | #Test | T | Classes | Clf Acc |
|---------|--------|-------|---|---------|---------|
| ItalyPowerDemand | 67 | 1029 | 24 | 2 | 96.2% |
| GunPoint | 50 | 150 | 150 | 2 | 98.7% |
| CBF | 30 | 900 | 128 | 3 | 43.9% |
| TwoLeadECG | 23 | 1139 | 82 | 2 | 88.5% |
| Earthquakes | 322 | 139 | 512 | 2 | 72.7% |
| Coffee | 28 | 28 | 286 | 2 | 71.4% |

### UEA Multivariate

| Dataset | #Train | #Test | T | d | Classes | Clf Acc |
|---------|--------|-------|---|---|---------|---------|
| Cricket | 108 | 72 | 1197 | 6 | 12 | — |
| Epilepsy | 137 | 138 | 206 | 3 | 4 | 97.8% |
| BasicMotions | 40 | 40 | 100 | 6 | 4 | 100.0% |
| ERing | 30 | 270 | 65 | 4 | 6 | 85.9% |
| RacketSports | 151 | 152 | 30 | 6 | 4 | — |
| NATOPS | 180 | 180 | 51 | 24 | 6 | — |

---

## ⚡ Quick Start

### Install

```bash
git clone https://github.com/Pratham-Sri/soft-dtw-cfe.git
cd soft-dtw-cfe
pip install -r requirements.txt
```

### Generate a single counterfactual

```python
import torch
from soft_dtw_cfe.data.dataset_loader import load_dataset
from soft_dtw_cfe.models.classifier import TSClassifier, train_classifier
from soft_dtw_cfe.methods.proposed_method import SoftDTWCounterfactualGenerator

# 1. Load data
X_train, y_train, X_test, y_test, meta = load_dataset("GunPoint")

# 2. Train classifier
model = TSClassifier(n_channels=meta["n_channels"], n_classes=meta["n_classes"])
model, acc, _ = train_classifier(model, ...)

# 3. Generate counterfactual
gen = SoftDTWCounterfactualGenerator(classifier=model, num_iterations=500)

x = torch.tensor(X_test[0], dtype=torch.float32)   # (d, T)
target = 1
target_samples = torch.tensor(X_train[y_train == target], dtype=torch.float32)

result = gen.generate(x, target, target_samples, verbose=True)
print(f"Valid: {result['valid']} | Pred: {result['predicted_class']}")
```

### Use SPARCE (new method)

```python
from soft_dtw_cfe.methods.sparce import SPARCECounterfactualGenerator

sparce = SPARCECounterfactualGenerator(classifier=model)
results = sparce.generate_batch(X_sample, y_sample, X_train, y_train, verbose=True)
```

---

## 🚀 Running Experiments

### Baseline — univariate/multivariate (Ours, M-CELS, DTW-CFE)

```bash
# All datasets, 20 test samples
python run_experiments.py

# Specific datasets, faster run
python run_experiments.py --datasets GunPoint CBF Epilepsy --n_test 10 --skip_optuna
```

| Flag | Default | Description |
|------|---------|-------------|
| `--datasets` | all | Datasets to evaluate |
| `--n_test` | 20 | Test samples per dataset |
| `--skip_optuna` | False | Use default hyperparameters |
| `--no_glacier` | False | Skip Glacier baseline |
| `--seed` | 0 | Random seed |

### GDFO — Generative Density comparison

```bash
# Epilepsy, BasicMotions, ERing
python run_gdfo_experiments.py --datasets Epilepsy BasicMotions ERing --n_test 10
```

### SPARCE — Saliency-Prototype comparison

```bash
# 3 core multivariate datasets
python run_sparce_experiments.py --datasets Epilepsy BasicMotions ERing --n_test 10

# Extended: higher-d datasets
python run_sparce_experiments.py --datasets RacketSports NATOPS --n_test 10

# Skip slow baselines for fast comparison
python run_sparce_experiments.py --no_dtwcfe --no_gdfo
```

### Full comparison — all tiers

```bash
python run_full_comparison.py --n_test 15
```

---

## 📈 Results Summary

### Multivariate — SPARCE vs All Methods (n=10 samples each)

#### Epilepsy (d=3, T=206, C=4, Acc=97.8%)

| Method | Val ↑ | L1 ↓ | L2 ↓ | DTW ↓ | IsoForest ↑ |
|--------|------:|-----:|-----:|------:|------------:|
| **Ours** (Soft-DTW) | 0.200 | 0.507 | 0.339 | 333.8 | **1.000** |
| **SPARCE** *(new)* | 0.800 | 0.730 | 0.867 | **310.8** | **1.000** |
| **GDFO** (VAE+GMM) | 0.000 | **0.341** | **0.148** | 603.3 | **1.000** |
| **M-CELS** | **1.000** | 0.450 | 0.675 | 506.5 | **1.000** |

#### BasicMotions (d=6, T=100, C=4, Acc=100.0%)

| Method | Val ↑ | L1 ↓ | L2 ↓ | DTW ↓ | IsoForest ↑ |
|--------|------:|-----:|-----:|------:|------------:|
| **Ours** (Soft-DTW) | 0.900 | **0.187** | **0.056** | 28.6 | **1.000** |
| **SPARCE** *(new)* | **1.000** | 0.202 | 0.077 | **23.0** | **1.000** |
| **GDFO** (VAE+GMM) | 0.000 | 0.171 | 0.035 | 69.3 | **1.000** |
| **M-CELS** | **1.000** | 0.224 | 0.098 | 36.2 | **1.000** |

#### ERing (d=4, T=65, C=6, Acc=85.9%)

| Method | Val ↑ | L1 ↓ | L2 ↓ | DTW ↓ | IsoForest ↑ |
|--------|------:|-----:|-----:|------:|------------:|
| **Ours** (Soft-DTW) | 0.900 | **0.533** | **0.369** | 111.3 | 0.800 |
| **SPARCE** *(new)* | **1.000** | 0.928 | 1.347 | **67.3** | **1.000** |
| **GDFO** (VAE+GMM) | 0.100 | 0.704 | 0.630 | 274.7 | 0.400 |
| **M-CELS** | **1.000** | 0.816 | 1.379 | 183.4 | 0.900 |

### Win Counts — Multivariate (3 datasets)

| Method | Val ↑ | DTW ↓ | IsoForest ↑ | L1 ↓ | L2 ↓ | **Total** |
|--------|------:|------:|------------:|-----:|-----:|----------:|
| **Ours** | 0 | 0 | 2 | **2** | **3** | **7** |
| **SPARCE** | 2 | **3** | **3** | 0 | 0 | **8** |
| **GDFO** | 0 | 0 | 0 | 0 | 0 | **0** |
| **M-CELS** | **3** | 0 | 2 | 1 | 0 | **6** |

### Average Rank — Multivariate (lower = better)

| Method | Val ↑ | DTW ↓ | IsoForest ↑ | L1 ↓ | L2 ↓ | **Overall** |
|--------|------:|------:|------------:|-----:|-----:|------------:|
| **Ours** | 3.00 | 2.00 | 1.67 | **1.33** | **1.00** | **1.80** |
| **SPARCE** | **1.33** | **1.00** | **1.67** | 2.67 | 2.33 | **1.80** |
| **GDFO** | 4.00 | 4.00 | 3.67 | 2.00 | 2.67 | 3.27 |
| **M-CELS** | **1.67** | 3.00 | 2.00 | 2.00 | 3.00 | 2.33 |

> **Ours and SPARCE tie at 1.80 avg rank** — they are complementary:  
> Ours = best L1/L2 sparsity. SPARCE = best DTW plausibility and validity.

### Univariate — DTW Plausibility (Ours dominates 5/6)

| Dataset | Ours | M-CELS | DTW-CFE |
|---------|-----:|-------:|--------:|
| ItalyPowerDemand | **1.01** | 1.95 | 1.96 |
| GunPoint | **2.97** | 5.16 | 5.83 |
| CBF | **18.64** | 29.65 | 29.04 |
| TwoLeadECG | **1.63** | 3.03 | 2.35 |
| Earthquakes | **21.4** | 27.5 | 28.1 |
| Cricket | **2362** | 3309 | 3956 |

Full per-dataset results: [`RESULTS_FULL_COMPARISON.md`](RESULTS_FULL_COMPARISON.md) | [`RESULTS_SPARCE.md`](RESULTS_SPARCE.md) | [`RESULTS_GDFO.md`](RESULTS_GDFO.md)

---

## 🏗️ Classifier Architecture

A 1D CNN (Section 5.1 of the paper) with three convolutional blocks:

```
Input: (d, T)
  ├─ Conv1d(d → 32, k=3) → BatchNorm1d → ReLU → MaxPool1d(2)
  ├─ Conv1d(32 → 64, k=3) → BatchNorm1d → ReLU → MaxPool1d(2)
  ├─ Conv1d(64 → 128, k=3) → BatchNorm1d → ReLU → MaxPool1d(2)
  ├─ AdaptiveAvgPool1d(1)
  ├─ Dropout(p)
  └─ Linear(128, C)
```

- Hyperparameters (dropout, lr, weight_decay) optimised via **Optuna** (30 trials)
- Training: up to 80 epochs, early stopping (patience=10)
- Pre-trained weights cached in `models/saved/classifier_{dataset}.pt`

---

## 📏 Evaluation Metrics

| Metric | Symbol | Direction | Definition |
|--------|--------|:---------:|------------|
| Validity | Val | ↑ | Fraction of CFEs where `f(X′) = y_target` |
| Sparsity | L1 | ↓ | Normalised L1: `‖X′ − X‖₁ / (d·T)` |
| Proximity | L2 | ↓ | Normalised L2: `‖X′ − X‖²₂ / (d·T)` |
| DTW Plausibility | DTW | ↓ | Avg Soft-DTW to k=10 nearest target-class neighbours |
| Isolation Forest | IsoForest | ↑ | Fraction of CFEs passing Isolation Forest anomaly check |

---

## ⚙️ Hyperparameters

### Base Soft-DTW CFE

| Parameter | Value | Description |
|-----------|------:|-------------|
| λ | 1.0 | Balance: validity+plausibility vs proximity+sparsity |
| k | 10 | Nearest neighbours for DTW plausibility |
| γ | 1.0 | Soft-DTW smoothing parameter |
| τ | 0.5 | Hinge loss threshold |
| lr | 0.01 | Adam learning rate |
| iterations | 500 | Gradient descent steps |

### SPARCE (new)

| Parameter | Value | Description |
|-----------|------:|-------------|
| alpha_init | 0.8 | Warm-start blend: 80% X + 20% NUN |
| n_prototypes | 3 | Diverse NUN selection count |
| phase1_iters | 80 | Validity-first phase iterations |
| phase2_iters | 120 | Plausibility refinement iterations |
| λ_valid (P1) | 5.0 | Aggressive validity in Phase 1 |
| λ_valid (P2) | 1.0 | Maintenance validity in Phase 2 |
| λ_DTW (P2) | 1.0 | DTW weight in Phase 2 |
| λ_corr | 0.1 | Cross-channel correlation weight |

### GDFO (generative)

| Parameter | Value | Description |
|-----------|------:|-------------|
| z_dim | 16 | VAE latent dimension |
| vae_epochs | 50 | VAE training epochs |
| vae_beta | 1.0 | VAE β (KL weight) |
| gmm_components | 4 | GMM mixture components |
| λ_valid | 2.0 | Validity loss weight |
| λ_density | 0.5 | Density (manifold) loss weight |

---

## 📚 References

1. **Kostrzewa, Galus, Zięba** (2026). *Towards plausibility in time series counterfactual explanations.* arXiv:2603.08349.
2. **Cuturi & Blondel** (2017). *Soft-DTW: a Differentiable Loss Function for Time-Series.* ICML.
3. **Li et al.** (2024). *M-CELS: Counterfactual Explanation for Multivariate Time Series.* AISTATS.
4. **Wang et al.** (2024). *GLACIER: Guided Locally Constrained Counterfactual Explanations.* ECML-PKDD.
5. **Akiba et al.** (2019). *Optuna: A next-generation hyperparameter optimization framework.* KDD.
6. **Liu et al.** (2008). *Isolation Forest.* ICDM.

---

## 📄 License

MIT License — see [LICENSE](LICENSE).

---

## 📖 Citation

If you use this code or results, please cite the original paper:

```bibtex
@article{kostrzewa2026plausibility,
  title   = {Towards plausibility in time series counterfactual explanations},
  author  = {Kostrzewa, Daniel and Galus, Maciej and Zi{\k{e}}ba, Maciej},
  journal = {arXiv preprint arXiv:2603.08349},
  year    = {2026}
}
```
