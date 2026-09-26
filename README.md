# Soft-DTW Counterfactual Explanations for Time Series

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Full implementation of:

> **Towards plausibility in time series counterfactual explanations**  
> Kostrzewa, Galus, Zięba (2026) — [arXiv:2603.08349v1](https://arxiv.org/abs/2603.08349)

This repository contains a complete, reproducible implementation of the Soft-DTW counterfactual explanation method for time series classification, along with two baseline methods (Glacier, M-CELS), evaluation metrics, and visualisation utilities.

---

## Method Overview

The proposed method generates counterfactual explanations X' for a classified time series X by minimising a four-component loss:

```
L_CF = L_prox + L_sparse + λ · (L_valid + L_DTW)
```

| Component | Equation | Role |
|-----------|----------|------|
| L_prox | (1/dT) ‖X'−X‖²₂ | Proximity — minimal perturbation |
| L_sparse | (1/dT) ‖X'−X‖₁ | Sparsity — localised changes |
| L_valid | max(0, τ − p_f(y_target\|X')) | Validity — hinge loss on target class |
| L_DTW | (1/k) Σ DTW_γ(X', Y) | Plausibility — soft-DTW alignment to k-NN |

The soft-DTW distance (Cuturi & Blondel, ICML 2017) makes L_DTW differentiable, enabling gradient-based optimisation directly in input space while the classifier remains frozen.

---

## Project Structure

```
soft_dtw_cfe/
├── __init__.py                  # Package root
├── config.py                    # All hyperparameters & paths
├── run_experiments.py           # Full experiment runner (CLI)
├── main.py                      # Alternative entry point
│
├── data/
│   └── dataset_loader.py        # UCR/UEA auto-download via aeon
│
├── models/
│   ├── classifier.py            # 1D CNN + early stopping
│   ├── optuna_tuner.py          # Optuna hyperparameter search
│   └── saved/                   # Saved model weights (.pt)
│
├── methods/
│   ├── soft_dtw.py              # Differentiable Soft-DTW
│   ├── proposed_method.py       # Main method (Soft-DTW CFE)
│   ├── glacier.py               # Baseline: Glacier (uniform)
│   └── m_cels.py                # Baseline: M-CELS
│
├── evaluation/
│   └── metrics.py               # Val, L1, L2, DTW, IsoForest
│
└── visualization/
    ├── plot.py                  # Counterfactual & diagnostic plots
    └── figures/                 # Saved PNG figures
```

---

## Datasets

All datasets are automatically downloaded from the [UCR](https://www.timeseriesclassification.com/) and [UEA](http://www.timeseriesclassification.com/dataset.php) archives using the `aeon` library.

| Dataset | Type | #Train | #Test | T | d | Classes |
|---------|------|--------|-------|---|---|---------|
| CBF | Univariate | 30 | 900 | 128 | 1 | 3 |
| TwoLeadECG | Univariate | 23 | 1139 | 82 | 1 | 2 |
| GunPoint | Univariate | 50 | 150 | 150 | 1 | 2 |
| Earthquakes | Univariate | 322 | 139 | 512 | 1 | 2 |
| Coffee | Univariate | 28 | 28 | 286 | 1 | 2 |
| ItalyPowerDemand | Univariate | 67 | 1029 | 24 | 1 | 2 |
| Cricket | Multivariate | 108 | 72 | 1197 | 6 | 12 |
| Epilepsy | Multivariate | 137 | 138 | 206 | 3 | 4 |

---

## Classifier Architecture

A 1D CNN (Section 5.1) with three convolutional blocks:

```
Input (d, T)
  → Conv1d(d, 32, k=3) → BatchNorm → ReLU → MaxPool(2)
  → Conv1d(32, 64, k=3) → BatchNorm → ReLU → MaxPool(2)
  → Conv1d(64, 128, k=3) → BatchNorm → ReLU → MaxPool(2)
  → AdaptiveAvgPool1d(1)
  → Dropout(p)
  → Linear(128, C)
```

Hyperparameters (dropout, lr, weight_decay) are optimised via **Optuna** (30 trials). Training runs for up to 80 epochs with early stopping (patience=10).

---

## Evaluation Metrics

| Metric | Symbol | Direction | Definition |
|--------|--------|-----------|------------|
| Validity | Val | ↑ | Fraction of CFEs where f(X') = y_target |
| Sparsity | L1 | ↓ | Normalised L1 distance ‖X'−X‖₁ / dT |
| Proximity | L2 | ↓ | Normalised L2 distance ‖X'−X‖²₂ / dT |
| DTW Plausibility | DTW | ↓ | Avg DTW distance to 10 nearest target-class neighbours |
| Isolation Forest | IsoForest | ↑ | Fraction of CFEs classified as nominal (non-outlier) |

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/soft-dtw-cfe.git
cd soft-dtw-cfe
pip install -r requirements.txt
```

**Requirements:**
- Python ≥ 3.9
- PyTorch ≥ 2.0
- numpy, matplotlib, scikit-learn, optuna, aeon

---

## Usage

### Run all experiments (full pipeline)

```bash
# From the repo root directory
python -m soft_dtw_cfe.run_experiments
```

This will:
1. Download datasets automatically
2. Train (or load cached) classifiers with Optuna tuning
3. Generate CFEs for 20 test samples per dataset
4. Evaluate all metrics
5. Save results to `soft_dtw_cfe/results/all_results.json`
6. Generate `RESULTS.md` report
7. Save visualisation figures to `soft_dtw_cfe/visualization/figures/`

### Options

```bash
python -m soft_dtw_cfe.run_experiments \
    --datasets GunPoint CBF TwoLeadECG \
    --n_test 10 \
    --skip_optuna \
    --no_glacier
```

| Flag | Default | Description |
|------|---------|-------------|
| `--datasets` | all 8 | Datasets to run |
| `--n_test` | 20 | Test samples per dataset |
| `--skip_optuna` | False | Use default hyperparameters |
| `--no_glacier` | False | Skip Glacier baseline |

### Single dataset (programmatic)

```python
from soft_dtw_cfe.run_experiments import run_dataset

results = run_dataset("GunPoint", n_test=5, skip_optuna=True)
print(results["methods"]["Ours"])
```

### Generate a single counterfactual

```python
import torch
from soft_dtw_cfe.methods.proposed_method import SoftDTWCounterfactualGenerator
from soft_dtw_cfe.data.dataset_loader import load_dataset

X_train, y_train, X_test, y_test, meta = load_dataset("GunPoint")

# Assume `model` is a trained TSClassifier
generator = SoftDTWCounterfactualGenerator(classifier=model, num_iterations=500)

x = torch.tensor(X_test[0], dtype=torch.float32)  # (d, T)
target_class = 1
target_samples = torch.tensor(X_train[y_train == target_class], dtype=torch.float32)

result = generator.generate(x, target_class, target_samples, verbose=True)
print(f"Valid: {result['valid']}  |  Predicted: {result['predicted_class']}")
```

---

## Results

See [`RESULTS.md`](RESULTS.md) for the full experimental results after running.

Key visualisation outputs saved in `soft_dtw_cfe/visualization/figures/`:
- `{dataset}_{method}_cf.png` — counterfactual vs original plot
- `{dataset}_{method}_loss.png` — optimisation loss curves (Ours method)
- `{dataset}_metrics_comparison.png` — bar chart comparing all metrics

---

## Hyperparameters

Default values from the paper (Section 5.1):

| Parameter | Value | Description |
|-----------|-------|-------------|
| λ | 1.0 | Balance between validity/plausibility vs proximity/sparsity |
| k | 10 | Nearest neighbours for DTW plausibility |
| γ | 1.0 | Soft-DTW smoothing |
| τ | 0.5 | Hinge loss threshold |
| lr | 0.01 | Adam learning rate for CFE optimisation |
| iterations | 500 | Gradient descent steps |

---

## References

1. Kostrzewa, Galus, Zięba (2026). *Towards plausibility in time series counterfactual explanations.* arXiv:2603.08349.
2. Cuturi, M., & Blondel, M. (2017). *Soft-DTW: a Differentiable Loss Function for Time-Series.* ICML.
3. Wang, Z., et al. (2024). *GLACIER: Guided Locally Constrained Counterfactual Explanations for Time Series.* ECML-PKDD.
4. Li, P., et al. (2024). *M-CELS: Counterfactual Explanation for Multivariate Time Series.* AISTATS.
5. Dempster, A., et al. (2020). *ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels.* Data Mining and Knowledge Discovery.
6. Akiba, T., et al. (2019). *Optuna: A next-generation hyperparameter optimization framework.* KDD.

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{kostrzewa2026plausibility,
  title={Towards plausibility in time series counterfactual explanations},
  author={Kostrzewa, Daniel and Galus, Maciej and Zi{\k{e}}ba, Maciej},
  journal={arXiv preprint arXiv:2603.08349},
  year={2026}
}
```
