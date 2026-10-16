# ECG5000 Experiment Results — Full 5-Method Comparison

> **Dataset**: UCR ECG5000 — Univariate heartbeat classification
> **Seeds**: [42] (1 runs)
> **Test samples**: 25 per seed

## Dataset Description

| Property | Value |
|----------|-------|
| Source | UCR Time Series Archive |
| Domain | ECG / Cardiology |
| Channels (d) | 1 |
| Length (T) | 140 |
| Classes (C) | 5 |
| Train samples | 500 |
| Classifier Accuracy | 94.42% ± 0.00% |

**Class semantics**: 5 heartbeat morphology types from the MIT-BIH
Arrhythmia Database — Normal (1), R-on-T PVC (2), PVC (3), SP (4), UB (5).
Binary-reducible to Normal vs Abnormal for clinical interpretability.

**Comparison to TwoLeadECG**: Both are ECG datasets from the UCR archive.
ECG5000 is larger (500 vs 23 train), longer (T=140 vs T=82), and
multi-class (C=5 vs C=2), providing a more challenging evaluation.

---

## Methods Compared

| Method | Type | Key Innovation |
|--------|------|---------------|
| **Ours** (Soft-DTW) | Gradient-based | Soft-DTW alignment for temporal plausibility |
| **SPARCE** | Prototype+Saliency | Two-phase curriculum with saliency masking |
| **GDFO** (VAE+GMM) | Generative density | VAE+GMM density-guided optimisation |
| **M-CELS** | Greedy baseline | Saliency-guided NUN substitution |
| **DTW-CFE** (CMA-ES) | Evolutionary | DTW-guided constrained deformation |

---

## Per-Seed Results

### Seed 42

- **Classifier Accuracy**: 94.42%
- **Test samples**: 25
- **Elapsed**: 3252.4s

| Method | Val ↑ | L1 ↓ | L2 ↓ | DTW ↓ | IsoForest ↑ | Time(s) | s/sample |
|--------|-------|------|------|-------|-------------|---------|----------|
| Ours | 0.280 | 0.3258 | 0.1730 | 15.7128 | 1.000 | 124.3 | 4.97 |
| SPARCE | 0.280 | 0.3917 | 0.3082 | 12.3997 | 1.000 | 2856.6 | 114.26 |
| GDFO | 0.040 | 0.3627 | 0.2008 | 31.6271 | 0.400 | 217.8 | 8.71 |
| M-CELS | 0.480 | 0.3708 | 0.3081 | 21.3504 | 0.840 | 0.7 | 0.03 |


---

## Aggregated Results (mean ± std, 1 seeds)

| Method | Val ↑ | L1 ↓ | L2 ↓ | DTW ↓ | IsoForest ↑ | Avg Time |
|--------|-------|------|------|-------|-------------|----------|
| **Ours** | 0.2800±0.0000 | 0.3258±0.0000 | 0.1730±0.0000 | 15.7128±0.0000 | 1.0000±0.0000 | 124.3±0.0 |
| **SPARCE** | 0.2800±0.0000 | 0.3917±0.0000 | 0.3082±0.0000 | 12.3997±0.0000 | 1.0000±0.0000 | 2856.6±0.0 |
| **GDFO** | 0.0400±0.0000 | 0.3627±0.0000 | 0.2008±0.0000 | 31.6271±0.0000 | 0.4000±0.0000 | 217.8±0.0 |
| **M-CELS** | 0.4800±0.0000 | 0.3708±0.0000 | 0.3081±0.0000 | 21.3504±0.0000 | 0.8400±0.0000 | 0.7±0.0 |

---

## Best Method per Metric (aggregated mean)

| Metric | Best Method | Value |
|--------|-------------|-------|
| Val ↑ | **M-CELS** | 0.4800 |
| DTW ↓ | **SPARCE** | 12.3997 |
| IsoForest ↑ | **Ours** | 1.0000 |
| L1 ↓ | **Ours** | 0.3258 |
| L2 ↓ | **Ours** | 0.1730 |

## Average Rank across 1 seeds (lower is better)

| Method | Val ↑ | DTW ↓ | IsoForest ↑ | L1 ↓ | L2 ↓ | **Avg** |
|--------|-------|-------|-------------|------|------|---------| 
| **Ours** | 2.00 | 2.00 | 1.00 | 1.00 | 1.00 | **1.40** |
| **SPARCE** | 3.00 | 1.00 | 2.00 | 4.00 | 4.00 | **2.80** |
| **GDFO** | 4.00 | 4.00 | 4.00 | 2.00 | 2.00 | **3.20** |
| **M-CELS** | 1.00 | 3.00 | 3.00 | 3.00 | 3.00 | **2.60** |
| **DTW-CFE** | - | - | - | - | - | **-** |

---

## Cross-Dataset Comparison: ECG5000 vs TwoLeadECG

| Property | TwoLeadECG | ECG5000 |
|----------|------------|---------|
| Archive | UCR | UCR |
| Domain | ECG | ECG |
| Channels (d) | 1 | 1 |
| Length (T) | 82 | 140 |
| Classes (C) | 2 | 5 |
| Train samples | 23 | 500 |
| Test samples | 1139 | 4500 |

ECG5000 provides a substantially more challenging evaluation due to its
multi-class nature (5 heartbeat types) and larger sample count, while
remaining in the same clinical ECG domain as TwoLeadECG.

---
*Generated automatically by `soft_dtw_cfe/run_ecg5000_experiments.py`*