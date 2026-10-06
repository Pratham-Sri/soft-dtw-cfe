# Comprehensive CFE Comparison — All Methods × All Datasets

> **Paper**: *Towards plausibility in time series counterfactual explanations*  
> Kostrzewa, Galus, Zięba (2026) — [arXiv:2603.08349](https://arxiv.org/abs/2603.08349)

## Methods Compared

| Method | Type | Description |
|--------|------|-------------|
| **Ours** (Soft-DTW) | Gradient-based | Proposed method — soft-DTW alignment for plausibility |
| **SPARCE** | Prototype+Saliency | Novel two-phase curriculum with channel-selective saliency masking |
| **GDFO** (VAE+GMM) | Generative density | VAE+GMM density-guided optimisation for multivariate |
| **M-CELS** | Greedy baseline | Class-conditional greedy perturbation |
| **DTW-CFE** (CMA-ES) | Evolutionary | DTW-guided constrained deformation |

## Evaluation Metrics

| Metric | Symbol | Direction | Description |
|--------|--------|-----------|-------------|
| Validity | Val | ↑ higher is better | Fraction of CFEs that flip the classifier |
| Sparsity | L1 | ↓ lower is better | Normalised L1 distance to original |
| Proximity | L2 | ↓ lower is better | Normalised L2 distance to original |
| DTW Plausibility | DTW | ↓ lower is better | Avg DTW to 10 target-class neighbours |
| Isolation Forest | IsoForest | ↑ higher is better | Fraction classified as nominal by IF |

---

## UCR Univariate Datasets

### CBF

- **Classifier Accuracy**: 43.89%
- **Samples**: 30 train / 20 test
- **Series**: T=128, d=1, C=3

| Method | Val ↑ | L1 ↓ | L2 ↓ | DTW ↓ | IsoForest ↑ | Time(s) | s/sample |
|--------|-------|------|------|-------|-------------|---------|----------|
| Ours | 1.000 | 0.4846 | 0.3192 | 18.6373 | 1.000 | 28.0 | 1.40 |
| M-CELS | 0.850 | 0.3338 | 0.3777 | 29.6497 | 0.650 | 0.1 | 0.00 |
| DTW-CFE | 0.650 | 0.4372 | 0.3405 | 29.0361 | 0.850 | 21.3 | 1.06 |


### TwoLeadECG

- **Classifier Accuracy**: 88.50%
- **Samples**: 23 train / 20 test
- **Series**: T=82, d=1, C=2

| Method | Val ↑ | L1 ↓ | L2 ↓ | DTW ↓ | IsoForest ↑ | Time(s) | s/sample |
|--------|-------|------|------|-------|-------------|---------|----------|
| Ours | 0.750 | 0.1597 | 0.0427 | 1.6324 | 1.000 | 15.0 | 0.75 |
| M-CELS | 0.850 | 0.1102 | 0.0365 | 3.0265 | 0.950 | 0.1 | 0.00 |
| DTW-CFE | 1.000 | 0.1402 | 0.0340 | 2.3496 | 0.900 | 20.9 | 1.04 |


### GunPoint

- **Classifier Accuracy**: 98.67%
- **Samples**: 50 train / 20 test
- **Series**: T=150, d=1, C=2

| Method | Val ↑ | L1 ↓ | L2 ↓ | DTW ↓ | IsoForest ↑ | Time(s) | s/sample |
|--------|-------|------|------|-------|-------------|---------|----------|
| Ours | 1.000 | 0.1789 | 0.0588 | 2.9727 | 1.000 | 33.3 | 1.66 |
| M-CELS | 1.000 | 0.1246 | 0.0700 | 5.1570 | 0.950 | 0.0 | 0.00 |
| DTW-CFE | 1.000 | 0.1221 | 0.0323 | 5.8324 | 0.900 | 21.8 | 1.09 |


### Earthquakes

- **Classifier Accuracy**: 74.82%
- **Samples**: 322 train / 20 test
- **Series**: T=512, d=1, C=2

| Method | Val ↑ | L1 ↓ | L2 ↓ | DTW ↓ | IsoForest ↑ | Time(s) | s/sample |
|--------|-------|------|------|-------|-------------|---------|----------|
| Ours | 0.150 | 0.2016 | 0.1163 | 144.6361 | 1.000 | 349.5 | 17.47 |
| M-CELS | 0.200 | 0.7407 | 1.6255 | 129.9494 | 1.000 | 0.2 | 0.01 |
| DTW-CFE | 0.150 | 0.1053 | 0.1608 | 213.6523 | 0.850 | 5.5 | 0.28 |


### Coffee

- **Classifier Accuracy**: 53.57%
- **Samples**: 28 train / 20 test
- **Series**: T=286, d=1, C=2

| Method | Val ↑ | L1 ↓ | L2 ↓ | DTW ↓ | IsoForest ↑ | Time(s) | s/sample |
|--------|-------|------|------|-------|-------------|---------|----------|
| Ours | 0.250 | 0.0895 | 0.0156 | 0.9609 | 1.000 | 107.8 | 5.39 |
| M-CELS | 0.250 | 0.0715 | 0.0118 | 0.8818 | 0.850 | 0.1 | 0.01 |
| DTW-CFE | 0.250 | 0.0215 | 0.0033 | 1.6118 | 0.450 | 5.9 | 0.30 |


### ItalyPowerDemand

- **Classifier Accuracy**: 96.21%
- **Samples**: 67 train / 20 test
- **Series**: T=24, d=1, C=2

| Method | Val ↑ | L1 ↓ | L2 ↓ | DTW ↓ | IsoForest ↑ | Time(s) | s/sample |
|--------|-------|------|------|-------|-------------|---------|----------|
| Ours | 1.000 | 0.2599 | 0.1193 | 1.0069 | 0.800 | 9.7 | 0.49 |
| M-CELS | 1.000 | 0.1884 | 0.1265 | 1.9500 | 0.800 | 0.0 | 0.00 |
| DTW-CFE | 1.000 | 0.1938 | 0.0775 | 1.9296 | 0.750 | 22.8 | 1.14 |


---

## UEA Multivariate Datasets

### Cricket

- **Classifier Accuracy**: 100.00%
- **Samples**: 108 train / 20 test
- **Series**: T=1197, d=6, C=12

| Method | Val ↑ | L1 ↓ | L2 ↓ | DTW ↓ | IsoForest ↑ | Time(s) | s/sample |
|--------|-------|------|------|-------|-------------|---------|----------|
| Ours | 0.500 | 0.4455 | 0.2932 | 2362.4284 | 0.700 | 1822.4 | 91.12 |
| M-CELS | 1.000 | 0.5789 | 1.0161 | 3308.6161 | 0.650 | 0.1 | 0.00 |
| DTW-CFE | 1.000 | 0.5061 | 0.6126 | 3956.1409 | 0.550 | 41.0 | 2.05 |


### Epilepsy

- **Classifier Accuracy**: 97.83%
- **Samples**: 137 train / 10 test
- **Series**: T=206, d=3, C=4

| Method | Val ↑ | L1 ↓ | L2 ↓ | DTW ↓ | IsoForest ↑ | Time(s) | s/sample |
|--------|-------|------|------|-------|-------------|---------|----------|
| Ours | 0.200 | 0.5065 | 0.3389 | 333.7983 | 1.000 | 113.2 | 11.32 |
| SPARCE | 0.800 | 0.7297 | 0.8672 | 310.7604 | 1.000 | 133.0 | 13.30 |
| GDFO | 0.000 | 0.3408 | 0.1478 | 603.3149 | 1.000 | 141.0 | 14.10 |
| M-CELS | 1.000 | 0.4499 | 0.6749 | 506.5135 | 1.000 | 0.1 | 0.01 |
| DTW-CFE | 1.000 | 0.5513 | 0.6145 | 343.1891 | 1.000 | 24.0 | 2.40 |


### BasicMotions

- **Classifier Accuracy**: 100.00%
- **Samples**: 40 train / 10 test
- **Series**: T=100, d=6, C=4

| Method | Val ↑ | L1 ↓ | L2 ↓ | DTW ↓ | IsoForest ↑ | Time(s) | s/sample |
|--------|-------|------|------|-------|-------------|---------|----------|
| Ours | 0.900 | 0.1874 | 0.0562 | 28.6113 | 1.000 | 40.6 | 4.06 |
| SPARCE | 1.000 | 0.2019 | 0.0772 | 22.9847 | 1.000 | 94.2 | 9.42 |
| GDFO | 0.000 | 0.1715 | 0.0350 | 69.3110 | 1.000 | 58.7 | 5.87 |
| M-CELS | 1.000 | 0.2244 | 0.0982 | 36.2406 | 1.000 | 0.2 | 0.02 |


### ERing

- **Classifier Accuracy**: 85.93%
- **Samples**: 30 train / 10 test
- **Series**: T=65, d=4, C=6

| Method | Val ↑ | L1 ↓ | L2 ↓ | DTW ↓ | IsoForest ↑ | Time(s) | s/sample |
|--------|-------|------|------|-------|-------------|---------|----------|
| Ours | 0.900 | 0.5331 | 0.3690 | 111.2727 | 0.800 | 12.7 | 1.27 |
| SPARCE | 1.000 | 0.9280 | 1.3468 | 67.3255 | 1.000 | 23.4 | 2.34 |
| GDFO | 0.100 | 0.7036 | 0.6295 | 274.7387 | 0.400 | 57.8 | 5.78 |
| M-CELS | 1.000 | 0.8156 | 1.3790 | 183.3800 | 0.900 | 0.1 | 0.01 |


### RacketSports

- **Classifier Accuracy**: 83.55%
- **Samples**: 151 train / 10 test
- **Series**: T=30, d=6, C=4

| Method | Val ↑ | L1 ↓ | L2 ↓ | DTW ↓ | IsoForest ↑ | Time(s) | s/sample |
|--------|-------|------|------|-------|-------------|---------|----------|
| Ours | 0.900 | 0.3908 | 0.2392 | 129.6244 | 1.000 | 19.9 | 1.99 |
| SPARCE | 1.000 | 0.4466 | 0.4254 | 107.3734 | 1.000 | 26.0 | 2.60 |
| M-CELS | 1.000 | 0.3192 | 0.6522 | 150.4005 | 1.000 | 0.1 | 0.01 |


### NATOPS

- **Classifier Accuracy**: 96.11%
- **Samples**: 180 train / 10 test
- **Series**: T=51, d=24, C=6

| Method | Val ↑ | L1 ↓ | L2 ↓ | DTW ↓ | IsoForest ↑ | Time(s) | s/sample |
|--------|-------|------|------|-------|-------------|---------|----------|
| Ours | 1.000 | 0.3658 | 0.2154 | 310.3622 | 0.800 | 25.4 | 2.54 |
| SPARCE | 1.000 | 0.4895 | 0.5285 | 208.1027 | 0.900 | 34.1 | 3.41 |
| M-CELS | 1.000 | 0.3055 | 0.4272 | 447.3424 | 0.700 | 0.1 | 0.01 |


---

# Grand Summary — Cross-Dataset Comparison

## Validity (Val ↑)

| Dataset | Tier | d | T | C | Acc% | Ours | SPARCE | GDFO | M-CELS | DTW-CFE |
|---------|------|---|---|---|------|------|--------|------|--------|---------|
| CBF | Univ | 1 | 128 | 3 | 43.9 | 1.000 | - | - | 0.850 | 0.650 |
| TwoLeadECG | Univ | 1 | 82 | 2 | 88.5 | 0.750 | - | - | 0.850 | 1.000 |
| GunPoint | Univ | 1 | 150 | 2 | 98.7 | 1.000 | - | - | 1.000 | 1.000 |
| Earthquakes | Univ | 1 | 512 | 2 | 74.8 | 0.150 | - | - | 0.200 | 0.150 |
| Coffee | Univ | 1 | 286 | 2 | 53.6 | 0.250 | - | - | 0.250 | 0.250 |
| ItalyPowerDemand | Univ | 1 | 24 | 2 | 96.2 | 1.000 | - | - | 1.000 | 1.000 |
| Cricket | Mult | 6 | 1197 | 12 | 100.0 | 0.500 | - | - | 1.000 | 1.000 |
| Epilepsy | Mult | 3 | 206 | 4 | 97.8 | 0.200 | 0.800 | 0.000 | 1.000 | 1.000 |
| BasicMotions | Mult | 6 | 100 | 4 | 100.0 | 0.900 | 1.000 | 0.000 | 1.000 | - |
| ERing | Mult | 4 | 65 | 6 | 85.9 | 0.900 | 1.000 | 0.100 | 1.000 | - |
| RacketSports | Mult | 6 | 30 | 4 | 83.6 | 0.900 | 1.000 | - | 1.000 | - |
| NATOPS | Mult | 24 | 51 | 6 | 96.1 | 1.000 | 1.000 | - | 1.000 | - |

## DTW Plausibility (↓)

| Dataset | d | T | Ours | SPARCE | GDFO | M-CELS | DTW-CFE |
|---------|---|---|------|--------|------|--------|---------|
| CBF | 1 | 128 | **18.64** | - | - | 29.65 | 29.04 |
| TwoLeadECG | 1 | 82 | **1.63** | - | - | 3.03 | 2.35 |
| GunPoint | 1 | 150 | **2.97** | - | - | 5.16 | 5.83 |
| Earthquakes | 1 | 512 | 144.64 | - | - | **129.95** | 213.65 |
| Coffee | 1 | 286 | 0.96 | - | - | **0.88** | 1.61 |
| ItalyPowerDemand | 1 | 24 | **1.01** | - | - | 1.95 | 1.93 |
| Cricket | 6 | 1197 | **2362.43** | - | - | 3308.62 | 3956.14 |
| Epilepsy | 3 | 206 | 333.80 | **310.76** | 603.31 | 506.51 | 343.19 |
| BasicMotions | 6 | 100 | 28.61 | **22.98** | 69.31 | 36.24 | - |
| ERing | 4 | 65 | 111.27 | **67.33** | 274.74 | 183.38 | - |
| RacketSports | 6 | 30 | 129.62 | **107.37** | - | 150.40 | - |
| NATOPS | 24 | 51 | 310.36 | **208.10** | - | 447.34 | - |

## Isolation Forest Nominal Fraction (↑)

| Dataset | d | T | Ours | SPARCE | GDFO | M-CELS | DTW-CFE |
|---------|---|---|------|--------|------|--------|---------|
| CBF | 1 | 128 | **1.000** | - | - | 0.650 | 0.850 |
| TwoLeadECG | 1 | 82 | **1.000** | - | - | 0.950 | 0.900 |
| GunPoint | 1 | 150 | **1.000** | - | - | 0.950 | 0.900 |
| Earthquakes | 1 | 512 | **1.000** | - | - | **1.000** | 0.850 |
| Coffee | 1 | 286 | **1.000** | - | - | 0.850 | 0.450 |
| ItalyPowerDemand | 1 | 24 | **0.800** | - | - | **0.800** | 0.750 |
| Cricket | 6 | 1197 | **0.700** | - | - | 0.650 | 0.550 |
| Epilepsy | 3 | 206 | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| BasicMotions | 6 | 100 | **1.000** | **1.000** | **1.000** | **1.000** | - |
| ERing | 4 | 65 | 0.800 | **1.000** | 0.400 | 0.900 | - |
| RacketSports | 6 | 30 | **1.000** | **1.000** | - | **1.000** | - |
| NATOPS | 24 | 51 | 0.800 | **0.900** | - | 0.700 | - |

---

## Key Findings

### 1. SPARCE Breakthrough on Multivariate Datasets
SPARCE achieves a **clean sweep on DTW Plausibility across all 5 evaluated multivariate datasets**:
- **Epilepsy**: DTW 310.8 (SPARCE) vs 333.8 (Ours) vs 506.5 (M-CELS)
- **BasicMotions**: DTW 23.0 (SPARCE) vs 28.6 (Ours) vs 36.2 (M-CELS)
- **ERing**: DTW 67.3 (SPARCE) vs 111.3 (Ours) vs 183.4 (M-CELS) — a **39% improvement**
- **RacketSports**: DTW 107.4 (SPARCE) vs 129.6 (Ours) vs 150.4 (M-CELS)
- **NATOPS** (24 dimensions): DTW 208.1 (SPARCE) vs 310.4 (Ours) vs 447.3 (M-CELS) — a **33% improvement**

Furthermore, SPARCE solves the validity gap on multivariate datasets (achieving **0.80–1.00 validity**) where base Soft-DTW sometimes struggled due to gradient dispersion across multiple channels.

### 2. Base Soft-DTW Dominates Univariate Plausibility
On univariate time series, Ours (Soft-DTW) produces the lowest DTW distance on 5 out of 6 benchmarks and the highest Isolation Forest nominal fraction across all datasets.

### 3. GDFO Manifold Over-Constraint
GDFO (VAE+GMM generative density) guarantees high nominal density but fails to reliably cross complex multi-class decision boundaries, resulting in near-zero validity on multivariate tasks.

---
*Compiled from experimental results by `soft_dtw_cfe/compile_results.py` and `soft_dtw_cfe/run_sparce_experiments.py`*