# SPARCE Comparison — Multivariate Time Series CFE

> Saliency-guided Prototype-Aligned Refinement for Counterfactual Explanations
> vs. existing methods on multivariate time series datasets.

## Methods Compared

| Method | Type | Key Innovation |
|--------|------|---------------|
| **Ours** (Soft-DTW) | Gradient-based | Soft-DTW alignment for temporal plausibility |
| **SPARCE** | Prototype+Saliency | Two-phase curriculum with channel-selective saliency masking |
| **GDFO** (VAE+GMM) | Generative density | VAE+GMM density-guided optimisation |
| **M-CELS** | Greedy baseline | Saliency-guided NUN substitution |
| **DTW-CFE** (CMA-ES) | Evolutionary | DTW-guided constrained deformation |

## SPARCE Key Innovations

1. **Prototype warm-start**: Initialise from saliency-weighted blend of original + nearest unlike neighbour
2. **Channel-selective saliency mask**: Learnable mask focuses perturbations on discriminative channels
3. **Two-phase curriculum**: Phase 1 = validity-first (aggressive), Phase 2 = plausibility refinement
4. **Cross-channel correlation**: Preserves inter-channel covariance structure

---

## Results per Dataset

### Epilepsy

- **Classifier Accuracy**: 97.83%
- **Samples**: 137 train / 10 test
- **Series**: T=206, d=3, C=4
- **Elapsed**: 278.9s

| Method | Val ↑ | L1 ↓ | L2 ↓ | DTW ↓ | IsoForest ↑ | Time(s) | s/sample |
|--------|-------|------|------|-------|-------------|---------|----------|
| Ours | 0.200 | 0.5065 | 0.3389 | 333.7983 | 1.000 | 113.2 | 11.32 |
| SPARCE | 0.800 | 0.7297 | 0.8672 | 310.7604 | 1.000 | 133.0 | 13.30 |
| M-CELS | 1.000 | 0.4499 | 0.6749 | 506.5135 | 1.000 | 0.1 | 0.01 |


### BasicMotions

- **Classifier Accuracy**: 100.00%
- **Samples**: 40 train / 10 test
- **Series**: T=100, d=6, C=4
- **Elapsed**: 165.8s

| Method | Val ↑ | L1 ↓ | L2 ↓ | DTW ↓ | IsoForest ↑ | Time(s) | s/sample |
|--------|-------|------|------|-------|-------------|---------|----------|
| Ours | 0.900 | 0.1874 | 0.0562 | 28.6113 | 1.000 | 40.6 | 4.06 |
| SPARCE | 1.000 | 0.2019 | 0.0772 | 22.9847 | 1.000 | 94.2 | 9.42 |
| M-CELS | 1.000 | 0.2244 | 0.0982 | 36.2406 | 1.000 | 0.2 | 0.02 |


### ERing

- **Classifier Accuracy**: 85.93%
- **Samples**: 30 train / 10 test
- **Series**: T=65, d=4, C=6
- **Elapsed**: 45.4s

| Method | Val ↑ | L1 ↓ | L2 ↓ | DTW ↓ | IsoForest ↑ | Time(s) | s/sample |
|--------|-------|------|------|-------|-------------|---------|----------|
| Ours | 0.900 | 0.5331 | 0.3690 | 111.2727 | 0.800 | 12.7 | 1.27 |
| SPARCE | 1.000 | 0.9280 | 1.3468 | 67.3255 | 1.000 | 23.4 | 2.34 |
| M-CELS | 1.000 | 0.8156 | 1.3790 | 183.3800 | 0.900 | 0.1 | 0.01 |


### RacketSports

- **Classifier Accuracy**: 83.55%
- **Samples**: 151 train / 10 test
- **Series**: T=30, d=6, C=4
- **Elapsed**: 77.1s

| Method | Val ↑ | L1 ↓ | L2 ↓ | DTW ↓ | IsoForest ↑ | Time(s) | s/sample |
|--------|-------|------|------|-------|-------------|---------|----------|
| Ours | 0.900 | 0.3908 | 0.2392 | 129.6244 | 1.000 | 19.9 | 1.99 |
| SPARCE | 1.000 | 0.4466 | 0.4254 | 107.3734 | 1.000 | 26.0 | 2.60 |
| M-CELS | 1.000 | 0.3192 | 0.6522 | 150.4005 | 1.000 | 0.1 | 0.01 |


### NATOPS

- **Classifier Accuracy**: 96.11%
- **Samples**: 180 train / 10 test
- **Series**: T=51, d=24, C=6
- **Elapsed**: 344.3s

| Method | Val ↑ | L1 ↓ | L2 ↓ | DTW ↓ | IsoForest ↑ | Time(s) | s/sample |
|--------|-------|------|------|-------|-------------|---------|----------|
| Ours | 1.000 | 0.3658 | 0.2154 | 310.3622 | 0.800 | 25.4 | 2.54 |
| SPARCE | 1.000 | 0.4895 | 0.5285 | 208.1027 | 0.900 | 34.1 | 3.41 |
| M-CELS | 1.000 | 0.3055 | 0.4272 | 447.3424 | 0.700 | 0.1 | 0.01 |


---

## Grand Summary — SPARCE vs All Methods

### Validity (Val ↑)

| Dataset | d | T | C | Ours | SPARCE | GDFO | M-CELS | DTW-CFE |
|---------|---|---|---|------|--------|------|--------|---------|
| Epilepsy | 3 | 206 | 4 | 0.200 | 0.800 | - | 1.000 | - |
| BasicMotions | 6 | 100 | 4 | 0.900 | 1.000 | - | 1.000 | - |
| ERing | 4 | 65 | 6 | 0.900 | 1.000 | - | 1.000 | - |
| RacketSports | 6 | 30 | 4 | 0.900 | 1.000 | - | 1.000 | - |
| NATOPS | 24 | 51 | 6 | 1.000 | 1.000 | - | 1.000 | - |

### DTW Plausibility (↓)

| Dataset | d | T | Ours | SPARCE | GDFO | M-CELS | DTW-CFE |
|---------|---|---|------|--------|------|--------|---------|
| Epilepsy | 3 | 206 | 333.80 | 310.76 | - | 506.51 | - |
| BasicMotions | 6 | 100 | 28.61 | 22.98 | - | 36.24 | - |
| ERing | 4 | 65 | 111.27 | 67.33 | - | 183.38 | - |
| RacketSports | 6 | 30 | 129.62 | 107.37 | - | 150.40 | - |
| NATOPS | 24 | 51 | 310.36 | 208.10 | - | 447.34 | - |

### IsoForest Plausibility (↑)

| Dataset | d | T | Ours | SPARCE | GDFO | M-CELS | DTW-CFE |
|---------|---|---|------|--------|------|--------|---------|
| Epilepsy | 3 | 206 | 1.000 | 1.000 | - | 1.000 | - |
| BasicMotions | 6 | 100 | 1.000 | 1.000 | - | 1.000 | - |
| ERing | 4 | 65 | 0.800 | 1.000 | - | 0.900 | - |
| RacketSports | 6 | 30 | 1.000 | 1.000 | - | 1.000 | - |
| NATOPS | 24 | 51 | 0.800 | 0.900 | - | 0.700 | - |

---

## Method Win Counts (across 5 multivariate datasets)

| Method | Val ↑ | DTW ↓ | IsoForest ↑ | L1 ↓ | L2 ↓ | **Total** |
|--------|-------|-------|-------------|------|------|-----------|
| **Ours** | 1 | 0 | 3 | 2 | 5 | **11** |
| **SPARCE** | 4 | 5 | 5 | 0 | 0 | **14** |
| **GDFO** | 0 | 0 | 0 | 0 | 0 | **0** |
| **M-CELS** | 5 | 0 | 3 | 3 | 0 | **11** |
| **DTW-CFE** | 0 | 0 | 0 | 0 | 0 | **0** |

## Average Rank (lower is better)

| Method | Val ↑ | DTW ↓ | IsoForest ↑ | L1 ↓ | L2 ↓ | **Avg** |
|--------|-------|-------|-------------|------|------|---------|
| **Ours** | 2.60 | 2.00 | 1.60 | 1.60 | 1.00 | **1.76** |
| **SPARCE** | 1.40 | 1.00 | 1.60 | 2.80 | 2.40 | **1.84** |
| **GDFO** | - | - | - | - | - | **-** |
| **M-CELS** | 2.00 | 3.00 | 2.80 | 1.60 | 2.60 | **2.40** |
| **DTW-CFE** | - | - | - | - | - | **-** |

---

## SPARCE Analysis

### Why SPARCE works on multivariate data

1. **Prototype warm-start solves the validity gap**: By initialising from a
   saliency-weighted blend of the original + nearest target-class sample,
   SPARCE starts close to the decision boundary. This avoids the GDFO problem
   where density pulls the CFE back before validity is achieved.

2. **Channel saliency focuses perturbations**: Instead of perturbing all channels
   equally, the learnable saliency mask identifies discriminative channels and
   concentrates modifications there — preserving non-discriminative channels intact.

3. **Two-phase curriculum**: Phase 1 aggressively achieves validity with strong λ_valid;
   Phase 2 refines plausibility (DTW + correlation) while maintaining the class flip.

4. **Cross-channel correlation preservation**: The covariance penalty ensures
   inter-channel correlations (e.g., x/y/z accelerometer axes) are maintained.

---
*Generated automatically by `soft_dtw_cfe/run_sparce_experiments.py`*