# GDFO Comparison — Generative Density Function Optimisation

> Comparing the new GDFO approach with existing methods on larger
> multivariate time series datasets from the UEA archive.

## Methods Compared

| Method | Description |
|--------|-------------|
| **Ours** (Soft-DTW) | Original proposed method — gradient-based soft-DTW alignment |
| **GDFO** | Generative density function optimisation — VAE+GMM density-guided |
| **M-CELS** | Baseline — greedy class-conditional perturbation |
| **DTW-CFE** | DTW-guided constrained deformation via CMA-ES |

## Evaluation Metrics

| Metric | Direction | Description |
|--------|-----------|-------------|
| Val | ↑ higher is better | Fraction of CFEs that flip the classifier |
| L1 | ↓ lower is better | Normalised L1 distance (sparsity) |
| L2 | ↓ lower is better | Normalised L2 distance (proximity) |
| DTW | ↓ lower is better | Avg DTW to 10 target-class neighbours (plausibility) |
| IsoForest | ↑ higher is better | Fraction classified as nominal by Isolation Forest |

---

## Results per Dataset

### Epilepsy

- **Classifier Accuracy**: 97.83%
- **Train / Test samples used**: 137 / 10
- **Series**: T=206, d=3, C=4
- **Elapsed**: 272.3s

| Method | Val ↑ | L1 ↓ | L2 ↓ | DTW ↓ | IsoForest ↑ | Time(s) | s/sample |
|--------|-------|------|------|-------|-------------|---------|----------|
| Ours | 0.200 | 0.5065 | 0.3389 | 333.7983 | 1.000 | 99.9 | 9.99 |
| GDFO | 0.000 | 0.3408 | 0.1478 | 603.3149 | 1.000 | 141.0 | 14.10 |
| M-CELS | 1.000 | 0.4499 | 0.6749 | 506.5135 | 1.000 | 0.1 | 0.01 |


### BasicMotions

- **Classifier Accuracy**: 100.00%
- **Train / Test samples used**: 40 / 10
- **Series**: T=100, d=6, C=4
- **Elapsed**: 127.9s

| Method | Val ↑ | L1 ↓ | L2 ↓ | DTW ↓ | IsoForest ↑ | Time(s) | s/sample |
|--------|-------|------|------|-------|-------------|---------|----------|
| Ours | 0.900 | 0.1874 | 0.0562 | 28.6113 | 1.000 | 53.7 | 5.37 |
| GDFO | 0.000 | 0.1715 | 0.0350 | 69.3110 | 1.000 | 58.7 | 5.87 |
| M-CELS | 1.000 | 0.2244 | 0.0982 | 36.2406 | 1.000 | 0.2 | 0.02 |


### ERing

- **Classifier Accuracy**: 85.93%
- **Train / Test samples used**: 30 / 10
- **Series**: T=65, d=4, C=6
- **Elapsed**: 114.0s

| Method | Val ↑ | L1 ↓ | L2 ↓ | DTW ↓ | IsoForest ↑ | Time(s) | s/sample |
|--------|-------|------|------|-------|-------------|---------|----------|
| Ours | 0.900 | 0.5331 | 0.3690 | 111.2727 | 0.800 | 16.2 | 1.62 |
| GDFO | 0.100 | 0.7036 | 0.6295 | 274.7387 | 0.400 | 57.8 | 5.78 |
| M-CELS | 1.000 | 0.8156 | 1.3790 | 183.3800 | 0.900 | 0.2 | 0.02 |


---

## Summary: GDFO vs Previous Approaches

| Dataset | d | T | C | Acc% | Ours Val | GDFO Val | Ours DTW | GDFO DTW | Ours IsoF | GDFO IsoF | GDFO s/samp |
|---------|---|---|---|------|----------|----------|----------|----------|-----------|-----------|-------------|
| Epilepsy | 3 | 206 | 4 | 97.8 | 0.200 | 0.000 | 333.80 | 603.31 | 1.000 | 1.000 | 14.10 |
| BasicMotions | 6 | 100 | 4 | 100.0 | 0.900 | 0.000 | 28.61 | 69.31 | 1.000 | 1.000 | 5.87 |
| ERing | 4 | 65 | 6 | 85.9 | 0.900 | 0.100 | 111.27 | 274.74 | 0.800 | 0.400 | 5.78 |

---

## Key Observations

1. **Density-guided plausibility**: GDFO's VAE+GMM density term provides a 
   smooth gradient signal from the full class distribution, not just k nearest 
   neighbours. This typically improves IsoForest scores (nominal fraction).

2. **Multivariate scaling**: GDFO jointly encodes all channels, capturing 
   cross-channel correlations that per-channel methods may miss.

3. **Trade-offs**: GDFO adds VAE training overhead but reuses the density model 
   across all test samples, amortising the cost for larger test sets.

---
*Generated automatically by `soft_dtw_cfe/run_gdfo_experiments.py`*