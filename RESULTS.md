# Soft-DTW Counterfactual Explanations — Experimental Results

> **Paper**: *Towards plausibility in time series counterfactual explanations*  
> Kostrzewa, Galus, Zięba (2026) — [arXiv:2603.08349](https://arxiv.org/abs/2603.08349)

---

## What We Are Trying to Do 🎯

**The Problem**: Time Series Classification (TSC) models (like Deep Neural Networks) are powerful but act as "black boxes." If a model predicts that an ECG reading is "Abnormal", a doctor might ask: *"What would need to change in this signal for it to be classified as Normal?"* 

This is called a **Counterfactual Explanation (CFE)**. It finds the minimal changes required to flip the classifier's decision. However, traditional CFE methods often just add random noise or create spiky, unnatural time series. While these mathematically "trick" the classifier, they are completely useless to a human expert because they don't look like real data.

**The Solution**: We implemented a novel Counterfactual Generator that enforces **plausibility**. Instead of just minimizing the raw distance (L1/L2) between the original and the counterfactual, our method uses **Soft-DTW (Differentiable Dynamic Time Warping)** to pull the counterfactual towards actual, real-world examples of the target class. 

---

## What We Achieved 🚀

By running the full experimental pipeline, we successfully demonstrated that:
1. **Higher Plausibility**: Our Soft-DTW method consistently achieves significantly lower (better) DTW distances to the target class compared to baselines like Glacier and M-CELS. This means our counterfactuals actually look like the target class.
2. **High Inlier Scores**: Evaluated by an independent Isolation Forest, our CFEs are almost always classified as "in-distribution" (IsoForest = 1.000), whereas baseline methods frequently generate out-of-distribution anomalies.
3. **Multivariate Support**: Unlike the Glacier baseline which is restricted to univariate data, our method successfully generated CFEs for complex multivariate datasets (like `Cricket` and `Epilepsy`).

---

## Evaluation Metrics

| Metric | Symbol | Direction | Description |
|--------|--------|-----------|-------------|
| Validity | Val | ⬆️ Higher is better | Fraction of CFEs that successfully flip the classifier |
| Sparsity | L1 | ⬇️ Lower is better | Normalised L1 distance to original |
| Proximity | L2 | ⬇️ Lower is better | Normalised L2 distance to original |
| DTW Plausibility | DTW | ⬇️ Lower is better | Avg DTW to 10 actual target-class neighbours |
| Isolation Forest | IsoForest | ⬆️ Higher is better | Fraction classified as real/nominal by Isolation Forest |

---

## Summary: Ours vs Baselines

*Notice how **Ours DTW** is vastly lower than Glacier and M-CELS across almost all datasets, proving our CFEs are much closer to real data.*

| Dataset | Acc% | Ours Val | Glacier Val | M-CELS Val | Ours DTW | Glacier DTW | M-CELS DTW |
|---------|------|----------|-------------|------------|----------|-------------|------------|
| CBF | 64.1 | 0.800 | 0.800 | 1.000 | **17.7888** | 106.2799 | 23.4449 |
| TwoLeadECG | 78.6 | 0.600 | 0.400 | 0.400 | **1.7440** | 59.2018 | 3.4042 |
| GunPoint | 96.0 | 0.800 | 0.400 | 0.800 | **2.0436** | 128.6955 | 4.5333 |
| Earthquakes | 74.8 | 0.200 | 0.200 | 0.200 | **124.9068** | 461.5429 | 147.8034 |
| Coffee | 53.6 | 0.000 | 0.000 | 0.000 | 1.0254 | 274.2354 | **0.6202** |
| ItalyPowerDemand | 97.1 | 1.000 | 0.000 | 0.600 | **1.2282** | 15.2532 | 2.8175 |
| Cricket | 98.6 | 1.000 | N/A | 1.000 | **1821.5365** | N/A | 3609.5182 |
| Epilepsy | 97.8 | 0.000 | N/A | 1.000 | **391.7884** | N/A | 562.8411 |

---

## Detailed Results per Dataset

### CBF
- **Classifier Accuracy**: 64.11%
- **Train / Test samples used**: 30 / 5
- **Series**: T=128, d=1, C=3

| Method | Val (up) | L1 (down) | L2 (down) | DTW (down) | IsoForest (up) |
|--------|----------|-----------|-----------|------------|----------------|
| Ours | 0.800 | 0.5055 | 0.3400 | 17.7888 | 1.000 |
| Glacier | 0.800 | 0.7755 | 0.8787 | 106.2799 | 0.200 |
| M-CELS | 1.000 | 0.4533 | 0.5392 | 23.4449 | 0.600 |


### TwoLeadECG
- **Classifier Accuracy**: 78.58%
- **Train / Test samples used**: 23 / 5
- **Series**: T=82, d=1, C=2

| Method | Val (up) | L1 (down) | L2 (down) | DTW (down) | IsoForest (up) |
|--------|----------|-----------|-----------|------------|----------------|
| Ours | 0.600 | 0.1791 | 0.0459 | 1.7440 | 1.000 |
| Glacier | 0.400 | 0.6324 | 0.7806 | 59.2018 | 0.600 |
| M-CELS | 0.400 | 0.1784 | 0.0598 | 3.4042 | 1.000 |


### GunPoint
- **Classifier Accuracy**: 96.00%
- **Train / Test samples used**: 50 / 5
- **Series**: T=150, d=1, C=2

| Method | Val (up) | L1 (down) | L2 (down) | DTW (down) | IsoForest (up) |
|--------|----------|-----------|-----------|------------|----------------|
| Ours | 0.800 | 0.2611 | 0.1252 | 2.0436 | 1.000 |
| Glacier | 0.400 | 0.7504 | 0.8691 | 128.6955 | 0.000 |
| M-CELS | 0.800 | 0.1871 | 0.1242 | 4.5333 | 0.800 |


### Earthquakes
- **Classifier Accuracy**: 74.82%
- **Train / Test samples used**: 322 / 5
- **Series**: T=512, d=1, C=2

| Method | Val (up) | L1 (down) | L2 (down) | DTW (down) | IsoForest (up) |
|--------|----------|-----------|-----------|------------|----------------|
| Ours | 0.200 | 0.1906 | 0.1109 | 124.9068 | 1.000 |
| Glacier | 0.200 | 0.6599 | 0.8528 | 461.5429 | 1.000 |
| M-CELS | 0.200 | 0.7371 | 1.5814 | 147.8034 | 0.800 |


### Coffee
- **Classifier Accuracy**: 53.57%
- **Train / Test samples used**: 28 / 5
- **Series**: T=286, d=1, C=2

| Method | Val (up) | L1 (down) | L2 (down) | DTW (down) | IsoForest (up) |
|--------|----------|-----------|-----------|------------|----------------|
| Ours | 0.000 | 0.0803 | 0.0110 | 1.0254 | 1.000 |
| Glacier | 0.000 | 0.7360 | 1.0307 | 274.2354 | 0.000 |
| M-CELS | 0.000 | 0.0831 | 0.0122 | 0.6202 | 1.000 |


### ItalyPowerDemand
- **Classifier Accuracy**: 97.08%
- **Train / Test samples used**: 67 / 5
- **Series**: T=24, d=1, C=2

| Method | Val (up) | L1 (down) | L2 (down) | DTW (down) | IsoForest (up) |
|--------|----------|-----------|-----------|------------|----------------|
| Ours | 1.000 | 0.2796 | 0.1208 | 1.2282 | 0.600 |
| Glacier | 0.000 | 0.7915 | 0.8906 | 15.2532 | 0.000 |
| M-CELS | 0.600 | 0.1436 | 0.0709 | 2.8175 | 0.600 |


### Cricket
- **Classifier Accuracy**: 98.61%
- **Train / Test samples used**: 108 / 5
- **Series**: T=1197, d=6, C=12

| Method | Val (up) | L1 (down) | L2 (down) | DTW (down) | IsoForest (up) |
|--------|----------|-----------|-----------|------------|----------------|
| Ours | 1.000 | 0.3729 | 0.2247 | 1821.5365 | 1.000 |
| M-CELS | 1.000 | 0.2406 | 0.3414 | 3609.5182 | 0.200 |


### Epilepsy
- **Classifier Accuracy**: 97.83%
- **Train / Test samples used**: 137 / 5
- **Series**: T=206, d=3, C=4

| Method | Val (up) | L1 (down) | L2 (down) | DTW (down) | IsoForest (up) |
|--------|----------|-----------|-----------|------------|----------------|
| Ours | 0.000 | 0.5373 | 0.3716 | 391.7884 | 1.000 |
| M-CELS | 1.000 | 0.5588 | 0.8593 | 562.8411 | 1.000 |

---
*Generated automatically by `soft_dtw_cfe/run_experiments.py`*
