# Soft-DTW Counterfactual Explanations — Experimental Results

> **Paper**: *Towards plausibility in time series counterfactual explanations*  
> Kostrzewa, Galus, Zięba (2026) — [arXiv:2603.08349](https://arxiv.org/abs/2603.08349)

## Evaluation Metrics

| Metric | Symbol | Direction | Description |
|--------|--------|-----------|-------------|
| Validity | Val | up higher is better | Fraction of CFEs that flip the classifier |
| Sparsity | L1 | down lower is better | Normalised L1 distance to original |
| Proximity | L2 | down lower is better | Normalised L2 distance to original |
| DTW Plausibility | DTW | down lower is better | Avg DTW to 10 target-class neighbours |
| Isolation Forest | IsoForest | up higher is better | Fraction classified as nominal by IF |

---

## Results per Dataset

### ItalyPowerDemand

- **Classifier Accuracy**: 96.21%
- **Train / Test samples used**: 67 / 20
- **Series**: T=24, d=1, C=2
- **Elapsed**: 36.1s

| Method | Val (up) | L1 (down) | L2 (down) | DTW (down) | IsoForest (up) | Time(s) | s/sample |
|--------|----------|-----------|-----------|------------|----------------|---------|----------|
| Ours | 1.000 | 0.2599 | 0.1193 | 1.0069 | 0.800 | 9.7 | 0.49 |
| M-CELS | 1.000 | 0.1884 | 0.1265 | 1.9500 | 0.800 | 0.0 | 0.00 |
| DTW-CFE | 1.000 | 0.1938 | 0.0775 | 1.9296 | 0.750 | 22.8 | 1.14 |


### GunPoint

- **Classifier Accuracy**: 98.67%
- **Train / Test samples used**: 50 / 20
- **Series**: T=150, d=1, C=2
- **Elapsed**: 65.1s

| Method | Val (up) | L1 (down) | L2 (down) | DTW (down) | IsoForest (up) | Time(s) | s/sample |
|--------|----------|-----------|-----------|------------|----------------|---------|----------|
| Ours | 1.000 | 0.1789 | 0.0588 | 2.9727 | 1.000 | 33.3 | 1.66 |
| M-CELS | 1.000 | 0.1246 | 0.0700 | 5.1570 | 0.950 | 0.0 | 0.00 |
| DTW-CFE | 1.000 | 0.1221 | 0.0323 | 5.8324 | 0.900 | 21.8 | 1.09 |


### Coffee

- **Classifier Accuracy**: 53.57%
- **Train / Test samples used**: 28 / 20
- **Series**: T=286, d=1, C=2
- **Elapsed**: 142.1s

| Method | Val (up) | L1 (down) | L2 (down) | DTW (down) | IsoForest (up) | Time(s) | s/sample |
|--------|----------|-----------|-----------|------------|----------------|---------|----------|
| Ours | 0.250 | 0.0895 | 0.0156 | 0.9609 | 1.000 | 107.8 | 5.39 |
| M-CELS | 0.250 | 0.0715 | 0.0118 | 0.8818 | 0.850 | 0.1 | 0.01 |
| DTW-CFE | 0.250 | 0.0215 | 0.0033 | 1.6118 | 0.450 | 5.9 | 0.30 |


### Earthquakes

- **Classifier Accuracy**: 74.82%
- **Train / Test samples used**: 322 / 20
- **Series**: T=512, d=1, C=2
- **Elapsed**: 441.1s

| Method | Val (up) | L1 (down) | L2 (down) | DTW (down) | IsoForest (up) | Time(s) | s/sample |
|--------|----------|-----------|-----------|------------|----------------|---------|----------|
| Ours | 0.150 | 0.2016 | 0.1163 | 144.6361 | 1.000 | 349.5 | 17.47 |
| M-CELS | 0.200 | 0.7407 | 1.6255 | 129.9494 | 1.000 | 0.2 | 0.01 |
| DTW-CFE | 0.150 | 0.1053 | 0.1608 | 213.6523 | 0.850 | 5.5 | 0.28 |


### CBF

- **Classifier Accuracy**: 43.89%
- **Train / Test samples used**: 30 / 20
- **Series**: T=128, d=1, C=3
- **Elapsed**: 245.0s

| Method | Val (up) | L1 (down) | L2 (down) | DTW (down) | IsoForest (up) | Time(s) | s/sample |
|--------|----------|-----------|-----------|------------|----------------|---------|----------|
| Ours | 1.000 | 0.4846 | 0.3192 | 18.6373 | 1.000 | 28.0 | 1.40 |
| M-CELS | 0.850 | 0.3338 | 0.3777 | 29.6497 | 0.650 | 0.1 | 0.00 |
| DTW-CFE | 0.650 | 0.4372 | 0.3405 | 29.0361 | 0.850 | 21.3 | 1.06 |


### TwoLeadECG

- **Classifier Accuracy**: 88.50%
- **Train / Test samples used**: 23 / 20
- **Series**: T=82, d=1, C=2
- **Elapsed**: 199.8s

| Method | Val (up) | L1 (down) | L2 (down) | DTW (down) | IsoForest (up) | Time(s) | s/sample |
|--------|----------|-----------|-----------|------------|----------------|---------|----------|
| Ours | 0.750 | 0.1597 | 0.0427 | 1.6324 | 1.000 | 15.0 | 0.75 |
| M-CELS | 0.850 | 0.1102 | 0.0365 | 3.0265 | 0.950 | 0.1 | 0.00 |
| DTW-CFE | 1.000 | 0.1402 | 0.0340 | 2.3496 | 0.900 | 20.9 | 1.04 |


### Epilepsy

- **Classifier Accuracy**: 97.83%
- **Train / Test samples used**: 137 / 20
- **Series**: T=206, d=3, C=4
- **Elapsed**: 227.2s

| Method | Val (up) | L1 (down) | L2 (down) | DTW (down) | IsoForest (up) | Time(s) | s/sample |
|--------|----------|-----------|-----------|------------|----------------|---------|----------|
| Ours | 0.300 | 0.4782 | 0.3084 | 266.2740 | 1.000 | 65.6 | 3.28 |
| M-CELS | 1.000 | 0.4404 | 0.5822 | 424.4845 | 1.000 | 0.0 | 0.00 |
| DTW-CFE | 1.000 | 0.5513 | 0.6145 | 343.1891 | 1.000 | 24.0 | 1.20 |


### Cricket

- **Classifier Accuracy**: 100.00%
- **Train / Test samples used**: 108 / 20
- **Series**: T=1197, d=6, C=12
- **Elapsed**: 2885.4s

| Method | Val (up) | L1 (down) | L2 (down) | DTW (down) | IsoForest (up) | Time(s) | s/sample |
|--------|----------|-----------|-----------|------------|----------------|---------|----------|
| Ours | 0.500 | 0.4455 | 0.2932 | 2362.4284 | 0.700 | 1822.4 | 91.12 |
| M-CELS | 1.000 | 0.5789 | 1.0161 | 3308.6161 | 0.650 | 0.1 | 0.00 |
| DTW-CFE | 1.000 | 0.5061 | 0.6126 | 3956.1409 | 0.550 | 41.0 | 2.05 |


---

## Summary: Ours vs Baselines

| Dataset | Acc% | Ours Val | M-CELS Val | DTW-CFE Val | Ours DTW | M-CELS DTW | DTW-CFE DTW | Ours s/samp | DTW-CFE s/samp |
|---------|------|----------|------------|-------------|----------|------------|-------------|-------------|----------------|
| ItalyPowerDemand | 96.2 | 1.000 | 1.000 | 1.000 | 1.0069 | 1.9500 | 1.9296 | 0.49 | 1.14 |
| GunPoint | 98.7 | 1.000 | 1.000 | 1.000 | 2.9727 | 5.1570 | 5.8324 | 1.66 | 1.09 |
| Coffee | 53.6 | 0.250 | 0.250 | 0.250 | 0.9609 | 0.8818 | 1.6118 | 5.39 | 0.30 |
| Earthquakes | 74.8 | 0.150 | 0.200 | 0.150 | 144.6361 | 129.9494 | 213.6523 | 17.47 | 0.28 |
| CBF | 43.9 | 1.000 | 0.850 | 0.650 | 18.6373 | 29.6497 | 29.0361 | 1.40 | 1.06 |
| TwoLeadECG | 88.5 | 0.750 | 0.850 | 1.000 | 1.6324 | 3.0265 | 2.3496 | 0.75 | 1.04 |
| Epilepsy | 97.8 | 0.300 | 1.000 | 1.000 | 266.2740 | 424.4845 | 343.1891 | 3.28 | 1.20 |
| Cricket | 100.0 | 0.500 | 1.000 | 1.000 | 2362.4284 | 3308.6161 | 3956.1409 | 91.12 | 2.05 |

---
*Generated automatically by `soft_dtw_cfe/run_experiments.py`*