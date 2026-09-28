"""
Evaluation metrics for counterfactual explanations.

Implements the five metrics described in Section 5.1:
  - Val ↑: Validity (Equation 4)
  - L1 ↓: Sparsity (Equation 5, p=1)
  - L2 ↓: Proximity (Equation 5, p=2)
  - DTW ↓: Plausibility (avg DTW to k nearest target-class neighbours)
  - Iso Forest Score ↑: Fraction of CFs classified as nominal (non-outliers)
"""

import numpy as np
from sklearn.ensemble import IsolationForest

from soft_dtw_cfe.methods.soft_dtw import compute_dtw_distance
from soft_dtw_cfe.config import EVAL_CONFIG

def compute_validity(results: list) -> float:
    """
    Validity (Val ↑): Fraction of counterfactuals that successfully
    flip the classifier's prediction to the target class.

    Args:
        results: List of result dicts from CFE generators.
    """
    if not results:
        return 0.0
    valid_count = sum(1 for r in results if r["valid"])
    return valid_count / len(results)

def compute_validity_any(results: list) -> float:
    """
    Any-flip validity (paper Eq.4): fraction where predicted class
    differs from original class, regardless of target.
    """
    if not results:
        return 0.0
    n = sum(1 for r in results
            if r.get("predicted_class") != r.get("original_class"))
    return n / len(results)

def compute_proximity_sparsity(results: list):
    """
    Proximity (L2 ↓) and Sparsity (L1 ↓), normalized by dT,
    plus paper-Table-2 raw (unnormalized) variants L1_raw/L2_raw.

    Args:
        results: List of result dicts.

    Returns:
        mean_l1, mean_l2, mean_l1_raw, mean_l2_raw
    """
    l1_distances = []
    l2_distances = []
    l1_raw = []
    l2_raw = []

    for r in results:
        # Tensors to numpy, shape (d, T)
        x = r["original"].numpy()
        x_cf = r["counterfactual"].numpy()

        # d*T
        size = x.size

        l1r = float(np.sum(np.abs(x_cf - x)))
        l2r = float(np.sum((x_cf - x)**2))

        l1_distances.append(l1r / size)
        l2_distances.append(l2r / size)
        l1_raw.append(l1r)
        l2_raw.append(l2r)

    return (float(np.mean(l1_distances)), float(np.mean(l2_distances)),
            float(np.mean(l1_raw)), float(np.mean(l2_raw)))

def compute_plausibility_dtw(results: list, X_train: np.ndarray, y_train: np.ndarray, k: int = None) -> float:
    """
    Plausibility (DTW ↓): Average DTW distance to the k nearest neighbours
    from the target class.
    
    Args:
        results: List of result dicts.
        X_train: (N, d, T) — training data.
        y_train: (N,) — training labels.
        k: number of neighbours (default from config).
    """
    if k is None:
        k = EVAL_CONFIG["dtw_k"]
        
    dtw_scores = []
    
    for r in results:
        x_cf = r["counterfactual"].numpy()  # (d, T)
        target_class = r["target_class"]
        
        # Get target class samples
        mask = (y_train == int(target_class))
        target_samples = X_train[mask]  # (N_target, d, T)
        
        # We need the k nearest neighbours based on DTW, or we approximate
        # by finding the k nearest neighbours using L2, and then computing DTW
        # The paper says: "average DTW distance to the 10 nearest neighbors from the target class"
        # It's common to find the neighbours via L2 for speed, then compute DTW.
        # Let's compute L2 distances to find the top k
        diffs = target_samples - np.expand_dims(x_cf, 0)
        l2_dists = np.sum(diffs**2, axis=(1, 2))
        
        k_actual = min(k, len(target_samples))
        top_k_idx = np.argsort(l2_dists)[:k_actual]
        nn_samples = target_samples[top_k_idx]
        
        # Compute DTW to these k neighbours
        # DTW function expects (T, d)
        x_cf_td = x_cf.T 
        nn_dtw_sum = 0.0
        for nn_samp in nn_samples:
            nn_td = nn_samp.T
            nn_dtw_sum += compute_dtw_distance(x_cf_td, nn_td)
            
        dtw_scores.append(nn_dtw_sum / k_actual)
        
    return np.mean(dtw_scores)

def compute_isolation_forest_score(results: list, X_train: np.ndarray, y_train: np.ndarray) -> float:
    """
    Isolation Forest Score ↑: Measures the fraction of counterfactuals
    classified as nominal (non-outliers) by an Isolation Forest trained
    on the target class.
    
    Args:
        results: List of result dicts.
        X_train: (N, d, T) — training data.
        y_train: (N,) — training labels.
    """
    if not results:
        return 0.0
        
    nominal_count = 0
    total = len(results)
    
    # We can train one IF per class
    unique_classes = np.unique(y_train)
    iso_forests = {}
    
    for c in unique_classes:
        mask = (y_train == int(c))
        X_c = X_train[mask]  # (N_c, d, T)
        
        # Flatten time series for Isolation Forest
        X_c_flat = X_c.reshape(X_c.shape[0], -1)
        
        clf = IsolationForest(
            contamination=EVAL_CONFIG["isolation_forest_contamination"],
            random_state=42
        )
        clf.fit(X_c_flat)
        iso_forests[c] = clf
        
    for r in results:
        x_cf = r["counterfactual"].numpy()
        target_class = r["target_class"]
        
        x_cf_flat = x_cf.reshape(1, -1)
        
        # Predict: 1 for nominal, -1 for outlier
        pred = iso_forests[target_class].predict(x_cf_flat)[0]
        
        if pred == 1:
            nominal_count += 1
            
    return nominal_count / total

def evaluate_all_metrics(results: list, X_train: np.ndarray, y_train: np.ndarray):
    """
    Compute and return all evaluation metrics as a dictionary.
    """
    if not results:
        return {}
        
    val = compute_validity(results)
    val_any = compute_validity_any(results)
    l1, l2, l1_raw, l2_raw = compute_proximity_sparsity(results)
    dtw = compute_plausibility_dtw(results, X_train, y_train)
    iso = compute_isolation_forest_score(results, X_train, y_train)

    return {
        "Val": val,
        "Val_any": val_any,
        "L1": l1,
        "L2": l2,
        "L1_raw": l1_raw,
        "L2_raw": l2_raw,
        "DTW": dtw,
        "IsoForest": iso
    }
