"""
Complete ECG Evaluation Script
==============================
Runs complete evaluation of CFE methods across all ECG datasets:
1. TwoLeadECG (Univariate: d=1, T=82, C=2) -> evaluates SPARCE + DTW-CFE + Ours + M-CELS
2. ECG5000 (Univariate: d=1, T=140, C=5) -> evaluates DTW-CFE (to complete 5-method comparison)
3. PTB_XL (Multivariate 12-lead: d=12, T=250, C=2) -> evaluates 10 test samples on Ours, SPARCE, M-CELS, DTW-CFE

Saves unified results into results/ecg_comprehensive_results.json.
"""

import os
import sys
import json
import time
import torch
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from soft_dtw_cfe.config import (
    DEVICE, RESULTS_DIR, CFE_CONFIG, DTWCFE_CONFIG, GDFO_CONFIG
)
from soft_dtw_cfe.data.dataset_loader import load_dataset, get_dataloaders
from soft_dtw_cfe.models.classifier import TSClassifier, load_classifier, evaluate_classifier
from soft_dtw_cfe.methods.proposed_method import SoftDTWCounterfactualGenerator
from soft_dtw_cfe.methods.sparce import SPARCECounterfactualGenerator
from soft_dtw_cfe.methods.m_cels import MCELS
from soft_dtw_cfe.methods.dtw_guided.dtw_cfe import DTWGuidedCFE
from soft_dtw_cfe.evaluation.metrics import evaluate_all_metrics
from soft_dtw_cfe.visualization.plot import (
    plot_counterfactual, plot_paper_metrics,
)

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_twoleadecg(n_test=20, seed=42):
    print("\n" + "="*70)
    print(" RUNNING TWOLEADECG (Univariate: d=1, T=82, C=2)")
    print("="*70)
    set_seed(seed)
    X_train, y_train, X_test, y_test, meta = load_dataset("TwoLeadECG")
    d, c, T = meta["n_channels"], meta["n_classes"], meta["seq_len"]

    model = TSClassifier(n_channels=d, n_classes=c)
    model = load_classifier(model, "TwoLeadECG")
    model.eval()

    n_actual = min(n_test, len(X_test))
    X_sample = torch.tensor(X_test[:n_actual], dtype=torch.float32)
    y_sample = torch.tensor(y_test[:n_actual], dtype=torch.long)

    results = {}
    elapsed = {}

    # 1. SPARCE
    print("  -> Running SPARCE...")
    t0 = time.time()
    sparce = SPARCECounterfactualGenerator(classifier=model)
    res_sparce = sparce.generate_batch(X_sample, y_sample, X_train, y_train, verbose=False)
    elapsed["SPARCE"] = time.time() - t0
    results["SPARCE"] = evaluate_all_metrics(res_sparce, X_train, y_train)

    # 2. DTW-CFE
    print("  -> Running DTW-CFE...")
    t0 = time.time()
    cfg = dict(DTWCFE_CONFIG)
    cfg["seed"] = seed
    dtwcfe = DTWGuidedCFE(classifier=model, **cfg)
    res_dtwcfe = dtwcfe.generate_batch(X_sample, y_sample, X_train, y_train, verbose=False)
    elapsed["DTW-CFE"] = time.time() - t0
    results["DTW-CFE"] = evaluate_all_metrics(res_dtwcfe, X_train, y_train)

    # 3. Soft-DTW (Ours)
    print("  -> Running Soft-DTW (Ours)...")
    t0 = time.time()
    ours = SoftDTWCounterfactualGenerator(classifier=model, num_iterations=CFE_CONFIG["num_iterations"])
    res_ours = ours.generate_batch(X_sample, y_sample, X_train, y_train, verbose=False)
    elapsed["Ours"] = time.time() - t0
    results["Ours"] = evaluate_all_metrics(res_ours, X_train, y_train)

    # 4. M-CELS
    print("  -> Running M-CELS...")
    t0 = time.time()
    mcels = MCELS(classifier=model)
    res_mcels = mcels.generate_batch(X_sample, y_sample, X_train, y_train, verbose=False)
    elapsed["M-CELS"] = time.time() - t0
    results["M-CELS"] = evaluate_all_metrics(res_mcels, X_train, y_train)

    # Plot sample counterfactuals
    if res_sparce:
        r = res_sparce[0]
        t_samples = torch.tensor(X_train[y_train == int(r["target_class"])], dtype=torch.float32)
        plot_counterfactual(
            r["original"], r["counterfactual"], t_samples,
            "TwoLeadECG", "SPARCE", r["original_class"], r["target_class"],
            "TwoLeadECG_SPARCE_sparce_cf"
        )

    out = {
        "dataset": "TwoLeadECG",
        "type": "Univariate",
        "d": d, "T": T, "C": c,
        "n_train": len(X_train), "n_test_used": n_actual,
        "methods": results,
        "elapsed": elapsed
    }
    print("  TwoLeadECG Results:", json.dumps(results, indent=2))
    return out

def run_ecg5000_dtwcfe(n_test=25, seed=42):
    print("\n" + "="*70)
    print(" RUNNING ECG5000 DTW-CFE (Univariate: d=1, T=140, C=5)")
    print("="*70)
    set_seed(seed)
    X_train, y_train, X_test, y_test, meta = load_dataset("ECG5000")
    d, c, T = meta["n_channels"], meta["n_classes"], meta["seq_len"]

    model = TSClassifier(n_channels=d, n_classes=c)
    model = load_classifier(model, "ECG5000")
    model.eval()

    n_actual = min(n_test, len(X_test))
    X_sample = torch.tensor(X_test[:n_actual], dtype=torch.float32)
    y_sample = torch.tensor(y_test[:n_actual], dtype=torch.long)

    cfg = dict(DTWCFE_CONFIG)
    cfg["seed"] = seed
    dtwcfe = DTWGuidedCFE(classifier=model, **cfg)
    t0 = time.time()
    res = dtwcfe.generate_batch(X_sample, y_sample, X_train, y_train, verbose=False)
    elapsed = time.time() - t0
    metrics = evaluate_all_metrics(res, X_train, y_train)
    print(f"  DTW-CFE ECG5000 completed in {elapsed:.1f}s. Metrics:", metrics)

    # Save visualization for DTW-CFE on ECG5000
    if res:
        r = res[0]
        t_samples = torch.tensor(X_train[y_train == int(r["target_class"])], dtype=torch.float32)
        plot_counterfactual(
            r["original"], r["counterfactual"], t_samples,
            "ECG5000", "DTW-CFE", r["original_class"], r["target_class"],
            "ECG5000_DTW-CFE_ecg5000_cf"
        )
    return {"metrics": metrics, "elapsed": elapsed}

def run_ptbxl_extended(n_test=10, seed=42):
    print("\n" + "="*70)
    print(" RUNNING PTB_XL EXTENDED (Multivariate 12-lead: d=12, T=250, C=2)")
    print("="*70)
    set_seed(seed)
    X_train, y_train, X_test, y_test, meta = load_dataset("PTB_XL")
    d, c, T = meta["n_channels"], meta["n_classes"], meta["seq_len"]

    model = TSClassifier(n_channels=d, n_classes=c)
    model = load_classifier(model, "PTB_XL")
    model.eval()

    n_actual = min(n_test, len(X_test))
    X_sample = torch.tensor(X_test[:n_actual], dtype=torch.float32)
    y_sample = torch.tensor(y_test[:n_actual], dtype=torch.long)

    results = {}
    elapsed = {}

    # 1. SPARCE
    print(f"  -> Running SPARCE on {n_actual} samples...")
    t0 = time.time()
    sparce = SPARCECounterfactualGenerator(classifier=model)
    res_sparce = sparce.generate_batch(X_sample, y_sample, X_train, y_train, verbose=False)
    elapsed["SPARCE"] = time.time() - t0
    results["SPARCE"] = evaluate_all_metrics(res_sparce, X_train, y_train)

    # 2. Soft-DTW (Ours)
    print(f"  -> Running Soft-DTW (Ours) on {n_actual} samples...")
    t0 = time.time()
    ours = SoftDTWCounterfactualGenerator(classifier=model, num_iterations=CFE_CONFIG["num_iterations"])
    res_ours = ours.generate_batch(X_sample, y_sample, X_train, y_train, verbose=False)
    elapsed["Ours"] = time.time() - t0
    results["Ours"] = evaluate_all_metrics(res_ours, X_train, y_train)

    # 3. M-CELS
    print(f"  -> Running M-CELS on {n_actual} samples...")
    t0 = time.time()
    mcels = MCELS(classifier=model)
    res_mcels = mcels.generate_batch(X_sample, y_sample, X_train, y_train, verbose=False)
    elapsed["M-CELS"] = time.time() - t0
    results["M-CELS"] = evaluate_all_metrics(res_mcels, X_train, y_train)

    # 4. DTW-CFE
    print(f"  -> Running DTW-CFE on {n_actual} samples...")
    t0 = time.time()
    cfg = dict(DTWCFE_CONFIG)
    cfg["seed"] = seed
    dtwcfe = DTWGuidedCFE(classifier=model, **cfg)
    res_dtwcfe = dtwcfe.generate_batch(X_sample, y_sample, X_train, y_train, verbose=False)
    elapsed["DTW-CFE"] = time.time() - t0
    results["DTW-CFE"] = evaluate_all_metrics(res_dtwcfe, X_train, y_train)

    out = {
        "dataset": "PTB_XL",
        "type": "Multivariate_12Lead",
        "d": d, "T": T, "C": c,
        "n_train": len(X_train), "n_test_used": n_actual,
        "methods": results,
        "elapsed": elapsed
    }
    print("  PTB_XL Extended Results:", json.dumps(results, indent=2))
    return out

def main():
    print("[START] Running complete ECG evaluation suite...")
    
    # 1. TwoLeadECG
    twoleadecg_res = run_twoleadecg(n_test=20)
    
    # 2. ECG5000 DTW-CFE
    ecg5000_dtw_res = run_ecg5000_dtwcfe(n_test=25)
    
    # 3. PTB_XL Extended
    ptbxl_res = run_ptbxl_extended(n_test=10)

    # 4. Load existing ECG5000 results and integrate DTW-CFE
    ecg5000_path = os.path.join(RESULTS_DIR, "ecg5000_results.json")
    with open(ecg5000_path, "r") as f:
        ecg5000_data = json.load(f)[0]
    
    ecg5000_data["methods"]["DTW-CFE"] = ecg5000_dtw_res["metrics"]
    ecg5000_data["method_elapsed"]["DTW-CFE"] = ecg5000_dtw_res["elapsed"]
    ecg5000_data["type"] = "Univariate"

    comprehensive = {
        "TwoLeadECG": twoleadecg_res,
        "ECG5000": ecg5000_data,
        "PTB_XL": ptbxl_res
    }

    out_file = os.path.join(RESULTS_DIR, "ecg_comprehensive_results.json")
    with open(out_file, "w") as f:
        json.dump(comprehensive, f, indent=2)
    print(f"\n[DONE] Successfully saved comprehensive ECG results to {out_file}!")

if __name__ == "__main__":
    main()
