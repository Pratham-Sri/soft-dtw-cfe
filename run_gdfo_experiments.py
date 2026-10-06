"""
GDFO Experiment Runner
======================
Runs the Generative Density Function Optimisation (GDFO) experiments on
larger multivariate datasets and compares with the existing approaches.

Usage:
    python -m soft_dtw_cfe.run_gdfo_experiments [--datasets DATASET1 ...]
                                                  [--n_test N]
                                                  [--skip_optuna]

This script:
  1. Loads larger multivariate datasets from the UEA archive
  2. Trains/loads a classifier per dataset
  3. Generates CFEs using:
       - Ours (Soft-DTW)         — the paper's proposed method
       - GDFO (density-guided)   — the new generative density approach
       - M-CELS                  — baseline
       - DTW-CFE                 — DTW-guided constrained deformation
  4. Evaluates: Val, L1, L2, DTW, IsoForest
  5. Produces a comparison report and figures
"""

import os
import sys
import json
import time
import argparse
import traceback
import numpy as np
import torch

# --- Make the repo importable as `soft_dtw_cfe` ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
import types as _types
if "soft_dtw_cfe" not in sys.modules:
    _pkg = _types.ModuleType("soft_dtw_cfe")
    _pkg.__path__ = [_THIS_DIR]
    sys.modules["soft_dtw_cfe"] = _pkg

from soft_dtw_cfe.config import (
    GDFO_DATASETS, MULTIVARIATE_DATASETS, LARGE_MULTIVARIATE_DATASETS,
    ALL_DATASETS, DEVICE, RESULTS_DIR, CFE_CONFIG, DTWCFE_CONFIG,
    GDFO_CONFIG,
)
from soft_dtw_cfe.data.dataset_loader import load_dataset, get_dataloaders
from soft_dtw_cfe.models.classifier import (
    TSClassifier, train_classifier, save_classifier,
    load_classifier, evaluate_classifier,
)
from soft_dtw_cfe.models.optuna_tuner import optimise_classifier
from soft_dtw_cfe.methods.proposed_method import SoftDTWCounterfactualGenerator
from soft_dtw_cfe.methods.gdfo import GDFOCounterfactualGenerator
from soft_dtw_cfe.methods.dtw_guided.dtw_cfe import DTWGuidedCFE
from soft_dtw_cfe.methods.m_cels import MCELS
from soft_dtw_cfe.evaluation.metrics import evaluate_all_metrics
from soft_dtw_cfe.visualization.plot import (
    plot_counterfactual, plot_loss_curves, plot_metrics_comparison,
    plot_paper_metrics,
)


def _model_path(dataset_name: str) -> str:
    from soft_dtw_cfe.config import MODELS_DIR
    return os.path.join(MODELS_DIR, f"classifier_{dataset_name}.pt")


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_classifier(dataset_name, d, c, train_loader, test_loader,
                     skip_optuna=False):
    """Load or train classifier."""
    path = _model_path(dataset_name)

    if os.path.exists(path):
        print(f"  -> Loading existing classifier from {path}")
        model = TSClassifier(n_channels=d, n_classes=c, dropout=0.3)
        model = load_classifier(model, dataset_name)
    else:
        print(f"  -> Training new classifier (skip_optuna={skip_optuna}) ...")
        if not skip_optuna:
            best_params = optimise_classifier(d, c, train_loader, test_loader)
            dropout = best_params["dropout"]
            lr = best_params["lr"]
            wd = best_params["weight_decay"]
        else:
            dropout, lr, wd = 0.3, 1e-3, 1e-4

        model = TSClassifier(n_channels=d, n_classes=c, dropout=dropout)
        model, best_acc, _ = train_classifier(
            model, train_loader, test_loader, lr=lr, weight_decay=wd
        )
        save_classifier(model, dataset_name)
        print(f"  -> Best val acc: {best_acc*100:.2f}%")

    return model


def run_dataset(
    dataset_name: str,
    n_test: int = 20,
    skip_optuna: bool = True,
    run_dtwcfe: bool = True,
    seed: int = 42,
) -> dict:
    """
    Run the full GDFO comparison pipeline for one dataset.

    Methods tested:
      1. Ours (Soft-DTW) — existing proposed method
      2. GDFO            — generative density function optimisation (new)
      3. M-CELS          — baseline
      4. DTW-CFE         — DTW-guided constrained deformation
    """
    set_seed(seed)
    t_start = time.time()
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name} (seed={seed})")
    print(f"{'='*60}")

    # 1. Data
    X_train, y_train, X_test, y_test, metadata = load_dataset(dataset_name)
    train_loader, test_loader = get_dataloaders(X_train, y_train, X_test, y_test)

    d = metadata["n_channels"]
    c = metadata["n_classes"]
    T = metadata["seq_len"]

    print(f"  Shape: n_train={metadata['n_train']}, n_test={metadata['n_test']}, "
          f"d={d}, T={T}, C={c}")

    # 2. Classifier
    model = setup_classifier(
        dataset_name, d, c, train_loader, test_loader, skip_optuna
    )
    acc = evaluate_classifier(model, test_loader)
    print(f"  Classifier accuracy: {acc*100:.2f}%")

    # 3. Select test samples
    n_actual = min(n_test, len(X_test))
    X_sample = torch.tensor(X_test[:n_actual], dtype=torch.float32)
    y_sample = torch.tensor(y_test[:n_actual], dtype=torch.long)

    # 4. Generate CFEs
    method_results = {}
    method_elapsed = {}

    # ── 4a. Proposed Method (Soft-DTW) ────────────────────────────────────
    print("\n  [1/4] Proposed Method (Soft-DTW CFE)")
    proposed = SoftDTWCounterfactualGenerator(
        classifier=model,
        num_iterations=CFE_CONFIG["num_iterations"],
    )
    _t0 = time.time()
    res_proposed = proposed.generate_batch(
        X_sample, y_sample, X_train, y_train, verbose=True
    )
    method_results["Ours"] = res_proposed
    method_elapsed["Ours"] = time.time() - _t0

    # ── 4b. GDFO (new) ───────────────────────────────────────────────────
    print("\n  [2/4] GDFO (Generative Density Function Optimisation)")
    gdfo = GDFOCounterfactualGenerator(
        classifier=model,
        n_channels=d,
        seq_len=T,
        z_dim=GDFO_CONFIG["z_dim"],
        vae_epochs=GDFO_CONFIG["vae_epochs"],
        vae_beta=GDFO_CONFIG["vae_beta"],
        gmm_components=GDFO_CONFIG["gmm_components"],
        lambda_valid=GDFO_CONFIG["lambda_valid"],
        lambda_density=GDFO_CONFIG["lambda_density"],
        lambda_dtw=GDFO_CONFIG["lambda_dtw"],
        tau=GDFO_CONFIG["tau"],
        lr=GDFO_CONFIG["lr"],
        num_iterations=GDFO_CONFIG["num_iterations"],
        k=GDFO_CONFIG["k"],
        gamma=GDFO_CONFIG["gamma"],
        use_dtw=GDFO_CONFIG["use_dtw"],
    )
    _t0 = time.time()
    res_gdfo = gdfo.generate_batch(
        X_sample, y_sample, X_train, y_train, verbose=True
    )
    method_results["GDFO"] = res_gdfo
    method_elapsed["GDFO"] = time.time() - _t0

    # ── 4c. M-CELS ───────────────────────────────────────────────────────
    print("\n  [3/4] M-CELS Method")
    mcels = MCELS(classifier=model, max_iterations=20)
    _t0 = time.time()
    res_mcels = mcels.generate_batch(
        X_sample, y_sample, X_train, y_train, verbose=True
    )
    method_results["M-CELS"] = res_mcels
    method_elapsed["M-CELS"] = time.time() - _t0

    # ── 4d. DTW-CFE ──────────────────────────────────────────────────────
    if run_dtwcfe:
        print("\n  [4/4] DTW-Guided Constrained Deformation")
        try:
            cfg = dict(DTWCFE_CONFIG)
            cfg["seed"] = seed
            dtwcfe = DTWGuidedCFE(classifier=model, **cfg)
            _t0 = time.time()
            res_dtwcfe = dtwcfe.generate_batch(
                X_sample, y_sample, X_train, y_train, verbose=True
            )
            method_results["DTW-CFE"] = res_dtwcfe
            method_elapsed["DTW-CFE"] = time.time() - _t0
        except Exception as e:
            print(f"\n  [DTW-CFE ERROR] {e}")
            traceback.print_exc()
    else:
        print("\n  [4/4] DTW-CFE skipped")

    # 5. Evaluate metrics
    print(f"\n  {'Method':<12} {'Val':>6} {'L1':>8} {'L2':>8} "
          f"{'DTW':>10} {'IsoForest':>10} {'Time(s)':>9} {'s/sample':>9}")
    print(f"  {'-'*77}")

    all_metrics = {}
    for mname, mresults in method_results.items():
        metrics = evaluate_all_metrics(mresults, X_train, y_train)
        all_metrics[mname] = metrics
        v   = metrics.get("Val", 0)
        l1  = metrics.get("L1", 0)
        l2  = metrics.get("L2", 0)
        dtw = metrics.get("DTW", 0)
        iso = metrics.get("IsoForest", 0)
        t = method_elapsed.get(mname, 0.0)
        print(f"  {mname:<12} {v:>6.3f} {l1:>8.4f} {l2:>8.4f} "
              f"{dtw:>10.4f} {iso:>10.3f} {t:>9.1f} {t/max(n_actual,1):>9.2f}")

    # 6. Visualizations
    unique_classes = np.unique(y_train)
    target_samples_dict = {}
    for cl in unique_classes:
        mask = (y_train == int(cl))
        target_samples_dict[cl] = torch.tensor(X_train[mask], dtype=torch.float32)

    for mname, mresults in method_results.items():
        if mresults:
            r = mresults[0]
            target_samps = target_samples_dict[int(r["target_class"])]
            save_name = f"{dataset_name}_{mname}_gdfo_cf"
            plot_counterfactual(
                r["original"], r["counterfactual"], target_samps,
                dataset_name, mname,
                r["original_class"], r["target_class"],
                save_name,
            )
            if "losses" in r:
                plot_loss_curves(r["losses"], dataset_name, f"{mname}_gdfo")

    plot_metrics_comparison(all_metrics, f"{dataset_name}_gdfo")
    plot_paper_metrics(all_metrics, f"{dataset_name}_gdfo")

    elapsed = time.time() - t_start
    print(f"\n  Done in {elapsed:.1f}s")

    return {
        "dataset": dataset_name,
        "seed": seed,
        "accuracy": acc,
        "n_train": metadata["n_train"],
        "n_test_used": n_actual,
        "seq_len": T,
        "n_channels": d,
        "n_classes": c,
        "methods": all_metrics,
        "method_elapsed": method_elapsed,
        "elapsed": elapsed,
    }


def generate_gdfo_report(all_results: list, output_path: str):
    """Write a Markdown comparison report for GDFO experiments."""
    lines = [
        "# GDFO Comparison — Generative Density Function Optimisation",
        "",
        "> Comparing the new GDFO approach with existing methods on larger",
        "> multivariate time series datasets from the UEA archive.",
        "",
        "## Methods Compared",
        "",
        "| Method | Description |",
        "|--------|-------------|",
        "| **Ours** (Soft-DTW) | Original proposed method — gradient-based soft-DTW alignment |",
        "| **GDFO** | Generative density function optimisation — VAE+GMM density-guided |",
        "| **M-CELS** | Baseline — greedy class-conditional perturbation |",
        "| **DTW-CFE** | DTW-guided constrained deformation via CMA-ES |",
        "",
        "## Evaluation Metrics",
        "",
        "| Metric | Direction | Description |",
        "|--------|-----------|-------------|",
        "| Val | ↑ higher is better | Fraction of CFEs that flip the classifier |",
        "| L1 | ↓ lower is better | Normalised L1 distance (sparsity) |",
        "| L2 | ↓ lower is better | Normalised L2 distance (proximity) |",
        "| DTW | ↓ lower is better | Avg DTW to 10 target-class neighbours (plausibility) |",
        "| IsoForest | ↑ higher is better | Fraction classified as nominal by Isolation Forest |",
        "",
        "---",
        "",
        "## Results per Dataset",
        "",
    ]

    for res in all_results:
        ds = res["dataset"]
        err = res.get("error", None)
        lines += [f"### {ds}", ""]

        if err:
            lines += [f"**ERROR**: `{err}`", "", ""]
            continue

        lines += [
            f"- **Classifier Accuracy**: {res['accuracy']*100:.2f}%",
            f"- **Train / Test samples used**: {res['n_train']} / {res['n_test_used']}",
            f"- **Series**: T={res['seq_len']}, d={res['n_channels']}, C={res['n_classes']}",
            f"- **Elapsed**: {res['elapsed']:.1f}s",
            "",
            "| Method | Val ↑ | L1 ↓ | L2 ↓ | DTW ↓ | IsoForest ↑ | Time(s) | s/sample |",
            "|--------|-------|------|------|-------|-------------|---------|----------|",
        ]

        mel = res.get("method_elapsed", {})
        n_used = max(res.get("n_test_used", 1), 1)
        for mname, m in res["methods"].items():
            v   = m.get("Val", 0)
            l1  = m.get("L1", 0)
            l2  = m.get("L2", 0)
            dtw = m.get("DTW", 0)
            iso = m.get("IsoForest", 0)
            t = mel.get(mname, 0.0)
            # Bold the best values
            lines.append(
                f"| {mname} | {v:.3f} | {l1:.4f} | {l2:.4f} | "
                f"{dtw:.4f} | {iso:.3f} | {t:.1f} | {t/n_used:.2f} |"
            )

        lines += ["", ""]

    # Summary comparison table
    lines += [
        "---",
        "",
        "## Summary: GDFO vs Previous Approaches",
        "",
        "| Dataset | d | T | C | Acc% | Ours Val | GDFO Val | Ours DTW | GDFO DTW | Ours IsoF | GDFO IsoF | GDFO s/samp |",
        "|---------|---|---|---|------|----------|----------|----------|----------|-----------|-----------|-------------|",
    ]

    for res in all_results:
        if res.get("error"):
            lines.append(f"| {res['dataset']} | - | - | - | ERROR | - | - | - | - | - | - | - |")
            continue
        m = res["methods"]
        mel = res.get("method_elapsed", {})
        n = max(res.get("n_test_used", 1), 1)
        d, T, c = res["n_channels"], res["seq_len"], res["n_classes"]
        acc = res["accuracy"] * 100

        ov  = f"{m['Ours']['Val']:.3f}"  if "Ours" in m else "-"
        gv  = f"{m['GDFO']['Val']:.3f}"  if "GDFO" in m else "-"
        od  = f"{m['Ours']['DTW']:.2f}"  if "Ours" in m else "-"
        gd  = f"{m['GDFO']['DTW']:.2f}"  if "GDFO" in m else "-"
        oi  = f"{m['Ours']['IsoForest']:.3f}" if "Ours" in m else "-"
        gi  = f"{m['GDFO']['IsoForest']:.3f}" if "GDFO" in m else "-"
        gt  = f"{mel.get('GDFO', 0)/n:.2f}" if "GDFO" in m else "-"

        lines.append(
            f"| {res['dataset']} | {d} | {T} | {c} | {acc:.1f} | "
            f"{ov} | {gv} | {od} | {gd} | {oi} | {gi} | {gt} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Key Observations",
        "",
        "1. **Density-guided plausibility**: GDFO's VAE+GMM density term provides a ",
        "   smooth gradient signal from the full class distribution, not just k nearest ",
        "   neighbours. This typically improves IsoForest scores (nominal fraction).",
        "",
        "2. **Multivariate scaling**: GDFO jointly encodes all channels, capturing ",
        "   cross-channel correlations that per-channel methods may miss.",
        "",
        "3. **Trade-offs**: GDFO adds VAE training overhead but reuses the density model ",
        "   across all test samples, amortising the cost for larger test sets.",
        "",
        "---",
        "*Generated automatically by `soft_dtw_cfe/run_gdfo_experiments.py`*",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[REPORT] GDFO comparison report saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="GDFO Experiment Runner — density-guided CFE comparison"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=GDFO_DATASETS,
        help="Datasets to run (default: all multivariate + large multivariate)",
    )
    parser.add_argument(
        "--n_test", type=int, default=20,
        help="Number of test samples to generate CFEs for (default: 20)",
    )
    parser.add_argument(
        "--skip_optuna", action="store_true", default=True,
        help="Skip Optuna tuning (default: True for speed)",
    )
    parser.add_argument(
        "--no_dtwcfe", action="store_true",
        help="Skip DTW-CFE method",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Device: {DEVICE}")
    print(f"Datasets: {args.datasets}")
    print(f"Test samples per dataset: {args.n_test}")
    print(f"Seed: {args.seed}")
    print(f"{'='*60}")
    print(f"  GDFO COMPARISON EXPERIMENT")
    print(f"{'='*60}")

    all_results = []

    for ds in args.datasets:
        try:
            result = run_dataset(
                ds,
                n_test=args.n_test,
                skip_optuna=args.skip_optuna,
                run_dtwcfe=(not args.no_dtwcfe),
                seed=args.seed,
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n[ERROR] Failed on {ds}: {e}")
            traceback.print_exc()
            all_results.append({
                "dataset": ds,
                "seed": args.seed,
                "accuracy": 0.0,
                "n_train": 0, "n_test_used": 0,
                "seq_len": 0, "n_channels": 0, "n_classes": 0,
                "methods": {},
                "method_elapsed": {},
                "elapsed": 0.0,
                "error": str(e),
            })

    # Save JSON results
    json_path = os.path.join(RESULTS_DIR, "gdfo_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[SAVE] JSON results saved to {json_path}")

    # Generate Markdown report
    report_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "RESULTS_GDFO.md"
    )
    generate_gdfo_report(all_results, report_path)

    print("\n" + "=" * 60)
    print("  GDFO experiments complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
