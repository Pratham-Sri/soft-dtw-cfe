"""
Full Experiment Runner
======================
Runs the complete experimental pipeline from the paper:
  "Towards plausibility in time series counterfactual explanations"
  Kostrzewa, Galus, Zięba (2026) — arXiv:2603.08349v1

Usage:
    python -m soft_dtw_cfe.run_experiments [--datasets DATASET1 DATASET2 ...]
                                            [--n_test N]
                                            [--skip_optuna]
                                            [--no_glacier]

This script:
  1. Trains/loads a classifier per dataset (with optional Optuna tuning)
  2. Generates CFEs for N test samples using:
       - Ours (Soft-DTW)
       - Glacier (uniform, univariate only)
       - M-CELS
  3. Evaluates all metrics: Val, L1, L2, DTW, IsoForest
  4. Saves results to JSON and a Markdown report
  5. Saves visualisation figures for the first sample per method
"""

import os
import sys
import json
import time
import argparse
import traceback
import numpy as np
import torch

# --- Make sure the package root is on the path ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from soft_dtw_cfe.config import ALL_DATASETS, UNIVARIATE_DATASETS, DEVICE, RESULTS_DIR, CFE_CONFIG
from soft_dtw_cfe.data.dataset_loader import load_dataset, get_dataloaders
from soft_dtw_cfe.models.classifier import (
    TSClassifier, train_classifier, save_classifier,
    load_classifier, evaluate_classifier,
)
from soft_dtw_cfe.models.optuna_tuner import optimise_classifier
from soft_dtw_cfe.methods.proposed_method import SoftDTWCounterfactualGenerator
from soft_dtw_cfe.methods.glacier import GlacierCFE
from soft_dtw_cfe.methods.m_cels import MCELS
from soft_dtw_cfe.evaluation.metrics import evaluate_all_metrics
from soft_dtw_cfe.visualization.plot import (
    plot_counterfactual, plot_loss_curves, plot_metrics_comparison,
)


def _model_path(dataset_name: str) -> str:
    from soft_dtw_cfe.config import MODELS_DIR
    return os.path.join(MODELS_DIR, f"classifier_{dataset_name}.pt")


def setup_classifier(dataset_name, d, c, train_loader, test_loader, skip_optuna=False):
    """Load or train classifier for a dataset."""
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
        model, best_acc, _ = train_classifier(model, train_loader, test_loader, lr=lr, weight_decay=wd)
        save_classifier(model, dataset_name)
        print(f"  -> Best val acc: {best_acc*100:.2f}%")

    return model


def run_dataset(
    dataset_name: str,
    n_test: int = 20,
    skip_optuna: bool = False,
    run_glacier: bool = True,
) -> dict:
    """
    Run the full pipeline for one dataset.
    """
    t_start = time.time()
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'='*60}")

    # 1. Data
    X_train, y_train, X_test, y_test, metadata = load_dataset(dataset_name)
    train_loader, test_loader = get_dataloaders(X_train, y_train, X_test, y_test)

    d = metadata["n_channels"]
    c = metadata["n_classes"]
    T = metadata["seq_len"]

    # 2. Classifier
    model = setup_classifier(dataset_name, d, c, train_loader, test_loader, skip_optuna)
    acc = evaluate_classifier(model, test_loader)
    print(f"  Classifier accuracy: {acc*100:.2f}%")

    # 3. Select test samples
    n_actual = min(n_test, len(X_test))
    X_sample = torch.tensor(X_test[:n_actual], dtype=torch.float32)
    y_sample = torch.tensor(y_test[:n_actual], dtype=torch.long)

    # 4. Generate CFEs
    method_results = {}

    print("\n  [1/3] Proposed Method (Soft-DTW CFE)")
    proposed = SoftDTWCounterfactualGenerator(
        classifier=model,
        num_iterations=CFE_CONFIG["num_iterations"],
    )
    res_proposed = proposed.generate_batch(X_sample, y_sample, X_train, y_train, verbose=True)
    method_results["Ours"] = res_proposed

    if d == 1 and run_glacier:
        print("\n  [2/3] Glacier Method (uniform variant)")
        glacier = GlacierCFE(
            classifier=model,
            seq_len=T,
            ae_epochs=10,
            cfe_iterations=100,
        )
        res_glacier = glacier.generate_batch(X_sample, y_sample, X_train, y_train, verbose=True)
        method_results["Glacier"] = res_glacier
    else:
        if d > 1:
            print("\n  [2/3] Glacier skipped (multivariate dataset)")
        else:
            print("\n  [2/3] Glacier skipped (--no_glacier flag)")

    print("\n  [3/3] M-CELS Method")
    mcels = MCELS(classifier=model, max_iterations=20)
    res_mcels = mcels.generate_batch(X_sample, y_sample, X_train, y_train, verbose=True)
    method_results["M-CELS"] = res_mcels

    # 5. Evaluate metrics
    print(f"\n  Evaluation Results ({n_actual} samples):")
    print(f"  {'Method':<12} {'Val':>6} {'L1':>8} {'L2':>8} {'DTW':>10} {'IsoForest':>10}")
    print(f"  {'-'*60}")

    all_metrics = {}
    for mname, mresults in method_results.items():
        metrics = evaluate_all_metrics(mresults, X_train, y_train)
        all_metrics[mname] = metrics
        v   = metrics.get("Val", 0)
        l1  = metrics.get("L1", 0)
        l2  = metrics.get("L2", 0)
        dtw = metrics.get("DTW", 0)
        iso = metrics.get("IsoForest", 0)
        print(f"  {mname:<12} {v:>6.3f} {l1:>8.4f} {l2:>8.4f} {dtw:>10.4f} {iso:>10.3f}")

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
            save_name = f"{dataset_name}_{mname}_cf"
            plot_counterfactual(
                r["original"],
                r["counterfactual"],
                target_samps,
                dataset_name,
                mname,
                r["original_class"],
                r["target_class"],
                save_name,
            )

            # Plot loss curves if available
            if "losses" in r:
                plot_loss_curves(r["losses"], dataset_name, mname)

    # Metrics comparison bar chart
    plot_metrics_comparison(all_metrics, dataset_name)

    elapsed = time.time() - t_start
    print(f"\n  Done in {elapsed:.1f}s")

    return {
        "dataset": dataset_name,
        "accuracy": acc,
        "n_train": metadata["n_train"],
        "n_test_used": n_actual,
        "seq_len": T,
        "n_channels": d,
        "n_classes": c,
        "methods": all_metrics,
        "elapsed": elapsed,
    }


def generate_markdown_report(all_results: list, output_path: str):
    """Write a detailed Markdown report with all results."""
    lines = [
        "# Soft-DTW Counterfactual Explanations — Experimental Results",
        "",
        "> **Paper**: *Towards plausibility in time series counterfactual explanations*  ",
        "> Kostrzewa, Galus, Zięba (2026) — [arXiv:2603.08349](https://arxiv.org/abs/2603.08349)",
        "",
        "## Evaluation Metrics",
        "",
        "| Metric | Symbol | Direction | Description |",
        "|--------|--------|-----------|-------------|",
        "| Validity | Val | up higher is better | Fraction of CFEs that flip the classifier |",
        "| Sparsity | L1 | down lower is better | Normalised L1 distance to original |",
        "| Proximity | L2 | down lower is better | Normalised L2 distance to original |",
        "| DTW Plausibility | DTW | down lower is better | Avg DTW to 10 target-class neighbours |",
        "| Isolation Forest | IsoForest | up higher is better | Fraction classified as nominal by IF |",
        "",
        "---",
        "",
        "## Results per Dataset",
        "",
    ]

    for res in all_results:
        ds = res["dataset"]
        err = res.get("error", None)
        lines += [
            f"### {ds}",
            "",
        ]
        if err:
            lines += [f"**ERROR**: `{err}`", "", ""]
            continue

        lines += [
            f"- **Classifier Accuracy**: {res['accuracy']*100:.2f}%",
            f"- **Train / Test samples used**: {res['n_train']} / {res['n_test_used']}",
            f"- **Series**: T={res['seq_len']}, d={res['n_channels']}, C={res['n_classes']}",
            f"- **Elapsed**: {res['elapsed']:.1f}s",
            "",
            "| Method | Val (up) | L1 (down) | L2 (down) | DTW (down) | IsoForest (up) |",
            "|--------|----------|-----------|-----------|------------|----------------|",
        ]

        for mname, m in res["methods"].items():
            v   = m.get("Val", 0)
            l1  = m.get("L1", 0)
            l2  = m.get("L2", 0)
            dtw = m.get("DTW", 0)
            iso = m.get("IsoForest", 0)
            lines.append(f"| {mname} | {v:.3f} | {l1:.4f} | {l2:.4f} | {dtw:.4f} | {iso:.3f} |")

        lines += ["", ""]

    lines += [
        "---",
        "",
        "## Summary: Ours vs Baselines",
        "",
        "| Dataset | Acc% | Ours Val | Glacier Val | M-CELS Val | Ours DTW | Glacier DTW | M-CELS DTW |",
        "|---------|------|----------|-------------|------------|----------|-------------|------------|",
    ]

    for res in all_results:
        if res.get("error"):
            lines.append(f"| {res['dataset']} | ERROR | - | - | - | - | - | - |")
            continue
        acc = res["accuracy"] * 100
        m = res["methods"]
        ours_v = f"{m['Ours']['Val']:.3f}"        if "Ours" in m else "-"
        glac_v = f"{m['Glacier']['Val']:.3f}"     if "Glacier" in m else "N/A"
        mcel_v = f"{m['M-CELS']['Val']:.3f}"      if "M-CELS" in m else "-"
        ours_d = f"{m['Ours']['DTW']:.4f}"        if "Ours" in m else "-"
        glac_d = f"{m['Glacier']['DTW']:.4f}"     if "Glacier" in m else "N/A"
        mcel_d = f"{m['M-CELS']['DTW']:.4f}"      if "M-CELS" in m else "-"
        lines.append(
            f"| {res['dataset']} | {acc:.1f} | {ours_v} | {glac_v} | {mcel_v} | {ours_d} | {glac_d} | {mcel_d} |"
        )

    lines += [
        "",
        "---",
        "*Generated automatically by `soft_dtw_cfe/run_experiments.py`*",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[REPORT] Markdown report saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Soft-DTW CFE Experiment Runner")
    parser.add_argument(
        "--datasets", nargs="+", default=ALL_DATASETS,
        help="Datasets to run (default: all 8)",
    )
    parser.add_argument(
        "--n_test", type=int, default=20,
        help="Number of test samples to generate CFEs for (default: 20)",
    )
    parser.add_argument(
        "--skip_optuna", action="store_true",
        help="Skip Optuna tuning and use default hyperparameters",
    )
    parser.add_argument(
        "--no_glacier", action="store_true",
        help="Skip Glacier method",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Device: {DEVICE}")
    print(f"Datasets: {args.datasets}")
    print(f"Test samples per dataset: {args.n_test}")
    print(f"Skip Optuna: {args.skip_optuna}")

    all_results = []

    for ds in args.datasets:
        try:
            result = run_dataset(
                ds,
                n_test=args.n_test,
                skip_optuna=args.skip_optuna,
                run_glacier=(not args.no_glacier),
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n[ERROR] Failed on {ds}: {e}")
            traceback.print_exc()
            all_results.append({
                "dataset": ds,
                "accuracy": 0.0,
                "n_train": 0, "n_test_used": 0,
                "seq_len": 0, "n_channels": 0, "n_classes": 0,
                "methods": {},
                "elapsed": 0.0,
                "error": str(e),
            })

    # Save JSON results
    json_path = os.path.join(RESULTS_DIR, "all_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[SAVE] JSON results saved to {json_path}")

    # Generate Markdown report
    report_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "RESULTS.md"
    )
    generate_markdown_report(all_results, report_path)

    print("\n" + "="*60)
    print("  All experiments complete!")
    print("="*60)


if __name__ == "__main__":
    main()
