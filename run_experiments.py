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

# --- Make the repo importable as `soft_dtw_cfe` despite hyphenated dir name ---
# The repo root (this file's dir) IS the package, but `soft_dtw_cfe`
# imports expect a package of that name. Alias it here before any such import.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
import types as _types
if "soft_dtw_cfe" not in sys.modules:
    _pkg = _types.ModuleType("soft_dtw_cfe")
    _pkg.__path__ = [_THIS_DIR]
    sys.modules["soft_dtw_cfe"] = _pkg

from soft_dtw_cfe.config import ALL_DATASETS, UNIVARIATE_DATASETS, DEVICE, RESULTS_DIR, CFE_CONFIG, DTWCFE_CONFIG
from soft_dtw_cfe.data.dataset_loader import load_dataset, get_dataloaders
from soft_dtw_cfe.models.classifier import (
    TSClassifier, train_classifier, save_classifier,
    load_classifier, evaluate_classifier,
)
from soft_dtw_cfe.models.optuna_tuner import optimise_classifier
from soft_dtw_cfe.methods.proposed_method import SoftDTWCounterfactualGenerator
from soft_dtw_cfe.methods.dtw_guided.dtw_cfe import DTWGuidedCFE
from soft_dtw_cfe.methods.glacier import GlacierCFE
from soft_dtw_cfe.methods.m_cels import MCELS
from soft_dtw_cfe.evaluation.metrics import evaluate_all_metrics
from soft_dtw_cfe.visualization.plot import (
    plot_counterfactual, plot_loss_curves, plot_metrics_comparison,
    plot_paper_metrics, plot_all_counterfactuals_grid,
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


def set_seed(seed: int):
    """Seed python/numpy/torch for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_dataset(
    dataset_name: str,
    n_test: int = 20,
    skip_optuna: bool = False,
    run_glacier: bool = True,
    run_dtwcfe: bool = True,
    seed: int = 42,
) -> dict:
    """
    Run the full pipeline for one dataset.
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
    method_elapsed = {}

    print("\n  [1/4] Proposed Method (Soft-DTW CFE)")
    proposed = SoftDTWCounterfactualGenerator(
        classifier=model,
        num_iterations=CFE_CONFIG["num_iterations"],
    )
    _t0 = time.time()
    res_proposed = proposed.generate_batch(X_sample, y_sample, X_train, y_train, verbose=True)
    method_results["Ours"] = res_proposed
    method_elapsed["Ours"] = time.time() - _t0

    if d == 1 and run_glacier:
        print("\n  [2/4] Glacier Method (uniform variant)")
        glacier = GlacierCFE(
            classifier=model,
            seq_len=T,
            ae_epochs=10,
            cfe_iterations=100,
        )
        _t0 = time.time()
        res_glacier = glacier.generate_batch(X_sample, y_sample, X_train, y_train, verbose=True)
        method_results["Glacier"] = res_glacier
        method_elapsed["Glacier"] = time.time() - _t0
    else:
        if d > 1:
            print("\n  [2/4] Glacier skipped (multivariate dataset)")
        else:
            print("\n  [2/4] Glacier skipped (--no_glacier flag)")

    print("\n  [3/4] M-CELS Method")
    mcels = MCELS(classifier=model, max_iterations=20)
    _t0 = time.time()
    res_mcels = mcels.generate_batch(X_sample, y_sample, X_train, y_train, verbose=True)
    method_results["M-CELS"] = res_mcels
    method_elapsed["M-CELS"] = time.time() - _t0

    if run_dtwcfe:
        print("\n  [4/4] DTW-Guided Constrained Deformation (novelty)")
        try:
            cfg = dict(DTWCFE_CONFIG)
            cfg["seed"] = seed
            dtwcfe = DTWGuidedCFE(classifier=model, **cfg)
            _t0 = time.time()
            res_dtwcfe = dtwcfe.generate_batch(X_sample, y_sample, X_train, y_train, verbose=True)
            method_results["DTW-CFE"] = res_dtwcfe
            method_elapsed["DTW-CFE"] = time.time() - _t0
        except Exception as e:
            print(f"\n  [DTW-CFE ERROR] {e}")
            traceback.print_exc()
    else:
        print("\n  [4/4] DTW-CFE skipped (--no_dtwcfe flag)")

    # 5. Evaluate metrics
    print(f"\n  Evaluation Results ({n_actual} samples):")
    print(f"  {'Method':<12} {'Val':>6} {'L1':>8} {'L2':>8} {'DTW':>10} {'IsoForest':>10} {'Time(s)':>9} {'s/sample':>9}")
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
        print(f"  {mname:<12} {v:>6.3f} {l1:>8.4f} {l2:>8.4f} {dtw:>10.4f} {iso:>10.3f} {t:>9.1f} {t/max(n_actual,1):>9.2f}")

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
    plot_paper_metrics(all_metrics, dataset_name)
    # Paper §5.3 style side-by-side grid (univariate only)
    if d == 1 and len(method_results) > 1:
        try:
            plot_all_counterfactuals_grid(method_results, dataset_name)
        except Exception as e:
            print(f"  [VIS] grid plot skipped: {e}")

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


def aggregate_seeds(results_by_seed: dict) -> list:
    """
    Aggregate per-seed result dicts into mean±std per dataset/method/metric.

    Args:
        results_by_seed: {seed: [run_dataset dicts]}
    Returns:
        list of {dataset, accuracy_mean/std, n, seq_len, ..., methods: {m: {metric: {mean, std}}},
                 method_elapsed: {m: {mean, std}}} with per-sample timing derived.
    """
    from collections import defaultdict
    grouped = defaultdict(list)
    for seed, res_list in results_by_seed.items():
        for r in res_list:
            if r.get("error"):
                continue
            grouped[r["dataset"]].append(r)

    summary = []
    for ds, runs in grouped.items():
        n_seeds = len(runs)
        methods = sorted({m for r in runs for m in r["methods"]})
        agg_methods, agg_el = {}, {}
        n_used = runs[0]["n_test_used"]
        for m in methods:
            agg_methods[m] = {}
            for metric in ("Val", "Val_any", "L1", "L2", "L1_raw",
                           "L2_raw", "DTW", "IsoForest"):
                vals = np.array([float(r["methods"][m].get(metric, 0))
                                 for r in runs if m in r["methods"]])
                agg_methods[m][metric] = {"mean": float(vals.mean()),
                                          "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}
            tvals = np.array([float(r.get("method_elapsed", {}).get(m, 0.0))
                              for r in runs if m in r.get("method_elapsed", {})])
            if len(tvals):
                agg_el[m] = {"mean": float(tvals.mean()),
                             "std": float(tvals.std(ddof=1)) if len(tvals) > 1 else 0.0,
                             "per_sample_mean": float(tvals.mean() / max(n_used, 1))}
            else:
                agg_el[m] = {"mean": 0.0, "std": 0.0, "per_sample_mean": 0.0}
        accs = np.array([r["accuracy"] for r in runs])
        summary.append({
            "dataset": ds,
            "n_seeds": n_seeds,
            "accuracy_mean": float(accs.mean()),
            "accuracy_std": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
            "n_test_used": n_used,
            "seq_len": runs[0]["seq_len"],
            "n_channels": runs[0]["n_channels"],
            "n_classes": runs[0]["n_classes"],
            "methods": agg_methods,
            "method_elapsed": agg_el,
        })
    return summary


def plot_scaling(summary: list, save_name: str = "scaling_time_vs_T"):
    """Log-log s/sample vs T for Ours / M-CELS / DTW-CFE with std bars."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[PLOT] matplotlib missing, skipping scaling plot")
        return
    from soft_dtw_cfe.config import FIGURES_DIR
    order = sorted(summary, key=lambda r: r["seq_len"])
    fig, ax = plt.subplots(figsize=(10, 6))
    for mname, color, marker in (("Ours", "#2c5f8a", "o"),
                                 ("M-CELS", "#5aa469", "s"),
                                 ("DTW-CFE", "#e05c5c", "^")):
        xs, ys, yerr, labels = [], [], [], []
        for r in order:
            if mname in r["method_elapsed"]:
                xs.append(r["seq_len"])
                ys.append(r["method_elapsed"][mname]["per_sample_mean"])
                n = max(r["n_seeds"], 1)
                se = r["method_elapsed"][mname]["std"] / max(r["n_test_used"], 1) / np.sqrt(n)
                yerr.append(se)
                labels.append(r["dataset"])
        if xs:
            ax.errorbar(xs, ys, yerr=yerr, label=mname, color=color,
                        marker=marker, capsize=4, linewidth=1.8)
            for x, y, lb in zip(xs, ys, labels):
                ax.annotate(lb, (x, y), fontsize=7, xytext=(4, 4),
                            textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sequence length T (log)")
    ax.set_ylabel("Seconds / sample (log)")
    ax.set_title("Scaling: per-sample CFE time vs sequence length (mean ± SE over seeds)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"{save_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  [VIS] Saved scaling figure -> {path}")
    plt.close(fig)


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
            "| Method | Val (up) | L1 (down) | L2 (down) | DTW (down) | IsoForest (up) | Time(s) | s/sample |",
            "|--------|----------|-----------|-----------|------------|----------------|---------|----------|",
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
            lines.append(f"| {mname} | {v:.3f} | {l1:.4f} | {l2:.4f} | {dtw:.4f} | {iso:.3f} | {t:.1f} | {t/n_used:.2f} |")

        lines += ["", ""]

    lines += [
        "---",
        "",
        "## Summary: Ours vs Baselines",
        "",
        "| Dataset | Acc% | Ours Val | M-CELS Val | DTW-CFE Val | Ours DTW | M-CELS DTW | DTW-CFE DTW | Ours s/samp | DTW-CFE s/samp |",
        "|---------|------|----------|------------|-------------|----------|------------|-------------|-------------|----------------|",
    ]

    for res in all_results:
        if res.get("error"):
            lines.append(f"| {res['dataset']} | ERROR | - | - | - | - | - | - | - | - |")
            continue
        acc = res["accuracy"] * 100
        m = res["methods"]
        mel = res.get("method_elapsed", {})
        n_used = max(res.get("n_test_used", 1), 1)
        ours_v = f"{m['Ours']['Val']:.3f}"        if "Ours" in m else "-"
        mcel_v = f"{m['M-CELS']['Val']:.3f}"      if "M-CELS" in m else "-"
        dcfe_v = f"{m['DTW-CFE']['Val']:.3f}"     if "DTW-CFE" in m else "-"
        ours_d = f"{m['Ours']['DTW']:.4f}"        if "Ours" in m else "-"
        mcel_d = f"{m['M-CELS']['DTW']:.4f}"      if "M-CELS" in m else "-"
        dcfe_d = f"{m['DTW-CFE']['DTW']:.4f}"     if "DTW-CFE" in m else "-"
        ours_t = f"{mel.get('Ours', 0.0)/n_used:.2f}" if "Ours" in m else "-"
        dcfe_t = f"{mel.get('DTW-CFE', 0.0)/n_used:.2f}" if "DTW-CFE" in m else "-"
        lines.append(
            f"| {res['dataset']} | {acc:.1f} | {ours_v} | {mcel_v} | {dcfe_v} | {ours_d} | {mcel_d} | {dcfe_d} | {ours_t} | {dcfe_t} |"
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
    parser.add_argument(
        "--no_dtwcfe", action="store_true",
        help="Skip DTW-CFE novelty method",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42],
        help="Random seeds to repeat each dataset with (default: 42)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Device: {DEVICE}")
    print(f"Datasets: {args.datasets}")
    print(f"Test samples per dataset: {args.n_test}")
    print(f"Skip Optuna: {args.skip_optuna}")
    print(f"Seeds: {args.seeds}")

    results_by_seed = {}

    for seed in args.seeds:
        print(f"\n{'#'*60}\n  SEED {seed}\n{'#'*60}")
        all_results = []

        for ds in args.datasets:
            try:
                result = run_dataset(
                    ds,
                    n_test=args.n_test,
                    skip_optuna=args.skip_optuna,
                    run_glacier=(not args.no_glacier),
                    run_dtwcfe=(not args.no_dtwcfe),
                    seed=seed,
                )
                all_results.append(result)
            except Exception as e:
                print(f"\n[ERROR] Failed on {ds}: {e}")
                traceback.print_exc()
                all_results.append({
                    "dataset": ds,
                    "seed": seed,
                    "accuracy": 0.0,
                    "n_train": 0, "n_test_used": 0,
                    "seq_len": 0, "n_channels": 0, "n_classes": 0,
                    "methods": {},
                    "method_elapsed": {},
                    "elapsed": 0.0,
                    "error": str(e),
                })

        results_by_seed[seed] = all_results

        # Save per-seed JSON results
        json_path = os.path.join(RESULTS_DIR, f"all_results_seed{seed}.json")
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n[SAVE] JSON results saved to {json_path}")
        # Backward-compat copy of first seed
        if seed == args.seeds[0]:
            compat = os.path.join(RESULTS_DIR, "all_results.json")
            with open(compat, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

    # Backward-compat single-seed report on first seed
    first = args.seeds[0]
    report_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "RESULTS.md"
    )
    generate_markdown_report(results_by_seed[first], report_path)

    # Multi-seed aggregation + scaling plot + aggregated report
    if len(args.seeds) > 1:
        summary = aggregate_seeds(results_by_seed)
        agg_path = os.path.join(RESULTS_DIR, "aggregated_results.json")
        with open(agg_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n[SAVE] Aggregated results saved to {agg_path}")
        plot_scaling(summary)
        # Mean-value metrics comparison plot per dataset
        for entry in summary:
            means = {m: {k: v["mean"] for k, v in mets.items()}
                     for m, mets in entry["methods"].items()}
            plot_metrics_comparison(means, f"{entry['dataset']}_mean")
            plot_paper_metrics(means, f"{entry['dataset']}_mean_paper")
        agg_md = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "RESULTS_AGGREGATED.md"
        )
        with open(agg_md, "w") as f:
            f.write("# Aggregated Results (mean ± std over seeds)\n\n")
            for entry in summary:
                f.write(f"## {entry['dataset']} (T={entry['seq_len']}, "
                        f"n={entry['n_test_used']}, seeds={entry['n_seeds']}, "
                        f"acc={entry['accuracy_mean']*100:.1f}±{entry['accuracy_std']*100:.1f}%)\n\n")
                f.write("| Method | Val | Val_any | L1 | L2 | L1_raw | L2_raw | DTW | IsoForest | s/sample |\n")
                f.write("|---|---|---|---|---|---|---|---|---|---|\n")
                for m, mets in entry["methods"].items():
                    t = entry["method_elapsed"].get(m, {})
                    f.write(f"| {m} | {mets['Val']['mean']:.3f}±{mets['Val']['std']:.3f} | "
                            f"{mets['Val_any']['mean']:.3f}±{mets['Val_any']['std']:.3f} | "
                            f"{mets['L1']['mean']:.4f}±{mets['L1']['std']:.4f} | "
                            f"{mets['L2']['mean']:.4f}±{mets['L2']['std']:.4f} | "
                            f"{mets['L1_raw']['mean']:.2f}±{mets['L1_raw']['std']:.2f} | "
                            f"{mets['L2_raw']['mean']:.3f}±{mets['L2_raw']['std']:.3f} | "
                            f"{mets['DTW']['mean']:.3f}±{mets['DTW']['std']:.3f} | "
                            f"{mets['IsoForest']['mean']:.3f}±{mets['IsoForest']['std']:.3f} | "
                            f"{t.get('per_sample_mean', 0):.2f}±{t.get('std', 0)/max(entry['n_test_used'],1):.2f} |\n")
                f.write("\n")
        print(f"[REPORT] Aggregated report saved to {agg_md}")

    print("\n" + "="*60)
    print("  All experiments complete!")
    print("="*60)


if __name__ == "__main__":
    main()
