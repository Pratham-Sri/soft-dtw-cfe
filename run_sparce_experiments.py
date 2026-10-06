"""
SPARCE Experiment Runner
========================
Runs the SPARCE (Saliency-guided Prototype-Aligned Refinement) method
against all existing baselines on multivariate datasets.

Compares: Ours (Soft-DTW), SPARCE, GDFO, M-CELS, DTW-CFE

Usage:
    python -m soft_dtw_cfe.run_sparce_experiments [--datasets DATASET1 ...]
                                                   [--n_test N]
                                                   [--seed SEED]
"""

import os
import sys
import json
import time
import argparse
import traceback
import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
import types as _types
if "soft_dtw_cfe" not in sys.modules:
    _pkg = _types.ModuleType("soft_dtw_cfe")
    _pkg.__path__ = [_THIS_DIR]
    sys.modules["soft_dtw_cfe"] = _pkg

from soft_dtw_cfe.config import (
    MULTIVARIATE_DATASETS, LARGE_MULTIVARIATE_DATASETS,
    DEVICE, RESULTS_DIR, CFE_CONFIG, DTWCFE_CONFIG, GDFO_CONFIG,
)
from soft_dtw_cfe.data.dataset_loader import load_dataset, get_dataloaders
from soft_dtw_cfe.models.classifier import (
    TSClassifier, train_classifier, save_classifier,
    load_classifier, evaluate_classifier,
)
from soft_dtw_cfe.models.optuna_tuner import optimise_classifier
from soft_dtw_cfe.methods.proposed_method import SoftDTWCounterfactualGenerator
from soft_dtw_cfe.methods.sparce import SPARCECounterfactualGenerator
from soft_dtw_cfe.methods.gdfo import GDFOCounterfactualGenerator
from soft_dtw_cfe.methods.m_cels import MCELS
from soft_dtw_cfe.evaluation.metrics import evaluate_all_metrics
from soft_dtw_cfe.visualization.plot import (
    plot_counterfactual, plot_loss_curves, plot_metrics_comparison,
    plot_paper_metrics,
)

# Datasets for SPARCE evaluation — all multivariate
SPARCE_DATASETS = [
    "Epilepsy",       # d=3, T=206, C=4 — core multivariate benchmark
    "BasicMotions",   # d=6, T=100, C=4 — motion sensor
    "ERing",          # d=4, T=65,  C=6 — electric ring
    "RacketSports",   # d=6, T=30,  C=4 — IMU (short + wide)
    "NATOPS",         # d=24, T=51, C=6 — body sensor (high-dimensional)
    "Cricket",        # d=6, T=1197, C=12 — long multivariate
]


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
                     skip_optuna=True):
    """Load or train classifier."""
    path = _model_path(dataset_name)
    if os.path.exists(path):
        print(f"  -> Loading existing classifier from {path}")
        model = TSClassifier(n_channels=d, n_classes=c, dropout=0.3)
        model = load_classifier(model, dataset_name)
    else:
        print(f"  -> Training new classifier ...")
        if not skip_optuna:
            best_params = optimise_classifier(d, c, train_loader, test_loader)
            dropout, lr, wd = best_params["dropout"], best_params["lr"], best_params["weight_decay"]
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
    n_test: int = 15,
    skip_optuna: bool = True,
    run_dtwcfe: bool = True,
    run_gdfo: bool = True,
    seed: int = 42,
) -> dict:
    """
    Run all methods including SPARCE on one multivariate dataset.

    Methods: Ours, SPARCE, GDFO, M-CELS, DTW-CFE
    """
    set_seed(seed)
    t_start = time.time()
    print(f"\n{'='*70}")
    print(f"  Dataset: {dataset_name}  |  seed={seed}")
    print(f"{'='*70}")

    # 1. Data
    X_train, y_train, X_test, y_test, metadata = load_dataset(dataset_name)
    train_loader, test_loader = get_dataloaders(X_train, y_train, X_test, y_test)

    d = metadata["n_channels"]
    c = metadata["n_classes"]
    T = metadata["seq_len"]
    print(f"  Shape: n_train={metadata['n_train']}, n_test={metadata['n_test']}, "
          f"d={d}, T={T}, C={c}")

    # 2. Classifier
    model = setup_classifier(dataset_name, d, c, train_loader, test_loader, skip_optuna)
    acc = evaluate_classifier(model, test_loader)
    print(f"  Classifier accuracy: {acc*100:.2f}%")

    # 3. Select test samples
    n_actual = min(n_test, len(X_test))
    X_sample = torch.tensor(X_test[:n_actual], dtype=torch.float32)
    y_sample = torch.tensor(y_test[:n_actual], dtype=torch.long)

    method_results = {}
    method_elapsed = {}

    # ── Method 1: Ours (Soft-DTW) ─────────────────────────────────────────
    print(f"\n  [1/5] Proposed Method (Soft-DTW CFE)")
    try:
        proposed = SoftDTWCounterfactualGenerator(
            classifier=model, num_iterations=CFE_CONFIG["num_iterations"],
        )
        _t0 = time.time()
        res = proposed.generate_batch(X_sample, y_sample, X_train, y_train, verbose=True)
        method_results["Ours"] = res
        method_elapsed["Ours"] = time.time() - _t0
    except Exception as e:
        print(f"  [Ours ERROR] {e}")
        traceback.print_exc()

    # ── Method 2: SPARCE (new) ────────────────────────────────────────────
    print(f"\n  [2/5] SPARCE (Saliency-guided Prototype-Aligned Refinement)")
    try:
        sparce = SPARCECounterfactualGenerator(classifier=model)
        _t0 = time.time()
        res = sparce.generate_batch(X_sample, y_sample, X_train, y_train, verbose=True)
        method_results["SPARCE"] = res
        method_elapsed["SPARCE"] = time.time() - _t0
    except Exception as e:
        print(f"  [SPARCE ERROR] {e}")
        traceback.print_exc()

    # ── Method 3: GDFO ────────────────────────────────────────────────────
    if run_gdfo:
        print(f"\n  [3/5] GDFO (Generative Density Function Optimisation)")
        try:
            gdfo = GDFOCounterfactualGenerator(
                classifier=model, n_channels=d, seq_len=T,
                z_dim=GDFO_CONFIG["z_dim"], vae_epochs=GDFO_CONFIG["vae_epochs"],
                vae_beta=GDFO_CONFIG["vae_beta"], gmm_components=GDFO_CONFIG["gmm_components"],
                lambda_valid=GDFO_CONFIG["lambda_valid"], lambda_density=GDFO_CONFIG["lambda_density"],
                lambda_dtw=GDFO_CONFIG["lambda_dtw"], tau=GDFO_CONFIG["tau"],
                lr=GDFO_CONFIG["lr"], num_iterations=GDFO_CONFIG["num_iterations"],
                k=GDFO_CONFIG["k"], gamma=GDFO_CONFIG["gamma"], use_dtw=GDFO_CONFIG["use_dtw"],
            )
            _t0 = time.time()
            res = gdfo.generate_batch(X_sample, y_sample, X_train, y_train, verbose=True)
            method_results["GDFO"] = res
            method_elapsed["GDFO"] = time.time() - _t0
        except Exception as e:
            print(f"  [GDFO ERROR] {e}")
            traceback.print_exc()
    else:
        print(f"\n  [3/5] GDFO skipped")

    # ── Method 4: M-CELS ─────────────────────────────────────────────────
    print(f"\n  [4/5] M-CELS Method")
    try:
        mcels = MCELS(classifier=model, max_iterations=20)
        _t0 = time.time()
        res = mcels.generate_batch(X_sample, y_sample, X_train, y_train, verbose=True)
        method_results["M-CELS"] = res
        method_elapsed["M-CELS"] = time.time() - _t0
    except Exception as e:
        print(f"  [M-CELS ERROR] {e}")
        traceback.print_exc()

    # ── Method 5: DTW-CFE ─────────────────────────────────────────────────
    if run_dtwcfe:
        print(f"\n  [5/5] DTW-Guided Constrained Deformation")
        try:
            cfg = dict(DTWCFE_CONFIG)
            cfg["seed"] = seed
            if T > 500:
                cfg["window"] = min(T // 4, 200)
            from soft_dtw_cfe.methods.dtw_guided.dtw_cfe import DTWGuidedCFE
            dtwcfe = DTWGuidedCFE(classifier=model, **cfg)
            _t0 = time.time()
            res = dtwcfe.generate_batch(X_sample, y_sample, X_train, y_train, verbose=True)
            method_results["DTW-CFE"] = res
            method_elapsed["DTW-CFE"] = time.time() - _t0
        except Exception as e:
            print(f"  [DTW-CFE ERROR] {e}")
            traceback.print_exc()
    else:
        print(f"\n  [5/5] DTW-CFE skipped")

    # 5. Evaluate metrics
    header = f"  {'Method':<12} {'Val':>6} {'L1':>8} {'L2':>8} {'DTW':>10} {'IsoForest':>10} {'Time(s)':>9} {'s/sample':>9}"
    print(f"\n{header}")
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
            try:
                plot_counterfactual(
                    r["original"], r["counterfactual"], target_samps,
                    dataset_name, mname,
                    r["original_class"], r["target_class"],
                    f"{dataset_name}_{mname}_sparce_cf",
                )
            except Exception:
                pass
            if "losses" in r:
                try:
                    plot_loss_curves(r["losses"], dataset_name, f"{mname}_sparce")
                except Exception:
                    pass

    try:
        plot_metrics_comparison(all_metrics, f"{dataset_name}_sparce")
        plot_paper_metrics(all_metrics, f"{dataset_name}_sparce")
    except Exception:
        pass

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


def generate_sparce_report(all_results: list, output_path: str):
    """Generate Markdown comparison report with SPARCE."""
    lines = [
        "# SPARCE Comparison — Multivariate Time Series CFE",
        "",
        "> Saliency-guided Prototype-Aligned Refinement for Counterfactual Explanations",
        "> vs. existing methods on multivariate time series datasets.",
        "",
        "## Methods Compared",
        "",
        "| Method | Type | Key Innovation |",
        "|--------|------|---------------|",
        "| **Ours** (Soft-DTW) | Gradient-based | Soft-DTW alignment for temporal plausibility |",
        "| **SPARCE** | Prototype+Saliency | Two-phase curriculum with channel-selective saliency masking |",
        "| **GDFO** (VAE+GMM) | Generative density | VAE+GMM density-guided optimisation |",
        "| **M-CELS** | Greedy baseline | Saliency-guided NUN substitution |",
        "| **DTW-CFE** (CMA-ES) | Evolutionary | DTW-guided constrained deformation |",
        "",
        "## SPARCE Key Innovations",
        "",
        "1. **Prototype warm-start**: Initialise from saliency-weighted blend of original + nearest unlike neighbour",
        "2. **Channel-selective saliency mask**: Learnable mask focuses perturbations on discriminative channels",
        "3. **Two-phase curriculum**: Phase 1 = validity-first (aggressive), Phase 2 = plausibility refinement",
        "4. **Cross-channel correlation**: Preserves inter-channel covariance structure",
        "",
        "---",
        "",
        "## Results per Dataset",
        "",
    ]

    method_order = ["Ours", "SPARCE", "GDFO", "M-CELS", "DTW-CFE"]

    for res in all_results:
        ds = res["dataset"]
        err = res.get("error")
        lines += [f"### {ds}", ""]

        if err:
            lines += [f"**ERROR**: `{err}`", "", ""]
            continue

        lines += [
            f"- **Classifier Accuracy**: {res['accuracy']*100:.2f}%",
            f"- **Samples**: {res['n_train']} train / {res['n_test_used']} test",
            f"- **Series**: T={res['seq_len']}, d={res['n_channels']}, C={res['n_classes']}",
            f"- **Elapsed**: {res['elapsed']:.1f}s",
            "",
            "| Method | Val ↑ | L1 ↓ | L2 ↓ | DTW ↓ | IsoForest ↑ | Time(s) | s/sample |",
            "|--------|-------|------|------|-------|-------------|---------|----------|",
        ]

        mel = res.get("method_elapsed", {})
        n_used = max(res.get("n_test_used", 1), 1)

        for mname in method_order:
            if mname not in res["methods"]:
                continue
            m = res["methods"][mname]
            v   = m.get("Val", 0)
            l1  = m.get("L1", 0)
            l2  = m.get("L2", 0)
            dtw = m.get("DTW", 0)
            iso = m.get("IsoForest", 0)
            t   = mel.get(mname, 0.0)
            lines.append(
                f"| {mname} | {v:.3f} | {l1:.4f} | {l2:.4f} | "
                f"{dtw:.4f} | {iso:.3f} | {t:.1f} | {t/n_used:.2f} |"
            )
        lines += ["", ""]

    # ── Grand Summary ─────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## Grand Summary — SPARCE vs All Methods",
        "",
        "### Validity (Val ↑)",
        "",
        "| Dataset | d | T | C | Ours | SPARCE | GDFO | M-CELS | DTW-CFE |",
        "|---------|---|---|---|------|--------|------|--------|---------|",
    ]

    for res in all_results:
        if res.get("error"):
            continue
        m = res["methods"]
        d, T, c = res["n_channels"], res["seq_len"], res["n_classes"]
        cols = []
        for mname in method_order:
            if mname in m:
                cols.append(f"{m[mname].get('Val', 0):.3f}")
            else:
                cols.append("-")
        lines.append(f"| {res['dataset']} | {d} | {T} | {c} | {' | '.join(cols)} |")

    lines += [
        "",
        "### DTW Plausibility (↓)",
        "",
        "| Dataset | d | T | Ours | SPARCE | GDFO | M-CELS | DTW-CFE |",
        "|---------|---|---|------|--------|------|--------|---------|",
    ]

    for res in all_results:
        if res.get("error"):
            continue
        m = res["methods"]
        d, T = res["n_channels"], res["seq_len"]
        cols = []
        for mname in method_order:
            if mname in m:
                cols.append(f"{m[mname].get('DTW', 0):.2f}")
            else:
                cols.append("-")
        lines.append(f"| {res['dataset']} | {d} | {T} | {' | '.join(cols)} |")

    lines += [
        "",
        "### IsoForest Plausibility (↑)",
        "",
        "| Dataset | d | T | Ours | SPARCE | GDFO | M-CELS | DTW-CFE |",
        "|---------|---|---|------|--------|------|--------|---------|",
    ]

    for res in all_results:
        if res.get("error"):
            continue
        m = res["methods"]
        d, T = res["n_channels"], res["seq_len"]
        cols = []
        for mname in method_order:
            if mname in m:
                cols.append(f"{m[mname].get('IsoForest', 0):.3f}")
            else:
                cols.append("-")
        lines.append(f"| {res['dataset']} | {d} | {T} | {' | '.join(cols)} |")

    # ── Win counts ────────────────────────────────────────────────────────
    metrics_dir = {"Val": "max", "DTW": "min", "IsoForest": "max", "L1": "min", "L2": "min"}
    wins = {m: {metric: 0 for metric in metrics_dir} for m in method_order}
    total_ds = 0

    for res in all_results:
        if res.get("error"):
            continue
        m = res.get("methods", {})
        if not m:
            continue
        total_ds += 1
        for metric, direction in metrics_dir.items():
            scores = {}
            for mname in method_order:
                if mname in m and metric in m[mname]:
                    scores[mname] = m[mname][metric]
            if not scores:
                continue
            best = max(scores.values()) if direction == "max" else min(scores.values())
            for mname, val in scores.items():
                if abs(val - best) < 1e-10:
                    wins[mname][metric] += 1

    lines += [
        "",
        "---",
        "",
        f"## Method Win Counts (across {total_ds} multivariate datasets)",
        "",
        "| Method | Val ↑ | DTW ↓ | IsoForest ↑ | L1 ↓ | L2 ↓ | **Total** |",
        "|--------|-------|-------|-------------|------|------|-----------|",
    ]
    for mname in method_order:
        w = wins[mname]
        total = sum(w.values())
        lines.append(
            f"| **{mname}** | {w['Val']} | {w['DTW']} | "
            f"{w['IsoForest']} | {w['L1']} | {w['L2']} | **{total}** |"
        )

    # ── Average Rank ──────────────────────────────────────────────────────
    ranks = {m: {metric: [] for metric in metrics_dir} for m in method_order}
    for res in all_results:
        if res.get("error"):
            continue
        m = res.get("methods", {})
        if not m:
            continue
        for metric, direction in metrics_dir.items():
            scores = []
            for mname in method_order:
                if mname in m and metric in m[mname]:
                    scores.append((m[mname][metric], mname))
            if not scores:
                continue
            scores.sort(key=lambda x: x[0], reverse=(direction == "max"))
            for rank_idx, (_, mname) in enumerate(scores, start=1):
                ranks[mname][metric].append(rank_idx)

    lines += [
        "",
        "## Average Rank (lower is better)",
        "",
        "| Method | Val ↑ | DTW ↓ | IsoForest ↑ | L1 ↓ | L2 ↓ | **Avg** |",
        "|--------|-------|-------|-------------|------|------|---------|",
    ]
    for mname in method_order:
        r = ranks[mname]
        rank_strs = []
        all_ranks = []
        for metric in ["Val", "DTW", "IsoForest", "L1", "L2"]:
            if r[metric]:
                avg = np.mean(r[metric])
                rank_strs.append(f"{avg:.2f}")
                all_ranks.extend(r[metric])
            else:
                rank_strs.append("-")
        overall = f"{np.mean(all_ranks):.2f}" if all_ranks else "-"
        lines.append(f"| **{mname}** | {' | '.join(rank_strs)} | **{overall}** |")

    # ── SPARCE Analysis ───────────────────────────────────────────────────
    lines += [
        "",
        "---",
        "",
        "## SPARCE Analysis",
        "",
        "### Why SPARCE works on multivariate data",
        "",
        "1. **Prototype warm-start solves the validity gap**: By initialising from a",
        "   saliency-weighted blend of the original + nearest target-class sample,",
        "   SPARCE starts close to the decision boundary. This avoids the GDFO problem",
        "   where density pulls the CFE back before validity is achieved.",
        "",
        "2. **Channel saliency focuses perturbations**: Instead of perturbing all channels",
        "   equally, the learnable saliency mask identifies discriminative channels and",
        "   concentrates modifications there — preserving non-discriminative channels intact.",
        "",
        "3. **Two-phase curriculum**: Phase 1 aggressively achieves validity with strong λ_valid;",
        "   Phase 2 refines plausibility (DTW + correlation) while maintaining the class flip.",
        "",
        "4. **Cross-channel correlation preservation**: The covariance penalty ensures",
        "   inter-channel correlations (e.g., x/y/z accelerometer axes) are maintained.",
        "",
        "---",
        "*Generated automatically by `soft_dtw_cfe/run_sparce_experiments.py`*",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[REPORT] SPARCE comparison report saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="SPARCE Experiment Runner")
    parser.add_argument("--datasets", nargs="+", default=SPARCE_DATASETS,
                        help="Datasets to run")
    parser.add_argument("--n_test", type=int, default=15,
                        help="Number of test samples (default: 15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip_optuna", action="store_true", default=True)
    parser.add_argument("--no_dtwcfe", action="store_true", help="Skip DTW-CFE")
    parser.add_argument("--no_gdfo", action="store_true", help="Skip GDFO")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Device: {DEVICE}")
    print(f"Datasets: {args.datasets}")
    print(f"Test samples per dataset: {args.n_test}")
    print(f"Seed: {args.seed}")
    print(f"{'='*70}")
    print(f"  SPARCE COMPARISON EXPERIMENT")
    print(f"{'='*70}")

    all_results = []

    for ds in args.datasets:
        try:
            result = run_dataset(
                ds, n_test=args.n_test, skip_optuna=args.skip_optuna,
                run_dtwcfe=(not args.no_dtwcfe),
                run_gdfo=(not args.no_gdfo),
                seed=args.seed,
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n[ERROR] Failed on {ds}: {e}")
            traceback.print_exc()
            all_results.append({
                "dataset": ds, "seed": args.seed, "accuracy": 0.0,
                "n_train": 0, "n_test_used": 0, "seq_len": 0,
                "n_channels": 0, "n_classes": 0, "methods": {},
                "method_elapsed": {}, "elapsed": 0.0, "error": str(e),
            })

    # Save JSON (merging with existing datasets if present)
    json_path = os.path.join(RESULTS_DIR, "sparce_results.json")
    merged_dict = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
                for item in prev_data:
                    if isinstance(item, dict) and "dataset" in item:
                        merged_dict[item["dataset"]] = item
        except Exception:
            pass
    for item in all_results:
        merged_dict[item["dataset"]] = item
    merged_results = list(merged_dict.values())

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(merged_results, f, indent=2, default=str)
    print(f"\n[SAVE] JSON results ({len(merged_results)} datasets) saved to {json_path}")

    # Generate report with all accumulated results
    report_path = os.path.join(_THIS_DIR, "RESULTS_SPARCE.md")
    generate_sparce_report(merged_results, report_path)

    print("\n" + "=" * 70)
    print("  SPARCE experiments complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
