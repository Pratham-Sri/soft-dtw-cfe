"""
ECG5000 Experiment Runner — Paper-Quality Evaluation
=====================================================
Runs all 5 CFE methods on UCR ECG5000 (univariate ECG, d=1, T=140, C=5).

This dataset is a standard benchmark in time-series XAI literature, comparable
to TwoLeadECG but with:
  - 5 classes (heartbeat morphologies) vs 2
  - 500 train / 4500 test  (excellent statistical power)
  - T=140 time steps (comparable to TwoLeadECG T=82)

Methods: Ours (Soft-DTW), SPARCE, GDFO, M-CELS, DTW-CFE

Usage:
    python -m soft_dtw_cfe.run_ecg5000_experiments [--n_test N] [--seeds S1 S2 ...]
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

DATASET_NAME = "ECG5000"


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


def run_single_seed(
    n_test: int = 25,
    skip_optuna: bool = True,
    seed: int = 42,
) -> dict:
    """
    Run all 5 methods on ECG5000 for a single seed.

    Returns structured result dict with all metrics.
    """
    set_seed(seed)
    t_start = time.time()
    print(f"\n{'='*70}")
    print(f"  ECG5000 Experiment  |  seed={seed}  |  n_test={n_test}")
    print(f"{'='*70}")

    # 1. Data
    X_train, y_train, X_test, y_test, metadata = load_dataset(DATASET_NAME)
    train_loader, test_loader = get_dataloaders(X_train, y_train, X_test, y_test)

    d = metadata["n_channels"]
    c = metadata["n_classes"]
    T = metadata["seq_len"]
    print(f"  Shape: n_train={metadata['n_train']}, n_test={metadata['n_test']}, "
          f"d={d}, T={T}, C={c}")

    # 2. Classifier
    model = setup_classifier(DATASET_NAME, d, c, train_loader, test_loader, skip_optuna)
    acc = evaluate_classifier(model, test_loader)
    print(f"  Classifier accuracy: {acc*100:.2f}%")

    # 3. Select test samples — use seed-based shuffling for different subsets
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(X_test))[:n_test]
    n_actual = len(indices)
    X_sample = torch.tensor(X_test[indices], dtype=torch.float32)
    y_sample = torch.tensor(y_test[indices], dtype=torch.long)

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

    # ── Method 2: SPARCE ──────────────────────────────────────────────────
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
    print(f"\n  [5/5] DTW-Guided Constrained Deformation")
    try:
        cfg = dict(DTWCFE_CONFIG)
        cfg["seed"] = seed
        from soft_dtw_cfe.methods.dtw_guided.dtw_cfe import DTWGuidedCFE
        dtwcfe = DTWGuidedCFE(classifier=model, **cfg)
        _t0 = time.time()
        res = dtwcfe.generate_batch(X_sample, y_sample, X_train, y_train, verbose=True)
        method_results["DTW-CFE"] = res
        method_elapsed["DTW-CFE"] = time.time() - _t0
    except Exception as e:
        print(f"  [DTW-CFE ERROR] {e}")
        traceback.print_exc()

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

    # 6. Visualizations (seed 42 only — representative examples)
    if seed == 42:
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
                        DATASET_NAME, mname,
                        r["original_class"], r["target_class"],
                        f"{DATASET_NAME}_{mname}_ecg5000_cf",
                    )
                except Exception:
                    pass
                if "losses" in r:
                    try:
                        plot_loss_curves(r["losses"], DATASET_NAME, f"{mname}_ecg5000")
                    except Exception:
                        pass

        try:
            plot_metrics_comparison(all_metrics, f"{DATASET_NAME}_ecg5000")
            plot_paper_metrics(all_metrics, f"{DATASET_NAME}_ecg5000")
        except Exception:
            pass

    elapsed = time.time() - t_start
    print(f"\n  Done in {elapsed:.1f}s")

    return {
        "dataset": DATASET_NAME,
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


def aggregate_seeds(all_seed_results: list) -> dict:
    """Compute mean ± std across seeds for each method & metric."""
    method_order = ["Ours", "SPARCE", "GDFO", "M-CELS", "DTW-CFE"]
    metric_names = ["Val", "L1", "L2", "DTW", "IsoForest"]

    agg = {}
    for mname in method_order:
        agg[mname] = {}
        for metric in metric_names:
            values = []
            for res in all_seed_results:
                if mname in res["methods"] and metric in res["methods"][mname]:
                    values.append(res["methods"][mname][metric])
            if values:
                agg[mname][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "values": values,
                    "n_seeds": len(values),
                }
        # Aggregate timing
        times = [res["method_elapsed"].get(mname, 0) for res in all_seed_results
                 if mname in res.get("method_elapsed", {})]
        if times:
            agg[mname]["time_mean"] = float(np.mean(times))
            agg[mname]["time_std"] = float(np.std(times))

    return agg


def generate_ecg5000_report(all_seed_results: list, agg: dict, output_path: str):
    """Generate comprehensive Markdown report for the paper."""
    n_seeds = len(all_seed_results)
    method_order = ["Ours", "SPARCE", "GDFO", "M-CELS", "DTW-CFE"]
    metric_names = ["Val", "L1", "L2", "DTW", "IsoForest"]

    # Get dataset metadata from first result
    meta = all_seed_results[0] if all_seed_results else {}
    acc_values = [r["accuracy"] for r in all_seed_results]
    acc_mean = np.mean(acc_values) if acc_values else 0
    acc_std = np.std(acc_values) if acc_values else 0

    lines = [
        "# ECG5000 Experiment Results — Full 5-Method Comparison",
        "",
        "> **Dataset**: UCR ECG5000 — Univariate heartbeat classification",
        f"> **Seeds**: {[r['seed'] for r in all_seed_results]} ({n_seeds} runs)",
        f"> **Test samples**: {meta.get('n_test_used', '?')} per seed",
        "",
        "## Dataset Description",
        "",
        "| Property | Value |",
        "|----------|-------|",
        f"| Source | UCR Time Series Archive |",
        f"| Domain | ECG / Cardiology |",
        f"| Channels (d) | {meta.get('n_channels', 1)} |",
        f"| Length (T) | {meta.get('seq_len', 140)} |",
        f"| Classes (C) | {meta.get('n_classes', 5)} |",
        f"| Train samples | {meta.get('n_train', 500)} |",
        f"| Classifier Accuracy | {acc_mean*100:.2f}% ± {acc_std*100:.2f}% |",
        "",
        "**Class semantics**: 5 heartbeat morphology types from the MIT-BIH",
        "Arrhythmia Database — Normal (1), R-on-T PVC (2), PVC (3), SP (4), UB (5).",
        "Binary-reducible to Normal vs Abnormal for clinical interpretability.",
        "",
        "**Comparison to TwoLeadECG**: Both are ECG datasets from the UCR archive.",
        "ECG5000 is larger (500 vs 23 train), longer (T=140 vs T=82), and",
        "multi-class (C=5 vs C=2), providing a more challenging evaluation.",
        "",
        "---",
        "",
        "## Methods Compared",
        "",
        "| Method | Type | Key Innovation |",
        "|--------|------|---------------|",
        "| **Ours** (Soft-DTW) | Gradient-based | Soft-DTW alignment for temporal plausibility |",
        "| **SPARCE** | Prototype+Saliency | Two-phase curriculum with saliency masking |",
        "| **GDFO** (VAE+GMM) | Generative density | VAE+GMM density-guided optimisation |",
        "| **M-CELS** | Greedy baseline | Saliency-guided NUN substitution |",
        "| **DTW-CFE** (CMA-ES) | Evolutionary | DTW-guided constrained deformation |",
        "",
        "---",
        "",
    ]

    # ── Per-seed Results ──────────────────────────────────────────────────
    lines += ["## Per-Seed Results", ""]
    for res in all_seed_results:
        seed = res["seed"]
        lines += [
            f"### Seed {seed}",
            "",
            f"- **Classifier Accuracy**: {res['accuracy']*100:.2f}%",
            f"- **Test samples**: {res['n_test_used']}",
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

    # ── Aggregated Results (mean ± std) ───────────────────────────────────
    lines += [
        "---",
        "",
        f"## Aggregated Results (mean ± std, {n_seeds} seeds)",
        "",
        "| Method | Val ↑ | L1 ↓ | L2 ↓ | DTW ↓ | IsoForest ↑ | Avg Time |",
        "|--------|-------|------|------|-------|-------------|----------|",
    ]

    for mname in method_order:
        if mname not in agg or not agg[mname]:
            continue
        cols = []
        for metric in metric_names:
            if metric in agg[mname]:
                m = agg[mname][metric]
                cols.append(f"{m['mean']:.4f}±{m['std']:.4f}")
            else:
                cols.append("-")
        t_str = "-"
        if "time_mean" in agg[mname]:
            t_str = f"{agg[mname]['time_mean']:.1f}±{agg[mname].get('time_std',0):.1f}"
        lines.append(f"| **{mname}** | {' | '.join(cols)} | {t_str} |")

    # ── Best-performing Analysis ──────────────────────────────────────────
    metrics_dir = {"Val": "max", "DTW": "min", "IsoForest": "max", "L1": "min", "L2": "min"}

    lines += [
        "",
        "---",
        "",
        "## Best Method per Metric (aggregated mean)",
        "",
        "| Metric | Best Method | Value |",
        "|--------|-------------|-------|",
    ]

    for metric, direction in metrics_dir.items():
        best_val = None
        best_method = None
        for mname in method_order:
            if mname in agg and metric in agg[mname]:
                val = agg[mname][metric]["mean"]
                if best_val is None:
                    best_val = val
                    best_method = mname
                elif (direction == "max" and val > best_val) or \
                     (direction == "min" and val < best_val):
                    best_val = val
                    best_method = mname
        if best_method:
            sym = "↑" if direction == "max" else "↓"
            lines.append(f"| {metric} {sym} | **{best_method}** | {best_val:.4f} |")

    # ── Average Rank ──────────────────────────────────────────────────────
    ranks = {m: {metric: [] for metric in metrics_dir} for m in method_order}
    for res in all_seed_results:
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
        f"## Average Rank across {n_seeds} seeds (lower is better)",
        "",
        "| Method | Val ↑ | DTW ↓ | IsoForest ↑ | L1 ↓ | L2 ↓ | **Avg** |",
        "|--------|-------|-------|-------------|------|------|---------| ",
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

    # ── Comparison to TwoLeadECG ──────────────────────────────────────────
    lines += [
        "",
        "---",
        "",
        "## Cross-Dataset Comparison: ECG5000 vs TwoLeadECG",
        "",
        "| Property | TwoLeadECG | ECG5000 |",
        "|----------|------------|---------|",
        "| Archive | UCR | UCR |",
        "| Domain | ECG | ECG |",
        "| Channels (d) | 1 | 1 |",
        f"| Length (T) | 82 | {meta.get('seq_len', 140)} |",
        "| Classes (C) | 2 | 5 |",
        "| Train samples | 23 | 500 |",
        "| Test samples | 1139 | 4500 |",
        "",
        "ECG5000 provides a substantially more challenging evaluation due to its",
        "multi-class nature (5 heartbeat types) and larger sample count, while",
        "remaining in the same clinical ECG domain as TwoLeadECG.",
        "",
        "---",
        f"*Generated automatically by `soft_dtw_cfe/run_ecg5000_experiments.py`*",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[REPORT] ECG5000 report saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="ECG5000 Experiment Runner")
    parser.add_argument("--n_test", type=int, default=25,
                        help="Number of test samples per seed (default: 25)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 0, 1],
                        help="Random seeds for multi-seed evaluation (default: 42 0 1)")
    parser.add_argument("--skip_optuna", action="store_true", default=True)
    parser.add_argument("--single_seed", type=int, default=None,
                        help="Run only a single seed (for quick testing)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.single_seed is not None:
        args.seeds = [args.single_seed]

    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Test samples per seed: {args.n_test}")
    print(f"Seeds: {args.seeds}")
    print(f"{'='*70}")
    print(f"  ECG5000 — FULL 5-METHOD COMPARISON")
    print(f"{'='*70}")

    all_seed_results = []

    for seed in args.seeds:
        try:
            result = run_single_seed(
                n_test=args.n_test,
                skip_optuna=args.skip_optuna,
                seed=seed,
            )
            all_seed_results.append(result)
        except Exception as e:
            print(f"\n[ERROR] Failed on seed {seed}: {e}")
            traceback.print_exc()
            all_seed_results.append({
                "dataset": DATASET_NAME, "seed": seed, "accuracy": 0.0,
                "n_train": 0, "n_test_used": 0, "seq_len": 0,
                "n_channels": 0, "n_classes": 0, "methods": {},
                "method_elapsed": {}, "elapsed": 0.0, "error": str(e),
            })

    # ── Save all per-seed results ─────────────────────────────────────────
    json_path = os.path.join(RESULTS_DIR, "ecg5000_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_seed_results, f, indent=2, default=str)
    print(f"\n[SAVE] Per-seed JSON results saved to {json_path}")

    # ── Compute aggregated stats ──────────────────────────────────────────
    valid_results = [r for r in all_seed_results if "error" not in r]
    agg = aggregate_seeds(valid_results)

    agg_path = os.path.join(RESULTS_DIR, "ecg5000_aggregated.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, default=str)
    print(f"[SAVE] Aggregated JSON results saved to {agg_path}")

    # ── Also merge into sparce_results.json for unified reporting ─────────
    sparce_json_path = os.path.join(RESULTS_DIR, "sparce_results.json")
    merged_dict = {}
    if os.path.exists(sparce_json_path):
        try:
            with open(sparce_json_path, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
                for item in prev_data:
                    if isinstance(item, dict) and "dataset" in item:
                        merged_dict[item["dataset"]] = item
        except Exception:
            pass
    # Use seed 42 result as the canonical entry
    seed42_result = next((r for r in valid_results if r["seed"] == 42), valid_results[0] if valid_results else None)
    if seed42_result:
        merged_dict[DATASET_NAME] = seed42_result
        merged_results = list(merged_dict.values())
        with open(sparce_json_path, "w", encoding="utf-8") as f:
            json.dump(merged_results, f, indent=2, default=str)
        print(f"[SAVE] Merged into {sparce_json_path} ({len(merged_results)} datasets)")

    # ── Generate report ───────────────────────────────────────────────────
    report_path = os.path.join(_THIS_DIR, "RESULTS_ECG5000.md")
    generate_ecg5000_report(valid_results, agg, report_path)

    print("\n" + "=" * 70)
    print("  ECG5000 experiments complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
