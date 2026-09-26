"""
Visualization of time series counterfactuals.

Implements qualitative visualizations as shown in Section 5.3 
(Figures 2 and 3) plus additional diagnostic plots.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import torch

from soft_dtw_cfe.config import FIGURES_DIR


def plot_counterfactual(
    original: torch.Tensor,
    counterfactual: torch.Tensor,
    target_samples: torch.Tensor,
    dataset_name: str,
    method_name: str,
    original_class: int,
    target_class: int,
    save_name: str = None
):
    """
    Plot the original time series, the generated counterfactual, and 
    a few representative samples from the target class.

    Args:
        original: (d, T) tensor
        counterfactual: (d, T) tensor
        target_samples: (N, d, T) tensor (training samples of target class)
        dataset_name: name of the dataset
        method_name: name of the CFE generation method
        original_class: original predicted class
        target_class: target class
        save_name: filename to save the plot. If None, won't save.
    """
    # Convert to numpy for plotting
    x_orig = original.numpy() if hasattr(original, 'numpy') else np.array(original)
    x_cf   = counterfactual.numpy() if hasattr(counterfactual, 'numpy') else np.array(counterfactual)
    target_samps = target_samples.numpy() if hasattr(target_samples, 'numpy') else np.array(target_samples)

    d, T = x_orig.shape
    time_steps = np.arange(T)

    fig, axes = plt.subplots(d, 1, figsize=(12, 4 * d), squeeze=False)
    fig.suptitle(
        f"{dataset_name} — {method_name}\n"
        f"Original class: {original_class}  →  Target class: {target_class}",
        fontsize=13, fontweight='bold'
    )

    # Pick a few target samples to plot as background reference
    n_target_plot = min(8, len(target_samps))
    indices = np.random.choice(len(target_samps), n_target_plot, replace=False)

    for channel in range(d):
        ax = axes[channel, 0]

        # Plot target class samples (light grey/blue)
        for i, idx in enumerate(indices):
            ax.plot(
                time_steps,
                target_samps[idx, channel, :],
                color="#a8d8ea",
                alpha=0.5,
                linewidth=0.8,
                label="Target class samples" if i == 0 else ""
            )

        # Highlight the difference region
        diff = np.abs(x_cf[channel, :] - x_orig[channel, :])
        ax.fill_between(
            time_steps,
            x_orig[channel, :] - diff,
            x_orig[channel, :] + diff,
            alpha=0.12, color="orange", label="Change region"
        )

        # Plot original
        ax.plot(
            time_steps, x_orig[channel, :],
            color="#2c5f8a", linewidth=2.2,
            label=f"Original (class {original_class})"
        )

        # Plot counterfactual
        ax.plot(
            time_steps, x_cf[channel, :],
            color="#e05c5c", linestyle="--", linewidth=2.2,
            label=f"Counterfactual (class {target_class})"
        )

        ax.set_title(f"Channel {channel+1}" if d > 1 else "", fontsize=10)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_name:
        path = os.path.join(FIGURES_DIR, f"{save_name}.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  [VIS] Saved counterfactual figure -> {path}")

    plt.close(fig)


def plot_loss_curves(losses: dict, dataset_name: str, method_name: str, save_name: str = None):
    """
    Plot training loss curves for the CFE optimisation.

    Args:
        losses: dict with keys 'total', 'proximity', 'sparsity', 'validity', 'dtw'
        dataset_name: Dataset name (for title)
        method_name: Method name (for title)
        save_name: If None, auto-generated.
    """
    if not losses or not losses.get("total"):
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"CFE Optimisation Loss Curves — {dataset_name} ({method_name})",
        fontsize=13, fontweight='bold'
    )

    iters = range(1, len(losses["total"]) + 1)

    components = [
        ("total",     "Total Loss",    "#333333", axes[0, 0]),
        ("proximity", "Proximity (L2)","#2c5f8a", axes[0, 1]),
        ("sparsity",  "Sparsity (L1)", "#5aa469", axes[1, 0]),
        ("validity",  "Validity",       "#e05c5c", axes[1, 1]),
    ]

    for key, title, color, ax in components:
        vals = losses.get(key, [])
        if vals:
            ax.plot(iters, vals, color=color, linewidth=1.5)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)

    # Overlay DTW loss on total loss panel
    if losses.get("dtw"):
        ax0 = axes[0, 0]
        ax_twin = ax0.twinx()
        ax_twin.plot(iters, losses["dtw"], color="#f0a500", linewidth=1.2,
                     linestyle=":", label="DTW loss")
        ax_twin.set_ylabel("DTW Loss", color="#f0a500")
        ax_twin.tick_params(axis='y', labelcolor="#f0a500")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_name is None:
        save_name = f"{dataset_name}_{method_name}_loss"
    path = os.path.join(FIGURES_DIR, f"{save_name}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  [VIS] Saved loss curve figure -> {path}")
    plt.close(fig)


def plot_metrics_comparison(all_metrics: dict, dataset_name: str, save_name: str = None):
    """
    Bar chart comparing all methods across all metrics.

    Args:
        all_metrics: {method_name: {metric_name: value}}
        dataset_name: For the figure title.
        save_name: Output filename (auto-generated if None).
    """
    if not all_metrics:
        return

    methods = list(all_metrics.keys())
    metrics = ["Val", "L1", "L2", "DTW", "IsoForest"]
    metric_labels = ["Val ↑", "L1 ↓", "L2 ↓", "DTW ↓", "IsoForest ↑"]

    x = np.arange(len(metrics))
    width = 0.8 / max(len(methods), 1)

    colors = ["#2c5f8a", "#e05c5c", "#5aa469", "#f0a500", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(
        f"Metrics Comparison — {dataset_name}",
        fontsize=13, fontweight='bold'
    )

    for i, (mname, color) in enumerate(zip(methods, colors)):
        vals = [all_metrics[mname].get(m, 0) for m in metrics]
        offset = (i - len(methods) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9, label=mname,
                      color=color, alpha=0.85, edgecolor='white')

        # Value labels
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha='center', va='bottom', fontsize=7.5, rotation=45
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel("Value")
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, max(
        max(all_metrics[m].get(mt, 0) for m in methods for mt in metrics) * 1.3,
        0.1
    ))

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_name is None:
        save_name = f"{dataset_name}_metrics_comparison"
    path = os.path.join(FIGURES_DIR, f"{save_name}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  [VIS] Saved metrics comparison -> {path}")
    plt.close(fig)


def plot_all_counterfactuals_grid(results_per_method: dict, dataset_name: str, save_name: str = None):
    """
    Grid plot showing multiple CFE examples side-by-side for all methods.

    Args:
        results_per_method: {method_name: [result_dict, ...]} (univariate only)
        dataset_name: Dataset name
        save_name: Output filename
    """
    methods = list(results_per_method.keys())
    n_examples = min(5, min(len(v) for v in results_per_method.values()))
    if n_examples == 0:
        return

    n_rows = n_examples
    n_cols = len(methods)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 3 * n_rows),
        squeeze=False
    )
    fig.suptitle(f"Counterfactual Explanations — {dataset_name}", fontsize=14, fontweight='bold')

    for col, mname in enumerate(methods):
        results = results_per_method[mname]
        axes[0, col].set_title(mname, fontsize=12, fontweight='bold')
        for row in range(n_examples):
            ax = axes[row, col]
            r = results[row]
            x_orig = r["original"].numpy()
            x_cf   = r["counterfactual"].numpy()
            T = x_orig.shape[-1]
            t = np.arange(T)
            ax.plot(t, x_orig[0], color="#2c5f8a", linewidth=1.5,
                    label=f"Orig (c{r['original_class']})")
            ax.plot(t, x_cf[0], color="#e05c5c", linestyle="--", linewidth=1.5,
                    label=f"CF (c{r['target_class']})")
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel(f"Sample {row+1}", fontsize=9)
            if row == 0:
                ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_name is None:
        save_name = f"{dataset_name}_grid"
    path = os.path.join(FIGURES_DIR, f"{save_name}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  [VIS] Saved grid figure -> {path}")
    plt.close(fig)
