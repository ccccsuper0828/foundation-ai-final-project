"""
Plotting utilities for training curves, Pareto fronts, and comparison tables.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# Use a clean style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 12,
})

# Color palette for three families
FAMILY_COLORS = {
    "cnn": "#2196F3",     # Blue
    "vit": "#FF9800",     # Orange
    "mamba": "#4CAF50",   # Green
}

MODEL_MARKERS = {
    "resnet18": "o",
    "resnet50": "s",
    "convnext_tiny": "D",
    "deit_tiny_patch16_224": "^",
    "deit_small_patch16_224": "v",
    "vim_tiny": "P",
    "vim_small": "*",
}


def load_all_results(output_dir: str = "./outputs"):
    """Load results.json from all model subdirectories."""
    results = {}
    output_path = Path(output_dir)
    for model_dir in output_path.iterdir():
        if model_dir.is_dir():
            result_file = model_dir / "results.json"
            if result_file.exists():
                with open(result_file) as f:
                    results[model_dir.name] = json.load(f)
    return results


def plot_training_curves(results: dict, save_path: str = None):
    """Plot train loss and val accuracy curves for all models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for model_name, data in results.items():
        family = data.get("family", "unknown")
        color = FAMILY_COLORS.get(family, "gray")
        history = data.get("history", [])
        if not history:
            continue

        epochs = [h["epoch"] for h in history]
        train_loss = [h["train_loss"] for h in history]
        val_acc = [h["val_acc1"] for h in history]

        ax1.plot(epochs, train_loss, label=model_name, color=color, alpha=0.8)
        ax2.plot(epochs, val_acc, label=model_name, color=color, alpha=0.8)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss Curves")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val Accuracy (%)")
    ax2.set_title("Validation Accuracy Curves")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_pareto_front(efficiency_results: list, save_path: str = None):
    """
    Plot FLOPs vs Accuracy (Pareto front).

    Args:
        efficiency_results: List of dicts with keys:
            model_name, family, flops_G, test_acc1
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for r in efficiency_results:
        name = r["model_name"]
        family = r.get("family", "unknown")
        color = FAMILY_COLORS.get(family, "gray")
        marker = MODEL_MARKERS.get(name, "o")

        ax.scatter(
            r["flops_G"], r["test_acc1"],
            c=color, marker=marker, s=150, edgecolors="black", linewidth=0.5,
            zorder=5,
        )
        ax.annotate(
            name.replace("_patch16_224", ""),
            (r["flops_G"], r["test_acc1"]),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=9,
        )

    # Legend for families
    for family, color in FAMILY_COLORS.items():
        ax.scatter([], [], c=color, s=80, label=family.upper(), edgecolors="black", linewidth=0.5)
    ax.legend(fontsize=11)

    ax.set_xlabel("FLOPs (G)", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Efficiency–Accuracy Pareto Front", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_robustness_comparison(fgsm_results: dict, save_path: str = None):
    """
    Plot accuracy vs FGSM epsilon for all models.

    Args:
        fgsm_results: {model_name: {epsilon: accuracy}}
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, eps_acc in fgsm_results.items():
        family = "cnn" if "resnet" in model_name or "convnext" in model_name else \
                 "vit" if "deit" in model_name else "mamba"
        color = FAMILY_COLORS.get(family, "gray")

        epsilons = sorted(eps_acc.keys())
        accs = [eps_acc[e] for e in epsilons]

        ax.plot(epsilons, accs, marker="o", label=model_name.replace("_patch16_224", ""),
                color=color, linewidth=2)

    ax.set_xlabel("FGSM Epsilon", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Adversarial Robustness (FGSM)", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def generate_comparison_table(results: dict):
    """Generate a formatted comparison table string."""
    header = f"{'Model':<25} {'Family':<8} {'Params(M)':<10} {'Test Acc@1':<12} {'Test Acc@5':<12} {'Train Time':<12}"
    lines = [header, "-" * len(header)]

    for model_name, data in sorted(results.items()):
        family = data.get("family", "?")
        params_m = data.get("num_params", 0) / 1e6
        test_acc1 = data.get("test_acc1", 0)
        test_acc5 = data.get("test_acc5", 0)
        train_time = data.get("total_train_time", 0) / 60  # minutes

        lines.append(
            f"{model_name:<25} {family:<8} {params_m:<10.2f} {test_acc1:<12.2f} {test_acc5:<12.2f} {train_time:<12.1f}min"
        )

    return "\n".join(lines)
