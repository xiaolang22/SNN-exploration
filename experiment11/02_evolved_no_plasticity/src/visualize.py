from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from .experiment import ExperimentResult

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def _setup_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _save(fig, output_dir: Path, name: str):
    fig.savefig(output_dir / f"{name}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_learning_curve(round_accs: list[tuple[float, float]],
                        round_v100: list[tuple[float, float]],
                        output_dir: Path) -> None:
    _setup_style()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    rounds = np.arange(1, len(round_accs) + 1)

    means = [a[0] for a in round_accs]
    stds = [a[1] for a in round_accs]
    ax.errorbar(rounds, means, yerr=stds, fmt="o-", color=COLORS[0],
                capsize=3, markersize=5, label="Raw count features")

    means_v = [a[0] for a in round_v100]
    stds_v = [a[1] for a in round_v100]
    ax.errorbar(rounds, means_v, yerr=stds_v, fmt="s--", color=COLORS[1],
                capsize=3, markersize=5, label="V100 features")

    ax.axhline(1.0 / 6, color="gray", linestyle=":", linewidth=1, label="Chance level")
    ax.set_xlabel("Training round")
    ax.set_ylabel("Classification accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(rounds)
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, output_dir, "learning_curve")


def plot_firing_rate_evolution(result: ExperimentResult, n_rounds: int,
                               reps_per_mode: int, output_dir: Path) -> None:
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 3))
    trials = np.arange(len(result.trial_mfr))
    ax.plot(trials, result.trial_mfr, color="k", linewidth=0.4, alpha=0.7)

    window = 10
    if len(result.trial_mfr) > window:
        smoothed = np.convolve(result.trial_mfr, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window - 1, len(result.trial_mfr)), smoothed,
                color=COLORS[0], linewidth=1.5, label=f"Moving average (n={window})")

    trials_per_round = 6 * reps_per_mode
    for i in range(1, n_rounds):
        ax.axvline(i * trials_per_round, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)

    ax.set_xlabel("Trial")
    ax.set_ylabel("Mean firing rate (Hz)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, output_dir, "firing_rate_evolution")


def plot_channel_heatmap(result: ExperimentResult, n_rounds: int,
                         reps_per_mode: int, output_dir: Path) -> None:
    _setup_style()
    mat = np.array(result.trial_channel_counts)
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(mat.T, aspect="auto", cmap="inferno", interpolation="nearest")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Recording channel")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Spike count")

    trials_per_round = 6 * reps_per_mode
    for i in range(1, n_rounds):
        ax.axvline(i * trials_per_round - 0.5, color="cyan", linestyle="--", linewidth=0.6)

    fig.tight_layout()
    _save(fig, output_dir, "channel_heatmap")


def plot_weight_distribution(result: ExperimentResult, output_dir: Path) -> None:
    _setup_style()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bins = np.linspace(0, 12, 50)

    keys_to_plot = ["post_warmup"]
    for k in sorted(result.weight_snapshots.keys()):
        if k.startswith("post_round"):
            keys_to_plot.append(k)
    keys_to_plot = [keys_to_plot[0], keys_to_plot[len(keys_to_plot) // 2], keys_to_plot[-1]]

    labels_map = {"post_warmup": "Post-warmup"}
    for k in keys_to_plot:
        if k.startswith("post_round"):
            labels_map[k] = f"After round {k.replace('post_round', '')}"

    for i, key in enumerate(keys_to_plot):
        if key in result.weight_snapshots:
            ax.hist(result.weight_snapshots[key], bins=bins, alpha=0.6,
                    color=COLORS[i], density=True, label=labels_map.get(key, key))

    ax.set_xlabel("Excitatory synaptic weight")
    ax.set_ylabel("Probability density")
    ax.legend()
    fig.tight_layout()
    _save(fig, output_dir, "weight_distribution")


def plot_spatial_activity(result: ExperimentResult, n_rounds: int,
                          reps_per_mode: int, output_dir: Path) -> None:
    _setup_style()
    trials_per_round = 6 * reps_per_mode
    labels = result.mode_sequence
    last_round_start = (n_rounds - 1) * trials_per_round

    mode_names = ["A1_100mV", "A1_300mV", "A1_500mV", "A2_100mV", "A2_300mV", "A2_500mV"]
    positions = result.positions[result.record_indices]

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.flatten()

    for idx, mode_name in enumerate(mode_names):
        mode_trials = [i for i in range(last_round_start, len(labels)) if labels[i] == mode_name]
        if not mode_trials:
            continue
        counts = np.mean([result.trial_channel_counts[i] for i in mode_trials], axis=0)
        ax = axes[idx]
        sc = ax.scatter(positions[:, 0], positions[:, 1], c=counts, cmap="YlOrRd",
                        s=15, edgecolors="k", linewidths=0.2, vmin=0)
        for center in result.area_centers:
            ax.plot(center[0], center[1], "b*", markersize=10)
        ax.set_title(mode_name, fontsize=10)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        fig.colorbar(sc, ax=ax, shrink=0.7)

    fig.tight_layout()
    _save(fig, output_dir, "spatial_activity")


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], output_dir: Path) -> None:
    _setup_style()
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    _save(fig, output_dir, "confusion_matrix")
