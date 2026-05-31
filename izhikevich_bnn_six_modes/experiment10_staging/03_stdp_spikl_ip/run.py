"""Entry point for the Izhikevich BNN six-mode stimulation experiment."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.decoding import classify_round, compute_confusion_matrix, extract_v100
from src.experiment import Experiment
from src.visualize import (
    plot_channel_heatmap,
    plot_confusion_matrix,
    plot_firing_rate_evolution,
    plot_intrinsic_parameter_distribution,
    plot_ip_metric_evolution,
    plot_learning_curve,
    plot_rate_distribution,
    plot_spatial_activity,
    plot_weight_distribution,
)


def print_table(round_accs, round_v100):
    print("\n+-------+----------------------+----------------------+")
    print("| Round | Raw count features   | V100 features        |")
    print("+-------+----------------------+----------------------+")
    for i, (raw, v100) in enumerate(zip(round_accs, round_v100)):
        print(f"|  {i+1:<4} | {raw[0]*100:5.1f}% +/- {raw[1]*100:4.1f}%  |"
              f" {v100[0]*100:5.1f}% +/- {v100[1]*100:4.1f}%  |")
    print("+-------+----------------------+----------------------+")


def load_group_settings(run_dir: Path) -> dict:
    settings_path = run_dir / "group_settings.json"
    if not settings_path.exists():
        return {}
    with open(settings_path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_group_settings(cfg: Config, settings: dict) -> None:
    if "seed" in settings:
        cfg.experiment.seed = int(settings["seed"])

    if "stdp_enabled" in settings:
        cfg.plasticity.stdp.enabled = bool(settings["stdp_enabled"])

    ip_cfg = cfg.plasticity.ip
    if "ip_enabled" in settings:
        ip_cfg.enabled = bool(settings["ip_enabled"])
    if "ip_method" in settings:
        ip_cfg.method = str(settings["ip_method"])
    if "target_mu_hz" in settings:
        ip_cfg.target_mu_hz = float(settings["target_mu_hz"])
    if "tau_cal_ms" in settings:
        ip_cfg.tau_cal_ms = float(settings["tau_cal_ms"])
    if "epsilon_hz" in settings:
        ip_cfg.epsilon_hz = float(settings["epsilon_hz"])
    if "eta_bias" in settings:
        ip_cfg.eta_bias = float(settings["eta_bias"])
    if "eta_gain" in settings:
        ip_cfg.eta_gain = float(settings["eta_gain"])
    if "bias_min" in settings:
        ip_cfg.bias_min = float(settings["bias_min"])
    if "bias_max" in settings:
        ip_cfg.bias_max = float(settings["bias_max"])
    if "gain_min" in settings:
        ip_cfg.gain_min = float(settings["gain_min"])
    if "gain_max" in settings:
        ip_cfg.gain_max = float(settings["gain_max"])
    if "stimulus_current_gain" in settings:
        cfg.network.stimulus_current_gain = float(settings["stimulus_current_gain"])


def main():
    run_dir = Path(__file__).parent
    output_dir = run_dir / "output"
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    settings = load_group_settings(run_dir)
    apply_group_settings(cfg, settings)
    exp = Experiment(cfg)
    n_rounds = cfg.experiment.n_rounds
    reps = cfg.experiment.reps_per_mode
    group_name = settings.get("group_name", run_dir.name)

    print("=" * 50)
    print("  Experiment 10: Izhikevich BNN + Intrinsic Plasticity")
    print("=" * 50)
    print(f"  Group: {group_name}")
    print(f"  Neurons: {cfg.network.n_total} ({cfg.network.n_exc}E + {cfg.network.n_inh}I)")
    print(f"  Recording channels: {cfg.network.n_record}")
    print(f"  Synapses: {exp.model.n_edges}")
    print(f"  Time step: {cfg.experiment.dt_ms} ms")
    print(f"  Training: {n_rounds} rounds x {reps} trials/mode x 6 modes")
    print(f"  STDP: {'on' if cfg.plasticity.stdp.enabled else 'off'}")
    print(
        "  IP: "
        f"{'on' if cfg.plasticity.ip.enabled else 'off'}"
        f" ({cfg.plasticity.ip.method})"
    )
    print()

    t0 = time.time()
    print("[1/4] Warming up network...", end=" ", flush=True)
    warmup_counts, mfr = exp.warmup()
    print(f"done (mean firing rate: {mfr:.1f} Hz, elapsed: {time.time()-t0:.1f}s)")

    t1 = time.time()
    print("[2/4] Calibrating stimuli...", end=" ", flush=True)
    dist, alpha, sigmas = exp.select_and_calibrate(warmup_counts)
    print(f"done (hotspot distance: {dist:.2f} mm, alpha={alpha:.4f})")

    print(f"[3/4] Training ({n_rounds} rounds):")

    def on_round_done(rnd, total):
        elapsed = time.time() - t1
        print(f"      Round {rnd}/{total} done ({elapsed:.0f}s)")

    result = exp.run_rounds(callback=on_round_done)

    print("[4/4] Analyzing and plotting...", end=" ", flush=True)
    trials_per_round = 6 * reps
    round_accs = []
    round_v100 = []
    for rnd in range(n_rounds):
        start = rnd * trials_per_round
        end = start + trials_per_round
        counts_list = result.trial_channel_counts[start:end]
        labels = np.array(result.mode_sequence[start:end])
        features_raw = np.array(counts_list, dtype=float)
        features_v100 = np.array([extract_v100(c) for c in counts_list])
        round_accs.append(classify_round(features_raw, labels))
        round_v100.append(classify_round(features_v100, labels))

    last_start = (n_rounds - 1) * trials_per_round
    last_counts = result.trial_channel_counts[last_start:]
    last_labels = np.array(result.mode_sequence[last_start:])
    last_features = np.array(last_counts, dtype=float)
    cm = compute_confusion_matrix(last_features, last_labels)

    plot_learning_curve(round_accs, round_v100, fig_dir)
    plot_firing_rate_evolution(result, n_rounds, reps, fig_dir)
    plot_channel_heatmap(result, n_rounds, reps, fig_dir)
    plot_weight_distribution(result, fig_dir)
    plot_spatial_activity(result, n_rounds, reps, fig_dir)
    plot_confusion_matrix(cm, sorted(set(result.mode_sequence)), fig_dir)
    plot_ip_metric_evolution(result, fig_dir)
    plot_rate_distribution(result, fig_dir, cfg.plasticity.ip.target_mu_hz)
    plot_intrinsic_parameter_distribution(result, fig_dir)
    print("done")

    print("\nClassification results (10-fold CV, chance level=16.7%):")
    print_table(round_accs, round_v100)

    print("\nFunctional connectivity metrics (channel activity correlation):")
    print("+-------+-----------------+-------------------+------------+")
    print("| Round | Mean correlation | Global efficiency | Modularity |")
    print("+-------+-----------------+-------------------+------------+")
    for i, fm in enumerate(result.functional_metrics):
        print(f"|  {i+1:<4} |     {fm['avg_correlation']:.4f}      |"
              f"      {fm['global_efficiency']:.4f}       |   {fm['modularity']:.4f}   |")
    print("+-------+-----------------+-------------------+------------+")

    if result.ip_metrics:
        print("\nIntrinsic plasticity metrics:")
        print("+-------+------------+------------+--------------+-------------+")
        print("| Round | KL div.    | Silent %   | Burst-like % | Mean rateHz |")
        print("+-------+------------+------------+--------------+-------------+")
        for i, ipm in enumerate(result.ip_metrics):
            print(
                f"|  {i+1:<4} |   {ipm['kl_divergence']:7.4f} |"
                f"   {ipm['silent_neuron_ratio']*100:6.2f}% |"
                f"    {ipm['burst_like_trial_ratio']*100:6.2f}% |"
                f"   {ipm['mean_rate_hz']:8.2f} |"
            )
        print("+-------+------------+------------+--------------+-------------+")

    results_data = {
        "group": group_name,
        "config": {
            "n_total": cfg.network.n_total,
            "n_rounds": n_rounds,
            "reps_per_mode": reps,
            "stimulus_gain": cfg.network.stimulus_current_gain,
            "seed": cfg.experiment.seed,
            "stdp_enabled": cfg.plasticity.stdp.enabled,
            "ip_enabled": cfg.plasticity.ip.enabled,
            "ip_method": cfg.plasticity.ip.method,
            "ip_target_mu_hz": cfg.plasticity.ip.target_mu_hz,
            "ip_tau_cal_ms": cfg.plasticity.ip.tau_cal_ms,
            "ip_eta_bias": cfg.plasticity.ip.eta_bias,
            "ip_eta_gain": cfg.plasticity.ip.eta_gain,
        },
        "hotspots": {
            "distance_mm": float(dist),
            "alpha": float(alpha),
        },
        "classification": {
            "raw_counts": [{"mean": a[0], "std": a[1]} for a in round_accs],
            "v100": [{"mean": a[0], "std": a[1]} for a in round_v100],
        },
        "functional_metrics": result.functional_metrics,
        "warmup_ip_metrics": result.warmup_ip_metrics,
        "ip_metrics": result.ip_metrics,
    }
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    total_time = time.time() - t0
    print(f"\nTotal elapsed time: {total_time:.1f}s")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
