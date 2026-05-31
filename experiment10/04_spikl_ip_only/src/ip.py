from __future__ import annotations

import numpy as np

from .config import IPParams


def update_calcium(
    calcium: np.ndarray,
    fired: np.ndarray,
    dt_ms: float,
    tau_cal_ms: float,
) -> None:
    decay = np.exp(-dt_ms / tau_cal_ms)
    calcium *= decay
    if len(fired) > 0:
        calcium[fired] += 1.0


def rates_from_calcium(calcium: np.ndarray, tau_cal_ms: float) -> np.ndarray:
    return 1000.0 * calcium / max(tau_cal_ms, 1e-9)


def _clip_update(error: np.ndarray, ip_cfg: IPParams) -> np.ndarray:
    scale = max(ip_cfg.target_mu_hz, 1e-9)
    normalized = error / scale
    return np.clip(normalized, -ip_cfg.update_clip, ip_cfg.update_clip)


def apply_rate_target_ip(
    bias_current: np.ndarray,
    input_gain: np.ndarray,
    rates_hz: np.ndarray,
    ip_cfg: IPParams,
) -> None:
    update = _clip_update(ip_cfg.target_mu_hz - rates_hz, ip_cfg)
    bias_current += ip_cfg.eta_bias * update
    input_gain += ip_cfg.eta_gain * update

    silent_mask = rates_hz <= ip_cfg.epsilon_hz
    if np.any(silent_mask):
        bias_current[silent_mask] += ip_cfg.eta_bias * ip_cfg.silent_bias_boost
        input_gain[silent_mask] += ip_cfg.eta_gain * ip_cfg.silent_gain_boost

    np.clip(bias_current, ip_cfg.bias_min, ip_cfg.bias_max, out=bias_current)
    np.clip(input_gain, ip_cfg.gain_min, ip_cfg.gain_max, out=input_gain)


def apply_spikl_ip(
    bias_current: np.ndarray,
    input_gain: np.ndarray,
    rates_hz: np.ndarray,
    ip_cfg: IPParams,
) -> None:
    n_neurons = len(rates_hz)
    if n_neurons == 0:
        return

    order = np.argsort(rates_hz)
    sorted_rates = rates_hz[order]
    quantiles = (np.arange(n_neurons, dtype=float) + 0.5) / n_neurons
    target_sorted = -ip_cfg.target_mu_hz * np.log(np.clip(1.0 - quantiles, 1e-9, 1.0))

    sorted_error = target_sorted - sorted_rates
    update = np.empty_like(sorted_error)
    update[order] = _clip_update(sorted_error, ip_cfg)

    bias_current += ip_cfg.eta_bias * update
    input_gain += ip_cfg.eta_gain * update

    silent_mask = rates_hz <= ip_cfg.epsilon_hz
    if np.any(silent_mask):
        bias_current[silent_mask] += ip_cfg.eta_bias * ip_cfg.silent_bias_boost
        input_gain[silent_mask] += ip_cfg.eta_gain * ip_cfg.silent_gain_boost

    np.clip(bias_current, ip_cfg.bias_min, ip_cfg.bias_max, out=bias_current)
    np.clip(input_gain, ip_cfg.gain_min, ip_cfg.gain_max, out=input_gain)


def apply_intrinsic_plasticity(
    bias_current: np.ndarray,
    input_gain: np.ndarray,
    rates_hz: np.ndarray,
    ip_cfg: IPParams,
) -> None:
    if not ip_cfg.enabled or ip_cfg.method == "none":
        return
    if ip_cfg.method == "rate_target":
        apply_rate_target_ip(bias_current, input_gain, rates_hz, ip_cfg)
        return
    if ip_cfg.method == "spikl":
        apply_spikl_ip(bias_current, input_gain, rates_hz, ip_cfg)
        return
    raise ValueError(f"Unsupported IP method: {ip_cfg.method}")


def _distribution_entropy_bits(values: np.ndarray, bins: np.ndarray) -> float:
    hist, _ = np.histogram(values, bins=bins)
    probs = hist.astype(float)
    total = probs.sum()
    if total <= 0:
        return 0.0
    probs /= total
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def kl_divergence_to_exponential(rates_hz: np.ndarray, target_mu_hz: float) -> float:
    if len(rates_hz) == 0:
        return 0.0
    upper = max(float(rates_hz.max()), 5.0 * target_mu_hz, 1.0)
    bins = np.linspace(0.0, upper, 41)
    hist, edges = np.histogram(rates_hz, bins=bins)
    empirical = hist.astype(float)
    empirical_sum = empirical.sum()
    if empirical_sum <= 0:
        return 0.0
    empirical /= empirical_sum

    scale = max(target_mu_hz, 1e-9)
    target_cdf = 1.0 - np.exp(-edges / scale)
    target = np.diff(target_cdf)
    target /= max(target.sum(), 1e-9)

    mask = empirical > 0
    return float(np.sum(empirical[mask] * np.log(empirical[mask] / np.clip(target[mask], 1e-12, None))))


def summarize_ip_state(
    rates_hz: np.ndarray,
    bias_current: np.ndarray,
    input_gain: np.ndarray,
    ip_cfg: IPParams,
) -> dict[str, float]:
    if len(rates_hz) == 0:
        return {
            "mean_rate_hz": 0.0,
            "silent_neuron_ratio": 0.0,
            "distribution_entropy_bits": 0.0,
            "kl_divergence": 0.0,
            "bias_mean": 0.0,
            "bias_std": 0.0,
            "gain_mean": 0.0,
            "gain_std": 0.0,
        }

    upper = max(float(rates_hz.max()), 5.0 * ip_cfg.target_mu_hz, 1.0)
    bins = np.linspace(0.0, upper, 41)
    return {
        "mean_rate_hz": float(rates_hz.mean()),
        "silent_neuron_ratio": float(np.mean(rates_hz <= ip_cfg.epsilon_hz)),
        "distribution_entropy_bits": _distribution_entropy_bits(rates_hz, bins),
        "kl_divergence": kl_divergence_to_exponential(rates_hz, ip_cfg.target_mu_hz),
        "bias_mean": float(bias_current.mean()),
        "bias_std": float(bias_current.std()),
        "gain_mean": float(input_gain.mean()),
        "gain_std": float(input_gain.std()),
    }
