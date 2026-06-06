from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN, KMeans

from .config import Config


def select_hotspots(positions: np.ndarray, record_indices: np.ndarray,
                    spike_counts: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    stim = cfg.stim
    mfr = spike_counts.astype(float)
    n_top = max(2, int(len(mfr) * stim.top_fraction))
    top_idx = np.argsort(mfr)[-n_top:]
    top_positions = positions[record_indices[top_idx]]
    top_mfr = mfr[top_idx]

    labels = DBSCAN(eps=stim.dbscan_eps_mm, min_samples=stim.dbscan_min_samples).fit_predict(top_positions)
    unique_labels = set(labels) - {-1}
    if len(unique_labels) < 2:
        labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(top_positions)
        unique_labels = {0, 1}

    clusters = []
    for lab in unique_labels:
        mask = labels == lab
        weights = top_mfr[mask]
        if weights.sum() > 0:
            center = np.average(top_positions[mask], axis=0, weights=weights)
        else:
            center = top_positions[mask].mean(axis=0)
        clusters.append((center, weights.sum()))
    clusters.sort(key=lambda x: -x[1])

    best_pair, best_score = None, -1.0
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            dist = np.linalg.norm(clusters[i][0] - clusters[j][0])
            if dist < stim.min_distance_mm:
                continue
            score = clusters[i][1] + clusters[j][1]
            if score > best_score:
                best_score = score
                best_pair = (i, j)
    if best_pair is None:
        best_pair = (0, 1)
    return clusters[best_pair[0]][0], clusters[best_pair[1]][0]


def calibrate_stimuli(positions: np.ndarray, area_centers: list[np.ndarray],
                      cfg: Config) -> tuple[float, dict[float, float]]:
    stim = cfg.stim
    net = cfg.network
    best_error, best_alpha = float("inf"), stim.alpha_grid[0]
    best_sigmas: dict[float, float] = {}
    for alpha in stim.alpha_grid:
        for sigma in stim.sigma_grid:
            total_err = 0.0
            for center in area_centers:
                dists = np.linalg.norm(positions - center, axis=1)
                for volt, target in zip(stim.voltages_mv, stim.target_counts):
                    field = alpha * volt * np.exp(-dists**2 / (2 * sigma**2))
                    total_err += abs((field >= net.recruitment_threshold).sum() - target)
            if total_err < best_error:
                best_error = total_err
                best_alpha = alpha
                best_sigmas = {v: sigma for v in stim.voltages_mv}
    return best_alpha, best_sigmas


def build_pulse_currents(positions: np.ndarray, center: np.ndarray, voltage_mv: float,
                         alpha: float, sigma: float, gain: float,
                         pulse_phases: tuple[tuple[int, float], ...]) -> dict[int, np.ndarray]:
    dists = np.linalg.norm(positions - center, axis=1)
    base_field = alpha * voltage_mv * np.exp(-dists**2 / (2 * sigma**2)) * gain
    currents: dict[int, np.ndarray] = {}
    t = 0
    for duration_steps, scale in pulse_phases:
        for _ in range(duration_steps):
            if scale != 0.0:
                currents[t] = base_field * scale
            t += 1
    return currents
