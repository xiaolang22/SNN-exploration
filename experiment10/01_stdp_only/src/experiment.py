from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import networkx as nx
import community as community_louvain

from .config import Config
from .network import ReservoirModel
from .stimulation import build_pulse_currents, calibrate_stimuli, select_hotspots


@dataclass
class ModeSpec:
    mode_id: int
    area_center: np.ndarray
    voltage_mv: float
    label: str


@dataclass
class ExperimentResult:
    mode_sequence: list[str] = field(default_factory=list)
    trial_mfr: list[float] = field(default_factory=list)
    trial_channel_counts: list[np.ndarray] = field(default_factory=list)
    weight_snapshots: dict[str, np.ndarray] = field(default_factory=dict)
    intrinsic_snapshots: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    functional_metrics: list[dict[str, float]] = field(default_factory=list)
    ip_metrics: list[dict[str, float]] = field(default_factory=list)
    area_centers: list[np.ndarray] = field(default_factory=list)
    positions: np.ndarray | None = None
    record_indices: np.ndarray | None = None
    area_distance_mm: float = 0.0
    alpha: float = 0.0
    sigmas: dict[float, float] = field(default_factory=dict)
    warmup_ip_metrics: dict[str, float] = field(default_factory=dict)


class Experiment:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.experiment.seed)
        self.model = ReservoirModel(cfg, self.rng)
        self.alpha: float = 0.0
        self.sigmas: dict[float, float] = {}
        self.modes: list[ModeSpec] = []
        self.result = ExperimentResult()
        self.result.positions = self.model.positions
        self.result.record_indices = self.model.record_indices

    def warmup(self) -> tuple[np.ndarray, float]:
        exp = self.cfg.experiment
        total_steps = int(exp.warmup_ms / exp.dt_ms)
        window_steps = int(exp.warmup_mfr_window_ms / exp.dt_ms)
        start_record = total_steps - window_steps
        trial_steps = max(1, int(exp.trial_duration_ms / exp.dt_ms))
        n_rec = len(self.model.record_indices)
        counts = np.zeros(n_rec, dtype=np.int32)

        for t in range(total_steps):
            fired = self.model.step(plasticity=True)
            if (t + 1) % trial_steps == 0:
                self.model.apply_ip_update()
            if t >= start_record:
                rec_fired = np.isin(fired, self.model.record_indices)
                for nidx in fired[rec_fired]:
                    counts[np.searchsorted(self.model.record_indices, nidx)] += 1

        if total_steps % trial_steps != 0:
            self.model.apply_ip_update()
        self.result.weight_snapshots["post_warmup"] = self.model.get_exc_weights()
        self.result.intrinsic_snapshots["post_warmup"] = self.model.get_ip_snapshot()
        self.result.warmup_ip_metrics = self.model.get_ip_metrics()
        mfr = counts.mean() / (exp.warmup_mfr_window_ms / 1000.0)
        return counts, mfr

    def select_and_calibrate(self, warmup_counts: np.ndarray) -> tuple[float, float, dict]:
        area1, area2 = select_hotspots(
            self.model.positions, self.model.record_indices, warmup_counts, self.cfg
        )
        dist = np.linalg.norm(area1 - area2)
        self.result.area_centers = [area1, area2]
        self.result.area_distance_mm = dist

        self.alpha, self.sigmas = calibrate_stimuli(
            self.model.positions, [area1, area2], self.cfg
        )
        self.result.alpha = self.alpha
        self.result.sigmas = self.sigmas

        voltages = self.cfg.stim.voltages_mv
        self.modes = [
            ModeSpec(1, area1, voltages[0], "A1_100mV"),
            ModeSpec(2, area1, voltages[1], "A1_300mV"),
            ModeSpec(3, area1, voltages[2], "A1_500mV"),
            ModeSpec(4, area2, voltages[0], "A2_100mV"),
            ModeSpec(5, area2, voltages[1], "A2_300mV"),
            ModeSpec(6, area2, voltages[2], "A2_500mV"),
        ]
        return dist, self.alpha, self.sigmas

    def _compute_functional_metrics(self, round_counts: list[np.ndarray]) -> dict[str, float]:
        mat = np.array(round_counts, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.corrcoef(mat.T)
        np.fill_diagonal(corr, 0)
        corr = np.nan_to_num(corr, nan=0.0)

        n = corr.shape[0]
        triu_idx = np.triu_indices(n, k=1)
        avg_corr = float(corr[triu_idx].mean())

        threshold = 0.1
        adj = (np.abs(corr) > threshold).astype(float)
        np.fill_diagonal(adj, 0)

        G = nx.from_numpy_array(adj)
        efficiency = nx.global_efficiency(G)
        partition = community_louvain.best_partition(G)
        modularity = community_louvain.modularity(partition, G)

        return {
            "avg_correlation": avg_corr,
            "global_efficiency": float(efficiency),
            "modularity": float(modularity),
        }

    def _compute_round_ip_metrics(self, round_mfr: list[float]) -> dict[str, float]:
        metrics = self.model.get_ip_metrics()
        if not round_mfr:
            metrics["burst_like_trial_ratio"] = 0.0
            metrics["max_trial_mfr"] = 0.0
            metrics["trial_mfr_cv"] = 0.0
            return metrics

        mfr_arr = np.asarray(round_mfr, dtype=float)
        metrics["burst_like_trial_ratio"] = float(
            np.mean(mfr_arr > self.cfg.plasticity.ip.burst_trial_threshold_hz)
        )
        metrics["max_trial_mfr"] = float(mfr_arr.max())
        metrics["trial_mfr_cv"] = float(mfr_arr.std() / mfr_arr.mean()) if mfr_arr.mean() > 0 else 0.0
        return metrics

    def run_rounds(self, callback=None) -> ExperimentResult:
        exp = self.cfg.experiment
        n_rounds = exp.n_rounds
        reps = exp.reps_per_mode
        mode_ids = [1, 2, 3, 4, 5, 6]
        trials_per_round = 6 * reps

        for rnd in range(n_rounds):
            trial_plan = []
            for mid in mode_ids:
                trial_plan.extend([mid] * reps)

            round_mfr: list[float] = []
            for mode_id in trial_plan:
                mode = self.modes[mode_id - 1]
                sigma = self.sigmas[mode.voltage_mv]
                pulse_currents = build_pulse_currents(
                    self.model.positions, mode.area_center, mode.voltage_mv,
                    self.alpha, sigma, self.cfg.network.stimulus_current_gain,
                    self.cfg.stim.pulse_phases,
                )
                counts = self.model.run_segment(
                    exp.trial_duration_ms, pulse_currents, plasticity=True,
                    response_window_ms=(exp.response_window_start_ms, exp.response_window_ms),
                )
                response_duration_s = (
                    exp.response_window_ms - exp.response_window_start_ms
                ) / 1000.0
                mfr = counts.sum() / response_duration_s / len(counts)
                self.model.apply_ip_update()
                self.result.trial_mfr.append(mfr)
                round_mfr.append(mfr)
                self.result.trial_channel_counts.append(counts)
                self.result.mode_sequence.append(mode.label)

            self.result.weight_snapshots[f"post_round{rnd + 1}"] = self.model.get_exc_weights()
            self.result.intrinsic_snapshots[f"post_round{rnd + 1}"] = self.model.get_ip_snapshot()
            round_start = rnd * trials_per_round
            round_counts = self.result.trial_channel_counts[round_start:round_start + trials_per_round]
            self.result.functional_metrics.append(self._compute_functional_metrics(round_counts))
            self.result.ip_metrics.append(self._compute_round_ip_metrics(round_mfr))
            if callback:
                callback(rnd + 1, n_rounds)

        return self.result
