from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NetworkParams:
    n_total: int = 1000
    n_exc: int = 800
    n_inh: int = 200
    record_exc: int = 80
    record_inh: int = 20
    plane_width_mm: float = 3.85
    plane_height_mm: float = 2.10
    connection_prob: float = 0.05
    distance_lambda_mm: float = 0.90
    dc_offset: float = 1.5
    noise_std: float = 1.0
    synaptic_tau_ms: float = 5.0
    synaptic_current_limit: float = 50.0
    stimulus_current_gain: float = 40.0
    recruitment_threshold: float = 0.5

    @property
    def n_record(self) -> int:
        return self.record_exc + self.record_inh


@dataclass
class WeightParams:
    exc_mu: float = 0.0
    exc_sigma: float = 0.25
    inh_mu: float = 0.4
    inh_sigma: float = 0.2
    w_min: float = 0.05
    w_max: float = 3.0


@dataclass
class STDPParams:
    enabled: bool = False
    a_plus: float = 0.004
    a_minus: float = 0.005
    tau_pre_ms: float = 20.0
    tau_post_ms: float = 20.0


@dataclass
class PlasticityParams:
    stdp: STDPParams = field(default_factory=STDPParams)


@dataclass
class TopologyPlasticityParams:
    enabled: bool = True
    update_interval_rounds: int = 1
    max_add_per_update: int = 48
    max_prune_per_update: int = 48
    max_edge_fraction_per_update: float = 0.002
    min_pair_cooccurrence: int = 3
    delay_window_ms: tuple[float, float] = (1.0, 20.0)
    class_specificity_weight: float = 0.4
    pair_count_weight: float = 0.9
    consistency_weight: float = 0.55
    distance_penalty: float = 0.1
    inactive_quantile: float = 0.2
    protect_min_in_degree: int = 1
    protect_min_out_degree: int = 1
    max_candidate_distance_mm: float = 1.5
    keep_total_edges_constant: bool = True


@dataclass
class StimParams:
    top_fraction: float = 0.10
    min_distance_mm: float = 1.0
    dbscan_eps_mm: float = 0.30
    dbscan_min_samples: int = 3
    voltages_mv: tuple[float, ...] = (100.0, 300.0, 500.0)
    target_counts: tuple[int, ...] = (20, 80, 200)
    alpha_grid: tuple[float, ...] = (0.005, 0.010, 0.015, 0.020, 0.030, 0.040, 0.050)
    sigma_grid: tuple[float, ...] = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00)
    pulse_phases: tuple[tuple[int, float], ...] = ((1, 1.0), (1, -1.0), (3, 0.0))


@dataclass
class ExperimentParams:
    dt_ms: float = 1.0
    warmup_ms: float = 20000.0
    warmup_mfr_window_ms: float = 5000.0
    trial_duration_ms: float = 200.0
    response_window_start_ms: float = 5.0
    response_window_ms: float = 55.0
    n_rounds: int = 3
    reps_per_mode: int = 20
    seed: int = 42


@dataclass
class Config:
    network: NetworkParams = field(default_factory=NetworkParams)
    weights: WeightParams = field(default_factory=WeightParams)
    plasticity: PlasticityParams = field(default_factory=PlasticityParams)
    topology_plasticity: TopologyPlasticityParams = field(default_factory=TopologyPlasticityParams)
    stim: StimParams = field(default_factory=StimParams)
    experiment: ExperimentParams = field(default_factory=ExperimentParams)
