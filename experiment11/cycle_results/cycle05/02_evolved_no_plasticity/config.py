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
    enabled: bool = True
    a_plus: float = 0.004
    a_minus: float = 0.005
    tau_pre_ms: float = 20.0
    tau_post_ms: float = 20.0


@dataclass
class PlasticityParams:
    stdp: STDPParams = field(default_factory=STDPParams)


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
class StructureEvolutionParams:
    topology_mode: str = "random"
    population_size: int = 16
    elite_size: int = 4
    offspring_per_elite: int = 3
    generations: int = 12
    innovation_count: int = 5
    mutation_rate: float = 0.01
    short_eval_warmup_ms: float = 2500.0
    short_eval_trials_per_mode: int = 4
    short_eval_trial_duration_ms: float = 200.0
    short_response_window_start_ms: float = 5.0
    short_response_window_end_ms: float = 55.0
    adaptation_trials_per_mode: int = 1
    post_adaptation_trials_per_mode: int = 2
    fitness_rank_weight: float = 0.05
    fitness_pattern_weight: float = 0.15
    fitness_raw_distance_weight: float = 0.20
    fitness_decode_weight: float = 0.15
    fitness_plasticity_decode_weight: float = 0.50
    quick_decode_folds: int = 3
    fitness_activity_weight: float = 0.06
    activity_target: float = 0.72
    activity_tolerance: float = 0.24
    fitness_burst_weight: float = 0.60
    fitness_density_weight: float = 0.30
    pattern_ratio_floor: float = 1.10
    pattern_ratio_ceiling: float = 1.90
    burst_active_fraction_threshold: float = 0.075
    burst_trial_active_fraction_threshold: float = 0.16
    silent_repair_additions: int = 2
    local_rewire_count: int = 4
    bridge_rewire_count: int = 8
    degree_swap_factor: int = 5
    evaluation_seed_offset: int = 1000
    max_rank_normalization_trials: int = 24


@dataclass
class Config:
    network: NetworkParams = field(default_factory=NetworkParams)
    weights: WeightParams = field(default_factory=WeightParams)
    plasticity: PlasticityParams = field(default_factory=PlasticityParams)
    stim: StimParams = field(default_factory=StimParams)
    experiment: ExperimentParams = field(default_factory=ExperimentParams)
    structure: StructureEvolutionParams = field(default_factory=StructureEvolutionParams)
