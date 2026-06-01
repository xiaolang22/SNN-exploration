from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold

from src.config import Config
from src.experiment import Experiment
from src.network import ReservoirModel
from src.stimulation import build_pulse_currents


@dataclass
class TopologySpec:
    label: str
    adjacency: np.ndarray
    blueprint: dict

    def to_override(self) -> dict:
        override = copy.deepcopy(self.blueprint)
        override["adjacency"] = self.adjacency.copy()
        return override


@dataclass
class CandidateMetrics:
    fitness: float
    separation_score: float
    rank_score: float
    distance_ratio: float
    pattern_distance_ratio: float
    decode_score: float
    plasticity_decode_score: float
    activity_score: float
    silent_neuron_ratio: float
    burst_penalty: float
    density_penalty: float
    total_synapses: int
    active_neurons: np.ndarray
    silent_neurons: np.ndarray
    state_matrix: np.ndarray
    record_counts: np.ndarray

    def to_summary(self) -> dict[str, float]:
        return {
            "fitness": float(self.fitness),
            "separation_score": float(self.separation_score),
            "rank_score": float(self.rank_score),
            "distance_ratio": float(self.distance_ratio),
            "pattern_distance_ratio": float(self.pattern_distance_ratio),
            "decode_score": float(self.decode_score),
            "plasticity_decode_score": float(self.plasticity_decode_score),
            "activity_score": float(self.activity_score),
            "silent_neuron_ratio": float(self.silent_neuron_ratio),
            "burst_penalty": float(self.burst_penalty),
            "density_penalty": float(self.density_penalty),
            "total_synapses": int(self.total_synapses),
        }


@dataclass
class CandidateResult:
    topology: TopologySpec
    metrics: CandidateMetrics


@dataclass
class EvolutionBundle:
    selected_topology: TopologySpec
    selected_metrics: CandidateMetrics
    evolved_topology: TopologySpec
    evolved_metrics: CandidateMetrics
    degree_matched_topology: TopologySpec | None
    degree_matched_metrics: CandidateMetrics | None
    history: list[dict[str, float]]
    baseline_topology: TopologySpec
    baseline_metrics: CandidateMetrics
    area_centers: list[np.ndarray]
    alpha: float
    sigmas: dict[float, float]

    def to_summary(self) -> dict:
        payload = {
            "selected_label": self.selected_topology.label,
            "selected_metrics": self.selected_metrics.to_summary(),
            "evolved_metrics": self.evolved_metrics.to_summary(),
            "baseline_metrics": self.baseline_metrics.to_summary(),
            "history": self.history,
            "alpha": float(self.alpha),
            "sigmas": {str(k): float(v) for k, v in self.sigmas.items()},
            "area_centers": [center.tolist() for center in self.area_centers],
        }
        if self.degree_matched_topology is not None and self.degree_matched_metrics is not None:
            payload["degree_matched_metrics"] = self.degree_matched_metrics.to_summary()
        return payload


def summarize_topology(adjacency: np.ndarray, positions: np.ndarray) -> dict[str, float]:
    out_degree = adjacency.sum(axis=1)
    in_degree = adjacency.sum(axis=0)
    total_synapses = int(adjacency.sum())
    if total_synapses > 0:
        src_idx, tgt_idx = np.nonzero(adjacency)
        mean_distance = float(np.linalg.norm(positions[src_idx] - positions[tgt_idx], axis=1).mean())
    else:
        mean_distance = 0.0
    return {
        "total_synapses": total_synapses,
        "avg_in_degree": float(in_degree.mean()),
        "avg_out_degree": float(out_degree.mean()),
        "std_in_degree": float(in_degree.std()),
        "std_out_degree": float(out_degree.std()),
        "mean_connection_distance_mm": mean_distance,
    }


class StructureEvolver:
    def __init__(self, cfg: Config) -> None:
        self.cfg = copy.deepcopy(cfg)
        self.cfg.plasticity.stdp.enabled = False
        self.structure_cfg = self.cfg.structure
        self.base_seed = self.cfg.experiment.seed
        self.eval_seed_base = self.base_seed + self.structure_cfg.evaluation_seed_offset
        self._setup_reference_state()

    def _setup_reference_state(self) -> None:
        base_exp = Experiment(copy.deepcopy(self.cfg))
        warmup_counts, _ = base_exp.warmup()
        base_exp.select_and_calibrate(warmup_counts)
        self.reference_positions = base_exp.model.positions.copy()
        self.reference_record_indices = base_exp.model.record_indices.copy()
        self.reference_blueprint = base_exp.model.export_blueprint()
        self.reference_adjacency = base_exp.model.get_connectivity_mask()
        self.target_synapse_count = int(self.reference_adjacency.sum())
        self.area_centers = [center.copy() for center in base_exp.result.area_centers]
        self.alpha = float(base_exp.alpha)
        self.sigmas = dict(base_exp.sigmas)
        self.distance_matrix = np.linalg.norm(
            self.reference_positions[:, None, :] - self.reference_positions[None, :, :],
            axis=2,
        )
        self.prob_matrix = self.cfg.network.connection_prob * (
            0.35 + 0.65 * np.exp(-self.distance_matrix / self.cfg.network.distance_lambda_mm)
        )
        np.fill_diagonal(self.prob_matrix, 0.0)
        self.hotspot_neighborhoods = [
            self._nearest_neurons(center, 80) for center in self.area_centers
        ]
        self.baseline_topology = TopologySpec(
            label="baseline_random",
            adjacency=self.reference_adjacency.copy(),
            blueprint=self.reference_blueprint,
        )
        self.baseline_metrics = self.evaluate_topology(
            self.baseline_topology.adjacency,
            label=self.baseline_topology.label,
            eval_index=0,
        ).metrics

    def _nearest_neurons(self, center: np.ndarray, count: int) -> np.ndarray:
        dists = np.linalg.norm(self.reference_positions - center, axis=1)
        return np.argsort(dists)[:count]

    def _build_override(self, adjacency: np.ndarray) -> dict:
        override = copy.deepcopy(self.reference_blueprint)
        override["adjacency"] = adjacency.copy()
        return override

    def _make_model(self, adjacency: np.ndarray, eval_index: int) -> ReservoirModel:
        rng = np.random.default_rng(self.eval_seed_base + eval_index)
        eval_cfg = copy.deepcopy(self.cfg)
        return ReservoirModel(eval_cfg, rng, topology_override=self._build_override(adjacency))

    def build_random_topology(self, seed_offset: int = 0) -> np.ndarray:
        rng = np.random.default_rng(self.base_seed + seed_offset)
        adjacency = rng.random(self.prob_matrix.shape) < self.prob_matrix
        np.fill_diagonal(adjacency, False)
        return adjacency

    def evaluate_topology(self, adjacency: np.ndarray, label: str, eval_index: int) -> CandidateResult:
        model = self._make_model(adjacency, eval_index=eval_index)
        scfg = self.structure_cfg
        if scfg.short_eval_warmup_ms > 0:
            model.run_segment(
                scfg.short_eval_warmup_ms,
                plasticity=False,
            )

        states = []
        record_counts = []
        active_union = np.zeros(model.n_total, dtype=bool)
        peak_fractions = []
        labels = []
        mode_defs = [
            ("A1_100mV", self.area_centers[0], self.cfg.stim.voltages_mv[0]),
            ("A1_300mV", self.area_centers[0], self.cfg.stim.voltages_mv[1]),
            ("A1_500mV", self.area_centers[0], self.cfg.stim.voltages_mv[2]),
            ("A2_100mV", self.area_centers[1], self.cfg.stim.voltages_mv[0]),
            ("A2_300mV", self.area_centers[1], self.cfg.stim.voltages_mv[1]),
            ("A2_500mV", self.area_centers[1], self.cfg.stim.voltages_mv[2]),
        ]
        response_window = (
            scfg.short_response_window_start_ms,
            scfg.short_response_window_end_ms,
        )
        pulse_cache = {
            (mode_label, voltage): build_pulse_currents(
                model.positions,
                center,
                voltage,
                self.alpha,
                self.sigmas[voltage],
                self.cfg.network.stimulus_current_gain,
                self.cfg.stim.pulse_phases,
            )
            for mode_label, center, voltage in mode_defs
        }

        for mode_label, center, voltage in mode_defs:
            pulse_currents = pulse_cache[(mode_label, voltage)]
            for _ in range(scfg.short_eval_trials_per_mode):
                detail = model.run_segment_detailed(
                    scfg.short_eval_trial_duration_ms,
                    pulse_currents=pulse_currents,
                    plasticity=False,
                    response_window_ms=response_window,
                )
                record_counts.append(detail["record_counts"])
                states.append(detail["active_mask"].astype(float))
                active_union |= detail["active_mask"]
                peak_fractions.append(float(detail["peak_window_activity"]) / model.n_total)
                labels.append(mode_label)

        state_matrix = np.array(states, dtype=float)
        feature_matrix = np.array(record_counts, dtype=float)
        normalized_feature_matrix = self._normalize_count_features(feature_matrix)
        rank_denom = min(max(1, state_matrix.shape[0]), scfg.max_rank_normalization_trials)
        rank_score = float(np.linalg.matrix_rank(state_matrix) / rank_denom)
        distance_ratio = self._compute_distance_ratio(feature_matrix, np.array(labels))
        pattern_distance_ratio = self._compute_distance_ratio(normalized_feature_matrix, np.array(labels))
        pattern_score = self._ratio_to_unit_score(pattern_distance_ratio)
        raw_distance_score = self._ratio_to_unit_score(distance_ratio)
        decode_score = self._quick_decode_score(feature_matrix, np.array(labels))
        plasticity_decode_score = self._evaluate_plasticity_decode(
            model, mode_defs, pulse_cache, response_window
        )
        separation_score = (
            scfg.fitness_rank_weight * rank_score
            + scfg.fitness_pattern_weight * pattern_score
            + scfg.fitness_raw_distance_weight * raw_distance_score
            + scfg.fitness_decode_weight * decode_score
            + scfg.fitness_plasticity_decode_weight * plasticity_decode_score
        )
        activity_score = float(active_union.mean())
        silent_neurons = np.flatnonzero(~active_union)
        burst_penalty = self._compute_burst_penalty(np.array(peak_fractions, dtype=float))
        density_penalty = abs(int(adjacency.sum()) - self.target_synapse_count) / max(1, self.target_synapse_count)
        fitness = (
            separation_score
            + scfg.fitness_activity_weight * activity_score
            - scfg.fitness_burst_weight * burst_penalty
            - scfg.fitness_density_weight * density_penalty
        )

        metrics = CandidateMetrics(
            fitness=float(fitness),
            separation_score=separation_score,
            rank_score=rank_score,
            distance_ratio=float(distance_ratio),
            pattern_distance_ratio=float(pattern_distance_ratio),
            decode_score=float(decode_score),
            plasticity_decode_score=float(plasticity_decode_score),
            activity_score=activity_score,
            silent_neuron_ratio=float(1.0 - activity_score),
            burst_penalty=burst_penalty,
            density_penalty=float(density_penalty),
            total_synapses=int(adjacency.sum()),
            active_neurons=np.flatnonzero(active_union),
            silent_neurons=silent_neurons,
            state_matrix=state_matrix,
            record_counts=feature_matrix,
        )
        return CandidateResult(
            topology=TopologySpec(label=label, adjacency=adjacency.copy(), blueprint=self.reference_blueprint),
            metrics=metrics,
        )

    def _compute_distance_ratio(self, features: np.ndarray, labels: np.ndarray) -> float:
        n_samples = len(labels)
        if n_samples < 2:
            return 0.0
        between = []
        within = []
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = float(np.linalg.norm(features[i] - features[j]))
                if labels[i] == labels[j]:
                    within.append(dist)
                else:
                    between.append(dist)
        if not between or not within:
            return 0.0
        within_mean = float(np.mean(within))
        if within_mean <= 1e-8:
            return 0.0
        return float(np.mean(between) / within_mean)

    def _normalize_count_features(self, features: np.ndarray) -> np.ndarray:
        totals = features.sum(axis=1, keepdims=True)
        totals[totals <= 1e-8] = 1.0
        return features / totals

    def _ratio_to_unit_score(self, ratio: float) -> float:
        scfg = self.structure_cfg
        floor = scfg.pattern_ratio_floor
        ceiling = max(floor + 1e-6, scfg.pattern_ratio_ceiling)
        scaled = (ratio - floor) / (ceiling - floor)
        return float(np.clip(scaled, 0.0, 1.0))

    def _compute_burst_penalty(self, peak_fractions: np.ndarray) -> float:
        if peak_fractions.size == 0:
            return 0.0
        threshold = self.structure_cfg.burst_active_fraction_threshold
        penalties = np.maximum(0.0, peak_fractions - threshold) / max(1e-6, 1.0 - threshold)
        return float(np.mean(penalties))

    def _quick_decode_score(self, features: np.ndarray, labels: np.ndarray) -> float:
        unique, counts = np.unique(labels, return_counts=True)
        if len(unique) < 2 or counts.min() < 2:
            return 0.0
        n_folds = min(self.structure_cfg.quick_decode_folds, int(counts.min()))
        if n_folds < 2:
            return 0.0
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.base_seed)
        scores = []
        for train_idx, test_idx in cv.split(features, labels):
            clf = RidgeClassifier(alpha=1.0)
            clf.fit(features[train_idx], labels[train_idx])
            scores.append(float(clf.score(features[test_idx], labels[test_idx])))
        return float(np.mean(scores)) if scores else 0.0

    def _evaluate_plasticity_decode(self, model: ReservoirModel,
                                    mode_defs: list[tuple[str, np.ndarray, float]],
                                    pulse_cache: dict[tuple[str, float], dict[int, np.ndarray]],
                                    response_window: tuple[float, float]) -> float:
        scfg = self.structure_cfg
        if scfg.adaptation_trials_per_mode <= 0 or scfg.post_adaptation_trials_per_mode <= 0:
            return 0.0

        for mode_label, _center, voltage in mode_defs:
            pulse_currents = pulse_cache[(mode_label, voltage)]
            for _ in range(scfg.adaptation_trials_per_mode):
                model.run_segment(
                    scfg.short_eval_trial_duration_ms,
                    pulse_currents=pulse_currents,
                    plasticity=True,
                    response_window_ms=response_window,
                )

        features = []
        labels = []
        for mode_label, _center, voltage in mode_defs:
            pulse_currents = pulse_cache[(mode_label, voltage)]
            for _ in range(scfg.post_adaptation_trials_per_mode):
                detail = model.run_segment_detailed(
                    scfg.short_eval_trial_duration_ms,
                    pulse_currents=pulse_currents,
                    plasticity=False,
                    response_window_ms=response_window,
                )
                features.append(detail["record_counts"])
                labels.append(mode_label)

        if not features:
            return 0.0
        return self._quick_decode_score(np.array(features, dtype=float), np.array(labels))

    def _rewire_source(self, adjacency: np.ndarray, source: int, target: int, rng: np.random.Generator) -> bool:
        if source == target or adjacency[source, target]:
            return False
        outgoing = np.flatnonzero(adjacency[source])
        if len(outgoing) == 0:
            adjacency[source, target] = True
            return True
        removable = outgoing[outgoing != target]
        if len(removable) == 0:
            return False
        old_target = int(rng.choice(removable))
        adjacency[source, old_target] = False
        adjacency[source, target] = True
        return True

    def _add_edge_with_budget(self, adjacency: np.ndarray, source: int, target: int,
                              rng: np.random.Generator) -> bool:
        if source == target or adjacency[source, target]:
            return False
        adjacency[source, target] = True
        edge_src, edge_tgt = np.nonzero(adjacency)
        removable = np.where((edge_src != source) | (edge_tgt != target))[0]
        if len(removable) == 0:
            return True
        remove_idx = int(rng.choice(removable))
        adjacency[edge_src[remove_idx], edge_tgt[remove_idx]] = False
        return True

    def _mutate_silent_repair(self, adjacency: np.ndarray, metrics: CandidateMetrics,
                              rng: np.random.Generator) -> None:
        if len(metrics.silent_neurons) == 0 or len(metrics.active_neurons) == 0:
            return
        n_ops = min(self.structure_cfg.silent_repair_additions, len(metrics.silent_neurons))
        for _ in range(n_ops):
            silent = int(rng.choice(metrics.silent_neurons))
            active_dists = self.distance_matrix[silent, metrics.active_neurons]
            order = np.argsort(active_dists)[: max(1, min(20, len(active_dists)))]
            source = int(rng.choice(metrics.active_neurons[order]))
            if not self._rewire_source(adjacency, source, silent, rng):
                self._add_edge_with_budget(adjacency, source, silent, rng)

    def _mutate_local_rewire(self, adjacency: np.ndarray, rng: np.random.Generator) -> None:
        edge_src, edge_tgt = np.nonzero(adjacency)
        if len(edge_src) == 0:
            return
        n_ops = min(self.structure_cfg.local_rewire_count, len(edge_src))
        for _ in range(n_ops):
            idx = int(rng.integers(len(edge_src)))
            source = int(edge_src[idx])
            old_target = int(edge_tgt[idx])
            candidate_order = np.argsort(self.distance_matrix[source])
            for new_target in candidate_order:
                if new_target == source or adjacency[source, new_target]:
                    continue
                adjacency[source, old_target] = False
                adjacency[source, new_target] = True
                break

    def _mutate_hotspot_bridge(self, adjacency: np.ndarray, rng: np.random.Generator) -> None:
        n_ops = self.structure_cfg.bridge_rewire_count
        for _ in range(n_ops):
            source_pool = self.hotspot_neighborhoods[int(rng.integers(2))]
            target_pool = self.hotspot_neighborhoods[int(rng.integers(2))]
            source = int(rng.choice(source_pool))
            target = int(rng.choice(target_pool))
            if source == target:
                continue
            self._rewire_source(adjacency, source, target, rng)

    def _restore_edge_budget(self, adjacency: np.ndarray, rng: np.random.Generator) -> None:
        current = int(adjacency.sum())
        if current > self.target_synapse_count:
            edge_src, edge_tgt = np.nonzero(adjacency)
            drop_count = current - self.target_synapse_count
            chosen = rng.choice(len(edge_src), size=drop_count, replace=False)
            adjacency[edge_src[chosen], edge_tgt[chosen]] = False
        elif current < self.target_synapse_count:
            add_count = self.target_synapse_count - current
            for _ in range(add_count):
                source = int(rng.integers(self.cfg.network.n_total))
                weights = self.prob_matrix[source].copy()
                weights[adjacency[source]] = 0.0
                weights[source] = 0.0
                if weights.sum() <= 0:
                    continue
                target = int(rng.choice(self.cfg.network.n_total, p=weights / weights.sum()))
                adjacency[source, target] = True

    def mutate_topology(self, parent: CandidateResult, rng: np.random.Generator) -> np.ndarray:
        adjacency = parent.topology.adjacency.copy()
        self._mutate_silent_repair(adjacency, parent.metrics, rng)
        self._mutate_local_rewire(adjacency, rng)
        self._mutate_hotspot_bridge(adjacency, rng)
        self._restore_edge_budget(adjacency, rng)
        np.fill_diagonal(adjacency, False)
        return adjacency

    def degree_match_randomize(self, adjacency: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.base_seed + 999)
        randomized = adjacency.copy()
        edge_src, edge_tgt = np.nonzero(randomized)
        if len(edge_src) < 2:
            return randomized
        n_swaps = min(
            self.structure_cfg.degree_swap_factor * len(edge_src),
            max(10, 2 * len(edge_src)),
        )
        swaps = 0
        attempts = 0
        while swaps < n_swaps and attempts < n_swaps * 20:
            attempts += 1
            idx1 = int(rng.integers(len(edge_src)))
            idx2 = int(rng.integers(len(edge_src)))
            if idx1 == idx2:
                continue
            s1, t1 = int(edge_src[idx1]), int(edge_tgt[idx1])
            s2, t2 = int(edge_src[idx2]), int(edge_tgt[idx2])
            if len({s1, s2, t1, t2}) < 4:
                continue
            if randomized[s1, t2] or randomized[s2, t1]:
                continue
            if s1 == t2 or s2 == t1:
                continue
            randomized[s1, t1] = False
            randomized[s2, t2] = False
            randomized[s1, t2] = True
            randomized[s2, t1] = True
            edge_tgt[idx1] = t2
            edge_tgt[idx2] = t1
            swaps += 1
        return randomized

    def evolve(self) -> EvolutionBundle:
        scfg = self.structure_cfg
        population = [
            self.evaluate_topology(self.reference_adjacency.copy(), "seed_random", eval_index=1)
        ]
        for idx in range(scfg.population_size - 1):
            adjacency = self.build_random_topology(seed_offset=100 + idx)
            population.append(
                self.evaluate_topology(adjacency, f"random_{idx:02d}", eval_index=2 + idx)
            )

        history = []
        child_eval_index = 5000
        innovation_eval_index = 9000
        for generation in range(scfg.generations):
            population.sort(key=lambda item: item.metrics.fitness, reverse=True)
            history.append({
                "generation": float(generation),
                "best_fitness": float(population[0].metrics.fitness),
                "mean_fitness": float(np.mean([item.metrics.fitness for item in population])),
                "best_silent_neuron_ratio": float(population[0].metrics.silent_neuron_ratio),
                "best_burst_penalty": float(population[0].metrics.burst_penalty),
            })

            elites = population[: scfg.elite_size]
            candidate_pool = elites.copy()
            rng = np.random.default_rng(self.base_seed + generation * 17 + 7)

            for elite in elites:
                for _ in range(scfg.offspring_per_elite):
                    child_adj = self.mutate_topology(elite, rng)
                    candidate_pool.append(
                        self.evaluate_topology(
                            child_adj,
                            f"gen{generation:02d}_offspring",
                            eval_index=child_eval_index,
                        )
                    )
                    child_eval_index += 1

            for idx in range(scfg.innovation_count):
                adjacency = self.build_random_topology(seed_offset=2000 + generation * 100 + idx)
                candidate_pool.append(
                    self.evaluate_topology(
                        adjacency,
                        f"gen{generation:02d}_innovation",
                        eval_index=innovation_eval_index,
                    )
                )
                innovation_eval_index += 1

            candidate_pool.sort(key=lambda item: item.metrics.fitness, reverse=True)
            population = candidate_pool[: scfg.population_size]

        population.sort(key=lambda item: item.metrics.fitness, reverse=True)
        evolved = population[0]
        degree_adj = self.degree_match_randomize(evolved.topology.adjacency)
        degree_result = self.evaluate_topology(
            degree_adj,
            "degree_matched_random",
            eval_index=200000,
        )

        selected = evolved
        if self.structure_cfg.topology_mode == "degree_matched_random":
            selected = degree_result

        return EvolutionBundle(
            selected_topology=selected.topology,
            selected_metrics=selected.metrics,
            evolved_topology=evolved.topology,
            evolved_metrics=evolved.metrics,
            degree_matched_topology=degree_result.topology,
            degree_matched_metrics=degree_result.metrics,
            history=history,
            baseline_topology=self.baseline_topology,
            baseline_metrics=self.baseline_metrics,
            area_centers=self.area_centers,
            alpha=self.alpha,
            sigmas=self.sigmas,
        )


def prepare_topology(cfg: Config, output_dir: Path) -> tuple[dict | None, dict | None]:
    if cfg.structure.topology_mode == "random":
        return None, None

    evolver = StructureEvolver(cfg)
    bundle = evolver.evolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "topology_artifacts.npz",
        baseline_random=bundle.baseline_topology.adjacency.astype(np.uint8),
        evolved=bundle.evolved_topology.adjacency.astype(np.uint8),
        degree_matched=bundle.degree_matched_topology.adjacency.astype(np.uint8),
        positions=evolver.reference_positions,
        record_indices=evolver.reference_record_indices,
    )
    with open(output_dir / "structure_evolution.json", "w", encoding="utf-8") as f:
        json.dump(bundle.to_summary(), f, indent=2, ensure_ascii=False)
    return bundle.selected_topology.to_override(), bundle.to_summary()
