from __future__ import annotations

from collections import defaultdict

import numpy as np


class OnlineTopologyPlasticity:
    def __init__(self, cfg, model) -> None:
        self.cfg = cfg
        self.model = model

    def update(
        self,
        round_spike_records: list[list[np.ndarray]],
        round_mode_labels: list[str],
    ) -> dict[str, object]:
        params = self.cfg.topology_plasticity
        if not params.enabled:
            return {
                "added": 0,
                "removed": 0,
                "candidate_pairs": 0,
                "selected_pairs": [],
                "pruned_edges": [],
            }

        pair_stats = self._collect_pair_stats(round_spike_records, round_mode_labels)
        if not pair_stats:
            return {
                "added": 0,
                "removed": 0,
                "candidate_pairs": 0,
                "selected_pairs": [],
                "pruned_edges": [],
            }

        add_edges, add_meta = self._select_shortcuts(pair_stats)
        remove_edge_ids, remove_meta = self._select_edges_to_prune(round_spike_records, add_edges)

        if params.keep_total_edges_constant:
            n_ops = min(len(add_edges), len(remove_edge_ids))
            add_edges = add_edges[:n_ops]
            add_meta = add_meta[:n_ops]
            remove_edge_ids = remove_edge_ids[:n_ops]
            remove_meta = remove_meta[:n_ops]

        ops = self.model.rewire_edges(add_edges, remove_edge_ids)
        return {
            "added": ops["added"],
            "removed": ops["removed"],
            "candidate_pairs": len(pair_stats),
            "selected_pairs": add_meta,
            "pruned_edges": remove_meta,
            "topology_metrics": self.model.get_topology_metrics(),
        }

    def _collect_pair_stats(
        self,
        round_spike_records: list[list[np.ndarray]],
        round_mode_labels: list[str],
    ) -> dict[tuple[int, int], dict[str, object]]:
        params = self.cfg.topology_plasticity
        adjacency = self.model.get_adjacency_mask()
        stats: dict[tuple[int, int], dict[str, object]] = {}

        for spikes_by_step, label in zip(round_spike_records, round_mode_labels):
            first_spike_time = {}
            for step_idx, fired in enumerate(spikes_by_step):
                if len(fired) == 0:
                    continue
                for neuron in fired.tolist():
                    first_spike_time.setdefault(int(neuron), step_idx)

            if len(first_spike_time) < 2:
                continue

            items = sorted(first_spike_time.items(), key=lambda x: x[1])
            for src_idx, (src, src_step) in enumerate(items):
                if src >= self.model.n_exc:
                    continue
                for tgt, tgt_step in items[src_idx + 1:]:
                    delay_ms = (tgt_step - src_step) * self.cfg.experiment.dt_ms
                    if delay_ms > params.delay_window_ms[1]:
                        break
                    if not (params.delay_window_ms[0] <= delay_ms <= params.delay_window_ms[1]):
                        continue
                    if adjacency[src, tgt]:
                        continue
                    dist = np.linalg.norm(self.model.positions[src] - self.model.positions[tgt])
                    if dist > params.max_candidate_distance_mm:
                        continue

                    key = (src, tgt)
                    entry = stats.setdefault(
                        key,
                        {
                            "count": 0,
                            "delays": [],
                            "labels": defaultdict(int),
                            "distance_mm": float(dist),
                        },
                    )
                    entry["count"] += 1
                    entry["delays"].append(float(delay_ms))
                    entry["labels"][label] += 1

        return {
            key: value
            for key, value in stats.items()
            if value["count"] >= params.min_pair_cooccurrence
        }

    def _select_shortcuts(
        self,
        pair_stats: dict[tuple[int, int], dict[str, object]],
    ) -> tuple[list[tuple[int, int]], list[dict[str, object]]]:
        params = self.cfg.topology_plasticity
        scored = []
        for (src, tgt), entry in pair_stats.items():
            delays = np.array(entry["delays"], dtype=float)
            label_counts = entry["labels"]
            dominant = max(label_counts.values())
            specificity = dominant / entry["count"]
            consistency = 1.0 / (1.0 + float(delays.std())) if len(delays) > 1 else 1.0
            score = (
                params.pair_count_weight * entry["count"]
                + params.consistency_weight * consistency
                + params.class_specificity_weight * specificity
                - params.distance_penalty * entry["distance_mm"]
            )
            scored.append(
                {
                    "src": int(src),
                    "tgt": int(tgt),
                    "score": float(score),
                    "count": int(entry["count"]),
                    "consistency": float(consistency),
                    "specificity": float(specificity),
                    "distance_mm": float(entry["distance_mm"]),
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        max_updates = max(
            1,
            int(self.model.n_edges * params.max_edge_fraction_per_update),
        )
        limit = min(params.max_add_per_update, max_updates, len(scored))
        selected = scored[:limit]
        add_edges = [(item["src"], item["tgt"]) for item in selected]
        return add_edges, selected

    def _select_edges_to_prune(
        self,
        round_spike_records: list[list[np.ndarray]],
        add_edges: list[tuple[int, int]],
    ) -> tuple[list[int], list[dict[str, object]]]:
        params = self.cfg.topology_plasticity
        if not add_edges:
            return [], []

        spike_counts = np.zeros(self.model.n_total, dtype=np.int32)
        pair_counts = defaultdict(int)
        for spikes_by_step in round_spike_records:
            first_spike_time = {}
            for step_idx, fired in enumerate(spikes_by_step):
                if len(fired) == 0:
                    continue
                spike_counts[fired] += 1
                for neuron in fired.tolist():
                    first_spike_time.setdefault(int(neuron), step_idx)

            items = sorted(first_spike_time.items(), key=lambda x: x[1])
            for src_idx, (src, src_step) in enumerate(items):
                if src >= self.model.n_exc:
                    continue
                for tgt, tgt_step in items[src_idx + 1:]:
                    delay_ms = (tgt_step - src_step) * self.cfg.experiment.dt_ms
                    if delay_ms > params.delay_window_ms[1]:
                        break
                    if params.delay_window_ms[0] <= delay_ms <= params.delay_window_ms[1]:
                        pair_counts[(src, tgt)] += 1

        threshold = np.quantile(spike_counts, params.inactive_quantile)
        removable = []
        for eid, (src, tgt, is_exc) in enumerate(
            zip(self.model.edge_src, self.model.edge_tgt, self.model.edge_is_exc)
        ):
            if not is_exc:
                continue
            if self.model.out_degree[src] <= params.protect_min_out_degree:
                continue
            if self.model.in_degree[tgt] <= params.protect_min_in_degree:
                continue
            usage = pair_counts.get((int(src), int(tgt)), 0)
            inactive_bonus = 1.0 if spike_counts[src] <= threshold or spike_counts[tgt] <= threshold else 0.0
            score = (1.0 / (1.0 + usage)) + inactive_bonus
            removable.append(
                {
                    "eid": int(eid),
                    "src": int(src),
                    "tgt": int(tgt),
                    "usage": int(usage),
                    "score": float(score),
                }
            )

        removable.sort(key=lambda x: x["score"], reverse=True)
        max_updates = max(
            1,
            int(self.model.n_edges * params.max_edge_fraction_per_update),
        )
        limit = min(params.max_prune_per_update, max_updates, len(add_edges), len(removable))
        selected = removable[:limit]
        return [item["eid"] for item in selected], selected
