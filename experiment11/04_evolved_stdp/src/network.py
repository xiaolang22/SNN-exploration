from __future__ import annotations

import numpy as np

from .config import Config


class ReservoirModel:
    def __init__(self, cfg: Config, rng: np.random.Generator,
                 topology_override: dict | None = None) -> None:
        self.cfg = cfg
        self.rng = rng
        self.topology_override = topology_override or {}
        net = cfg.network
        self.n_total = net.n_total
        self.n_exc = net.n_exc
        self.n_inh = net.n_inh
        self.dt = cfg.experiment.dt_ms

        self.positions = self._build_positions()
        self.is_exc = np.zeros(self.n_total, dtype=bool)
        self.is_exc[: self.n_exc] = True
        self.record_indices = self._select_record_channels()
        self.a, self.b, self.c, self.d = self._build_neuron_params()

        self.v = np.full(self.n_total, -65.0)
        self.u = self.b * self.v
        self.syn_current = np.zeros(self.n_total)
        if "bias_current" in self.topology_override:
            self.bias_current = self.topology_override["bias_current"].copy()
        else:
            self.bias_current = rng.normal(0.0, 0.4, size=self.n_total)
        self.pre_trace = np.zeros(self.n_total)
        self.post_trace = np.zeros(self.n_total)

        self._build_connectivity()
        self.time_ms = 0.0

    def _build_positions(self) -> np.ndarray:
        if "positions" in self.topology_override:
            return self.topology_override["positions"].copy()
        net = self.cfg.network
        return np.column_stack([
            self.rng.uniform(0.0, net.plane_width_mm, self.n_total),
            self.rng.uniform(0.0, net.plane_height_mm, self.n_total),
        ])

    def _select_record_channels(self) -> np.ndarray:
        if "record_indices" in self.topology_override:
            return self.topology_override["record_indices"].copy()
        net = self.cfg.network
        exc_sel = self.rng.choice(self.n_exc, net.record_exc, replace=False)
        inh_sel = self.rng.choice(
            np.arange(self.n_exc, self.n_total), net.record_inh, replace=False
        )
        return np.sort(np.concatenate([exc_sel, inh_sel]))

    def _build_neuron_params(self):
        if "neuron_params" in self.topology_override:
            params = self.topology_override["neuron_params"]
            return (
                params["a"].copy(),
                params["b"].copy(),
                params["c"].copy(),
                params["d"].copy(),
            )
        re = self.rng.random(self.n_exc)
        ri = self.rng.random(self.n_inh)
        a = np.empty(self.n_total)
        b = np.empty(self.n_total)
        c = np.empty(self.n_total)
        d = np.empty(self.n_total)
        a[: self.n_exc], b[: self.n_exc] = 0.02, 0.20
        c[: self.n_exc] = -65.0 + 15.0 * re**2
        d[: self.n_exc] = 8.0 - 6.0 * re**2
        a[self.n_exc:] = 0.02 + 0.08 * ri
        b[self.n_exc:] = 0.25 - 0.05 * ri
        c[self.n_exc:], d[self.n_exc:] = -65.0, 2.0
        return a, b, c, d

    def _build_connectivity(self) -> None:
        net = self.cfg.network
        n = self.n_total
        if "edge_src" in self.topology_override and "edge_tgt" in self.topology_override:
            sources = self.topology_override["edge_src"].astype(np.int32).copy()
            targets = self.topology_override["edge_tgt"].astype(np.int32).copy()
        elif "adjacency" in self.topology_override:
            conn_mask = self.topology_override["adjacency"].astype(bool).copy()
            np.fill_diagonal(conn_mask, False)
            sources, targets = np.nonzero(conn_mask)
            sources = sources.astype(np.int32)
            targets = targets.astype(np.int32)
        else:
            dists = np.linalg.norm(
                self.positions[:, None, :] - self.positions[None, :, :], axis=2
            )
            probs = net.connection_prob * (0.35 + 0.65 * np.exp(-dists / net.distance_lambda_mm))
            np.fill_diagonal(probs, 0.0)
            conn_mask = self.rng.random((n, n)) < probs
            sources, targets = np.nonzero(conn_mask)
            sources = sources.astype(np.int32)
            targets = targets.astype(np.int32)

        self.edge_src = sources
        self.edge_tgt = targets
        self.edge_is_exc = self.is_exc[self.edge_src]
        self.n_edges = len(self.edge_src)

        if "edge_w" in self.topology_override:
            self.edge_w = self.topology_override["edge_w"].copy()
        else:
            self.edge_w = self._sample_edge_weights()

        self._build_index_structures()

    def _sample_edge_weights(self) -> np.ndarray:
        wgt = self.cfg.weights
        n_edges = self.n_edges
        weights = np.empty(n_edges)
        n_exc_edges = self.edge_is_exc.sum()
        weights[self.edge_is_exc] = np.exp(
            self.rng.normal(wgt.exc_mu, wgt.exc_sigma, n_exc_edges)
        )
        weights[~self.edge_is_exc] = -np.exp(
            self.rng.normal(wgt.inh_mu, wgt.inh_sigma, n_edges - n_exc_edges)
        )
        return weights

    def _build_index_structures(self) -> None:
        n = self.n_total
        self.out_indptr = np.zeros(n + 1, dtype=np.int64)
        for s in self.edge_src:
            self.out_indptr[s + 1] += 1
        np.cumsum(self.out_indptr, out=self.out_indptr)
        self.out_edge_ids = np.argsort(self.edge_src, kind="stable").astype(np.int64)

        in_exc_counts = np.zeros(n, dtype=np.int64)
        exc_indices = np.where(self.edge_is_exc)[0]
        for idx in exc_indices:
            in_exc_counts[self.edge_tgt[idx]] += 1
        self.in_exc_indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(in_exc_counts, out=self.in_exc_indptr[1:])
        in_exc_order = np.argsort(self.edge_tgt[exc_indices], kind="stable")
        self.in_exc_edge_ids = exc_indices[in_exc_order].astype(np.int64)

    def step(self, external: np.ndarray | None = None, plasticity: bool = True) -> np.ndarray:
        net = self.cfg.network
        plas = self.cfg.plasticity
        dt = self.dt
        self.syn_current *= np.exp(-dt / net.synaptic_tau_ms)
        np.clip(
            self.syn_current,
            -net.synaptic_current_limit,
            net.synaptic_current_limit,
            out=self.syn_current,
        )
        if plasticity:
            self.pre_trace *= np.exp(-dt / plas.stdp.tau_pre_ms)
            self.post_trace *= np.exp(-dt / plas.stdp.tau_post_ms)

        I_total = (net.dc_offset + self.bias_current + self.syn_current
                   + self.rng.normal(0.0, net.noise_std, self.n_total))
        if external is not None:
            I_total += external
        for _ in range(2):
            np.clip(self.v, -100.0, 30.0, out=self.v)
            self.v += 0.5 * (
                0.04 * self.v**2 + 5.0 * self.v + 140.0 - self.u + I_total
            )
            np.nan_to_num(self.v, copy=False, nan=30.0, posinf=30.0, neginf=-100.0)
            np.clip(self.v, -100.0, 30.0, out=self.v)
        self.u += self.a * (self.b * self.v - self.u)

        fired = np.where(self.v >= 30.0)[0]
        if len(fired) > 0:
            self.v[fired] = self.c[fired]
            self.u[fired] += self.d[fired]
        if len(fired) > 0:
            self._propagate_spikes(fired)
        if plasticity and len(fired) > 0:
            self._apply_stdp(fired)
        self.time_ms += dt
        return fired

    def _propagate_spikes(self, fired: np.ndarray) -> None:
        for neuron in fired:
            start, end = self.out_indptr[neuron], self.out_indptr[neuron + 1]
            eids = self.out_edge_ids[start:end]
            if len(eids) == 0:
                continue
            np.add.at(self.syn_current, self.edge_tgt[eids], self.edge_w[eids])

    def _apply_stdp(self, fired: np.ndarray) -> None:
        stdp = self.cfg.plasticity.stdp
        if not stdp.enabled:
            return
        wgt = self.cfg.weights
        self.pre_trace[fired] += 1.0
        self.post_trace[fired] += 1.0
        for neuron in fired:
            start, end = self.out_indptr[neuron], self.out_indptr[neuron + 1]
            eids = self.out_edge_ids[start:end]
            exc_eids = eids[self.edge_is_exc[eids]]
            if len(exc_eids) > 0:
                self.edge_w[exc_eids] -= stdp.a_minus * self.post_trace[self.edge_tgt[exc_eids]]
        for neuron in fired:
            start, end = self.in_exc_indptr[neuron], self.in_exc_indptr[neuron + 1]
            eids = self.in_exc_edge_ids[start:end]
            if len(eids) > 0:
                self.edge_w[eids] += stdp.a_plus * self.pre_trace[self.edge_src[eids]]
        exc_edges = np.where(self.edge_is_exc)[0]
        np.clip(self.edge_w[exc_edges], wgt.w_min, wgt.w_max, out=self.edge_w[exc_edges])

    def run_segment(self, duration_ms: float, pulse_currents: dict[int, np.ndarray] | None = None,
                    plasticity: bool = True, response_window_ms: tuple[float, float] | None = None) -> np.ndarray:
        return self.run_segment_detailed(
            duration_ms,
            pulse_currents=pulse_currents,
            plasticity=plasticity,
            response_window_ms=response_window_ms,
        )["record_counts"]

    def run_segment_detailed(self, duration_ms: float,
                             pulse_currents: dict[int, np.ndarray] | None = None,
                             plasticity: bool = True,
                             response_window_ms: tuple[float, float] | None = None) -> dict[str, np.ndarray | float | bool]:
        n_steps = int(duration_ms / self.dt)
        n_rec = len(self.record_indices)
        spike_counts = np.zeros(n_rec, dtype=np.int32)
        neuron_spike_counts = np.zeros(self.n_total, dtype=np.int32)
        active_mask = np.zeros(self.n_total, dtype=bool)
        peak_window_activity = 0
        if response_window_ms is not None:
            win_start, win_end = int(response_window_ms[0] / self.dt), int(response_window_ms[1] / self.dt)
        else:
            win_start, win_end = 0, n_steps
        for t in range(n_steps):
            ext = pulse_currents.get(t) if pulse_currents else None
            fired = self.step(ext, plasticity=plasticity)
            if win_start <= t < win_end:
                if len(fired) > 0:
                    neuron_spike_counts[fired] += 1
                    active_mask[fired] = True
                    peak_window_activity = max(peak_window_activity, len(fired))
                rec_fired = np.isin(fired, self.record_indices)
                for nidx in fired[rec_fired]:
                    spike_counts[np.searchsorted(self.record_indices, nidx)] += 1
        return {
            "record_counts": spike_counts,
            "neuron_spike_counts": neuron_spike_counts,
            "active_mask": active_mask,
            "peak_window_activity": float(peak_window_activity),
            "burst_flag": peak_window_activity / self.n_total >= self.cfg.structure.burst_active_fraction_threshold,
        }

    def get_exc_weights(self) -> np.ndarray:
        return self.edge_w[self.edge_is_exc].copy()

    def get_connectivity_mask(self) -> np.ndarray:
        mask = np.zeros((self.n_total, self.n_total), dtype=bool)
        mask[self.edge_src, self.edge_tgt] = True
        return mask

    def export_blueprint(self) -> dict:
        return {
            "positions": self.positions.copy(),
            "record_indices": self.record_indices.copy(),
            "neuron_params": {
                "a": self.a.copy(),
                "b": self.b.copy(),
                "c": self.c.copy(),
                "d": self.d.copy(),
            },
            "bias_current": self.bias_current.copy(),
        }
