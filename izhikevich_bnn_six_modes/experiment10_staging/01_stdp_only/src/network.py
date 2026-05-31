from __future__ import annotations

import numpy as np

from .config import Config
from .ip import (
    apply_intrinsic_plasticity,
    rates_from_calcium,
    summarize_ip_state,
    update_calcium,
)


class ReservoirModel:
    def __init__(self, cfg: Config, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng
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
        self.bias_current = rng.normal(0.0, 0.4, size=self.n_total)
        self.input_gain = np.ones(self.n_total)
        self.calcium = np.zeros(self.n_total)
        self.current_rates_hz = np.zeros(self.n_total)
        self.pre_trace = np.zeros(self.n_total)
        self.post_trace = np.zeros(self.n_total)

        self._build_connectivity()
        self.time_ms = 0.0

    def _build_positions(self) -> np.ndarray:
        net = self.cfg.network
        return np.column_stack([
            self.rng.uniform(0.0, net.plane_width_mm, self.n_total),
            self.rng.uniform(0.0, net.plane_height_mm, self.n_total),
        ])

    def _select_record_channels(self) -> np.ndarray:
        net = self.cfg.network
        exc_sel = self.rng.choice(self.n_exc, net.record_exc, replace=False)
        inh_sel = self.rng.choice(
            np.arange(self.n_exc, self.n_total), net.record_inh, replace=False
        )
        return np.sort(np.concatenate([exc_sel, inh_sel]))

    def _build_neuron_params(self):
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
        wgt = self.cfg.weights
        n = self.n_total
        dists = np.linalg.norm(
            self.positions[:, None, :] - self.positions[None, :, :], axis=2
        )
        probs = net.connection_prob * (0.35 + 0.65 * np.exp(-dists / net.distance_lambda_mm))
        np.fill_diagonal(probs, 0.0)
        conn_mask = self.rng.random((n, n)) < probs
        sources, targets = np.nonzero(conn_mask)
        n_edges = len(sources)
        weights = np.empty(n_edges)
        exc_edge_mask = self.is_exc[sources]
        n_exc_edges = exc_edge_mask.sum()
        weights[exc_edge_mask] = np.exp(self.rng.normal(wgt.exc_mu, wgt.exc_sigma, n_exc_edges))
        weights[~exc_edge_mask] = -np.exp(self.rng.normal(wgt.inh_mu, wgt.inh_sigma, n_edges - n_exc_edges))

        self.edge_src = sources.astype(np.int32)
        self.edge_tgt = targets.astype(np.int32)
        self.edge_w = weights
        self.edge_is_exc = exc_edge_mask
        self.n_edges = n_edges

        self.out_indptr = np.zeros(n + 1, dtype=np.int64)
        for s in sources:
            self.out_indptr[s + 1] += 1
        np.cumsum(self.out_indptr, out=self.out_indptr)
        self.out_edge_ids = np.argsort(sources, kind="stable").astype(np.int64)

        in_exc_counts = np.zeros(n, dtype=np.int64)
        exc_indices = np.where(exc_edge_mask)[0]
        for idx in exc_indices:
            in_exc_counts[targets[idx]] += 1
        self.in_exc_indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(in_exc_counts, out=self.in_exc_indptr[1:])
        in_exc_order = np.argsort(targets[exc_indices], kind="stable")
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

        I_total = (
            net.dc_offset
            + self.bias_current
            + self.input_gain * self.syn_current
            + self.rng.normal(0.0, net.noise_std, self.n_total)
        )
        if external is not None:
            I_total += self.input_gain * external
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
        update_calcium(self.calcium, fired, dt, plas.ip.tau_cal_ms)
        self.current_rates_hz = rates_from_calcium(self.calcium, plas.ip.tau_cal_ms)
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
        n_steps = int(duration_ms / self.dt)
        n_rec = len(self.record_indices)
        spike_counts = np.zeros(n_rec, dtype=np.int32)
        if response_window_ms is not None:
            win_start, win_end = int(response_window_ms[0] / self.dt), int(response_window_ms[1] / self.dt)
        else:
            win_start, win_end = 0, n_steps
        for t in range(n_steps):
            ext = pulse_currents.get(t) if pulse_currents else None
            fired = self.step(ext, plasticity=plasticity)
            if win_start <= t < win_end:
                rec_fired = np.isin(fired, self.record_indices)
                for nidx in fired[rec_fired]:
                    spike_counts[np.searchsorted(self.record_indices, nidx)] += 1
        return spike_counts

    def get_exc_weights(self) -> np.ndarray:
        return self.edge_w[self.edge_is_exc].copy()

    def apply_ip_update(self) -> None:
        apply_intrinsic_plasticity(
            self.bias_current,
            self.input_gain,
            self.current_rates_hz,
            self.cfg.plasticity.ip,
        )

    def get_ip_metrics(self) -> dict[str, float]:
        return summarize_ip_state(
            self.current_rates_hz,
            self.bias_current,
            self.input_gain,
            self.cfg.plasticity.ip,
        )

    def get_ip_snapshot(self) -> dict[str, np.ndarray]:
        return {
            "rates_hz": self.current_rates_hz.copy(),
            "bias_current": self.bias_current.copy(),
            "input_gain": self.input_gain.copy(),
        }
