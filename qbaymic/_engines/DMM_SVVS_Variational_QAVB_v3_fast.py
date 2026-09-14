#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMM-SVVS VarQITE QAVB — Performance-Optimised Variant
=================================================================

"""
from __future__ import annotations

import os
import sys
from time import time

import numpy as np
from sklearn.cluster import MiniBatchKMeans

# ── Import v2 building blocks ──────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from DMM_SVVS_Variational_QAVB_v2 import (  # noqa: E402
    DMM_SVVS_VarQITE_QAVB,
    NumericalStability,
)


class DMM_SVVS_VarQITE_QAVB_Fast(DMM_SVVS_VarQITE_QAVB):
    """
    Performance-optimised VarQITE QAVB. Inherits the algorithm from
    DMM_SVVS_VarQITE_QAVB and overrides only the device-init / per-sample
    inner loop / cross-sample E-step.
    """

    def __init__(
        self,
        *args,
        device_name: str = "lightning.qubit",
        r_early_stop_tol: float | None = 1e-3,
        r_early_stop_check_every: int = 2,
        metric_refresh: int = 4,
        dedup_n_clusters=None,
        dedup_kmeans_max_iter: int = 20,
        dedup_min_unique_ratio: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.device_name           = str(device_name)
        self.r_early_stop_tol      = (None if r_early_stop_tol is None
                                      else float(r_early_stop_tol))
        self.r_early_stop_check_every = max(1, int(r_early_stop_check_every))
        self.metric_refresh        = max(1, int(metric_refresh))
        self.dedup_n_clusters      = dedup_n_clusters
        self.dedup_kmeans_max_iter = int(dedup_kmeans_max_iter)
        self.dedup_min_unique_ratio = float(dedup_min_unique_ratio)

        # Per-fit instrumentation (populated by _compute_r_annealed).
        self._varqite_stats = {
            "early_stops_per_iter": [],
            "dedup_fraction_per_iter": [],
            "metric_refreshes_per_iter": [],
            "inner_steps_per_iter": [],
        }

    # ── Layer 1: device selection ─────────────────────────────────────────

    def _init_pennylane_device(self):
        try:
            import pennylane as qml
            from pennylane import numpy as pnp
        except ImportError:
            raise ImportError("PennyLane required:  pip install pennylane>=0.30")

        self._qml = qml
        self._pnp = pnp
        self.n_sys = max(1, int(np.ceil(np.log2(max(self.K, 2)))))
        self.n_anc = self.n_sys
        self.K_pad = 1 << self.n_sys
        self.n_tot = 2 * self.n_sys + 1
        self.n_params = 2 * (2 * self.n_sys) * self.ansatz_depth

        # Try requested device first; fall back to default.qubit.
        try:
            self._dev = qml.device(self.device_name, wires=self.n_tot)
            self._active_device = self.device_name
        except Exception as exc:
            self._dev = qml.device("default.qubit", wires=self.n_tot)
            self._active_device = "default.qubit"
            if self.verbose >= 1:
                print(f"  [warn] device '{self.device_name}' unavailable "
                      f"({type(exc).__name__}); falling back to default.qubit")

        # Reset all caches (parameters depending on K).
        self._theta_cache = {}

    # ── Layer 3: cached/refreshing McLachlan step ─────────────────────────

    def _make_qnodes(self, hamiltonian):
        """Build (state_circuit, energy_circuit) once per per-sample inner
        loop so the QNode objects can be reused across all VarQITE substeps
        for that sample. Defined as a closure over `hamiltonian` so the
        observable is fixed for the trajectory."""
        qml = self._qml

        @qml.qnode(self._dev, interface="autograd")
        def state_circuit(th):
            self._ansatz(th)
            return qml.state()

        @qml.qnode(self._dev, interface="autograd")
        def energy_circuit(th):
            self._ansatz(th)
            return qml.expval(hamiltonian)

        return state_circuit, energy_circuit

    def _varqite_step_cached(self, theta, state_qnode, energy_qnode,
                             dtau, A_cache: dict):
        """
        McLachlan step that reuses the QFI metric A(theta) across nearby
        substeps. A is recomputed every `self.metric_refresh` calls; the
        gradient C is computed every step (it is cheaper and changes faster).
        """
        qml = self._qml
        pnp = self._pnp

        th_pnp = pnp.array(theta, requires_grad=True)

        # Refresh A on schedule.
        if A_cache.get("A") is None or A_cache.get("age", 0) >= self.metric_refresh:
            A = np.asarray(
                qml.metric_tensor(state_qnode, approx=self.metric_approx)(th_pnp),
                dtype=np.float64,
            )
            A_cache["A"]   = A
            A_cache["age"] = 0
            A_cache.setdefault("refresh_count", 0)
            A_cache["refresh_count"] += 1
        else:
            A = A_cache["A"]
            A_cache["age"] += 1

        C = 0.5 * np.asarray(qml.grad(energy_qnode)(th_pnp), dtype=np.float64)

        A_reg = A + self.regularization * np.eye(A.shape[0])
        try:
            dtheta = np.linalg.solve(A_reg, -C)
        except np.linalg.LinAlgError:
            dtheta = np.linalg.lstsq(A_reg, -C, rcond=None)[0]

        return theta + dtau * dtheta

    # ── Layer 2 + 3: per-sample responsibility with early stop ────────────

    def _varqite_responsibility(self, d_i, beta_t, s_t, sample_id=None):
        EPS = NumericalStability.EPS

        # Per-sample shift/scale/padding -- same convention as v2.
        d_shift = d_i - d_i.min()
        d_range = d_shift.max()
        if d_range > 1e-10:
            d_scaled_K = 4.0 * d_shift / d_range
        else:
            d_scaled_K = d_shift.copy()
        d_padded = np.full(self.K_pad, self.PHANTOM_PENALTY, dtype=np.float64)
        d_padded[: self.K] = d_scaled_K

        hamiltonian = self._build_hamiltonian(d_padded, s_t)
        state_qnode, energy_qnode = self._make_qnodes(hamiltonian)

        # Warm start.
        if (self.warm_start and sample_id is not None
                and sample_id in self._theta_cache
                and len(self._theta_cache[sample_id]) == self.n_params):
            theta = self._theta_cache[sample_id].copy()
        else:
            if self.init_perturbation > 0:
                theta = self._varqite_rng.normal(
                    0.0, self.init_perturbation, self.n_params
                ).astype(np.float64)
            else:
                theta = np.zeros(self.n_params, dtype=np.float64)

        tau_final = 0.5 * beta_t
        dtau = tau_final / max(self.n_varqite_steps, 1)

        A_cache: dict = {"A": None, "age": 0, "refresh_count": 0}
        r_prev = None
        steps_executed = 0
        early_stopped = False

        for step in range(self.n_varqite_steps):
            theta = self._varqite_step_cached(
                theta, state_qnode, energy_qnode, dtau, A_cache
            )
            steps_executed += 1

            # Layer-2 early stop: check r every `r_early_stop_check_every`
            # substeps once we are past the first few transient steps.
            if (self.r_early_stop_tol is not None
                    and (step + 1) % self.r_early_stop_check_every == 0
                    and step >= 2):
                probs = self._readout_probs(theta)
                r_curr = probs[: self.K]
                if r_prev is not None:
                    delta = float(np.max(np.abs(r_curr - r_prev)))
                    if delta < self.r_early_stop_tol:
                        early_stopped = True
                        break
                r_prev = r_curr

        if self.warm_start and sample_id is not None:
            self._theta_cache[sample_id] = theta

        # Final readout (may already be in r_prev if we early-stopped; recompute
        # to avoid one cached read with a stale theta).
        probs = self._readout_probs(theta)
        r = np.clip(probs[: self.K], EPS, None)
        r = r / r.sum()
        return r, {
            "steps_executed": steps_executed,
            "early_stopped": early_stopped,
            "metric_refreshes": A_cache["refresh_count"],
        }

    # ── Layer 4: E-step over the dataset with optional dedup ──────────────

    def _resolve_dedup_n_clusters(self, N: int) -> int | None:
        if self.dedup_n_clusters is None:
            return None
        if self.dedup_n_clusters == "auto":
            return max(self.K, min(N // 5, 30))
        return int(self.dedup_n_clusters)

    def _compute_r_annealed(self, X: np.ndarray, beta_t: float,
                            s_t: float) -> np.ndarray:
        E_log_pi = self._E_log_pi()
        ll = self._expected_log_lik_trigamma(X)
        D = -(E_log_pi[None, :] + ll)              # (N, K)
        N, K = D.shape

        n_centroids = self._resolve_dedup_n_clusters(N)

        # ── Decide between dedup and per-sample paths ─────────────────────
        use_dedup = (n_centroids is not None and n_centroids < N)
        if use_dedup:
            try:
                km = MiniBatchKMeans(
                    n_clusters=n_centroids,
                    max_iter=self.dedup_kmeans_max_iter,
                    n_init=1,
                    random_state=int(self.random_state),
                    batch_size=min(N, 256),
                )
                centroid_labels = km.fit_predict(D)
                centroids = km.cluster_centers_
                # Some clusters can be empty after MiniBatchKMeans;
                # keep only non-empty ones.
                unique = np.unique(centroid_labels)
                if len(unique) < self.dedup_min_unique_ratio * n_centroids:
                    use_dedup = False
                else:
                    # Re-map labels to contiguous 0..U-1 and centroid array.
                    label_map = {old: new for new, old in enumerate(unique)}
                    centroid_labels = np.array(
                        [label_map[c] for c in centroid_labels]
                    )
                    centroids = centroids[unique]
            except Exception:
                use_dedup = False

        early_stop_count   = 0
        refresh_total      = 0
        inner_steps_total  = 0

        if use_dedup:
            # ── Per-centroid VarQITE; broadcast to N samples ──────────────
            U = centroids.shape[0]
            r_centroid = np.zeros((U, K))
            for c_idx in range(U):

                r_c, stats = self._varqite_responsibility(
                    centroids[c_idx], beta_t, s_t,
                    sample_id=("centroid", c_idx),
                )
                r_centroid[c_idx] = r_c
                early_stop_count  += int(stats["early_stopped"])
                refresh_total     += stats["metric_refreshes"]
                inner_steps_total += stats["steps_executed"]
            r = r_centroid[centroid_labels]
            self._varqite_stats["dedup_fraction_per_iter"].append(U / N)
        else:
            r = np.zeros((N, K))
            for i in range(N):
                r[i], stats = self._varqite_responsibility(
                    D[i], beta_t, s_t, sample_id=i
                )
                early_stop_count  += int(stats["early_stopped"])
                refresh_total     += stats["metric_refreshes"]
                inner_steps_total += stats["steps_executed"]
            self._varqite_stats["dedup_fraction_per_iter"].append(1.0)

        # Record instrumentation for the post-fit summary.
        self._varqite_stats["early_stops_per_iter"].append(
            early_stop_count
        )
        self._varqite_stats["metric_refreshes_per_iter"].append(
            refresh_total
        )
        self._varqite_stats["inner_steps_per_iter"].append(
            inner_steps_total
        )
        return r

    # ── Reporting ─────────────────────────────────────────────────────────

    def _print_fit_header(self):
        trig = "ON" if self.use_trigamma_correction else "OFF"
        dn = self.dedup_n_clusters
        dn_str = (f"{dn}" if isinstance(dn, int)
                  else ("'auto'" if dn == "auto" else "OFF"))
        print(f"\nStarting VarQITE QAVB FAST — DMM-SVVS")
        print(f"  β0={self.beta0}, s0={self.s0}, "
              f"τ1={self.tau1}, τ2={self.tau2}, "
              f"prune_start={self.prune_start}, trigamma={trig}")
        print(f"  qubits: n_sys={self.n_sys}, n_anc={self.n_sys}, "
              f"aux=1, total={self.n_tot}")
        print(f"  ansatz_depth={self.ansatz_depth}, n_params={self.n_params}, "
              f"n_varqite_steps={self.n_varqite_steps}, "
              f"mixer={self.mixer}, δ={self.regularization}")
        print(f"  [perf] device={self._active_device}, "
              f"metric_refresh={self.metric_refresh}, "
              f"r_early_stop_tol={self.r_early_stop_tol}, "
              f"dedup={dn_str}")

    def perf_summary(self) -> dict:
        """Aggregate per-iteration instrumentation for post-fit reporting."""
        stats = self._varqite_stats
        if not stats["inner_steps_per_iter"]:
            return {}
        return {
            "mean_inner_steps_per_iter": float(np.mean(
                stats["inner_steps_per_iter"])),
            "mean_early_stops_per_iter": float(np.mean(
                stats["early_stops_per_iter"])),
            "mean_metric_refreshes_per_iter": float(np.mean(
                stats["metric_refreshes_per_iter"])),
            "mean_dedup_fraction": float(np.mean(
                stats["dedup_fraction_per_iter"])),
            "device": self._active_device,
        }


# ════════════════════════════════════════════════════════════════════════════
# Smoke + benchmark vs v2 on a small dataset
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from sklearn.metrics import (adjusted_rand_score,
                                  normalized_mutual_info_score)
    from DMM_SVVS_Variational_QAVB_v2 import DMM_SVVS_VarQITE_QAVB

    print("=" * 72)
    print("VarQITE QAVB v3_fast — benchmark vs v2 on small synthetic data")
    print("=" * 72)

    rng = np.random.default_rng(42)
    N, S, K_true = 400, 5000, 5
    block = S // K_true
    alpha = np.full((K_true, S), 0.1)
    for k in range(K_true):
        alpha[k, k * block:(k + 1) * block] = 3.0
    true_labels = rng.choice(K_true, size=N)
    X = np.array([
        rng.multinomial(2000, rng.dirichlet(alpha[true_labels[i]]))
        for i in range(N)
    ], dtype=float)

    results = []

    # 1. v2 baseline (reference).
    print("\n--- v2 baseline (default.qubit, no perf fixes) ---")
    t0 = time()
    m2 = DMM_SVVS_VarQITE_QAVB(
        K_max=4, nu='auto',
        max_iter=15, beta0=5.0, s0=1.0, tau1=5, tau2=10, prune_start=12,
        verbose=1, random_state=42,
        selection_prior=0.3, prune_threshold=0.2,
        use_trigamma_correction=False,
        n_varqite_steps=8, ansatz_depth=2,
        mixer="transverse_field", regularization=1e-4,
        warm_start=True, init_perturbation=0.05,
    )
    m2.fit(X)
    dt2 = time() - t0
    pred2 = m2.predict(X)
    results.append(("v2 baseline (default.qubit)",
                    adjusted_rand_score(true_labels, pred2),
                    normalized_mutual_info_score(true_labels, pred2),
                    m2.K, dt2))

    # 2. v3 with all layers EXCEPT dedup.
    print("\n--- v3_fast: lightning + early-stop + metric-refresh ---")
    t0 = time()
    m3a = DMM_SVVS_VarQITE_QAVB_Fast(
        K_max=20, nu='auto',
        max_iter=15, beta0=5.0, s0=1.0, tau1=5, tau2=10, prune_start=12,
        verbose=1, random_state=42,
        selection_prior=0.3, prune_threshold=0.2,
        use_trigamma_correction=False,
        n_varqite_steps=8, ansatz_depth=2,
        mixer="transverse_field", regularization=1e-4,
        warm_start=True, init_perturbation=0.05,
        device_name="lightning.qubit",
        r_early_stop_tol=1e-3,
        metric_refresh=4,
        dedup_n_clusters=None,
    )
    m3a.fit(X)
    dt3a = time() - t0
    pred3a = m3a.predict(X)
    results.append(("v3_fast (no dedup)",
                    adjusted_rand_score(true_labels, pred3a),
                    normalized_mutual_info_score(true_labels, pred3a),
                    m3a.K, dt3a))
    print(f"  perf summary: {m3a.perf_summary()}")

    # 3. v3 with all layers including dedup.
    print("\n--- v3_fast: + dedup (auto) ---")
    t0 = time()
    m3b = DMM_SVVS_VarQITE_QAVB_Fast(
        K_max=4, nu='auto',
        max_iter=15, beta0=5.0, s0=1.0, tau1=5, tau2=10, prune_start=12,
        verbose=1, random_state=42,
        selection_prior=0.3, prune_threshold=0.2,
        use_trigamma_correction=False,
        n_varqite_steps=8, ansatz_depth=2,
        mixer="transverse_field", regularization=1e-4,
        warm_start=True, init_perturbation=0.05,
        device_name="lightning.qubit",
        r_early_stop_tol=1e-3,
        metric_refresh=4,
        dedup_n_clusters="auto",
    )
    m3b.fit(X)
    dt3b = time() - t0
    pred3b = m3b.predict(X)
    results.append(("v3_fast (+ dedup auto)",
                    adjusted_rand_score(true_labels, pred3b),
                    normalized_mutual_info_score(true_labels, pred3b),
                    m3b.K, dt3b))
    print(f"  perf summary: {m3b.perf_summary()}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"Benchmark (N={N}, S={S}, K_true={K_true})")
    print(f"{'Method':<36} {'ARI':>8} {'NMI':>8} {'K':>5} {'time(s)':>10}")
    print("-" * 72)
    base_time = results[0][4]
    for name, ari, nmi, k, dt in results:
        speedup = base_time / dt if dt > 0 else float("inf")
        print(f"{name:<36} {ari:8.3f} {nmi:8.3f} {k:5d} "
              f"{dt:10.1f}  ({speedup:5.1f}x)")
    print("=" * 72)
