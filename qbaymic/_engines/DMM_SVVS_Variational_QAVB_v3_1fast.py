#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMM-SVVS VarQITE QAVB 
========================================================

"""
from __future__ import annotations

import os
import sys
from time import time

import numpy as np
from sklearn.cluster import MiniBatchKMeans


os.environ.setdefault("JAX_ENABLE_X64", "1")

def _resolve_cpu_device_count() -> int:
    env = os.environ.get("V3_1_CPU_DEVICES")
    if env is not None:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    cores = os.cpu_count() or 1
    return max(1, min(cores // 2, 32))


_V3_1_CPU_DEVICES = _resolve_cpu_device_count()
if _V3_1_CPU_DEVICES > 1:

    _flags = os.environ.get("XLA_FLAGS", "")
    if "xla_force_host_platform_device_count" not in _flags:
        os.environ["XLA_FLAGS"] = (
            (_flags + " " if _flags else "")
            + f"--xla_force_host_platform_device_count={_V3_1_CPU_DEVICES}"
        )

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from DMM_SVVS_Variational_QAVB_v2 import NumericalStability  # noqa: E402
from DMM_SVVS_Variational_QAVB_v3_fast import (  # noqa: E402
    DMM_SVVS_VarQITE_QAVB_Fast,
)


class DMM_SVVS_VarQITE_QAVB_JAX(DMM_SVVS_VarQITE_QAVB_Fast):
    """
    """

    def __init__(
        self,
        *args,
        sample_batch_size: int | None = None,
        jit_warmup: bool = True,

        phantom_penalty: float = 10.0,

        decouple_phantom_mixer: bool = True,

        dtau_max: float | None = 0.2,

        n_estep_restarts: int = 3,

        cpu_parallel="auto",

        cpu_parallel_min_batch: int = 64,
        **kwargs,
    ):

        requested = kwargs.get("device_name", "default.qubit")
        if str(requested) not in ("default.qubit", "default.qubit.jax"):
            if kwargs.get("verbose", 1) >= 1:
                print(f"  [v3.1] device '{requested}' is not a jax.jit target; "
                      f"using default.qubit (interface='jax').")
        kwargs["device_name"] = "default.qubit"
        super().__init__(*args, **kwargs)

        self.PHANTOM_PENALTY = float(phantom_penalty)

        self.decouple_phantom_mixer = bool(decouple_phantom_mixer)

        self.dtau_max = (None if dtau_max is None else float(dtau_max))
        self._n_steps_user = int(self.n_varqite_steps)
        self._dtau_clamp_applied = False    # one-time log gate

        self.n_estep_restarts = max(1, int(n_estep_restarts))

        self.sample_batch_size = (None if sample_batch_size is None
                                  else int(sample_batch_size))
        self.jit_warmup = bool(jit_warmup)
        self.cpu_parallel = cpu_parallel
        self.cpu_parallel_min_batch = max(1, int(cpu_parallel_min_batch))

        if self.r_early_stop_tol is not None and self.verbose >= 1:
            print("  [v3.1] note: r_early_stop_tol is ignored under JIT+vmap "
                  "(data-dependent breaks defeat single-compile). Inner loop "
                  "runs all n_varqite_steps.")

        # JAX handles (imported lazily in _init).
        self._jax = None
        self._jnp = None
        self._batched_kernel = None   # jitted vmap(scan(step)) -> theta_final
        self._probs_batched = None    # jitted vmap(readout)
        self._pmap_kernel = None      # pmap(vmap(scan(step))) -> theta_final
        self._pmap_probs = None       # pmap(vmap(readout))
        self._n_devices = 1           # set in _init_pennylane_device
        self._compiled_signature = None  # (n_sys, n_params, P, n_steps, refresh)
        self._theta_warm = None       # (N, n_params) JAX array carried across iters
        self._compile_seconds = 0.0

    # ── JAX-backed device + static Pauli basis ────────────────────────────

    def _init_pennylane_device(self):
        try:
            import pennylane as qml
        except ImportError:
            raise ImportError("PennyLane required:  pip install pennylane>=0.40")
        try:
            import jax
            import jax.numpy as jnp
            from jax import lax  # noqa: F401
        except ImportError:
            raise ImportError("JAX required for v3.1:  pip install -U 'jax[cpu]'")

        jax.config.update("jax_enable_x64", True)

        self._qml = qml
        self._jax = jax
        self._jnp = jnp

        self._n_devices = max(1, jax.local_device_count())

        self.n_sys = max(1, int(np.ceil(np.log2(max(self.K, 2)))))
        self.n_anc = self.n_sys
        self.K_pad = 1 << self.n_sys
        self.n_tot = 2 * self.n_sys + 1
        self.n_params = 2 * (2 * self.n_sys) * self.ansatz_depth

        self._dev = qml.device("default.qubit", wires=self.n_tot)
        self._active_device = "default.qubit (jax)"

        n_sys = self.n_sys

        def z_string_op(z):
            active = [q for q in range(n_sys)
                      if (z >> (n_sys - 1 - q)) & 1]
            if not active:
                return qml.Identity(0)
            op = qml.PauliZ(active[0])
            for q in active[1:]:
                op = op @ qml.PauliZ(q)
            return op

        diag_ops = [z_string_op(z) for z in range(self.K_pad)]

        self._dense_mix_static_c = None

        if self.mixer == "transverse_field":
            if self.decouple_phantom_mixer:

                H_mix = self._projected_transverse_field_matrix()
                terms = self._dense_to_pauli_terms(H_mix)
                mixer_ops = [op for _, op in terms]
                self._dense_mix_static_c = np.asarray(
                    [float(c) for c, _ in terms], dtype=np.float64)
            else:

                mixer_ops = [qml.PauliX(q) for q in range(n_sys)]
        elif self.mixer == "cyclic_shift":

            H_mix = self._cyclic_shift_padded_matrix()
            terms = self._dense_to_pauli_terms(H_mix)  # [(coeff, qml-op)]
            mixer_ops = [op for _c, op in terms]

            self._dense_mix_static_c = np.asarray(
                [float(_c) for _c, _op in terms], dtype=np.float64)
            # Legacy alias retained for any external code that may read it.
            self._cyclic_mix_static_c = self._dense_mix_static_c
        else:
            raise ValueError(f"Unknown mixer: {self.mixer!r}")

        self._all_pauli_ops = diag_ops + mixer_ops
        self._n_diag = len(diag_ops)
        self._n_mix = len(mixer_ops)
        self._P = len(self._all_pauli_ops)

        Kp = self.K_pad
        Hsign = np.empty((Kp, Kp), dtype=np.float64)
        for z in range(Kp):
            zb = [(z >> (n_sys - 1 - q)) & 1 for q in range(n_sys)]
            for k in range(Kp):
                kb = [(k >> (n_sys - 1 - q)) & 1 for q in range(n_sys)]
                par = sum(a & b for a, b in zip(zb, kb)) & 1
                Hsign[z, k] = -1.0 if par else 1.0
        self._walsh_sign = jnp.asarray(Hsign / Kp)

        self._build_jax_kernels()

        # Invalidate any carried warm-start array (K may have changed).
        self._theta_warm = None
        self._theta_cache = {}   # kept for API; unused on the JAX path

    # ── Fix B helper: dense projected transverse-field mixer ──────────────

    def _projected_transverse_field_matrix(self) -> np.ndarray:
        """
        Dense (K_pad, K_pad) matrix of  H_mixer = -Σ_q X_q  PROJECTED onto
        the K-real-cluster subspace (phantom rows/cols zeroed).

        Construction:
          1. Build M = -Σ_q X_q on n_sys qubits as a dense 2^n_sys × 2^n_sys
             matrix. This is the canonical transverse-field driver.
          2. Zero rows/cols whose index ≥ K. The result satisfies
                 P_K · M · P_K     for the diagonal projector P_K =
                 diag(1,...,1,0,...,0).
             So the mixer can only hop *within* the K-block; the phantom
             state is isolated.

        Sign convention matches v2's H_mixer = -Σ X_q so the same
        contribution rule  H_S += -s_t · H_mixer  applies (giving
        +s_t · Σ X_q on the K-block when expanded).
        """
        n_sys = self.n_sys
        dim = 1 << n_sys
        I_q = np.eye(2, dtype=np.float64)
        X_q = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)

        M = np.zeros((dim, dim), dtype=np.float64)
        for q in range(n_sys):

            op = None
            for w in range(n_sys):
                factor = X_q if w == q else I_q
                op = factor if op is None else np.kron(op, factor)
            M -= op

        # Project onto the K-block: zero phantom rows and columns.
        K = self.K
        if K < dim:
            M[K:, :] = 0.0
            M[:, K:] = 0.0
        return M

    # ── Build the static qnodes and the jitted vmap+scan kernel ───────────

    def _build_jax_kernels(self):
        qml = self._qml
        jax = self._jax
        jnp = self._jnp
        lax = jax.lax
        ALL_OPS = self._all_pauli_ops
        n_sys = self.n_sys
        P_obs = self._P
        delta = float(self.regularization)
        n_steps = max(int(self.n_varqite_steps), 1)
        refresh = max(int(self.metric_refresh), 1)

        @qml.qnode(self._dev, interface="jax")
        def _expvals(th):
            self._ansatz(th)
            return [qml.expval(o) for o in ALL_OPS]

        @qml.qnode(self._dev, interface="jax")
        def _state(th):
            self._ansatz(th)
            return qml.state()

        @qml.qnode(self._dev, interface="jax")
        def _probs(th):
            self._ansatz(th)
            return qml.probs(wires=list(range(n_sys)))

        def energy(th, c):
            e = jnp.stack(_expvals(th))
            return jnp.sum(c * e)

        def metric_A(th):
            psi = _state(th)
            J = jax.jacfwd(lambda t: _state(t))(th)          # (dim, p) complex
            G = jnp.einsum("ki,kj->ij", jnp.conj(J), J)
            v = jnp.einsum("ki,k->i", jnp.conj(J), psi)
            return jnp.real(G - jnp.outer(v, jnp.conj(v)))   # (p, p) real

        grad_energy = jax.grad(energy)

        def trajectory(theta0, c, dtau):
            """One sample's full inner loop as a lax.scan with carried A
            (§3.2 metric-refresh). Carry = (theta, A, age)."""
            p = theta0.shape[0]
            A0 = jnp.zeros((p, p))

            def step(carry, _):
                theta, A_cached, age = carry
                # Refresh A when age == 0 (mod refresh); else reuse carried A.
                need_refresh = (age % refresh) == 0
                A = lax.cond(need_refresh,
                             lambda t: metric_A(t),
                             lambda _t: A_cached,
                             theta)
                C = 0.5 * grad_energy(theta, c)
                A_reg = A + delta * jnp.eye(p)
                dth = jnp.linalg.solve(A_reg, -C)
                theta_new = theta + dtau * dth
                return (theta_new, A, age + 1), None

            (theta_final, _A, _age), _ = lax.scan(
                step, (theta0, A0, 0), None, length=n_steps)
            return theta_final

        # vmap over (theta0[i], c[i]); dtau broadcast as scalar.
        batched = jax.vmap(trajectory, in_axes=(0, 0, None))
        self._batched_kernel = jax.jit(batched)

        probs_batched = jax.vmap(_probs, in_axes=(0,))
        self._probs_batched = jax.jit(probs_batched)

        energy_batched = jax.vmap(energy, in_axes=(0, 0))
        self._energy_batched = jax.jit(energy_batched)


        if self._n_devices > 1:
            self._pmap_kernel = jax.pmap(
                jax.vmap(trajectory, in_axes=(0, 0, None)),
                in_axes=(0, 0, None))
            self._pmap_probs = jax.pmap(
                jax.vmap(_probs, in_axes=(0,)), in_axes=(0,))
            self._pmap_energy = jax.pmap(
                jax.vmap(energy, in_axes=(0, 0)), in_axes=(0, 0))
        else:
            self._pmap_kernel = None
            self._pmap_probs = None
            self._pmap_energy = None

        self._compiled_signature = (n_sys, self.n_params, P_obs,
                                    n_steps, refresh)

    # ── §3.1 per-sample coefficient vector (vectorised, JAX) ──────────────

    def _coefficients_batch(self, D_block: np.ndarray, s_t: float):
        """
        Map a batch of raw per-sample energy rows D_block (B, K) to the
        coefficient matrix C (B, P) indexed by the static Pauli basis,
        applying v2's EXACT per-sample shift/scale/phantom-padding.

        Returns a JAX array (B, P).
        """
        jnp = self._jnp
        K = self.K
        Kp = self.K_pad
        B = D_block.shape[0]

        D = np.asarray(D_block, dtype=np.float64)
        d_shift = D - D.min(axis=1, keepdims=True)
        d_range = d_shift.max(axis=1, keepdims=True)
        scaled = np.where(d_range > 1e-10, 4.0 * d_shift / np.maximum(d_range, 1e-300),
                          d_shift)
        # Pad to K_pad with PHANTOM_PENALTY.
        d_padded = np.full((B, Kp), self.PHANTOM_PENALTY, dtype=np.float64)
        d_padded[:, :K] = scaled
        d_padded = jnp.asarray(d_padded)

        c_diag = (1.0 - s_t) * (d_padded @ self._walsh_sign.T)   # (B, K_pad)

        if self._dense_mix_static_c is not None:
            base = jnp.asarray(self._dense_mix_static_c)
            c_mix = -float(s_t) * jnp.broadcast_to(base, (B, self._n_mix))
        else:
            # Legacy: mixer is -Σ X_q on the full K_pad basis (phantom-coupled).
            c_mix = jnp.full((B, self._n_mix), float(s_t))

        return jnp.concatenate([c_diag, c_mix], axis=1)            # (B, P)

    # ── Warm-start / init parameter batch (carried JAX array, §3.2) ───────

    def _init_theta_batch(self, n_rows: int, keys=None) -> np.ndarray:
        """Initial θ for each row: warm-started from the carried array when
        available and shaped consistently, else identity-block + Gaussian
        kick (v2 convention)."""
        if self.init_perturbation > 0:
            theta0 = self._varqite_rng.normal(
                0.0, self.init_perturbation,
                size=(n_rows, self.n_params)).astype(np.float64)
        else:
            theta0 = np.zeros((n_rows, self.n_params), dtype=np.float64)

        if (self.warm_start and self._theta_warm is not None
                and self._theta_warm.shape == (n_rows, self.n_params)):
            theta0 = np.asarray(self._theta_warm, dtype=np.float64)
        return theta0

    # ── Kernel dispatch: pmap over CPU shards, or single-device vmap ──────

    def _use_pmap(self, B: int) -> bool:
        """Decide whether to shard a batch of size B across devices."""
        if self._n_devices <= 1 or self._pmap_kernel is None:
            return False
        if self.cpu_parallel is False:
            return False
        if self.cpu_parallel == "auto":

            return B >= max(self.cpu_parallel_min_batch, 2 * self._n_devices)
        return True   # cpu_parallel is True

    def _run_kernels(self, theta0: np.ndarray, C_mat, dtau: float):
        """
        Evaluate (theta_final, probs, energy) for the FULL batch theta0/C_mat.

        """
        jnp = self._jnp
        B = theta0.shape[0]

        if self._use_pmap(B):
            D = self._n_devices

            pad = (-B) % D
            if pad:
                theta0 = np.concatenate(
                    [theta0, np.zeros((pad, self.n_params))], axis=0)
                C_pad = jnp.concatenate(
                    [C_mat, jnp.zeros((pad, C_mat.shape[1]))], axis=0)
            else:
                C_pad = C_mat
            Bp = B + pad
            per = Bp // D
            th_sh = jnp.asarray(theta0).reshape(D, per, self.n_params)
            c_sh = C_pad.reshape(D, per, C_pad.shape[1])

            thf = self._pmap_kernel(th_sh, c_sh, dtau)        # (D, per, p)
            pr = self._pmap_probs(thf)                        # (D, per, K_pad)
            en = self._pmap_energy(thf, c_sh)                 # (D, per)

            thf = np.asarray(thf).reshape(Bp, self.n_params)[:B]
            pr = np.asarray(pr).reshape(Bp, self.K_pad)[:B]
            en = np.asarray(en).reshape(Bp)[:B]
            return thf, pr, en

        # Single-device path: chunk by sample_batch_size.
        bs = self.sample_batch_size or B
        thf_all = np.empty((B, self.n_params), dtype=np.float64)
        pr_all = np.empty((B, self.K_pad), dtype=np.float64)
        for start in range(0, B, bs):
            stop = min(start + bs, B)
            th0 = jnp.asarray(theta0[start:stop])
            cc = C_mat[start:stop]
            thf = self._batched_kernel(th0, cc, dtau)
            pr = self._probs_batched(thf)
            thf_all[start:stop] = np.asarray(thf)
            pr_all[start:stop] = np.asarray(pr)
        en_all = np.asarray(self._energy_batched(jnp.asarray(thf_all), C_mat))
        return thf_all, pr_all, en_all

    # ── Batched VarQITE over a set of energy rows (samples or centroids) ──

    def _varqite_batch(self, D_block: np.ndarray, beta_t: float, s_t: float):
        """
        Run the full McLachlan inner loop for a batch of energy rows in one
        jitted vmap call (chunked by sample_batch_size if set).

        Returns
        -------
        r : (B, K) responsibilities (clip+renormalise on the K-block, v2).
        theta_final : (B, n_params) for warm-start carry.
        """
        EPS = NumericalStability.EPS
        B = D_block.shape[0]

        C_mat = self._coefficients_batch(D_block, s_t)            # (B, P) JAX

        tau_final = 0.5 * beta_t
        dtau = tau_final / max(self.n_varqite_steps, 1)
        R = self.n_estep_restarts


        thetas_per_R = np.empty((R, B, self.n_params), dtype=np.float64)
        probs_per_R  = np.empty((R, B, self.K_pad),  dtype=np.float64)
        energy_per_R = np.empty((R, B),               dtype=np.float64)

        for r_idx in range(R):

            if r_idx == 0:
                theta0 = self._init_theta_batch(B)
            else:
                theta0 = self._varqite_rng.normal(
                    0.0, self.init_perturbation,
                    size=(B, self.n_params)).astype(np.float64)

            thf, pr, en = self._run_kernels(theta0, C_mat, dtau)
            thetas_per_R[r_idx] = thf
            probs_per_R[r_idx]  = pr
            energy_per_R[r_idx] = en

        if R == 1:
            theta_final = thetas_per_R[0]
            probs       = probs_per_R[0]
        else:
            best_r = np.argmin(energy_per_R, axis=0)              # (B,)
            sample_idx = np.arange(B)
            theta_final = thetas_per_R[best_r, sample_idx]        # (B, p)
            probs       = probs_per_R[best_r, sample_idx]         # (B, K_pad)

        r = np.clip(probs[:, : self.K], EPS, None)
        r = r / r.sum(axis=1, keepdims=True)
        return r, theta_final

    # ── E-step over the dataset (single vmap; dedup optional) ─────────────

    def _compute_r_annealed(self, X: np.ndarray, beta_t: float,
                            s_t: float) -> np.ndarray:

        if self.dtau_max is not None and not self._dtau_clamp_applied:
            n_min_for_dtau = int(np.ceil((0.5 * self.beta0) / self.dtau_max))
            n_eff = max(self._n_steps_user, n_min_for_dtau)
            if n_eff > self.n_varqite_steps:
                if self.verbose >= 1:
                    print(f"  [v3.1] dtau clamp: n_varqite_steps "
                          f"{self.n_varqite_steps} → {n_eff}  "
                          f"(beta0={self.beta0}, dtau_max={self.dtau_max} → "
                          f"required dtau={0.5*self.beta0/n_eff:.4f})")
                self.n_varqite_steps = n_eff
            self._dtau_clamp_applied = True


        sig = (self.n_sys, self.n_params, self._P,
               max(int(self.n_varqite_steps), 1),
               max(int(self.metric_refresh), 1))
        if sig != self._compiled_signature:
            self._build_jax_kernels()

        E_log_pi = self._E_log_pi()
        ll = self._expected_log_lik_trigamma(X)
        D = -(E_log_pi[None, :] + ll)              # (N, K)
        N, K = D.shape

        n_centroids = self._resolve_dedup_n_clusters(N)
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
                unique = np.unique(centroid_labels)
                if len(unique) < self.dedup_min_unique_ratio * n_centroids:
                    use_dedup = False
                else:
                    label_map = {old: new for new, old in enumerate(unique)}
                    centroid_labels = np.array(
                        [label_map[c] for c in centroid_labels])
                    centroids = centroids[unique]
            except Exception:
                use_dedup = False

        # JIT warmup (paid once, reported separately).
        self._maybe_warmup(K)

        inner_steps = max(int(self.n_varqite_steps), 1)

        if use_dedup:
            U = centroids.shape[0]
            r_centroid, _theta_c = self._varqite_batch(centroids, beta_t, s_t)
            r = r_centroid[centroid_labels]

            self._varqite_stats["dedup_fraction_per_iter"].append(U / N)
            self._varqite_stats["inner_steps_per_iter"].append(U * inner_steps)
        else:
            r, theta_final = self._varqite_batch(D, beta_t, s_t)
            if self.warm_start:
                # Carry the (N, p) array for next iteration (§3.2).
                self._theta_warm = self._jnp.asarray(theta_final)
            self._varqite_stats["dedup_fraction_per_iter"].append(1.0)
            self._varqite_stats["inner_steps_per_iter"].append(N * inner_steps)

        # Early-stop is disabled under JIT; record 0 for the summary.
        self._varqite_stats["early_stops_per_iter"].append(0)
        refresh = max(int(self.metric_refresh), 1)
        rows = (centroids.shape[0] if use_dedup else N)
        self._varqite_stats["metric_refreshes_per_iter"].append(
            rows * ((inner_steps + refresh - 1) // refresh))
        return r

    def _maybe_warmup(self, K: int):
        if not self.jit_warmup or self._compile_seconds > 0:
            return
        jnp = self._jnp
        try:
            t0 = time()
            dummy_D = np.zeros((2, K), dtype=np.float64)
            dummy_D[0, 0] = 1.0
            self._varqite_batch(dummy_D, beta_t=2.0, s_t=0.5)
            self._compile_seconds = time() - t0
            if self.verbose >= 1:
                print(f"  [v3.1] JIT compile warmup: "
                      f"{self._compile_seconds:.2f}s "
                      f"(n_sys={self.n_sys}, P={self._P}, "
                      f"steps={self.n_varqite_steps}, "
                      f"refresh={self.metric_refresh})")
        except Exception as exc:
            if self.verbose >= 1:
                print(f"  [v3.1] warmup skipped ({type(exc).__name__})")

    # ── Reporting ─────────────────────────────────────────────────────────

    def _print_fit_header(self):
        trig = "ON" if self.use_trigamma_correction else "OFF"
        dn = self.dedup_n_clusters
        dn_str = (f"{dn}" if isinstance(dn, int)
                  else ("'auto'" if dn == "auto" else "OFF"))
        bs = self.sample_batch_size or "all-N"
        print(f"\nStarting VarQITE QAVB v3.1 (JAX+JIT+vmap) — DMM-SVVS")
        print(f"  β0={self.beta0}, s0={self.s0}, "
              f"τ1={self.tau1}, τ2={self.tau2}, "
              f"prune_start={self.prune_start}, trigamma={trig}")
        print(f"  qubits: n_sys={self.n_sys}, n_anc={self.n_sys}, "
              f"aux=1, total={self.n_tot}")
        print(f"  ansatz_depth={self.ansatz_depth}, n_params={self.n_params}, "
              f"n_varqite_steps={self.n_varqite_steps}, "
              f"mixer={self.mixer}, δ={self.regularization}")
        par = (f"pmap×{self._n_devices}"
               if (self._n_devices > 1 and self.cpu_parallel is not False)
               else "single-dev")
        print(f"  [perf] device={self._active_device}, "
              f"static_pauli_P={self._P}, "
              f"metric_refresh={self.metric_refresh}, "
              f"vmap_batch={bs}, cpu={par}, dedup={dn_str}")
        padded = (self.K != self.K_pad)
        decoup = ("ON" if self._dense_mix_static_c is not None
                  and self.mixer == "transverse_field"
                  and self.decouple_phantom_mixer else "OFF")
        print(f"  [phantom] K={self.K}/K_pad={self.K_pad} "
              f"({'PADDED' if padded else 'no padding'}), "
              f"penalty={self.PHANTOM_PENALTY:g}, "
              f"decouple_mixer={decoup}")
        if self.dtau_max is not None:
            dtau_at_beta0 = 0.5 * self.beta0 / max(self.n_varqite_steps, 1)
            print(f"  [integrator] dtau_max={self.dtau_max}, "
                  f"dtau@beta0={dtau_at_beta0:.4f}  "
                  f"(user n_steps={self._n_steps_user}, effective="
                  f"{self.n_varqite_steps})")
        if self.n_estep_restarts > 1:
            print(f"  [restarts] n_estep_restarts={self.n_estep_restarts}  "
                  f"(picks lowest <H> per sample)")

    def perf_summary(self) -> dict:
        base = super().perf_summary()
        if base:
            base["compile_seconds"] = float(self._compile_seconds)
        return base

    # ── Visualisation: optimized-ansatz output (per-qubit Bloch) ──────────

    def optimized_bloch_vectors(self, d_i, beta_t=None, s_t=0.0):
        """
        Run VarQITE on demand for ONE energy row d_i and return the per-qubit
        reduced Bloch vectors of the OPTIMIZED state self._ansatz(theta*).

        The model must already be fitted (so n_sys / n_params / kernels exist).
        We re-prepare the trained circuit and measure ⟨X⟩,⟨Y⟩,⟨Z⟩ on every
        one of the n_tot = 2·n_sys+1 wires.

        Parameters
        ----------
        d_i : (K,) array
            A per-sample energy row (e.g. one row of D from the E-step, or a
            centroid). Shift/scale/padding is applied exactly as in the fit.
        beta_t : float or None
            Inverse temperature for the imaginary-time horizon τ=β/2. Defaults
            to the final-schedule value self.beta0 (the trained regime).
        s_t : float, default 0.0
            Mixer strength. 0.0 = the diagonal-dominated end of the anneal
            (where the readout is the responsibility); 1.0 = pure mixer.

        Returns
        -------
        dict with:
            theta     : (n_params,) optimized parameters
            bloch     : (n_tot, 3) array of (⟨X⟩,⟨Y⟩,⟨Z⟩) per wire
            purity    : (n_tot,) single-qubit purity ½(1+|r|²) per wire
            probs     : (K,) readout responsibilities on the system register
            roles     : list[str] wire role labels ("S0..","A0..","aux")
        """
        if not hasattr(self, "n_sys") or self._batched_kernel is None:
            raise RuntimeError(
                "Model not initialised. Call .fit(X) first (or at least "
                "_initialize_parameters) so the device/kernels exist.")
        qml = self._qml
        jnp = self._jnp
        np_ = np

        beta_t = float(self.beta0 if beta_t is None else beta_t)
        d_i = np_.asarray(d_i, dtype=np_.float64).reshape(-1)
        if d_i.shape[0] != self.K:
            raise ValueError(f"d_i must have length K={self.K}, got {d_i.shape[0]}")

        # Optimize theta for this single row via the batched kernel (B=1).
        r_batch, theta_batch = self._varqite_batch(d_i[None, :], beta_t, s_t)
        theta = np_.asarray(theta_batch[0], dtype=np_.float64)
        probs = np_.asarray(r_batch[0], dtype=np_.float64)

        # Per-wire Bloch components from the optimized state. One QNode per
        # Pauli axis returning expvals on every wire (cheap, n_tot wires).
        n_tot = self.n_tot

        @qml.qnode(self._dev, interface="jax")
        def _bloch(th):
            # Return measurements as a flat tuple (X for all wires, then Y,
            # then Z); PennyLane stacks them. We cannot jnp.stack measurement
            # objects inside the qfunc — reshape the returned array outside.
            self._ansatz(th)
            return tuple(
                [qml.expval(qml.PauliX(w)) for w in range(n_tot)]
                + [qml.expval(qml.PauliY(w)) for w in range(n_tot)]
                + [qml.expval(qml.PauliZ(w)) for w in range(n_tot)])

        flat = np_.asarray(_bloch(jnp.asarray(theta)), dtype=np_.float64)
        bxyz = flat.reshape(3, n_tot)                       # rows: X, Y, Z
        bloch = bxyz.T                                       # (n_tot, 3)
        rnorm = np_.linalg.norm(bloch, axis=1)
        purity = 0.5 * (1.0 + rnorm ** 2)                   # single-qubit purity

        roles = ([f"S{q}" for q in range(self.n_sys)]
                 + [f"A{q}" for q in range(self.n_sys)]
                 + ["aux"])
        return dict(theta=theta, bloch=bloch, purity=purity,
                    probs=probs, roles=roles, beta_t=beta_t, s_t=float(s_t))

    def visualize_optimized_ansatz(self, d_i, beta_t=None, s_t=0.0,
                                   out_path="v3_1_optimized_bloch.png",
                                   title=None, show=False):
        """
        Render the OPTIMIZED-ansatz output for one energy row d_i as a row of
        per-qubit Bloch spheres (system / ancilla / aux colour-coded) plus a
        bar chart of the readout responsibilities.

        Saves a PNG to `out_path` and returns the path. Requires matplotlib.
        See optimized_bloch_vectors() for the parameter meanings.
        """
        try:
            import matplotlib
            if not show:
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        except ImportError as exc:
            raise ImportError("matplotlib required for visualisation: "
                              "pip install matplotlib") from exc

        res = self.optimized_bloch_vectors(d_i, beta_t=beta_t, s_t=s_t)
        bloch, purity = res["bloch"], res["purity"]
        roles, probs = res["roles"], res["probs"]
        n_tot = self.n_tot

        role_color = {"S": "#2266aa", "A": "#aa5522", "a": "#338855"}

        def col(role):
            return role_color["a" if role == "aux" else role[0]]

        # Layout: one 3D Bloch sphere per wire on the top row, a probability
        # bar chart spanning the bottom row.
        fig = plt.figure(figsize=(max(3.0 * n_tot, 6.0), 6.2))
        gs = fig.add_gridspec(2, n_tot, height_ratios=[3.0, 1.4],
                              hspace=0.35, wspace=0.25)

        # Reference wireframe sphere (computed once).
        import numpy as _np
        u = _np.linspace(0, 2 * _np.pi, 24)
        v = _np.linspace(0, _np.pi, 16)
        sx = _np.outer(_np.cos(u), _np.sin(v))
        sy = _np.outer(_np.sin(u), _np.sin(v))
        sz = _np.outer(_np.ones_like(u), _np.cos(v))

        for w in range(n_tot):
            ax = fig.add_subplot(gs[0, w], projection="3d")
            ax.plot_wireframe(sx, sy, sz, color="#dddddd", linewidth=0.4)
            # axes through the sphere
            for a0, a1 in [((-1, 0, 0), (1, 0, 0)),
                           ((0, -1, 0), (0, 1, 0)),
                           ((0, 0, -1), (0, 0, 1))]:
                ax.plot(*zip(a0, a1), color="#bbbbbb", linewidth=0.6)
            bx, by, bz = bloch[w]
            c = col(roles[w])
            ax.quiver(0, 0, 0, bx, by, bz, color=c, linewidth=2.2,
                      arrow_length_ratio=0.18)
            ax.scatter([bx], [by], [bz], color=c, s=30)
            ax.set_title(f"{roles[w]}\n|r|={_np.linalg.norm(bloch[w]):.2f}  "
                         f"P={purity[w]:.2f}", fontsize=9, color=c)
            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.set_box_aspect((1, 1, 1))

        # Responsibility bar chart spanning the bottom row.
        axp = fig.add_subplot(gs[1, :])
        axp.bar(range(len(probs)), probs, color="#2266aa", alpha=0.85)
        axp.set_xlabel("cluster k")
        axp.set_ylabel("responsibility r")
        axp.set_xticks(range(len(probs)))
        axp.set_title("Readout: diag(ρ_S) on system register "
                      f"(s_t={res['s_t']:.2f}, β={res['beta_t']:.2f})",
                      fontsize=10)
        axp.grid(axis="y", alpha=0.3)

        if title is None:
            title = (f"v3.1 optimized ansatz output — K={self.K}, "
                     f"n_sys={self.n_sys}, depth={self.ansatz_depth}, "
                     f"steps={self.n_varqite_steps}\n"
                     f"per-qubit reduced Bloch vectors of |ψ(θ*)⟩  "
                     f"(blue=system Sk, orange=ancilla Ak, green=aux)")
        fig.suptitle(title, fontsize=11)

        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return out_path

    # ── Animated anneal: per-qubit Bloch spheres as s_t sweeps 1 → 0 ──────

    def animate_anneal_bloch(self, d_i, beta_t=None,
                             s_start=1.0, s_end=0.0, n_frames=41,
                             out_path="v3_1_anneal_bloch.gif",
                             fps=8, dpi=110, title=None, trail=True):
        """
        Render an ANIMATED GIF of the optimized-ansatz per-qubit Bloch vectors
        as the quantum-annealing mixer strength s_t is swept from `s_start`
        (default 1.0, pure mixer) down to `s_end` (default 0.0, diagonal end —
        where the readout IS the responsibility).

        Each frame re-runs VarQITE for the same energy row d_i at that s_t and
        draws the same layout as visualize_optimized_ansatz (one 3D Bloch
        sphere per wire + the readout responsibility bars), so the GIF shows
        how the trained state — and the cluster responsibilities — evolve
        along the anneal.

        Parameters
        ----------
        d_i : (K,) array
            The per-sample energy row to animate (e.g. D[sample_idx, :K]).
        beta_t : float or None
            Inverse temperature for the imaginary-time horizon. Defaults to
            self.beta0 (the trained regime), held FIXED across frames so the
            only thing varying is s_t.
        s_start, s_end : float
            Mixer-strength sweep endpoints. Default 1.0 → 0.0 (the anneal
            direction). The sweep is inclusive of both ends.
        n_frames : int
            Number of s_t samples (frames). 41 → steps of 0.025 over [0,1].
        out_path : str
            Output GIF path.
        fps : int
            Frames per second of the GIF.
        dpi : int
            Render resolution per frame (lower = smaller file / faster).
        title : str or None
            Figure suptitle; a sensible default is built if None.
        trail : bool
            If True, draw a faint dot trail of each wire's Bloch-vector tip
            over previous frames, so the path traced during the anneal is
            visible. Set False for clean per-frame arrows only.

        Returns
        -------
        out_path : str  (the GIF that was written)

        Notes
        -----
        Axis limits ([-1,1]³) and the responsibility-bar y-axis are FIXED
        across all frames, so the animation shows genuine motion rather than
        autoscaling artefacts. Requires matplotlib + Pillow.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")          # offscreen — we assemble a GIF
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            from PIL import Image
        except ImportError as exc:
            raise ImportError("matplotlib + Pillow required for the GIF: "
                              "pip install matplotlib pillow") from exc

        _np = np
        beta_t = float(self.beta0 if beta_t is None else beta_t)
        d_i = _np.asarray(d_i, dtype=_np.float64).reshape(-1)
        n_tot = self.n_tot

        # s_t sweep (inclusive endpoints). Clip to [0,1] to stay in range.
        s_values = _np.clip(
            _np.linspace(float(s_start), float(s_end), int(n_frames)),
            0.0, 1.0)

        role_color = {"S": "#2266aa", "A": "#aa5522", "a": "#338855"}

        def col(role):
            return role_color["a" if role == "aux" else role[0]]

        # Reference wireframe sphere (computed once, reused per frame).
        u = _np.linspace(0, 2 * _np.pi, 24)
        v = _np.linspace(0, _np.pi, 16)
        sx = _np.outer(_np.cos(u), _np.sin(v))
        sy = _np.outer(_np.sin(u), _np.sin(v))
        sz = _np.outer(_np.ones_like(u), _np.cos(v))

        # First pass: compute Bloch vectors + probs at every s_t. Doing this
        # up front (a) fixes the responsibility-bar y-limit across frames and
        # (b) lets us draw the tip trail. Each call re-optimises VarQITE at
        # that s_t, so this is the bulk of the runtime.
        frames_data = []
        max_prob = 1e-6
        for s in s_values:
            res = self.optimized_bloch_vectors(d_i, beta_t=beta_t, s_t=float(s))
            frames_data.append(res)
            max_prob = max(max_prob, float(_np.max(res["probs"])))
        roles = frames_data[0]["roles"]
        prob_ylim = min(1.0, 1.15 * max_prob)

        if title is None:
            title = (f"VarQITE anneal  s_t: {s_start:.2f} → {s_end:.2f}  "
                     f"(K={self.K}, n_sys={self.n_sys}, depth={self.ansatz_depth}, "
                     f"β={beta_t:.1f})\n"
                     f"per-qubit Bloch vectors of |ψ(θ*)⟩  "
                     f"(blue=system Sk, orange=ancilla Ak, green=aux)")

        # Tip-trail accumulator (per wire, list of (x,y,z)).
        trails = [[] for _ in range(n_tot)]

        images = []
        fig_w = max(3.0 * n_tot, 6.0)
        for fi, (s, res) in enumerate(zip(s_values, frames_data)):
            bloch, purity, probs = res["bloch"], res["purity"], res["probs"]

            fig = plt.figure(figsize=(fig_w, 6.6))
            gs = fig.add_gridspec(2, n_tot, height_ratios=[3.0, 1.5],
                                  hspace=0.42, wspace=0.25)

            for w in range(n_tot):
                ax = fig.add_subplot(gs[0, w], projection="3d")
                ax.plot_wireframe(sx, sy, sz, color="#dddddd", linewidth=0.4)
                for a0, a1 in [((-1, 0, 0), (1, 0, 0)),
                               ((0, -1, 0), (0, 1, 0)),
                               ((0, 0, -1), (0, 0, 1))]:
                    ax.plot(*zip(a0, a1), color="#bbbbbb", linewidth=0.6)

                bx, by, bz = bloch[w]
                c = col(roles[w])

                # Faint tip trail of previous frames.
                if trail:
                    trails[w].append((bx, by, bz))
                    if len(trails[w]) > 1:
                        tp = _np.array(trails[w])
                        ax.plot(tp[:, 0], tp[:, 1], tp[:, 2],
                                color=c, linewidth=0.8, alpha=0.35)

                ax.quiver(0, 0, 0, bx, by, bz, color=c, linewidth=2.4,
                          arrow_length_ratio=0.18)
                ax.scatter([bx], [by], [bz], color=c, s=32)
                ax.set_title(f"{roles[w]}\n|r|={_np.linalg.norm(bloch[w]):.2f}  "
                             f"P={purity[w]:.2f}", fontsize=9, color=c)
                ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
                ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
                ax.set_box_aspect((1, 1, 1))

            # Responsibility bars (fixed y-limit across frames).
            axp = fig.add_subplot(gs[1, :])
            axp.bar(range(len(probs)), probs, color="#2266aa", alpha=0.85)
            axp.set_xlabel("cluster k")
            axp.set_ylabel("responsibility r")
            axp.set_xticks(range(len(probs)))
            axp.set_ylim(0.0, prob_ylim)
            axp.set_title("Readout: diag(ρ_S) on system register",
                          fontsize=10)
            axp.grid(axis="y", alpha=0.3)

            # s_t progress bar across the bottom of the figure.
            frac = fi / max(len(s_values) - 1, 1)
            fig.suptitle(f"{title}\n"
                         f"frame {fi+1}/{len(s_values)}    "
                         f"s_t = {float(s):.3f}   "
                         f"[{'█' * int(round(frac * 20)):<20}]",
                         fontsize=11, family="monospace")

            fig.canvas.draw()
            # Grab the rendered RGB buffer as a PIL image.
            buf = _np.asarray(fig.canvas.buffer_rgba())
            images.append(Image.fromarray(buf[..., :3].copy()))
            plt.close(fig)

        # Assemble the GIF. duration is per-frame in ms; loop=0 = forever.
        duration_ms = int(round(1000.0 / max(fps, 1)))
        images[0].save(
            out_path, save_all=True, append_images=images[1:],
            duration=duration_ms, loop=0, optimize=True,
        )
        return out_path


# ════════════════════════════════════════════════════════════════════════════
# Standalone demo: visualise the optimized-ansatz output
# ════════════════════════════════════════════════════════════════════════════

def demo_optimized_bloch(out_path="v3_1_optimized_bloch.png",
                         K_max=4, sample_index=0, s_t=0.0, seed=0):
    """Fit v3.1 on a small synthetic dataset and render the per-qubit Bloch
    spheres of the OPTIMIZED ansatz for one sample. Returns the PNG path."""
    import numpy as np_
    rng = np_.random.default_rng(seed)
    N, S, K_true = 80, 1000, 3
    block = S // K_true
    alpha = np_.full((K_true, S), 0.1)
    for k in range(K_true):
        alpha[k, k * block:(k + 1) * block] = 3.0
    labels = rng.choice(K_true, N)
    X = np_.array([rng.multinomial(1500, rng.dirichlet(alpha[labels[i]]))
                   for i in range(N)], float)

    m = DMM_SVVS_VarQITE_QAVB_JAX(
        K_max=K_max, max_iter=8, beta0=5.0, s0=1.0, tau1=4, tau2=8,
        n_varqite_steps=8, ansatz_depth=2, verbose=0, random_state=seed,
        use_trigamma_correction=False, jit_warmup=False)
    m.fit(X)

    E_log_pi = m._E_log_pi()
    ll = m._expected_log_lik_trigamma(X)
    D = -(E_log_pi[None, :] + ll)
    d_row = D[sample_index][: m.K]

    path = m.visualize_optimized_ansatz(d_row, s_t=s_t, out_path=out_path)
    print(f"[demo] fitted K={m.K}, n_sys={m.n_sys}, n_tot={m.n_tot}")
    print(f"[demo] wrote {path}")
    return path


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # `--viz` renders the optimized-ansatz Bloch figure and exits, skipping
    # the (slow) accuracy/speed benchmark.
    if "--viz" in sys.argv:
        demo_optimized_bloch()
        sys.exit(0)

    from sklearn.metrics import (adjusted_rand_score,
                                  normalized_mutual_info_score)
    from DMM_SVVS_Variational_QAVB_v2 import DMM_SVVS_ClassicalQAVB

    print("=" * 72)
    print("VarQITE QAVB v3.1 (JAX) — accuracy vs ClassicalQAVB, speed vs v3_fast")
    print("=" * 72)

    rng = np.random.default_rng(42)
    N, S, K_true = 400, 5000, 4
    block = S // K_true
    alpha = np.full((K_true, S), 0.1)
    for k in range(K_true):
        alpha[k, k * block:(k + 1) * block] = 3.0
    true_labels = rng.choice(K_true, size=N)
    X = np.array([
        rng.multinomial(2000, rng.dirichlet(alpha[true_labels[i]]))
        for i in range(N)
    ], dtype=float)

    # Shared schedule / model knobs. The VarQITE-specific keys are added only
    # to the VarQITE constructors (ClassicalQAVB does not accept them).
    base = dict(
        K_max=10, nu='auto', max_iter=500,
        beta0=30.0, s0=1.0, tau1=100, tau2=200,
        verbose=1, random_state=42,
        selection_prior=0.3, prune_threshold=0.2,
        use_trigamma_correction=False,
    )
    varqite = dict(
        n_varqite_steps=8, ansatz_depth=2,
        mixer="transverse_field", regularization=1e-4,
        warm_start=True, init_perturbation=0.05,
    )

    # ── (A) Accuracy reference: exact ClassicalQAVB ───────────────────────
    print("\n--- ClassicalQAVB (exact expm E-step — ACCURACY REFERENCE) ---")
    t0 = time()
    mc = DMM_SVVS_ClassicalQAVB(**base)
    mc.fit(X); dt_c = time() - t0
    pred_c = mc.predict(X)
    ari_c = adjusted_rand_score(true_labels, pred_c)
    nmi_c = normalized_mutual_info_score(true_labels, pred_c)

    results = []  # (name, ari_vs_true, nmi_vs_true, ari_vs_classical, K, time)

    # ── (B) Speed competitors ─────────────────────────────────────────────
    print("\n--- v3_fast (lightning + earlystop + refresh=4) ---")
    t0 = time()
    m3 = DMM_SVVS_VarQITE_QAVB_Fast(
        **base, **varqite, device_name="lightning.qubit",
        r_early_stop_tol=1e-3, metric_refresh=4, dedup_n_clusters="auto")
    m3.fit(X); dt3 = time() - t0
    pred3 = m3.predict(X)
    results.append(("v3_fast (no dedup)",
                    adjusted_rand_score(true_labels, pred3),
                    normalized_mutual_info_score(true_labels, pred3),
                    adjusted_rand_score(pred_c, pred3), m3.K, dt3))

    print("\n--- v3.1 JAX+JIT+vmap (no dedup, refresh=4) ---")
    t0 = time()
    m31 = DMM_SVVS_VarQITE_QAVB_JAX(
        **base, **varqite, metric_refresh=4, dedup_n_clusters=None)
    m31.fit(X); dt31 = time() - t0
    pred31 = m31.predict(X)
    results.append(("v3.1 JAX (no dedup)",
                    adjusted_rand_score(true_labels, pred31),
                    normalized_mutual_info_score(true_labels, pred31),
                    adjusted_rand_score(pred_c, pred31), m31.K, dt31))
    print(f"  perf summary: {m31.perf_summary()}")
    
    sample_index = 0
    E_log_pi = m31._E_log_pi()
    ll = m31._expected_log_lik_trigamma(X)
    D = -(E_log_pi[None, :] + ll)
    d_row = D[sample_index][: m31.K]

    path = m31.visualize_optimized_ansatz(d_row, s_t=0.0, out_path="v3_1_optimized_bloch.png")
    print(f"[demo] fitted K={m31.K}, n_sys={m31.n_sys}, n_tot={m31.n_tot}")

    print("\n--- v3.1 JAX+JIT+vmap (+ dedup auto) ---")
    t0 = time()
    m31d = DMM_SVVS_VarQITE_QAVB_JAX(
        **base, **varqite, metric_refresh=4, dedup_n_clusters="auto")
    m31d.fit(X); dt31d = time() - t0
    pred31d = m31d.predict(X)
    results.append(("v3.1 JAX (+ dedup auto)",
                    adjusted_rand_score(true_labels, pred31d),
                    normalized_mutual_info_score(true_labels, pred31d),
                    adjusted_rand_score(pred_c, pred31d), m31d.K, dt31d))
    print(f"  perf summary: {m31d.perf_summary()}")
    
    sample_index = 0
    E_log_pi = m31d._E_log_pi()
    ll = m31d._expected_log_lik_trigamma(X)
    D = -(E_log_pi[None, :] + ll)
    d_row = D[sample_index][: m31d.K]

    path = m31d.visualize_optimized_ansatz(d_row, s_t=0.810, out_path="v3_1_1optimized_bloch.png")
    print(f"[demo] fitted K={m31d.K}, n_sys={m31d.n_sys}, n_tot={m31d.n_tot}")

    # ── Report ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"Benchmark (N={N}, S={S}, K_true={K_true})")
    print("=" * 72)
    print("ACCURACY reference — ClassicalQAVB (exact expm):")
    print(f"    ARI(vs true)={ari_c:.3f}  NMI(vs true)={nmi_c:.3f}  "
          f"K={mc.K}  time={dt_c:.1f}s")
    print("-" * 72)
    print(f"{'Method':<26} {'ARI':>6} {'NMI':>6} {'ARI|cls':>8} "
          f"{'K':>3} {'time(s)':>8} {'vs v3f':>8}")
    print("    (ARI|cls = ARI of this method's labels against ClassicalQAVB)")
    print("-" * 72)
    v3f_time = results[0][5]
    for name, ari, nmi, ari_cls, k, dt in results:
        sp = v3f_time / dt if dt > 0 else float("inf")
        print(f"{name:<26} {ari:6.3f} {nmi:6.3f} {ari_cls:8.3f} "
              f"{k:3d} {dt:8.1f} {sp:7.2f}x")
    print("=" * 72)
    print("ARI|cls ≈ 1.0 ⇒ v3.1 reproduces the exact-method clustering.")
    print("'vs v3f' is the v3.1 speedup over the previous fastest variant.")
