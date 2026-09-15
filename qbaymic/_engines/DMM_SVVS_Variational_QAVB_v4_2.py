#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMM-SVVS VQT QAVB 
=====================================================
"""
from __future__ import annotations

import os
import sys
from time import time

import numpy as np
from sklearn.cluster import MiniBatchKMeans

os.environ.setdefault("JAX_ENABLE_X64", "1")

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from DMM_SVVS_Variational_v2 import NumericalStability  # noqa: E402
from DMM_SVVS_Variational_QAVB_v2 import (  # noqa: E402
    _safe_density_matrix_from_M,
)
from DMM_SVVS_Variational_QAVB_v4_1 import (  # noqa: E402
    DMM_SVVS_VQT_QAVB_Fast,
    _hamiltonian_matrix,
)


class DMM_SVVS_VQT_QAVB_JAX(DMM_SVVS_VQT_QAVB_Fast):

    def __init__(
        self,
        *args,
        sample_batch_size: int | None = None,
        jit_warmup: bool = True,
        **kwargs,
    ):
        # v4_2 forces the JAX-compatible device; warn if the user asked for
        # something else (mirrors v3_1).
        requested = kwargs.get("device_name", "default.qubit")
        if str(requested) not in ("default.qubit", "default.qubit.jax"):
            if kwargs.get("verbose", 1) >= 1:
                print(f"  [v4_2] device '{requested}' is not a jax.jit target; "
                      f"using default.qubit (interface='jax').")
        kwargs["device_name"] = "default.qubit"
        # Force the JAX engine regardless of any compute_backend passed in.
        kwargs["compute_backend"] = "numpy"   # harmless; the JAX path overrides
        super().__init__(*args, **kwargs)
        self.compute_backend = "jax"

        self.sample_batch_size = (None if sample_batch_size is None
                                  else int(sample_batch_size))
        self.jit_warmup = bool(jit_warmup)

        if ((self.free_energy_tol and self.free_energy_tol > 0.0)
                or (self.r_early_stop_tol and self.r_early_stop_tol > 0.0)) \
                and self.verbose >= 1:
            print("  [v4_2] note: free_energy_tol / r_early_stop_tol are "
                  "ignored under JIT+vmap (data-dependent breaks defeat "
                  "single-compile). Inner loop runs all n_vqt_steps.")

        # JAX handles (imported lazily in _init).
        self._jax = None
        self._jnp = None
        self._batched_kernel = None       # jitted vmap(scan(adam_step))
        self._probs_batched  = None       # jitted vmap(readout)
        self._compiled_signature = None
        self._theta_warm = None           # (N, n_params) carried across iters
        self._phi_warm   = None           # (N, K_pad)    carried across iters
        self._compile_seconds = 0.0

    # ── JAX-backed device + static Pauli basis (v3_1 §3.1 pattern) ─────────

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
            raise ImportError("JAX required for v4_2:  pip install -U 'jax[cpu]'")

        jax.config.update("jax_enable_x64", True)

        self._qml = qml
        self._jax = jax
        self._jnp = jnp

        self.n_sys = max(1, int(np.ceil(np.log2(max(self.K, 2)))))
        self.K_pad = 1 << self.n_sys

        self._apply_adaptive_expressivity()

        self.n_tot = self.n_sys
        self.n_params = 2 * self.n_sys * self.ansatz_depth

        self._dev = qml.device("default.qubit", wires=self.n_tot)
        self._active_device = "default.qubit (jax)"

        from DMM_SVVS_Variational_QAVB_v4_1 import _build_cnot_layer
        self._cnot_layer = _build_cnot_layer(self.n_sys)

        # ── §3.1 static Pauli enumeration ─────────────────────────────────
        n_sys = self.n_sys

        def z_string_op(z):
            active = [q for q in range(n_sys) if (z >> (n_sys - 1 - q)) & 1]
            if not active:
                return qml.Identity(0)
            op = qml.PauliZ(active[0])
            for q in active[1:]:
                op = op @ qml.PauliZ(q)
            return op

        diag_ops = [z_string_op(z) for z in range(self.K_pad)]

        self._dense_mix_static_c = None
        if self.mixer == "transverse_field":
            mixer_ops = [qml.PauliX(q) for q in range(n_sys)]
        elif self.mixer == "cyclic_shift":
            H_mix = self._cyclic_shift_padded_matrix()
            terms = self._dense_to_pauli_terms(H_mix)
            mixer_ops = [op for _c, op in terms]
            self._dense_mix_static_c = np.asarray(
                [float(_c) for _c, _op in terms], dtype=np.float64)
        else:
            raise ValueError(f"Unknown mixer: {self.mixer!r}")

        self._all_pauli_ops = diag_ops + mixer_ops
        self._n_diag = len(diag_ops)
        self._n_mix  = len(mixer_ops)
        self._P      = len(self._all_pauli_ops)

        Kp = self.K_pad
        Hsign = np.empty((Kp, Kp), dtype=np.float64)
        for z in range(Kp):
            zb = [(z >> (n_sys - 1 - q)) & 1 for q in range(n_sys)]
            for k in range(Kp):
                kb = [(k >> (n_sys - 1 - q)) & 1 for q in range(n_sys)]
                par = sum(a & b for a, b in zip(zb, kb)) & 1
                Hsign[z, k] = -1.0 if par else 1.0
        self._walsh_sign = jnp.asarray(Hsign / Kp)

        # Build the static qnodes + jitted kernels.
        self._build_jax_kernels()

        # Invalidate carried warm-start (K may have changed via pruning).
        self._theta_warm = None
        self._phi_warm   = None
        self._theta_cache = {}   # kept for API compatibility; unused on JAX path
        self._phi_cache   = {}

    # ── Build the static qnodes and the jitted vmap+scan Adam kernel ───────

    def _build_jax_kernels(self):
        qml  = self._qml
        jax  = self._jax
        jnp  = self._jnp
        lax  = jax.lax
        ALL_OPS = self._all_pauli_ops
        n_sys   = self.n_sys
        K_pad   = self.K_pad
        P_obs   = self._P
        n_steps = max(int(self.n_vqt_steps), 1)
        lr_th   = float(self.learning_rate)
        lr_ph   = float(self.learning_rate_phi)
        strategy   = self.update_strategy
        n_warmup   = int(self.n_phi_warmup)
        b1, b2, eps_adam = 0.9, 0.999, 1e-8

        basis_indices = list(range(K_pad))

        @qml.qnode(self._dev, interface="jax")
        def _expvals_in_state(th, x):
            # Prepare |x⟩ (wire 0 = MSB), apply the ansatz, return static
            # Pauli expvals. 
            for q in range(n_sys):
                if (x >> (n_sys - 1 - q)) & 1:
                    qml.PauliX(wires=q)
            self._ansatz(th)
            return [qml.expval(o) for o in ALL_OPS]

        @qml.qnode(self._dev, interface="jax")
        def _probs_in_state(th, x):
            for q in range(n_sys):
                if (x >> (n_sys - 1 - q)) & 1:
                    qml.PauliX(wires=q)
            self._ansatz(th)
            return qml.probs(wires=list(range(n_sys)))

        def energies(th, c):
            # E_x(θ) = Σ_p c_p ⟨x|U†P_pU|x⟩ for all x = 0..K_pad-1.
            rows = [jnp.sum(c * jnp.stack(_expvals_in_state(th, x)))
                    for x in basis_indices]
            return jnp.stack(rows)                              # (K_pad,)

        def free_energy(th, ph, c, beta):
            E = energies(th, c)
            p = jax.nn.softmax(ph)
            T = 1.0 / beta
            entropy = -jnp.sum(p * jnp.log(p + 1e-15))
            return jnp.sum(p * E) - T * entropy

        grad_F = jax.grad(free_energy, argnums=(0, 1))

        # Static per-step masks for the three update strategies. 
        steps = np.arange(1, n_steps + 1)
        if strategy == "joint":
            mask_th = np.ones(n_steps, dtype=bool)
            mask_ph = np.ones(n_steps, dtype=bool)
        elif strategy == "alternating":
            mask_th = (steps % 2 == 1)
            mask_ph = (steps % 2 == 0)
        elif strategy == "phi_first":
            mask_th = (steps > n_warmup)
            mask_ph = np.ones(n_steps, dtype=bool)
        else:
            raise ValueError(f"Unknown update_strategy: {strategy!r}")
        mask_th = jnp.asarray(mask_th)
        mask_ph = jnp.asarray(mask_ph)

        def trajectory(theta0, phi0, c, beta):
            """One sample's full Adam inner loop as a lax.scan.
            Carry = (θ, φ, m_θ, v_θ, m_φ, v_φ)."""
            z_th = jnp.zeros_like(theta0)
            z_ph = jnp.zeros_like(phi0)

            def step(carry, inp):
                th, ph, m_th, v_th, m_ph, v_ph = carry
                t, do_th, do_ph = inp
                g_th, g_ph = grad_F(th, ph, c, beta)

                # θ Adam update (gated by do_th; updates are no-ops otherwise).
                m_th_n = b1 * m_th + (1 - b1) * g_th
                v_th_n = b2 * v_th + (1 - b2) * (g_th ** 2)
                mhat = m_th_n / (1 - b1 ** t)
                vhat = v_th_n / (1 - b2 ** t)
                th_n = th - lr_th * mhat / (jnp.sqrt(vhat) + eps_adam)
                th   = jnp.where(do_th, th_n, th)
                m_th = jnp.where(do_th, m_th_n, m_th)
                v_th = jnp.where(do_th, v_th_n, v_th)

                # φ Adam update (gated by do_ph).
                m_ph_n = b1 * m_ph + (1 - b1) * g_ph
                v_ph_n = b2 * v_ph + (1 - b2) * (g_ph ** 2)
                mhat = m_ph_n / (1 - b1 ** t)
                vhat = v_ph_n / (1 - b2 ** t)
                ph_n = ph - lr_ph * mhat / (jnp.sqrt(vhat) + eps_adam)
                ph   = jnp.where(do_ph, ph_n, ph)
                m_ph = jnp.where(do_ph, m_ph_n, m_ph)
                v_ph = jnp.where(do_ph, v_ph_n, v_ph)

                return (th, ph, m_th, v_th, m_ph, v_ph), None

            t_arr = jnp.arange(1, n_steps + 1).astype(theta0.dtype)
            inputs = (t_arr, mask_th, mask_ph)
            (thf, phf, *_), _ = lax.scan(
                step, (theta0, phi0, z_th, z_th, z_ph, z_ph), inputs)
            return thf, phf

        # vmap over (theta0[i], phi0[i], c[i]); beta broadcast as scalar.
        batched = jax.vmap(trajectory, in_axes=(0, 0, 0, None))
        self._batched_kernel = jax.jit(batched)

        # Readout: diag(ρ)_k = Σ_x p_φ(x)·|⟨k|U|x⟩|².
        def readout(th, ph):
            p = jax.nn.softmax(ph)
            diag = jnp.zeros(K_pad)
            for x in basis_indices:
                diag = diag + p[x] * _probs_in_state(th, x)
            return diag

        self._probs_batched = jax.jit(jax.vmap(readout, in_axes=(0, 0)))

        # Batched final free energy (for the instrumentation summary).
        self._F_batched = jax.jit(jax.vmap(free_energy, in_axes=(0, 0, 0, None)))

        self._compiled_signature = (
            self.n_sys, self.n_params, P_obs, n_steps,
            self.update_strategy, lr_th, lr_ph, n_warmup,
        )

    # ── §3.1 per-sample coefficient vector (vectorised, JAX) ──────────────

    def _coefficients_batch(self, D_block: np.ndarray, s_t: float):
        
        jnp = self._jnp
        K = self.K
        Kp = self.K_pad
        B = D_block.shape[0]

        D = np.asarray(D_block, dtype=np.float64)
        d_shift = D - D.min(axis=1, keepdims=True)
        d_range = d_shift.max(axis=1, keepdims=True)
        scaled = np.where(
            d_range > 1e-10,
            4.0 * d_shift / np.maximum(d_range, 1e-300),
            d_shift,
        )
        d_padded = np.full((B, Kp), self.PHANTOM_PENALTY, dtype=np.float64)
        d_padded[:, :K] = scaled
        d_padded = jnp.asarray(d_padded)

        # Diagonal Walsh–Hadamard block: c_diag[b,z] = (1-s)·Σ_k sign[z,k]·d/Kp.
        c_diag = (1.0 - s_t) * (d_padded @ self._walsh_sign.T)     # (B, K_pad)

        # Mixer block.
        if self._dense_mix_static_c is not None:
            # cyclic_shift: dense decomposition; per-sample coeff = -s_t·static_c.
            base = jnp.asarray(self._dense_mix_static_c)
            c_mix = -float(s_t) * jnp.broadcast_to(base, (B, self._n_mix))
        else:
            # transverse_field: +s_t per X-string (matches v4 _build_hamiltonian).
            c_mix = jnp.full((B, self._n_mix), float(s_t))

        return jnp.concatenate([c_diag, c_mix], axis=1)            # (B, P)

    # ── Warm-start / init parameter batch (carried JAX arrays, §3.2) ───────

    def _init_theta_phi_batch(self, n_rows: int):
        """Initial (θ, φ) for each row. Warm-started from the carried arrays
        when available and shaped consistently, else v4's zeros init
        (θ=0 identity circuit; φ=0 uniform mixture)."""
        theta0 = np.zeros((n_rows, self.n_params), dtype=np.float64)
        phi0   = np.zeros((n_rows, self.K_pad),    dtype=np.float64)

        if (self.warm_start and self._theta_warm is not None
                and self._theta_warm.shape == (n_rows, self.n_params)
                and self._phi_warm is not None
                and self._phi_warm.shape == (n_rows, self.K_pad)):
            theta0 = np.asarray(self._theta_warm, dtype=np.float64)
            phi0   = np.asarray(self._phi_warm,   dtype=np.float64)
        return theta0, phi0

    # ── Batched VQT over a set of energy rows (samples or centroids) ──────

    def _vqt_batch(self, D_block: np.ndarray, beta_t: float, s_t: float):
        
        jnp = self._jnp
        EPS = NumericalStability.EPS
        B = D_block.shape[0]

        C_mat = self._coefficients_batch(D_block, s_t)            # (B, P) JAX
        theta0, phi0 = self._init_theta_phi_batch(B)
        bs = self.sample_batch_size or B

        theta_final = np.empty((B, self.n_params), dtype=np.float64)
        phi_final   = np.empty((B, self.K_pad),    dtype=np.float64)
        probs       = np.empty((B, self.K_pad),    dtype=np.float64)

        for start in range(0, B, bs):
            stop = min(start + bs, B)
            th0 = jnp.asarray(theta0[start:stop])
            ph0 = jnp.asarray(phi0[start:stop])
            cc  = C_mat[start:stop]
            thf, phf = self._batched_kernel(th0, ph0, cc, float(beta_t))
            pr = self._probs_batched(thf, phf)
            theta_final[start:stop] = np.asarray(thf)
            phi_final[start:stop]   = np.asarray(phf)
            probs[start:stop]       = np.asarray(pr)

        F_final = np.asarray(self._F_batched(
            jnp.asarray(theta_final), jnp.asarray(phi_final), C_mat,
            float(beta_t)))

        r = np.clip(probs[:, : self.K], EPS, None)
        r = r / r.sum(axis=1, keepdims=True)
        return r, theta_final, phi_final, F_final

    # ── E-step over the dataset (single vmap; dedup optional) ─────────────

    def _compute_r_annealed(self, X: np.ndarray, beta_t: float,
                            s_t: float) -> np.ndarray:
        
        sig = (self.n_sys, self.n_params, self._P,
               max(int(self.n_vqt_steps), 1),
               self.update_strategy, float(self.learning_rate),
               float(self.learning_rate_phi), int(self.n_phi_warmup))
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

        inner_steps = max(int(self.n_vqt_steps), 1)

        if use_dedup:
            U = centroids.shape[0]
            r_centroid, _th, _ph, F_c = self._vqt_batch(centroids, beta_t, s_t)
            r = r_centroid[centroid_labels]
            # Warm-start carry doesn't survive MiniBatchKMeans relabelling.
            self._vqt_stats["dedup_fraction_per_iter"].append(U / N)
            self._vqt_stats["inner_steps_per_iter"].append(U * inner_steps)
            final_F_mean = float(np.mean(F_c)) if U else 0.0
        else:
            r, theta_final, phi_final, F_all = self._vqt_batch(D, beta_t, s_t)
            if self.warm_start:
                self._theta_warm = self._jnp.asarray(theta_final)
                self._phi_warm   = self._jnp.asarray(phi_final)
            self._vqt_stats["dedup_fraction_per_iter"].append(1.0)
            self._vqt_stats["inner_steps_per_iter"].append(N * inner_steps)
            final_F_mean = float(np.mean(F_all)) if N else 0.0

        # Early-stop disabled under JIT; record 0 for the summary.
        self._vqt_stats["early_stops_per_iter"].append(0)
        self._vqt_stats["final_free_energy_per_iter"].append(final_F_mean)
        return r

    def _maybe_warmup(self, K: int):
        if not self.jit_warmup or self._compile_seconds > 0:
            return
        try:
            t0 = time()
            dummy_D = np.zeros((2, K), dtype=np.float64)
            dummy_D[0, 0] = 1.0
            self._vqt_batch(dummy_D, beta_t=2.0, s_t=0.5)
            self._compile_seconds = time() - t0
            if self.verbose >= 1:
                print(f"  [v4_2] JIT compile warmup: "
                      f"{self._compile_seconds:.2f}s "
                      f"(n_sys={self.n_sys}, P={self._P}, "
                      f"steps={self.n_vqt_steps}, "
                      f"strategy={self.update_strategy})")
        except Exception as exc:
            if self.verbose >= 1:
                print(f"  [v4_2] warmup skipped ({type(exc).__name__})")

    # ── Reporting ─────────────────────────────────────────────────────────

    def _print_fit_header(self):
        trig = "ON" if self.use_trigamma_correction else "OFF"
        dn = self.dedup_n_clusters
        dn_str = (f"{dn}" if isinstance(dn, int)
                  else ("'auto'" if dn == "auto" else "OFF"))
        bs = self.sample_batch_size or "all-N"
        print(f"\nStarting VQT QAVB v4_2 (JAX+JIT+vmap) — DMM-SVVS "
              f"(free-energy minimisation)")
        print(f"  β0={self.beta0}, s0={self.s0}, "
              f"τ1={self.tau1}, τ2={self.tau2}, "
              f"prune_start={self.prune_start}, trigamma={trig}")
        print(f"  qubits: n_sys={self.n_sys}, n_anc=0, aux=0, total={self.n_tot}  "
              f"(VarQITE would use {2*self.n_sys + 1})")
        print(f"  ansatz_depth={self.ansatz_depth}, n_params={self.n_params}, "
              f"K_pad={self.K_pad}")
        print(f"  n_vqt_steps={self.n_vqt_steps}, lr_θ={self.learning_rate}, "
              f"lr_φ={self.learning_rate_phi}, strategy={self.update_strategy}")
        print(f"  [perf] device={self._active_device}, "
              f"static_pauli_P={self._P}, vmap_batch={bs}, "
              f"mixer={self.mixer}, dedup={dn_str}")

    def perf_summary(self) -> dict:
        base = super().perf_summary()
        if base:
            base["compile_seconds"] = float(self._compile_seconds)
            base["backend"] = "jax"
            base["static_pauli_P"] = int(self._P)
        return base


def verify_vqt_gibbs_jax(
    K: int = 4,
    beta: float = 5.0,
    s_t: float = 0.5,
    ansatz_depth: int = 3,
    n_vqt_steps: int = 200,
    learning_rate: float = 0.05,
    learning_rate_phi: float = 0.1,
    update_strategy: str = "joint",
    mixer: str = "transverse_field",
    random_state: int = 0,
):
    
    rng = np.random.default_rng(random_state)
    d_raw = rng.standard_normal(K) * 2.0

    # Lightweight object via __new__ (skip the SVVS heavy init).
    m = DMM_SVVS_VQT_QAVB_JAX.__new__(DMM_SVVS_VQT_QAVB_JAX)
    m.K = K
    m.ansatz_depth      = ansatz_depth
    m.n_vqt_steps       = n_vqt_steps
    m.learning_rate     = learning_rate
    m.learning_rate_phi = learning_rate_phi
    m.update_strategy   = update_strategy
    m.n_phi_warmup      = 5
    m.free_energy_tol   = 0.0
    m.r_early_stop_tol  = 0.0
    m.enumerate_basis   = True
    m.M_samples         = 100
    m.warm_start        = False
    m.mixer             = mixer
    m.device_name       = "default.qubit"
    m.compute_backend   = "jax"
    m.sample_batch_size = None
    m.jit_warmup        = False
    m.verbose           = 0
    m._vqt_rng          = np.random.default_rng(random_state)
    m._theta_cache      = {}
    m._phi_cache        = {}
    m._theta_warm       = None
    m._phi_warm         = None
    m._compile_seconds  = 0.0
    m._compiled_signature = None
    m.PHANTOM_PENALTY   = DMM_SVVS_VQT_QAVB_JAX.PHANTOM_PENALTY
    m._init_pennylane_device()

    # Per-sample shift + scale + pad.
    d_shift = d_raw - d_raw.min()
    d_range = d_shift.max()
    d_scaled_K = 4.0 * d_shift / d_range if d_range > 1e-10 else d_shift.copy()
    d_padded = np.full(m.K_pad, m.PHANTOM_PENALTY, dtype=np.float64)
    d_padded[: m.K] = d_scaled_K

    # Classical reference ρ_β (dense H, same as v4_1).
    H_mat = _hamiltonian_matrix(d_padded, s_t, m.n_sys, mixer=mixer, cyclic_K=K)
    rho_exact_pad = _safe_density_matrix_from_M(-beta * H_mat)
    rho_exact_K = rho_exact_pad[: K, : K].copy()
    rho_exact_K = rho_exact_K / np.trace(rho_exact_K).real

    eig_e = np.linalg.eigvalsh(rho_exact_pad).clip(1e-15)
    S_exact = float(-np.sum(eig_e * np.log(eig_e)))
    F_exact = float(np.real(np.trace(rho_exact_pad @ H_mat))
                    - (1.0 / beta) * S_exact)

    # Run the JAX VQT on this single row (B=1).
    r_b, theta_b, phi_b, F_b = m._vqt_batch(d_raw[None, :], beta, s_t)
    r_vqt = np.clip(r_b[0], 1e-15, None); r_vqt /= r_vqt.sum()
    r_exact = np.real(np.diag(rho_exact_K)).clip(1e-15); r_exact /= r_exact.sum()

    # Reconstruct ρ_VQT on the K_pad register for trace distance, using the
    # v4_1 dense unitary builder for the optimised θ.
    from DMM_SVVS_Variational_QAVB_v4_1 import _ansatz_unitary
    U = _ansatz_unitary(np.asarray(theta_b[0]), m.n_sys, m.ansatz_depth,
                        m._cnot_layer)
    p_final = np.asarray(m._jax.nn.softmax(m._jnp.asarray(phi_b[0])))
    rho_pad = (U * p_final[None, :]) @ U.conj().T
    rho_vqt_K = rho_pad[: K, : K]
    rho_vqt_K = rho_vqt_K / np.trace(rho_vqt_K).real

    trace_dist = 0.5 * float(np.linalg.svd(
        rho_vqt_K - rho_exact_K, compute_uv=False).sum())
    overlap = float(np.minimum(r_vqt, r_exact).sum())

    return {
        "trace_distance":   trace_dist,
        "overlap":          overlap,
        "r_vqt":            r_vqt,
        "r_exact":          r_exact,
        "rho_vqt":          rho_vqt_K,
        "rho_exact":        rho_exact_K,
        "F_vqt_final":      float(F_b[0]),
        "F_exact":          F_exact,
    }


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from sklearn.metrics import (adjusted_rand_score,
                                  normalized_mutual_info_score)

    print("=" * 72)
    print("DMM-SVVS QAVB v4_2 — JAX VQT: regression + benchmark vs v4 / v4_1")
    print("=" * 72)

    # ── Regression: JAX primitives vs v4_1 dense primitives ───────────────
    print("\n--- Regression: v4_2 JAX vs v4_1 dense (energies, grad, readout) ---")
    from DMM_SVVS_Variational_QAVB_v4_1 import (
        DMM_SVVS_VQT_QAVB_Fast as _V41, _ansatz_unitary,
    )

    for K, s_t in [(2, 0.3), (4, 0.5), (4, 1.0), (8, 0.4)]:
        rng = np.random.default_rng(7)
        d_raw = rng.standard_normal(K) * 2.0

        # v4_1 dense object via __new__.
        m41 = _V41.__new__(_V41)
        m41.K = K; m41.ansatz_depth = 3; m41.mixer = "transverse_field"
        m41.PHANTOM_PENALTY = _V41.PHANTOM_PENALTY
        m41.device_name = "default.qubit"; m41.compute_backend = "numpy"
        m41._theta_cache = {}; m41._phi_cache = {}
        m41._init_pennylane_device()

        # v4_2 JAX object via __new__.
        m42 = DMM_SVVS_VQT_QAVB_JAX.__new__(DMM_SVVS_VQT_QAVB_JAX)
        m42.K = K; m42.ansatz_depth = 3; m42.mixer = "transverse_field"
        m42.PHANTOM_PENALTY = DMM_SVVS_VQT_QAVB_JAX.PHANTOM_PENALTY
        m42.n_vqt_steps = 40; m42.learning_rate = 0.05
        m42.learning_rate_phi = 0.1; m42.update_strategy = "joint"
        m42.n_phi_warmup = 5; m42.warm_start = False; m42.verbose = 0
        m42.sample_batch_size = None; m42.jit_warmup = False
        m42._theta_warm = None; m42._phi_warm = None
        m42._compile_seconds = 0.0; m42._compiled_signature = None
        m42._theta_cache = {}; m42._phi_cache = {}
        m42._init_pennylane_device()

        d_shift = d_raw - d_raw.min(); d_range = d_shift.max()
        d_scaled_K = 4.0 * d_shift / d_range if d_range > 1e-10 else d_shift.copy()
        d_padded = np.full(m41.K_pad, m41.PHANTOM_PENALTY)
        d_padded[: K] = d_scaled_K

        theta = rng.standard_normal(m41.n_params)
        phi   = rng.standard_normal(m41.K_pad)
        beta_t = 3.0

        # v4_1 dense reference.
        H_np = _hamiltonian_matrix(d_padded, s_t, m41.n_sys,
                                   mixer="transverse_field", cyclic_K=K)
        E_ref = m41._energies_numpy(theta, H_np)
        g_ref = m41._grad_theta_numpy(theta, phi, H_np)
        r_ref = m41._readout_numpy(theta, phi)

        # v4_2 JAX: build c, evaluate energies / grad / readout directly.
        jnp = m42._jnp; jax = m42._jax
        C = m42._coefficients_batch(d_raw[None, :], s_t)[0]   # (P,)
        # Energies via the static-Pauli expvals (rebuild a small local fn).
        qml = m42._qml
        @qml.qnode(m42._dev, interface="jax")
        def _ev(th, x):
            for q in range(m42.n_sys):
                if (x >> (m42.n_sys - 1 - q)) & 1:
                    qml.PauliX(wires=q)
            m42._ansatz(th)
            return [qml.expval(o) for o in m42._all_pauli_ops]
        def _E(th):
            return jnp.stack([jnp.sum(C * jnp.stack(_ev(th, x)))
                              for x in range(m42.K_pad)])
        def _F(th, ph):
            p = jax.nn.softmax(ph); T = 1.0 / beta_t
            return jnp.sum(p * _E(th)) - T * (-jnp.sum(p * jnp.log(p + 1e-15)))
        E_jax = np.asarray(_E(jnp.asarray(theta)))
        g_jax, _ = jax.grad(_F, argnums=(0, 1))(jnp.asarray(theta),
                                                jnp.asarray(phi))
        g_jax = np.asarray(g_jax)
        # readout via dense unitary (identical formula to v4_1).
        U = _ansatz_unitary(theta, m42.n_sys, m42.ansatz_depth, m42._cnot_layer)
        p = np.asarray(jax.nn.softmax(jnp.asarray(phi)))
        r_jax = (np.abs(U) ** 2) @ p

        dE = np.max(np.abs(E_ref - E_jax))
        dg = np.max(np.abs(g_ref - g_jax))
        dr = np.max(np.abs(r_ref - r_jax))
        ok = "OK " if max(dE, dg, dr) < 1e-9 else "*** MISMATCH ***"
        print(f"  K={K} s={s_t:<3} | ΔE={dE:.2e}  Δgrad={dg:.2e}  "
              f"Δreadout={dr:.2e}  [{ok}]")

    # ── Level-0/1: JAX-engine Gibbs verification ──────────────────────────
    print("\n--- Level-0/1: v4_2 JAX-engine Gibbs verification ---")
    res = verify_vqt_gibbs_jax(K=4, beta=2.0, s_t=0.5,
                               ansatz_depth=3, n_vqt_steps=150,
                               random_state=0)
    print(f"  F_exact     = {res['F_exact']:.6f}")
    print(f"  F_vqt_final = {res['F_vqt_final']:.6f}  "
          f"(gap: {res['F_vqt_final'] - res['F_exact']:.2e})")
    print(f"  trace_dist  = {res['trace_distance']:.4e}")
    print(f"  overlap     = {res['overlap']:.4f}")

    # ── End-to-end fit + timing: v4 (QNode) vs v4_1 (dense) vs v4_2 (JAX) ──
    print("\n--- End-to-end fit + timing ---")
    rng = np.random.default_rng(42)
    N, S, K_true = 400, 5000, 3
    block = S // K_true
    alpha = np.full((K_true, S), 0.1)
    for k in range(K_true):
        alpha[k, k * block:(k + 1) * block] = 3.0
    true_labels = rng.choice(K_true, size=N)
    X = np.array([
        rng.multinomial(3000, rng.dirichlet(alpha[true_labels[i]]))
        for i in range(N)
    ], dtype=float)
    print(f"  Dataset: N={N}, S={S}, K_true={K_true}")

    common = dict(
        K_max=4, nu='auto', max_iter=300,
        beta0=30.0, s0=1.0, tau1=60, tau2=120, prune_start=20,
        verbose=0, random_state=42,
        selection_prior=0.3, prune_threshold=0.2,
        use_trigamma_correction=False,
        ansatz_depth=2, n_vqt_steps=40,
        learning_rate=0.05, learning_rate_phi=0.1,
        update_strategy="joint",
        free_energy_tol=5e-4, r_early_stop_tol=2e-3,
        enumerate_basis=True, warm_start=True,
        mixer="transverse_field",
    )

    results = []

    # v4_1 dense (the fast non-JAX reference).
    print("  [1/3] v4_1 dense NumPy (no dedup) ...")
    t0 = time()
    m_a = _V41(compute_backend="numpy", dedup_n_clusters="auto", device_name="lightning.qubit", **common)
    m_a.fit(X)
    dt_a = time() - t0
    pred = m_a.predict(X)
    results.append(("v4_1 dense (no dedup)",
                    adjusted_rand_score(true_labels, pred),
                    normalized_mutual_info_score(true_labels, pred),
                    m_a.K, dt_a))

    # v4_2 JAX, no dedup.
    print("  [2/3] v4_2 JAX+JIT+vmap (no dedup) ...")
    t0 = time()
    m_b = DMM_SVVS_VQT_QAVB_JAX(dedup_n_clusters=None, **common)
    m_b.fit(X)
    dt_b = time() - t0
    pred = m_b.predict(X)
    results.append(("v4_2 JAX (no dedup)",
                    adjusted_rand_score(true_labels, pred),
                    normalized_mutual_info_score(true_labels, pred),
                    m_b.K, dt_b))
    print(f"        perf: {m_b.perf_summary()}")

    # v4_2 JAX + dedup auto.
    print("  [3/3] v4_2 JAX+JIT+vmap + dedup(auto) ...")
    t0 = time()
    m_c = DMM_SVVS_VQT_QAVB_JAX(dedup_n_clusters="auto", **common)
    m_c.fit(X)
    dt_c = time() - t0
    pred = m_c.predict(X)
    results.append(("v4_2 JAX + dedup(auto)",
                    adjusted_rand_score(true_labels, pred),
                    normalized_mutual_info_score(true_labels, pred),
                    m_c.K, dt_c))
    print(f"        perf: {m_c.perf_summary()}")

    print("\n" + "=" * 72)
    print(f"Benchmark (N={N}, S={S}, K_true={K_true})")
    print(f"{'Method':<28} {'ARI':>7} {'NMI':>7} {'K':>4} {'time(s)':>9} {'vs v4_1':>9}")
    print("-" * 72)
    base = results[0][4]
    for name, ari, nmi, k, dt in results:
        sp = base / dt if dt > 0 else float("inf")
        print(f"{name:<28} {ari:7.3f} {nmi:7.3f} {k:4d} {dt:9.1f} {sp:8.2f}x")
    print("=" * 72)
    print("'vs v4_1' is the v4_2 speedup over the dense-NumPy engine.")
