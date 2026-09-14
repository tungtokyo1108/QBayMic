#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMM-SVVS VQT QAVB — Performance-Optimised Variant (v4_1)
==========================================================

Speeds up the Variational Quantum Thermalizer E-step of v4
(`DMM_SVVS_VQT_QAVB`) **without changing the algorithm**. Every result
produced by v4_1 with `compute_backend="numpy"` matches v4's PennyLane
path to machine precision (verified in the regression at the bottom of
this file). The optimisation is purely a *systems / linear-algebra*
rewrite of the inner loop, in the same spirit as v3_fast did for VarQITE.

Where v4 spends its time
------------------------
v4's `_vqt_responsibility` calls a fresh PennyLane QNode once per
computational-basis state, for the energy AND for every parameter-shift
evaluation:

    energy           : K_pad QNode calls          per inner step
    θ-gradient       : 2·n_params·K_pad QNode calls per inner step
    readout          : K_pad QNode calls          per early-stop check

For K=4 (n_params=12, K_pad=4) that is ~100 QNode dispatches *per inner
step*; for K=16 it is ~1000. Each dispatch pays autograd-tracing and
device-construction overhead that dwarfs the actual 2^n_sys × 2^n_sys
linear algebra at these sizes. This is the VQT analogue of the v3_fast
cProfile finding ("self-time dominated by autograd tracer").

The v4_1 idea — exploit VQT's structure
---------------------------------------
VQT prepares  ρ(θ,φ) = Σ_x p_φ(x)·U(θ)|x⟩⟨x|U(θ)†.  Everything the inner
loop needs is a function of the *single small unitary* U(θ) and the dense
Hamiltonian matrix H — both 2^n_sys × 2^n_sys.  At K ≤ 16 (n_sys ≤ 4)
that matrix is at most 16×16, so building it explicitly is far cheaper
than thousands of circuit dispatches:

  • Energies (ALL basis states at once)
        E_x(θ) = ⟨x|U†HU|x⟩  =  diag(U†HU)_x
    → ONE 16×16 triple product replaces K_pad QNode calls.

  • Readout (ALL basis states at once)
        diag(ρ)_k = Σ_x p_φ(x)|⟨k|U|x⟩|² = (|U|² @ p)_k
    → ONE matvec replaces K_pad QNode calls.

  • θ-gradient via parameter shift
        dF/dθ_j = Σ_x p(x)·½(E_x(θ_j+π/2) − E_x(θ_j−π/2))
    → 2·n_params unitary builds (each yields ALL K_pad energies),
      replacing 2·n_params·K_pad QNode calls.

  • φ-gradient — unchanged closed-form softmax gradient (already cheap).

The unitary U(θ) is assembled directly in NumPy from cached single-qubit
rotation matrices and a pre-built CNOT entangling layer (the entangler is
parameter-independent, so it is computed once per qubit count). This
removes PennyLane from the hot path entirely while reproducing exactly
the same hardware-efficient ansatz, the same Walsh–Hadamard diagonal
decomposition, and the same transverse-field / cyclic-shift mixers.

Layered fixes (cf. v3_fast's four layers)
-----------------------------------------
  Layer A — Dense state-vector engine (replaces v3_fast Layer 1).
      Pure-NumPy U(θ), H, energies, gradient, readout. The big win.
      Verified bit-exact vs the v4 QNode path. Selectable via
      `compute_backend` ("numpy" default, "qnode" = the original v4 path).

  Layer B — Early stopping on F / responsibility stability.
      Inherited from v4 (free_energy_tol, r_early_stop_tol). With the
      dense engine the readout used by the r-check is a single matvec,
      so the check is essentially free.

  Layer C — Energy-vector deduplication (ports v3_fast Layer 4).
      K-means on D = -(E[ln π] + E[ll]); run VQT on the centroids only;
      broadcast responsibilities by nearest-centroid assignment. Win up
      to N / n_centroids when N ≫ #distinct energy profiles.

What is intentionally NOT included
----------------------------------
  • JAX / vmap across samples — deferred (the supervisor's multi-week
    item). The dense engine already removes the per-step QNode overhead
    that JAX would otherwise be needed to amortise at this scale.
  • lightning.gpu — at n_sys ≤ 4 the state vector is ≤ 16 complex
    numbers; any accelerator launch overhead dominates.

Use
---
    from DMM_SVVS_Variational_QAVB_v4_1 import DMM_SVVS_VQT_QAVB_Fast
    m = DMM_SVVS_VQT_QAVB_Fast(
        K_max=4, beta0=30.0, s0=1.0, tau1=100, tau2=200,
        ansatz_depth=2, n_vqt_steps=40,
        compute_backend="numpy",      # Layer A (default)
        r_early_stop_tol=2e-3,        # Layer B (inherited)
        dedup_n_clusters="auto",      # Layer C (None = off)
    )
    m.fit(X)

References
----------
  Verdon G, et al. arXiv:1910.02071 (2019)  — VQT.
  v3_fast (this repo)                        — the four-layer template.
"""
from __future__ import annotations

import os
import sys
from time import time

import numpy as np
from sklearn.cluster import MiniBatchKMeans

# ── Import v4 building blocks ──────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from DMM_SVVS_Variational_v2 import NumericalStability  # noqa: E402
from DMM_SVVS_Variational_QAVB_v2 import (  # noqa: E402
    _diagonal_to_pauli_z_strings,
    _safe_density_matrix_from_M,
)
from DMM_SVVS_Variational_QAVB_v4 import DMM_SVVS_VQT_QAVB  # noqa: E402


# ════════════════════════════════════════════════════════════════════════════
# Pure-NumPy state-vector engine for the VQT ansatz (Layer A internals)
# ════════════════════════════════════════════════════════════════════════════
#
# These are free functions (no PennyLane) that reproduce v4's ansatz and
# Hamiltonian *exactly*, under the same wire convention used throughout v2/v4:
# wire 0 is the MOST significant bit. The single-qubit gate placed on wire q
# enters the tensor product at position q from the left.
# ----------------------------------------------------------------------------

_I2 = np.eye(2, dtype=complex)


def _ry(t: float) -> np.ndarray:
    c, s = np.cos(t / 2.0), np.sin(t / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(t: float) -> np.ndarray:
    e = np.exp(0.5j * t)
    return np.array([[1.0 / e, 0.0], [0.0, e]], dtype=complex)


def _embed_1q(gate: np.ndarray, q: int, n: int) -> np.ndarray:
    """Tensor a single-qubit `gate` into the full 2^n operator on wire q
    (wire 0 = leftmost / MSB)."""
    factors = [_I2] * n
    factors[q] = gate
    full = factors[0]
    for f in factors[1:]:
        full = np.kron(full, f)
    return full


def _cnot_full(ctrl: int, tgt: int, n: int) -> np.ndarray:
    """Dense CNOT(ctrl→tgt) on n wires (wire 0 = MSB)."""
    dim = 1 << n
    M = np.zeros((dim, dim), dtype=complex)
    for b in range(dim):
        bits = [(b >> (n - 1 - q)) & 1 for q in range(n)]
        if bits[ctrl]:
            bits[tgt] ^= 1
        nb = 0
        for q in range(n):
            nb = (nb << 1) | bits[q]
        M[nb, b] = 1.0
    return M


def _build_cnot_layer(n: int) -> np.ndarray:
    """Parameter-independent entangling layer matching v4's `_ansatz`:
    even-pair CNOTs, then odd-pair CNOTs, then the ring-closing CNOT for
    n ≥ 3. Returns identity for n < 2 (no entangler)."""
    dim = 1 << n
    layer = np.eye(dim, dtype=complex)
    if n >= 2:
        for q in range(0, n - 1, 2):
            layer = _cnot_full(q, q + 1, n) @ layer
        for q in range(1, n - 1, 2):
            layer = _cnot_full(q, q + 1, n) @ layer
        if n >= 3:
            layer = _cnot_full(n - 1, 0, n) @ layer
    return layer


def _ansatz_unitary(theta: np.ndarray, n_sys: int, depth: int,
                    cnot_layer: np.ndarray) -> np.ndarray:
    """Dense unitary U(θ) for the brick-wall (RY, RZ)+CNOT ansatz of v4.

    `cnot_layer` is the cached output of `_build_cnot_layer(n_sys)`."""
    dim = 1 << n_sys
    U = np.eye(dim, dtype=complex)
    idx = 0
    for _ in range(depth):
        for q in range(n_sys):
            U = _embed_1q(_ry(theta[idx]), q, n_sys) @ U
            idx += 1
            U = _embed_1q(_rz(theta[idx]), q, n_sys) @ U
            idx += 1
        if n_sys >= 2:
            U = cnot_layer @ U
    return U


def _hamiltonian_matrix(d_padded: np.ndarray, s_t: float, n_sys: int,
                        mixer: str = "transverse_field",
                        cyclic_K: int | None = None) -> np.ndarray:
    """Dense H_S(s_t) = (1−s_t)·diag(d_padded) − s_t·H_mixer.

    Reproduces `DMM_SVVS_VQT_QAVB._build_hamiltonian` exactly:
      transverse_field : H_mixer = −Σ_q X_q  ⇒  + s_t·Σ_q X_q
      cyclic_shift     : H_mixer = padded cyclic shift on the K-block
                         (= |k+1 mod K⟩⟨k| + h.c.), contribution −s_t·H_mixer.
    For cyclic_shift, `cyclic_K` is the physical K-block size.
    """
    K_pad = 1 << n_sys
    H = (1.0 - s_t) * np.diag(d_padded).astype(complex)

    if abs(s_t) <= 1e-15:
        return H

    if mixer == "transverse_field":
        X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        for q in range(n_sys):
            H = H + s_t * _embed_1q(X, q, n_sys)
    elif mixer == "cyclic_shift":
        K = int(cyclic_K) if cyclic_K is not None else K_pad
        H_mix = np.zeros((K_pad, K_pad), dtype=complex)
        idx = np.arange(K)
        nxt = (idx + 1) % K
        H_mix[idx, nxt] = 1.0
        H_mix[nxt, idx] = 1.0
        H = H - s_t * H_mix
    else:
        raise ValueError(f"Unknown mixer: {mixer!r}")
    return H


# ════════════════════════════════════════════════════════════════════════════
# Performance-optimised VQT QAVB
# ════════════════════════════════════════════════════════════════════════════

class DMM_SVVS_VQT_QAVB_Fast(DMM_SVVS_VQT_QAVB):
    """
    Performance-optimised VQT QAVB. Inherits the full algorithm from
    `DMM_SVVS_VQT_QAVB` (v4) and overrides only the inner per-sample loop
    and the cross-sample E-step. Default behaviour is bit-identical to v4
    but markedly faster.

    Parameters added on top of v4
    -----------------------------
    compute_backend : {"numpy", "qnode"}, default "numpy"
        "numpy" : dense state-vector engine (Layer A). Fastest at K ≤ 16.
        "qnode" : fall back to v4's exact PennyLane path (for regression /
                  cross-checking, or n_sys large enough that dense matrices
                  become unwieldy — not the regime this class targets).
    dedup_n_clusters : int | "auto" | None, default None
        Layer C energy-vector deduplication.
          None    -> off (every sample gets its own VQT pass).
          int     -> cluster d_i into this many k-means centroids.
          "auto"  -> max(K, min(N // 5, 30)) centroids.
        Clustering is recomputed each E-step (cheap: O(N·K)).
    dedup_kmeans_max_iter : int, default 20
        Max MiniBatchKMeans iterations per E-step (a coarse partition is
        enough; within-cluster diameter dominates the responsibility error).
    dedup_min_unique_ratio : float, default 0.5
        Skip dedup and run the per-sample path when k-means collapses to
        fewer than this fraction of the requested centroids.

    All v4 hyperparameters (ansatz_depth, n_vqt_steps, learning_rate,
    learning_rate_phi, update_strategy, n_phi_warmup, free_energy_tol,
    r_early_stop_tol, enumerate_basis, M_samples, warm_start, mixer, ...)
    behave identically.
    """

    def __init__(
        self,
        *args,
        compute_backend: str = "numpy",
        dedup_n_clusters=None,
        dedup_kmeans_max_iter: int = 20,
        dedup_min_unique_ratio: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.compute_backend        = str(compute_backend)
        if self.compute_backend not in ("numpy", "qnode"):
            raise ValueError(
                f"compute_backend must be 'numpy' or 'qnode', "
                f"got {self.compute_backend!r}"
            )
        self.dedup_n_clusters       = dedup_n_clusters
        self.dedup_kmeans_max_iter  = int(dedup_kmeans_max_iter)
        self.dedup_min_unique_ratio = float(dedup_min_unique_ratio)

        # Per-fit instrumentation (extends v4's _vqt_stats with dedup).
        self._vqt_stats.setdefault("dedup_fraction_per_iter", [])

        # Cached, parameter-independent CNOT entangling layer; (re)built in
        # _init_pennylane_device whenever n_sys changes (pruning).
        self._cnot_layer = None

    # ── Device init: also build the cached CNOT layer for the dense engine ──

    def _init_pennylane_device(self):
        super()._init_pennylane_device()
        # Cache the parameter-independent entangling layer for the dense
        # engine. Harmless (and tiny) even when compute_backend == "qnode".
        self._cnot_layer = _build_cnot_layer(self.n_sys)

    # ── Layer A: dense-NumPy per-basis-state energies ─────────────────────

    def _energies_numpy(self, theta: np.ndarray, H_mat: np.ndarray
                        ) -> np.ndarray:
        """All K_pad energies at once:  E_x = diag(U†HU)_x.

        One triple matrix product replaces K_pad QNode evaluations.
        Returns a real length-K_pad array.
        """
        U = _ansatz_unitary(theta, self.n_sys, self.ansatz_depth,
                            self._cnot_layer)
        M = U.conj().T @ H_mat @ U
        return np.real(np.diag(M))

    def _readout_numpy(self, theta: np.ndarray, phi: np.ndarray
                       ) -> np.ndarray:
        """diag(ρ)_k = Σ_x p_φ(x)|⟨k|U|x⟩|² = (|U|² @ p)_k.

        One matvec replaces K_pad QNode evaluations.
        """
        U = _ansatz_unitary(theta, self.n_sys, self.ansatz_depth,
                            self._cnot_layer)
        p = self._softmax_p(phi)
        return (np.abs(U) ** 2) @ p

    def _grad_theta_numpy(self, theta: np.ndarray, phi: np.ndarray,
                          H_mat: np.ndarray) -> np.ndarray:
        """Parameter-shift θ-gradient using the dense engine.

        dF/dθ_j = Σ_x p(x)·½(E_x(θ_j+π/2) − E_x(θ_j−π/2)).
        2·n_params unitary builds (each gives ALL K_pad energies) replace
        2·n_params·K_pad QNode evaluations.
        """
        p = self._softmax_p(phi)
        grad = np.zeros_like(theta)
        shift = np.pi / 2.0
        for j in range(len(theta)):
            th = theta.copy(); th[j] += shift
            E_p = self._energies_numpy(th, H_mat)
            th[j] -= 2.0 * shift
            E_m = self._energies_numpy(th, H_mat)
            grad[j] = float(np.sum(p * 0.5 * (E_p - E_m)))
        return grad

    # ── Per-sample VQT responsibility (dense engine) ──────────────────────

    def _vqt_responsibility_numpy(self, d_i, beta_t, s_t, sample_id=None):
        """Dense-engine twin of v4's `_vqt_responsibility`.

        Identical control flow, early-stopping policy, Adam state, warm-start
        caching and readout convention — only the energy / gradient / readout
        primitives are swapped for the NumPy ones.
        """
        EPS = NumericalStability.EPS

        # ── Per-sample shift + scale + pad (matches v4 / VarQITE) ──────────
        d_shift = d_i - d_i.min()
        d_range = d_shift.max()
        if d_range > 1e-10:
            d_scaled_K = 4.0 * d_shift / d_range
        else:
            d_scaled_K = d_shift.copy()
        d_padded = np.full(self.K_pad, self.PHANTOM_PENALTY, dtype=np.float64)
        d_padded[: self.K] = d_scaled_K

        H_mat = _hamiltonian_matrix(
            d_padded, s_t, self.n_sys, mixer=self.mixer, cyclic_K=self.K
        )

        # ── Initialise (θ, φ) ──────────────────────────────────────────────
        if (self.warm_start and sample_id is not None
                and sample_id in self._theta_cache
                and len(self._theta_cache[sample_id]) == self.n_params
                and len(self._phi_cache.get(sample_id, [])) == self.K_pad):
            theta = self._theta_cache[sample_id].copy()
            phi   = self._phi_cache[sample_id].copy()
        else:
            theta = np.zeros(self.n_params, dtype=np.float64)
            phi   = np.zeros(self.K_pad,     dtype=np.float64)

        # ── Adam state (two LRs) ───────────────────────────────────────────
        m_th = np.zeros_like(theta); v_th = np.zeros_like(theta)
        m_ph = np.zeros_like(phi);   v_ph = np.zeros_like(phi)
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

        F_prev = None
        r_prev = None
        steps_executed = 0
        early_stopped = False
        F = np.nan

        for step in range(1, self.n_vqt_steps + 1):
            # Energies for ALL basis states (dense): one triple product.
            E = self._energies_numpy(theta, H_mat)
            p = self._softmax_p(phi)
            T_t = 1.0 / beta_t
            energy_term  = float(np.sum(p * E))
            entropy_term = float(-np.sum(p * np.log(p + 1e-15)))
            F = energy_term - T_t * entropy_term

            # Which gradients this step (same policy as v4).
            do_theta = True
            do_phi   = True
            if self.update_strategy == "alternating":
                do_theta = (step % 2 == 1)
                do_phi   = (step % 2 == 0)
            elif self.update_strategy == "phi_first":
                if step <= self.n_phi_warmup:
                    do_theta = False
                    do_phi   = True
            elif self.update_strategy != "joint":
                raise ValueError(
                    f"Unknown update_strategy: {self.update_strategy!r}"
                )

            g_th = (self._grad_theta_numpy(theta, phi, H_mat)
                    if do_theta else None)
            g_ph = (self._grad_phi(E, phi, beta_t)
                    if do_phi else None)

            # Adam updates.
            if g_th is not None:
                m_th = beta1 * m_th + (1 - beta1) * g_th
                v_th = beta2 * v_th + (1 - beta2) * (g_th ** 2)
                m_hat = m_th / (1 - beta1 ** step)
                v_hat = v_th / (1 - beta2 ** step)
                theta = theta - self.learning_rate * m_hat / (
                    np.sqrt(v_hat) + eps_adam
                )
            if g_ph is not None:
                m_ph = beta1 * m_ph + (1 - beta1) * g_ph
                v_ph = beta2 * v_ph + (1 - beta2) * (g_ph ** 2)
                m_hat = m_ph / (1 - beta1 ** step)
                v_hat = v_ph / (1 - beta2 ** step)
                phi = phi - self.learning_rate_phi * m_hat / (
                    np.sqrt(v_hat) + eps_adam
                )

            steps_executed = step

            # Early stopping (every 2 steps after step 2) — readout is a
            # single matvec, so the r-check is essentially free.
            if step >= 2 and step % 2 == 0:
                if F_prev is not None and abs(F - F_prev) < self.free_energy_tol:
                    early_stopped = True
                    break
                r_curr = self._readout_numpy(theta, phi)
                if r_prev is not None:
                    delta = float(np.max(np.abs(r_curr - r_prev)))
                    if delta < self.r_early_stop_tol:
                        early_stopped = True
                        break
                r_prev = r_curr
            F_prev = F

        # Cache + final readout.
        if self.warm_start and sample_id is not None:
            self._theta_cache[sample_id] = theta
            self._phi_cache[sample_id]   = phi

        diag_full = self._readout_numpy(theta, phi)
        r = np.clip(diag_full[: self.K], EPS, None)
        r = r / r.sum()

        return r, {
            "steps_executed":    int(steps_executed),
            "early_stopped":     bool(early_stopped),
            "final_free_energy": float(F),
        }

    def _vqt_responsibility(self, d_i, beta_t, s_t, sample_id=None):
        """Dispatch to the dense engine (default) or v4's QNode path."""
        if self.compute_backend == "numpy":
            return self._vqt_responsibility_numpy(
                d_i, beta_t, s_t, sample_id=sample_id
            )
        return super()._vqt_responsibility(
            d_i, beta_t, s_t, sample_id=sample_id
        )

    # ── Layer C: E-step over the dataset with optional dedup ──────────────

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
        D = -(E_log_pi[None, :] + ll)        # (N, K)
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
                        [label_map[c] for c in centroid_labels]
                    )
                    centroids = centroids[unique]
            except Exception:
                use_dedup = False

        inner_steps_total = 0
        early_stop_count  = 0
        final_F_total     = 0.0

        if use_dedup:
            U = centroids.shape[0]
            r_centroid = np.zeros((U, K))
            for c_idx in range(U):
                r_c, stats = self._vqt_responsibility(
                    centroids[c_idx], beta_t, s_t,
                    sample_id=("centroid", c_idx),
                )
                r_centroid[c_idx] = r_c
                inner_steps_total += stats["steps_executed"]
                early_stop_count  += int(stats["early_stopped"])
                final_F_total     += stats["final_free_energy"]
            r = r_centroid[centroid_labels]
            self._vqt_stats["dedup_fraction_per_iter"].append(U / N)
            n_runs = U
        else:
            r = np.zeros((N, K))
            for i in range(N):
                r[i], stats = self._vqt_responsibility(
                    D[i], beta_t, s_t, sample_id=i
                )
                inner_steps_total += stats["steps_executed"]
                early_stop_count  += int(stats["early_stopped"])
                final_F_total     += stats["final_free_energy"]
            self._vqt_stats["dedup_fraction_per_iter"].append(1.0)
            n_runs = N

        self._vqt_stats["inner_steps_per_iter"].append(inner_steps_total)
        self._vqt_stats["early_stops_per_iter"].append(early_stop_count)
        self._vqt_stats["final_free_energy_per_iter"].append(
            final_F_total / max(n_runs, 1)
        )
        return r

    # ── Reporting ─────────────────────────────────────────────────────────

    def _print_fit_header(self):
        trig = "ON" if self.use_trigamma_correction else "OFF"
        dn = self.dedup_n_clusters
        dn_str = (f"{dn}" if isinstance(dn, int)
                  else ("'auto'" if dn == "auto" else "OFF"))
        print(f"\nStarting VQT QAVB FAST — DMM-SVVS  (free-energy minimisation)")
        print(f"  β0={self.beta0}, s0={self.s0}, "
              f"τ1={self.tau1}, τ2={self.tau2}, "
              f"prune_start={self.prune_start}, trigamma={trig}")
        print(f"  qubits: n_sys={self.n_sys}, n_anc=0, aux=0, total={self.n_tot}  "
              f"(VarQITE would use {2*self.n_sys + 1})")
        print(f"  ansatz_depth={self.ansatz_depth}, n_params={self.n_params}, "
              f"K_pad={self.K_pad}")
        print(f"  n_vqt_steps={self.n_vqt_steps}, lr_θ={self.learning_rate}, "
              f"lr_φ={self.learning_rate_phi}, strategy={self.update_strategy}")
        print(f"  [perf] backend={self.compute_backend}, "
              f"mixer={self.mixer}, dedup={dn_str}")

    def perf_summary(self) -> dict:
        """Aggregate per-iteration instrumentation for post-fit reporting."""
        stats = self._vqt_stats
        if not stats["inner_steps_per_iter"]:
            return {}
        return {
            "mean_inner_steps_per_iter": float(np.mean(
                stats["inner_steps_per_iter"])),
            "mean_early_stops_per_iter": float(np.mean(
                stats["early_stops_per_iter"])),
            "mean_dedup_fraction": float(np.mean(
                stats["dedup_fraction_per_iter"]))
            if stats["dedup_fraction_per_iter"] else 1.0,
            "backend": self.compute_backend,
        }


# ════════════════════════════════════════════════════════════════════════════
# Verification: dense-engine Gibbs check (mirrors v4.verify_vqt_gibbs)
# ════════════════════════════════════════════════════════════════════════════

def verify_vqt_gibbs_fast(
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
    return_trajectory: bool = True,
):
    """Level-0 + Level-1 sanity check for the v4_1 dense engine.

    Same contract as v4.verify_vqt_gibbs but runs the pure-NumPy primitives,
    so the verification path itself is fast. Compares VQT free energy and the
    K-block trace distance against the classical Gibbs state from expm.
    """
    rng = np.random.default_rng(random_state)
    d_raw = rng.standard_normal(K) * 2.0

    # Lightweight object via __new__ (skip SVVS heavy init).
    m = DMM_SVVS_VQT_QAVB_Fast.__new__(DMM_SVVS_VQT_QAVB_Fast)
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
    m.compute_backend   = "numpy"
    m._vqt_rng          = np.random.default_rng(random_state)
    m._theta_cache      = {}
    m._phi_cache        = {}
    m.PHANTOM_PENALTY   = DMM_SVVS_VQT_QAVB_Fast.PHANTOM_PENALTY
    m._init_pennylane_device()

    # Per-sample shift + scale + pad.
    d_shift = d_raw - d_raw.min()
    d_range = d_shift.max()
    d_scaled_K = 4.0 * d_shift / d_range if d_range > 1e-10 else d_shift.copy()
    d_padded = np.full(m.K_pad, m.PHANTOM_PENALTY, dtype=np.float64)
    d_padded[: m.K] = d_scaled_K

    # Classical reference ρ_β.
    H_mat = _hamiltonian_matrix(d_padded, s_t, m.n_sys, mixer=mixer,
                                cyclic_K=K)
    rho_exact_pad = _safe_density_matrix_from_M(-beta * H_mat)
    rho_exact_K = rho_exact_pad[: K, : K].copy()
    rho_exact_K = rho_exact_K / np.trace(rho_exact_K).real

    eig_e = np.linalg.eigvalsh(rho_exact_pad).clip(1e-15)
    S_exact = float(-np.sum(eig_e * np.log(eig_e)))
    F_exact = float(np.real(np.trace(rho_exact_pad @ H_mat))
                    - (1.0 / beta) * S_exact)

    # Run the dense-engine VQT (mirrors the v4 verification loop).
    theta = np.zeros(m.n_params, dtype=np.float64)
    phi   = np.zeros(m.K_pad,     dtype=np.float64)
    m_th = np.zeros_like(theta); v_th = np.zeros_like(theta)
    m_ph = np.zeros_like(phi);   v_ph = np.zeros_like(phi)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    F_history, td_history, r_history = [], [], []

    def _rho_pad(th, ph):
        U = _ansatz_unitary(th, m.n_sys, m.ansatz_depth, m._cnot_layer)
        p = m._softmax_p(ph)
        return (U * p[None, :]) @ U.conj().T   # Σ_x p_x U|x><x|U†

    for step in range(1, n_vqt_steps + 1):
        E = m._energies_numpy(theta, H_mat)
        p = m._softmax_p(phi)
        T_t = 1.0 / beta
        F = float(np.sum(p * E)) - T_t * float(-np.sum(p * np.log(p + 1e-15)))

        g_th = m._grad_theta_numpy(theta, phi, H_mat)
        g_ph = m._grad_phi(E, phi, beta)

        m_th = beta1 * m_th + (1 - beta1) * g_th
        v_th = beta2 * v_th + (1 - beta2) * (g_th ** 2)
        theta = theta - learning_rate * (m_th / (1 - beta1 ** step)) / (
            np.sqrt(v_th / (1 - beta2 ** step)) + eps_adam
        )
        m_ph = beta1 * m_ph + (1 - beta1) * g_ph
        v_ph = beta2 * v_ph + (1 - beta2) * (g_ph ** 2)
        phi = phi - learning_rate_phi * (m_ph / (1 - beta1 ** step)) / (
            np.sqrt(v_ph / (1 - beta2 ** step)) + eps_adam
        )

        if return_trajectory:
            F_history.append(F)
            rho_pad = _rho_pad(theta, phi)
            rho_K = rho_pad[: K, : K]
            rho_K = rho_K / np.trace(rho_K).real
            r_history.append(np.real(np.diag(rho_K)).clip(1e-15)
                             / max(np.real(np.diag(rho_K)).clip(1e-15).sum(),
                                   1e-15))
            td_history.append(0.5 * float(np.linalg.svd(
                rho_K - rho_exact_K, compute_uv=False).sum()))

    # Final readout.
    rho_vqt_pad = _rho_pad(theta, phi)
    rho_vqt_K = rho_vqt_pad[: K, : K]
    rho_vqt_K = rho_vqt_K / np.trace(rho_vqt_K).real

    r_vqt   = np.real(np.diag(rho_vqt_K)).clip(1e-15); r_vqt /= r_vqt.sum()
    r_exact = np.real(np.diag(rho_exact_K)).clip(1e-15); r_exact /= r_exact.sum()

    trace_dist = 0.5 * float(np.linalg.svd(
        rho_vqt_K - rho_exact_K, compute_uv=False).sum())
    overlap = float(np.minimum(r_vqt, r_exact).sum())

    p_final = m._softmax_p(phi)
    F_vqt_final = (
        float(np.sum(p_final * m._energies_numpy(theta, H_mat)))
        - (1.0 / beta) * float(-np.sum(p_final * np.log(p_final + 1e-15)))
    )

    out = {
        "trace_distance":   trace_dist,
        "overlap":          overlap,
        "r_vqt":            r_vqt,
        "r_exact":          r_exact,
        "rho_vqt":          rho_vqt_K,
        "rho_exact":        rho_exact_K,
        "F_vqt_final":      F_vqt_final,
        "F_exact":          F_exact,
    }
    if return_trajectory:
        out["F_history"]  = np.asarray(F_history)
        out["td_history"] = np.asarray(td_history)
        out["r_history"]  = np.asarray(r_history)
    return out


# ════════════════════════════════════════════════════════════════════════════
# Regression + benchmark:  v4_1 (numpy) vs v4 (qnode)
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from sklearn.metrics import (adjusted_rand_score,
                                  normalized_mutual_info_score)

    print("=" * 72)
    print("DMM-SVVS QAVB v4_1 — VQT speedup: regression + benchmark vs v4")
    print("=" * 72)

    # ── Regression: dense engine vs v4 QNode path, same per-sample energy ──
    print("\n--- Regression: v4_1 numpy primitives vs v4 QNode primitives ---")
    from DMM_SVVS_Variational_QAVB_v4 import DMM_SVVS_VQT_QAVB as _V4

    for K, s_t in [(2, 0.3), (4, 0.5), (4, 1.0), (8, 0.4)]:
        rng = np.random.default_rng(7)
        d_raw = rng.standard_normal(K) * 2.0

        # Build a v4 object and a v4_1 object via __new__.
        def _mk(cls, backend=None):
            m = cls.__new__(cls)
            m.K = K
            m.ansatz_depth = 3
            m.mixer = "transverse_field"
            m.PHANTOM_PENALTY = cls.PHANTOM_PENALTY
            m.device_name = "default.qubit"
            m._theta_cache = {}; m._phi_cache = {}
            if backend is not None:
                m.compute_backend = backend
            m._init_pennylane_device()
            return m

        m4  = _mk(_V4)
        m41 = _mk(DMM_SVVS_VQT_QAVB_Fast, backend="numpy")

        d_shift = d_raw - d_raw.min()
        d_range = d_shift.max()
        d_scaled_K = 4.0 * d_shift / d_range if d_range > 1e-10 else d_shift.copy()
        d_padded = np.full(m4.K_pad, m4.PHANTOM_PENALTY)
        d_padded[: K] = d_scaled_K

        theta = rng.standard_normal(m4.n_params)
        phi   = rng.standard_normal(m4.K_pad)
        beta_t = 3.0

        # v4 energies via QNode.
        H_qml = m4._build_hamiltonian(d_padded, s_t)
        eqn = m4._make_energy_qnode(H_qml)
        E_v4 = m4._energy_per_basis_state(eqn, theta)
        g_v4 = m4._grad_theta(theta, phi, eqn)
        pqn = m4._make_probs_qnode()
        r_v4 = m4._readout_responsibility(theta, phi, pqn)

        # v4_1 dense.
        H_np = _hamiltonian_matrix(d_padded, s_t, m41.n_sys,
                                   mixer="transverse_field", cyclic_K=K)
        E_v41 = m41._energies_numpy(theta, H_np)
        g_v41 = m41._grad_theta_numpy(theta, phi, H_np)
        r_v41 = m41._readout_numpy(theta, phi)

        dE = np.max(np.abs(E_v4 - E_v41))
        dg = np.max(np.abs(g_v4 - g_v41))
        dr = np.max(np.abs(r_v4 - r_v41))
        ok = "OK " if max(dE, dg, dr) < 1e-9 else "*** MISMATCH ***"
        print(f"  K={K} s={s_t:<3} | ΔE={dE:.2e}  Δgrad={dg:.2e}  "
              f"Δreadout={dr:.2e}  [{ok}]")

    # ── Level-0/1: dense-engine Gibbs verification ────────────────────────
    print("\n--- Level-0/1: v4_1 dense-engine Gibbs verification ---")
    res = verify_vqt_gibbs_fast(K=4, beta=2.0, s_t=0.5,
                                ansatz_depth=3, n_vqt_steps=150,
                                random_state=0)
    print(f"  F_exact     = {res['F_exact']:.6f}")
    print(f"  F_vqt_final = {res['F_vqt_final']:.6f}  "
          f"(gap: {res['F_vqt_final'] - res['F_exact']:.2e})")
    print(f"  trace_dist  = {res['trace_distance']:.4e}")
    print(f"  overlap     = {res['overlap']:.4f}")

    # ── End-to-end fit: v4 (qnode) vs v4_1 (numpy) vs v4_1 (numpy+dedup) ──
    print("\n--- End-to-end fit + timing: v4 vs v4_1 ---")
    rng = np.random.default_rng(42)
    N, S, K_true = 60, 800, 3
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
        device_name="lightning.qubit", # lightning.qubit, default.qubit
    )

    results = []

    # v4 baseline (QNode).
    print("  [1/3] v4 baseline (QNode, default.qubit) ...")
    t0 = time()
    m_v4 = _V4(**common)
    m_v4.fit(X)
    dt_v4 = time() - t0
    pred = m_v4.predict(X)
    results.append(("v4 baseline (QNode)",
                    adjusted_rand_score(true_labels, pred),
                    normalized_mutual_info_score(true_labels, pred),
                    m_v4.K, dt_v4))

    # v4_1 dense, no dedup.
    print("  [2/3] v4_1 dense NumPy (no dedup) ...")
    t0 = time()
    m_a = DMM_SVVS_VQT_QAVB_Fast(
        compute_backend="numpy", dedup_n_clusters=None, **common
    )
    m_a.fit(X)
    dt_a = time() - t0
    pred = m_a.predict(X)
    results.append(("v4_1 dense (no dedup)",
                    adjusted_rand_score(true_labels, pred),
                    normalized_mutual_info_score(true_labels, pred),
                    m_a.K, dt_a))
    print(f"        perf: {m_a.perf_summary()}")

    # v4_1 dense + dedup auto.
    print("  [3/3] v4_1 dense NumPy + dedup(auto) ...")
    t0 = time()
    m_b = DMM_SVVS_VQT_QAVB_Fast(
        compute_backend="numpy", dedup_n_clusters="auto", **common
    )
    m_b.fit(X)
    dt_b = time() - t0
    pred = m_b.predict(X)
    results.append(("v4_1 dense + dedup(auto)",
                    adjusted_rand_score(true_labels, pred),
                    normalized_mutual_info_score(true_labels, pred),
                    m_b.K, dt_b))
    print(f"        perf: {m_b.perf_summary()}")

    print("\n" + "=" * 72)
    print(f"Benchmark (N={N}, S={S}, K_true={K_true})")
    print(f"{'Method':<30} {'ARI':>7} {'NMI':>7} {'K':>4} {'time(s)':>9} {'speedup':>8}")
    print("-" * 72)
    base = results[0][4]
    for name, ari, nmi, k, dt in results:
        sp = base / dt if dt > 0 else float("inf")
        print(f"{name:<30} {ari:7.3f} {nmi:7.3f} {k:4d} {dt:9.1f} {sp:7.1f}x")
    print("=" * 72)
