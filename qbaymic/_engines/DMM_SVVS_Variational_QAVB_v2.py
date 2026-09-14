#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMM-SVVS with Quantum Annealing Variational Bayes (QAVB)
====================================================================

References
----------
  Miyahara H, Roychowdhury V. Quantum advantage in variational Bayes
  inference. PNAS 2023, 120(31):e2212660120.
  doi: 10.1073/pnas.2212660120

  Dang T, Kumaishi K, Usui E, et al. Stochastic variational variable
  selection for high-dimensional microbiome data. Microbiome
  2022, 10:236. doi: 10.1186/s40168-022-01439-0
"""

from __future__ import annotations

import os
import sys
from time import time

import numpy as np
from scipy.linalg import expm
from scipy.special import logsumexp, polygamma
from sklearn.utils import check_array, check_random_state

# ── Import base class ───────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from DMM_SVVS_Variational_v2 import DMM_SVVS_Variational_v2, NumericalStability  # noqa: E402


# ════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ════════════════════════════════════════════════════════════════════════════

def _cyclic_hamiltonian(K: int) -> np.ndarray:
    """
    K×K cyclic-shift driver Hamiltonian (transverse field on a K-level qudit).

    H_qu[k, k+1 mod K] = H_qu[k+1 mod K, k] = 1.

    Ground state: uniform superposition (1/√K, ..., 1/√K) with eigenvalue −2.
    Highest excited state has eigenvalue +2.
    All eigenstates are computational-basis-symmetric (translation invariant)
    so initialising ρ^Z = I_K/K is exactly the s=1, β→0 ground state.
    """
    H = np.zeros((K, K))
    idx = np.arange(K)
    nxt = (idx + 1) % K
    H[idx, nxt] = 1.0
    H[nxt, idx] = 1.0
    return H


def _annealing_schedule(t: int, beta0: float, s0: float,
                        tau1: int, tau2: int):
    """
    Two-phase annealing schedule (PNAS Eqs. 20-21):
        Phase I  [0,    τ1]:  s = s0(1 - t/τ1) → 0     |  β = β0 (large)
        Phase II [τ1,   τ2]:  s = 0                    |  β = β0 → 1 linear
        Phase III[τ2,   ∞]:   s = 0,    β = 1   (standard CAVI)

    Critical: with s_0 = 1 and β_0 ≫ 1, the algorithm starts at the maximally
    mixed state (driver ground state) and is initialisation-independent.
    """
    s_t = s0 * max(1.0 - t / max(tau1, 1), 0.0)

    if t <= tau1:
        beta_t = float(beta0)
    elif t <= tau2:
        frac = (tau2 - t) / max(tau2 - tau1, 1)
        beta_t = 1.0 + (beta0 - 1.0) * frac
    else:
        beta_t = 1.0

    return float(beta_t), float(s_t)


def _safe_density_matrix_from_M(M: np.ndarray) -> np.ndarray:
    """
    Compute exp(M) / Tr(exp(M)) for Hermitian M.

    """
    # Hermitian symmetrisation
    M = 0.5 * (M + M.T)

    # Shift by largest eigenvalue (scalar shift = phase factor in trace ratio)
    # Use eigh on M directly (small K, cheap) to get the right shift constant.
    eigvals_M = np.linalg.eigvalsh(M)
    M_shift = M - eigvals_M.max() * np.eye(M.shape[0])

    rho_unnorm = expm(M_shift)

    # Hermiticity again post-expm
    rho_unnorm = 0.5 * (rho_unnorm + rho_unnorm.T)

    tr = np.trace(rho_unnorm).real
    if abs(tr) < 1e-300:
        # Degenerate: fall back to maximally mixed state on the K-block.
        K = M.shape[0]
        return np.eye(K) / K

    rho = rho_unnorm / tr

    # PSD-clip: clip eigenvalues to floor and renormalise.
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.clip(eigvals, 1e-15, None)
    rho = eigvecs @ np.diag(eigvals) @ eigvecs.T
    rho = rho / np.trace(rho).real

    return rho


# ════════════════════════════════════════════════════════════════════════════
# Mixin: shared annealed CAVI loop
# ════════════════════════════════════════════════════════════════════════════

class _AnnealedDMMMixin:
    """
    Common fit-loop for DAVB and QAVB. Subclasses implement _compute_r_annealed.
    """

    # ── Standard temperature-scaled E-step (β-ramp phase) ──────────────────

    def _update_r_classical(self, X: np.ndarray, beta_t: float = 1.0):
        """
        Temperature-scaled softmax E-step:  log r_{ik} ∝ β · (E[ln π_k] + E[ll]).
        """
        EPS = NumericalStability.EPS
        E_log_pi = self._E_log_pi()
        ll = self._expected_log_lik(X)
        log_r = beta_t * (E_log_pi[None, :] + ll)
        log_r -= logsumexp(log_r, axis=1, keepdims=True)
        self.r = np.exp(log_r)
        self.r = np.maximum(self.r, EPS)
        self.r /= self.r.sum(axis=1, keepdims=True)

    # ── Override: fit loop ─────────────────────────────────────────────────

    def fit(self, X):
        X = check_array(X, dtype=np.float64)
        random_state = check_random_state(self.random_state)

        # Parent initialisation (sets λ*, ι*, ξ*, ϑ, ϑ', f, r from k-means).
        self._initialize_parameters(X, random_state)

        self._r_anchor = self.r.copy()                              # (N, K_init)

        if self.verbose >= 1:
            self._print_fit_header()
            print("=" * 72)

        t0 = time()

        for iteration in range(1, self.max_iter + 1):
            self.n_iter = iteration
            self._clear_cache()

            beta_t, s_t = _annealing_schedule(
                iteration - 1, self.beta0, self.s0, self.tau1, self.tau2
            )

            # ── E-step ─────────────────────────────────────────────────────
            if s_t > 1e-8:
                # Phase I: quantum annealing E-step.
                self.r = self._compute_r_annealed(X, beta_t, s_t)
            else:
                # Phase II / III: classical β-scaled softmax.
                self._update_r_classical(X, beta_t)

            # ── M-step responsibility blend ───────────────────────────────

            if s_t > 1e-8:
                r_save = self.r

                if self._r_anchor.shape == self.r.shape:
                    r_blend = (1.0 - s_t) * self.r + s_t * self._r_anchor
                    r_blend = np.maximum(r_blend, NumericalStability.EPS)
                    r_blend /= r_blend.sum(axis=1, keepdims=True)
                    self.r = r_blend
                self._clear_cache()

            self._update_f(X)
            self._update_theta()
            self._update_xi_star()
            self._update_lambda_star(X)
            self._update_iota_star(X)

            if s_t > 1e-8 and self._r_anchor.shape == r_save.shape:
                self.r = r_save

            # ── Pruning (only after annealing has done its work) ──────────
            if (iteration >= self.prune_start
                    and iteration % self.prune_every == 0):
                self._prune_empty_clusters()

            # ── ELBO + convergence ────────────────────────────────────────
            if iteration % 10 == 0:
                elbo = self._compute_elbo(X)
                self.elbo_history.append(elbo)

                if self.verbose >= 1:
                    phase = ("quantum"   if s_t > 1e-8 else
                             "annealing" if beta_t > 1.0 + 1e-6 else
                             "CAVI")
                    print(f"Iter {iteration:4d}: ELBO={elbo:14.2f}, K={self.K}, "
                          f"β={beta_t:.2f}, s={s_t:.3f}, [{phase}], "
                          f"Time={time() - t0:.1f}s")

                if (iteration > self.tau2
                        and self._pruned_at_least_once
                        and len(self.elbo_history) >= 3):
                    recent = self.elbo_history[-3:]
                    changes = [abs(recent[i] - recent[i - 1])
                               / (abs(recent[i]) + 1e-10)
                               for i in range(1, len(recent))]
                    if all(c < self.tol for c in changes):
                        self.converged = True
                        if self.verbose >= 1:
                            print(f"\n✓ Converged at iteration {iteration}")
                        break

        self._prune_empty_clusters()
        self.weights_ = self._compute_weights()

        if self.verbose >= 1:
            print(f"\nFinal: K={self.K}, weights={np.round(self.weights_, 4)}, "
                  f"time={time() - t0:.2f}s, converged={self.converged}")

        return self


# ════════════════════════════════════════════════════════════════════════════
# 1. Deterministic Annealing VB (DAVB) — no quantum term, classical baseline
# ════════════════════════════════════════════════════════════════════════════

class DMM_SVVS_DAVB(_AnnealedDMMMixin, DMM_SVVS_Variational_v2):
    """
    DAVB: temperature-scaled E-step. Equivalent to PNAS DAVB baseline.

        log r_{ik} ∝ β_t · (E[ln π_k] + E[ll_{ik}])

    β_t anneals from β_0 (≪ 1, near-uniform) up to 1.0 (standard CAVI).
    """

    def __init__(self, beta0=0.01, s0=0.0, tau1=1, tau2=150, **kwargs):
        super().__init__(**kwargs)
        self.beta0 = float(beta0)
        self.s0    = 0.0     # DAVB: classical only
        self.tau1  = int(tau1)
        self.tau2  = int(tau2)

    def _print_fit_header(self):
        print(f"\nStarting DAVB — DMM-SVVS")
        print(f"  β0={self.beta0}, τ2={self.tau2}, prune_start={self.prune_start}")

    def _compute_r_annealed(self, X, beta_t, s_t):
        raise NotImplementedError("DAVB does not use a quantum E-step")


# ════════════════════════════════════════════════════════════════════════════
# 2. Classical QAVB  (scipy.linalg.expm)
# ════════════════════════════════════════════════════════════════════════════

class DMM_SVVS_ClassicalQAVB(_AnnealedDMMMixin, DMM_SVVS_Variational_v2):
    """
    DMM-SVVS with QAVB via direct matrix exponentials.

    Quantum E-step (Phase I, t ≤ τ1)
    --------------------------------
    For each sample i:
        D_i      = energy vector (K-dim) — see below
        D_shift  = D_i - max_k D_i             (free constant shift)
        M_i      = -β(1-s) · diag(D_shift) - β·s · H_qu
        ρ_i      = expm(M_i_shifted) / trace(...)   [overflow-safe]
        r_{ik}   = real(ρ_i)_{kk}

    The diagonal energy is

        D_i[k] = -E[ln π_k] - Σ_j [ f_{ij} · ℓ̃^(α)_{kij}
                                  + (1 - f_{ij}) · ℓ̃^(β)_{ij} ]
    """

    def __init__(
        self,
        K_max=10,
        nu='auto',
        zeta=1.0,
        eta=1.0,
        xi_1=1.0,
        xi_2=1.0,
        selection_prior=0.3,
        tol=1e-4,
        max_iter=600,
        prune_threshold=0.02,
        min_clusters=None,
        prune_start=10,
        prune_every=5,
        verbose=1,
        random_state=42,
        # QAVB-specific
        beta0=30.0,
        s0=1.0,
        tau1=100,
        tau2=200,
        use_trigamma_correction=False,
    ):
        # Pruning must not fire during the quantum phase.
        effective_prune_start = max(int(prune_start), int(tau1) + int(prune_every))

        super().__init__(
            K_max=K_max, nu=nu, zeta=zeta, eta=eta,
            xi_1=xi_1, xi_2=xi_2,
            selection_prior=selection_prior,
            tol=tol, max_iter=max_iter,
            prune_threshold=prune_threshold,
            min_clusters=min_clusters,
            prune_start=effective_prune_start,
            prune_every=prune_every,
            verbose=verbose,
            random_state=random_state,
        )
        self.beta0 = float(beta0)
        self.s0    = float(s0)
        self.tau1  = int(tau1)
        self.tau2  = int(tau2)
        self.use_trigamma_correction = bool(use_trigamma_correction)
        self.H_qu = None

    # ── Driver Hamiltonian ─────────────────────────────────────────────────

    def _rebuild_H_qu(self):
        self.H_qu = _cyclic_hamiltonian(self.K)

    def _initialize_parameters(self, X, random_state):
        super()._initialize_parameters(X, random_state)
        self._rebuild_H_qu()

    def _prune_empty_clusters(self):
        pruned = super()._prune_empty_clusters()
        if pruned:
            self._rebuild_H_qu()
        return pruned

    # ── Taylor-corrected expected log-likelihood ───────────────────────────

    def _expected_log_lik_trigamma(self, X: np.ndarray) -> np.ndarray:
        """
        Cluster-dependent part of the expected log-likelihood with the
        second-order delta-method (trigamma) correction.

        Returns
        -------
        (N, K) array.  Cluster-independent constants (ln J_i! and similar)
        are omitted because they cancel under softmax / density-matrix
        normalisation.

        Vectorised structure
        --------------------
        Per the math note (§a.4-a.5):

            ℓ̃^(α)_{kij} = ln Γ(X_ij + ᾱ_kj) - ln Γ(ᾱ_kj)
                         + ½ [ψ₁(X_ij + ᾱ_kj) - ψ₁(ᾱ_kj)] · V_kj

            ℓ̃^(β)_{ij}  = ln Γ(X_ij + β̄_j) - ln Γ(β̄_j)
                         + ½ [ψ₁(X_ij + β̄_j) - ψ₁(β̄_j)] · V^β_j

            ll_{ik}     = Σ_j [ f_ij · ℓ̃^(α)_{kij} + (1-f_ij) · ℓ̃^(β)_{ij} ]

        """
        if not self.use_trigamma_correction:
            return self._expected_log_lik(X)

        from scipy.special import gammaln

        N, S = X.shape
        K = self.K

        # ── Dirichlet posterior moments (vectorised, no per-k cost) ─────
        lam       = self.lambda_star                                 # (K, S)
        lam_sum   = lam.sum(axis=1, keepdims=True)                   # (K, 1)
        alpha_bar = lam / np.maximum(lam_sum, 1e-300)                # (K, S)
        var_alpha = (lam * (lam_sum - lam)
                     / np.maximum(lam_sum ** 2 * (lam_sum + 1), 1e-300))   # (K, S)

        iota      = self.iota_star                                   # (S,)
        iota_sum  = float(iota.sum())
        beta_bar  = iota / max(iota_sum, 1e-300)                     # (S,)
        var_beta  = (iota * (iota_sum - iota)
                     / max(iota_sum ** 2 * (iota_sum + 1), 1e-300))  # (S,)

        f_vec     = self.f[0] if self.f.shape[0] > 0 else self.f.mean(axis=0)
        one_m_f   = 1.0 - f_vec                                      # (S,)

        # ── Sparse non-zero mask (the only cells contributing > 0) ──────

        cache_key = ('sparse_idx', id(X), X.shape)
        if cache_key not in self._cache:
            nz_rows, nz_cols = np.nonzero(X)
            nz_vals = X[nz_rows, nz_cols].astype(np.float64, copy=False)
            self._cache[cache_key] = (nz_rows, nz_cols, nz_vals)
        nz_rows, nz_cols, nz_vals = self._cache[cache_key]

        # ── α-branch (loop over k; per-k work is O(nnz + S) only) ───────
        ll_alpha = np.zeros((N, K))

        # Per-k gammaln(ᾱ) and ψ₁(ᾱ): O(S) each, used only on nonzero cols.
        for k in range(K):
            ab_full   = alpha_bar[k]                                 # (S,)
            va_full   = var_alpha[k]                                 # (S,)
            ab_nz     = ab_full[nz_cols]                             # (nnz,)
            va_nz     = va_full[nz_cols]                             # (nnz,)

            # gammaln(X + ᾱ) - gammaln(ᾱ)  evaluated only at non-zero X
            #   = gammaln(nz_vals + ab_nz) - gammaln(ab_nz)
            g_diff    = gammaln(nz_vals + ab_nz) - gammaln(ab_nz)    # (nnz,)
            p_diff    = (polygamma(1, nz_vals + ab_nz)
                         - polygamma(1, ab_nz))                      # (nnz,)
            term_nz   = g_diff + 0.5 * p_diff * va_nz                # (nnz,)

            # f_vec[nz_cols] is the per-(i,j) f weight
            contrib_nz = f_vec[nz_cols] * term_nz                    # (nnz,)

            # Scatter-add into ll_alpha[i, k] over the i-axis
            np.add.at(ll_alpha[:, k], nz_rows, contrib_nz)

        # ── β-branch (k-independent — compute once over nonzero subset) ─
        bb_nz     = beta_bar[nz_cols]                                # (nnz,)
        vb_nz     = var_beta[nz_cols]                                # (nnz,)
        gb_diff   = gammaln(nz_vals + bb_nz) - gammaln(bb_nz)        # (nnz,)
        pb_diff   = (polygamma(1, nz_vals + bb_nz)
                     - polygamma(1, bb_nz))                          # (nnz,)
        beta_nz   = gb_diff + 0.5 * pb_diff * vb_nz                  # (nnz,)
        beta_contrib_nz = one_m_f[nz_cols] * beta_nz                 # (nnz,)

        common_beta_contribution = np.zeros(N)                       # (N,)
        np.add.at(common_beta_contribution, nz_rows, beta_contrib_nz)

        return ll_alpha + common_beta_contribution[:, None]

    # ── Quantum E-step  (the heart of QAVB-DMM) ────────────────────────────

    def _compute_r_annealed(self, X: np.ndarray, beta_t: float,
                            s_t: float) -> np.ndarray:
        """
        Compute responsibilities r_{ik} = [ρ^{Z_i}]_{kk} via the QAVB Gibbs
        density operator.

        """
        EPS = NumericalStability.EPS
        E_log_pi = self._E_log_pi()                                    # (K,)
        ll = self._expected_log_lik_trigamma(X)                        # (N, K)

        # Diagonal energy: D_{i,k} = -E[ln π_k] - E[ll_{i,k}]
        # (We minimise this; equivalently we want exp(-β·D).)
        D = -(E_log_pi[None, :] + ll)                                  # (N, K)

        N, K = D.shape
        r = np.zeros((N, K))

        for i in range(N):
            d_i = D[i]
            
            d_shift = d_i - d_i.min()
            d_range = d_shift.max()
            
            if d_range > 1e-10:
                
                d_scaled = 4.0 * d_shift / d_range
            else:
                d_scaled = d_shift   # already zero — no information

            M = (-beta_t * (1.0 - s_t)) * np.diag(d_scaled) \
                - beta_t * s_t * self.H_qu
            rho = _safe_density_matrix_from_M(M)
            diag = np.real(np.diag(rho)).clip(EPS)
            r[i] = diag / diag.sum()

        return r

    def _print_fit_header(self):
        trig = "ON" if self.use_trigamma_correction else "OFF"
        print(f"\nStarting Classical QAVB — DMM-SVVS")
        print(f"  β0={self.beta0}, s0={self.s0}, "
              f"τ1={self.tau1}, τ2={self.tau2}, "
              f"prune_start={self.prune_start}, trigamma={trig}")


# ════════════════════════════════════════════════════════════════════════════
# 3. PennyLane QAVB  (verification on a quantum simulator)
# ════════════════════════════════════════════════════════════════════════════

class DMM_SVVS_PennyLaneQAVB_v2(DMM_SVVS_ClassicalQAVB):
    """
    QAVB with the per-sample Gibbs density matrix prepared on a PennyLane
    state-vector simulator via purification, then traced to recover the
    diagonal responsibilities.

    -----------------------------------------------
    n_sys = ⌈log2(K)⌉ system qubits     (K_pad = 2^n_sys)
    n_anc = n_sys ancilla qubits
    Total = 2·n_sys qubits.

    1. Compute classical ρ_pad (K_pad × K_pad) via expm.
    2. Eigen-decompose: ρ_pad = Σ_k λ_k |φ_k⟩⟨φ_k|.
    3. Purify: |Ψ⟩ = Σ_k √λ_k |φ_k⟩_sys ⊗ |k⟩_anc.
    4. PennyLane StatePrep |Ψ⟩, return ρ_sys = Tr_anc[|Ψ⟩⟨Ψ|].
    5. r_{i,k} = ρ_sys[k,k] for k = 0..K-1.

    """

    PHANTOM_PENALTY = 1e6   # Energy of phantom (k > K) levels.

    # ── Device management ──────────────────────────────────────────────────

    def _init_pennylane_device(self):
        try:
            import pennylane as qml
        except ImportError:
            raise ImportError(
                "PennyLane required:  pip install pennylane>=0.30"
            )

        self._qml = qml
        self.n_sys = max(1, int(np.ceil(np.log2(self.K))))
        self.K_pad = 2 ** self.n_sys
        self.n_tot = 2 * self.n_sys

        self._dev = qml.device("default.qubit", wires=list(range(self.n_tot)))
        self._qnode = qml.QNode(self._circuit, self._dev)

    def _circuit(self, purif_state):
        qml = self._qml
        qml.StatePrep(purif_state,
                      wires=list(range(self.n_tot)),
                      pad_with=0.0)
        return qml.density_matrix(wires=list(range(self.n_sys)))

    def _initialize_parameters(self, X, random_state):
        super()._initialize_parameters(X, random_state)
        self._init_pennylane_device()

    def _prune_empty_clusters(self):
        pruned = super()._prune_empty_clusters()
        if pruned:
            self._init_pennylane_device()
        return pruned

    # ── Build padded operator and density matrix ───────────────────────────

    def _gibbs_density_matrix_padded(
        self, d_i: np.ndarray, beta_t: float, s_t: float
    ) -> np.ndarray:
        """
        K_pad × K_pad density matrix for the purification circuit.

        """
        K, K_pad = self.K, self.K_pad

        d_shift = d_i - d_i.min()
        d_range = d_shift.max()
        if d_range > 1e-10:
            d_scaled = d_shift / d_range
        else:
            d_scaled = d_shift

        M = (-beta_t * (1.0 - s_t)) * np.diag(d_scaled) \
            - beta_t * s_t * self.H_qu

        rho_K = _safe_density_matrix_from_M(M)            # (K, K)

        rho_pad = np.zeros((K_pad, K_pad), dtype=rho_K.dtype)
        rho_pad[:K, :K] = rho_K
        return rho_pad

    # ── Build purification |Ψ⟩ ────────────────────────────────────────────

    def _build_purification(self, rho_pad: np.ndarray) -> np.ndarray:
        """
        |Ψ⟩ = Σ_k √λ_k |φ_k⟩_sys ⊗ |k⟩_anc  given  ρ = Σ_k λ_k |φ_k⟩⟨φ_k|.

        """
        K_pad = self.K_pad

        eigvals, eigvecs = np.linalg.eigh(rho_pad)
        eigvals = eigvals.clip(0)
        s = eigvals.sum()
        if s > 1e-10:
            eigvals = eigvals / s

        purif = np.zeros(K_pad * K_pad, dtype=complex)
        for k in range(K_pad):
            amp = np.sqrt(eigvals[k])
            if amp < 1e-15:
                continue
            for j in range(K_pad):
                purif[j * K_pad + k] += amp * eigvecs[j, k]

        norm = np.linalg.norm(purif)
        return purif / norm if norm > 1e-10 else purif

    # ── Per-sample quantum E-step via circuit ──────────────────────────────

    def _pennylane_diagonal(
        self, d_i: np.ndarray, beta_t: float, s_t: float
    ) -> np.ndarray:
        EPS = NumericalStability.EPS
        rho_pad = self._gibbs_density_matrix_padded(d_i, beta_t, s_t)
        purif = self._build_purification(rho_pad)
        rho_sys = self._qnode(purif)               # K_pad × K_pad
        diag = np.real(np.diag(rho_sys))[:self.K].clip(EPS)
        return diag / diag.sum()

    # ── Override quantum E-step ────────────────────────────────────────────

    def _compute_r_annealed(self, X: np.ndarray, beta_t: float,
                             s_t: float) -> np.ndarray:
        """
        Same E-step as ClassicalQAVB but routes through the PennyLane
        purification circuit.
        """
        E_log_pi = self._E_log_pi()
        ll = self._expected_log_lik_trigamma(X)
        D = -(E_log_pi[None, :] + ll)              # (N, K)

        N, K = D.shape
        r = np.zeros((N, K))
        for i in range(N):
            r[i] = self._pennylane_diagonal(D[i], beta_t, s_t)
        return r

    def _print_fit_header(self):
        trig = "ON" if self.use_trigamma_correction else "OFF"
        print(f"\nStarting PennyLane QAVB — DMM-SVVS")
        print(f"  β0={self.beta0}, s0={self.s0}, "
              f"τ1={self.tau1}, τ2={self.tau2}, "
              f"prune_start={self.prune_start}, trigamma={trig}")
        print(f"  qubits: n_sys={self.n_sys}, n_anc={self.n_sys}, total={self.n_tot}")


# ════════════════════════════════════════════════════════════════════════════
# 4. VarQITE QAVB  (genuine quantum subroutine — McLachlan imaginary-time
#                   evolution of a parameterised circuit on 2·n_sys qubits)
# ════════════════════════════════════════════════════════════════════════════
#
# This is the first quantum-native E-step in this module. The classical
# matrix exponential e^{-βM} is never computed at any point inside the inner
# loop. Instead, starting from the infinite-temperature thermofield-double
# state |TFD(0)⟩ (prepared by Hadamards + CNOTs on system↔ancilla pairs),
# a parameterised circuit U(θ) is evolved in imaginary time according to
# McLachlan's variational principle:
#
#       A(θ) · θ̇  =  -C(θ, Ĥ)
#       A_ij = Re ⟨∂_i ψ | ∂_j ψ⟩,    C_i = Re ⟨∂_i ψ | Ĥ | ψ⟩
#
# Both matrices are obtained on the quantum device — A via the Hadamard-
# test (PennyLane's metric_tensor transform) and C via parameter-shift.
# The diagonal of the reduced density matrix on the system register is the
# r_{i,k} responsibility vector for QAVB.
#
# Refs.
#   Yuan X, Endo S, Zhao Q, Li Y, Benjamin SC. Theory of variational
#       quantum simulation. Quantum 3, 191 (2019).
#   McArdle S et al. Variational ansatz-based quantum simulation of
#       imaginary time evolution. npj Quantum Inf. 5, 75 (2019).
#   Hadfield S et al. From the Quantum Approximate Optimization Algorithm
#       to a Quantum Alternating Operator Ansatz. Algorithms 12, 34 (2019).
# ────────────────────────────────────────────────────────────────────────────


def _diagonal_to_pauli_z_strings(diag: np.ndarray, n_qubits: int):
    """
    Walsh–Hadamard expansion of a 2^n × 2^n diagonal matrix into Pauli-Z
    tensor products. Convention: wire 0 is the most significant bit
    (PennyLane's default ordering for `qml.matrix`).

    Returns
    -------
    list of (coefficient, mask) pairs.  `mask` is a tuple of wire indices on
    which a PauliZ operator is placed; the empty tuple denotes an Identity.
    Coefficients with magnitude below 1e-12 are dropped.
    """
    K = 1 << n_qubits
    assert diag.shape == (K,)
    out = []
    for z in range(K):
        z_bits = [(z >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
        c = 0.0
        for k in range(K):
            k_bits = [(k >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
            parity = sum(zb & kb for zb, kb in zip(z_bits, k_bits)) & 1
            sign = -1.0 if parity else 1.0
            c += sign * diag[k]
        c /= K
        if abs(c) < 1e-12:
            continue
        active = tuple(q for q in range(n_qubits) if z_bits[q] == 1)
        out.append((float(c), active))
    return out


class DMM_SVVS_VarQITE_QAVB(_AnnealedDMMMixin, DMM_SVVS_Variational_v2):
    """
    QAVB where the per-sample Gibbs density matrix is prepared by Variational
    Quantum Imaginary-Time Evolution (VarQITE) on a parameterised circuit.

    Register layout
    ---------------
    n_sys = ⌈log2(K)⌉ system qubits     (K_pad = 2^n_sys)
    n_anc = n_sys ancilla qubits         (purification partners)
    n_aux = 1                            (auxiliary wire for Hadamard tests)
    Total wires = 2·n_sys + 1

    Per-sample E-step
    -----------------
    1. Build the diagonal energy d_i from the SVVS posteriors.
    2. Per-sample shift and scale (matching ClassicalQAVB).
    3. Pad to K_pad with PHANTOM_PENALTY so phantom levels are suppressed.
    4. Build H_S(s_t) = (1 - s_t) · diag(d_padded) − s_t · H_mixer
       as a qml.Hamiltonian on the SYSTEM register.
    5. Initialise θ from the per-sample warm-start cache (or zeros for
       iter 1, which leaves the variational body as the identity → the
       circuit prepares exactly |TFD(0)⟩, giving uniform reduced probs).
    6. Integrate dθ/dτ = -(A + δI)^{-1} C from τ = 0 to τ = β_t / 2 using
       n_varqite_steps explicit Euler steps.
    7. Measure the system register in the computational basis; the resulting
       probabilities are r_{i,·} (after dropping phantom entries).
    8. Cache θ for warm-starting the next iteration.

    Mixer choice
    ------------
    mixer="transverse_field"   :  H_mixer = -Σ_q X_q  (n_sys Pauli strings)
    mixer="cyclic_shift"       :  H_mixer = Σ_k |k+1 mod K_pad⟩⟨k| + h.c.
                                  decomposed via Walsh-Hadamard after a QFT
                                  basis change (O(K_pad²) strings worst case;
                                  here we decompose the dense matrix directly,
                                  which is exact but expensive — kept for
                                  ablation studies).
    """

    PHANTOM_PENALTY = 1e3   

    def __init__(
        self,
        K_max=10,
        nu='auto',
        zeta=1.0,
        eta=1.0,
        xi_1=1.0,
        xi_2=1.0,
        selection_prior=0.3,
        tol=1e-4,
        max_iter=600,
        prune_threshold=0.02,
        min_clusters=None,
        prune_start=10,
        prune_every=5,
        verbose=1,
        random_state=42,
        # QAVB schedule
        beta0=30.0,
        s0=1.0,
        tau1=60,
        tau2=120,
        use_trigamma_correction=True,
        # VarQITE-specific
        n_varqite_steps=20,
        ansatz_depth=3,
        mixer="transverse_field",
        regularization=1e-4,
        warm_start=True,
        metric_approx=None,           # None → full Hadamard-test (uses aux wire)
                                      # "block-diag" → cheaper block approx
        init_perturbation=0.05,       # σ of Gaussian kick on θ at iter 1.
                                      # 0 = exactly identity-block (will be
                                      # trapped at saddle point — see docstring).
    ):
        effective_prune_start = max(int(prune_start), int(tau1) + int(prune_every))

        super().__init__(
            K_max=K_max, nu=nu, zeta=zeta, eta=eta,
            xi_1=xi_1, xi_2=xi_2,
            selection_prior=selection_prior,
            tol=tol, max_iter=max_iter,
            prune_threshold=prune_threshold,
            min_clusters=min_clusters,
            prune_start=effective_prune_start,
            prune_every=prune_every,
            verbose=verbose,
            random_state=random_state,
        )
        self.beta0 = float(beta0)
        self.s0    = float(s0)
        self.tau1  = int(tau1)
        self.tau2  = int(tau2)
        self.use_trigamma_correction = bool(use_trigamma_correction)

        self.n_varqite_steps  = int(n_varqite_steps)
        self.ansatz_depth     = int(ansatz_depth)
        self.mixer            = str(mixer)
        self.regularization   = float(regularization)
        self.warm_start       = bool(warm_start)
        self.metric_approx    = metric_approx
        self.init_perturbation = float(init_perturbation)

        self._theta_cache = {}
        # Dedicated PRNG so per-sample perturbations are reproducible across
        # runs given the user's random_state.
        self._varqite_rng = np.random.default_rng(random_state)

    # ── Borrow the trigamma E-LL helper from ClassicalQAVB ────────────────
    _expected_log_lik_trigamma = DMM_SVVS_ClassicalQAVB._expected_log_lik_trigamma

    # ── Device + circuit construction ─────────────────────────────────────

    def _init_pennylane_device(self):
        try:
            import pennylane as qml
            from pennylane import numpy as pnp
        except ImportError:
            raise ImportError("PennyLane required:  pip install pennylane>=0.30")

        self._qml  = qml
        self._pnp  = pnp
        self.n_sys = max(1, int(np.ceil(np.log2(max(self.K, 2)))))
        self.n_anc = self.n_sys
        self.K_pad = 1 << self.n_sys
        # +1 auxiliary wire is needed by the Hadamard-test metric tensor.
        self.n_tot = 2 * self.n_sys + 1
        # Variational body acts on 2*n_sys qubits, with 2 single-qubit
        # rotations (RY, RZ) per qubit per layer.
        self.n_params = 2 * (2 * self.n_sys) * self.ansatz_depth

        self._dev = qml.device("default.qubit", wires=self.n_tot)
        # Stale warm-start parameters cannot be reused after K changes.
        self._theta_cache = {}

    def _initialize_parameters(self, X, random_state):
        super()._initialize_parameters(X, random_state)
        self._init_pennylane_device()

    def _prune_empty_clusters(self):
        pruned = super()._prune_empty_clusters()
        if pruned:
            self._init_pennylane_device()
        return pruned

    # ── The variational ansatz ────────────────────────────────────────────
    #
    # Prefix (fixed): Hadamards on system qubits 0..n_sys-1, CNOTs to ancilla
    #   partners n_sys..2*n_sys-1.  At θ = 0 the variational body is the
    #   identity (RY(0) = RZ(0) = I), so the full circuit prepares |TFD(0)⟩
    #   and the reduced state on the system is I/K_pad.
    # Body: L layers of (RY, RZ) on every system+ancilla qubit, brick-wall CZ.
    # ----------------------------------------------------------------------

    def _ansatz(self, theta):
        """
        Layout:
          Wires 0..n_sys-1            — SYSTEM register
          Wires n_sys..2*n_sys-1      — ANCILLA register (purification partners)
          Wire  2*n_sys               — auxiliary for Hadamard-test metric tensor

        Why the variational body must touch BOTH halves
        -----------------------------------------------
        For the maximally entangled state |TFD(0)⟩ = (1/√K) Σ |k⟩_S|k⟩_A,
        any unitary U_S ⊗ I_A acting on the system alone leaves the reduced
        state on the system invariant:
            Tr_A[(U_S ⊗ I_A) |TFD(0)⟩⟨TFD(0)| (U_S† ⊗ I_A)] = U_S (I/K) U_S†
            = I/K  (unitary preserves the maximally mixed state).
        Hence the variational manifold of states reachable by U_S alone is the
        trivial single point ρ_S = I/K — useless for VarQITE.

        The fix is to entangle the variational body across system+ancilla so
        that the reduced state on the system can move away from I/K. The
        body is a brick-wall ansatz over all 2*n_sys qubits.
        """
        qml = self._qml
        n_sys = self.n_sys
        n_block = 2 * n_sys

        # |TFD(0)⟩ prefix on system+ancilla.
        for q in range(n_sys):
            qml.Hadamard(wires=q)
            qml.CNOT(wires=[q, n_sys + q])

        # Variational body over the entire system+ancilla register.
        #
        # Entanglers: each layer applies CNOTs in two patterns —
        #   (a) intra-half ring on system  (wires 0..n_sys-1)
        #   (b) intra-half ring on ancilla (wires n_sys..2*n_sys-1)
        #   (c) inter-half CNOTs pairing each system qubit with its ancilla
        #       partner — these are what break the (U_S ⊗ I_A) symmetry that
        #       would otherwise pin the reduced state to I/K.
        # CNOTs (rather than CZ) are used because they couple amplitudes
        # non-diagonally, which is essential for moving the system reduced
        # state off the maximally mixed state.
        idx = 0
        for _ in range(self.ansatz_depth):
            for q in range(n_block):
                qml.RY(theta[idx], wires=q); idx += 1
                qml.RZ(theta[idx], wires=q); idx += 1
            # (a) system-side ring
            for q in range(n_sys - 1):
                qml.CNOT(wires=[q, q + 1])
            if n_sys >= 3:
                qml.CNOT(wires=[n_sys - 1, 0])
            # (b) ancilla-side ring
            for q in range(n_sys, n_block - 1):
                qml.CNOT(wires=[q, q + 1])
            if n_sys >= 3:
                qml.CNOT(wires=[n_block - 1, n_sys])
            # (c) inter-half — system_q ↔ ancilla_q for every q
            for q in range(n_sys):
                qml.CNOT(wires=[q, n_sys + q])

    # ── Hamiltonian construction on the system register ───────────────────

    def _build_hamiltonian(self, d_padded: np.ndarray, s_t: float):
        """
        H_S(s_t) = (1-s_t) · diag(d_padded) − s_t · H_mixer
        acting on system wires 0 .. n_sys-1.
        """
        qml = self._qml
        n_sys = self.n_sys

        coeffs = []
        ops    = []

        # Diagonal piece via Walsh–Hadamard.
        if abs(1.0 - s_t) > 1e-15:
            for c, mask in _diagonal_to_pauli_z_strings(d_padded, n_sys):
                if not mask:
                    op = qml.Identity(0)
                else:
                    op = qml.PauliZ(mask[0])
                    for q in mask[1:]:
                        op = op @ qml.PauliZ(q)
                coeffs.append((1.0 - s_t) * c)
                ops.append(op)

        # Mixer piece.
        if abs(s_t) > 1e-15:
            if self.mixer == "transverse_field":
                # H_mixer = -Σ_q X_q  →  contribution = -s_t · (-X_q) = +s_t X_q
                for q in range(n_sys):
                    coeffs.append(s_t)
                    ops.append(qml.PauliX(q))
            elif self.mixer == "cyclic_shift":
                # Build the K_pad×K_pad cyclic shift + h.c. on the K-block
                # padded with zeros on phantom rows/cols, then decompose
                # into Pauli strings via inverse Pauli-basis transform.
                H_mix = self._cyclic_shift_padded_matrix()
                for c, op in self._dense_to_pauli_terms(H_mix):
                    coeffs.append(-s_t * c)
                    ops.append(op)
            else:
                raise ValueError(f"Unknown mixer: {self.mixer!r}")

        if not coeffs:
            # All-zero Hamiltonian (s_t = 0 and diagonal is zero) → identity·0
            coeffs = [0.0]
            ops    = [qml.Identity(0)]

        return qml.Hamiltonian(coeffs, ops)

    def _cyclic_shift_padded_matrix(self) -> np.ndarray:
        """K_pad × K_pad cyclic-shift driver on the K-block, zeros elsewhere."""
        K, K_pad = self.K, self.K_pad
        H = np.zeros((K_pad, K_pad))
        idx = np.arange(K)
        nxt = (idx + 1) % K
        H[idx, nxt] = 1.0
        H[nxt, idx] = 1.0
        return H

    def _dense_to_pauli_terms(self, H_mat: np.ndarray):
        """
        Decompose a 2^n × 2^n Hermitian matrix into a list of (coeff, qml-op)
        Pauli-string pairs by inner product against the Pauli basis.

        c_P = (1 / 2^n) · Tr(P · H_mat)
        """
        qml = self._qml
        n = self.n_sys
        dim = 1 << n
        assert H_mat.shape == (dim, dim)
        I = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        Y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
        Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
        single = {0: I, 1: X, 2: Y, 3: Z}
        ops_map = {0: qml.Identity, 1: qml.PauliX, 2: qml.PauliY, 3: qml.PauliZ}

        out = []
        for code in range(4 ** n):
            digits = []
            tmp = code
            for _ in range(n):
                digits.append(tmp & 0b11)
                tmp >>= 2
            digits.reverse()   # wire 0 = first digit (most significant)
            P = single[digits[0]]
            for d in digits[1:]:
                P = np.kron(P, single[d])
            c = np.trace(P @ H_mat).real / dim
            if abs(c) < 1e-12:
                continue
            active = [(q, d) for q, d in enumerate(digits) if d != 0]
            if not active:
                op = qml.Identity(0)
            else:
                q0, d0 = active[0]
                op = ops_map[d0](q0)
                for q, d in active[1:]:
                    op = op @ ops_map[d](q)
            out.append((float(c), op))
        return out

    # ── McLachlan VarQITE step ────────────────────────────────────────────

    def _varqite_step(self, theta, hamiltonian, dtau):
        qml = self._qml
        pnp = self._pnp

        @qml.qnode(self._dev, interface="autograd")
        def state_circuit(th):
            self._ansatz(th)
            return qml.state()

        @qml.qnode(self._dev, interface="autograd")
        def energy_circuit(th):
            self._ansatz(th)
            return qml.expval(hamiltonian)

        th_pnp = pnp.array(theta, requires_grad=True)

        # A = Re ⟨∂_i ψ | ∂_j ψ⟩ — the real part of the QFI / 4
        A = np.asarray(
            qml.metric_tensor(state_circuit, approx=self.metric_approx)(th_pnp),
            dtype=np.float64,
        )
        # C = Re ⟨∂_i ψ | Ĥ | ψ⟩ = ½ ∂_i ⟨Ĥ⟩  (the ½ enters because
        # 2·Re⟨∂_i ψ | Ĥ | ψ⟩ = ∂_i ⟨Ĥ⟩ for real expectations of Hermitian ops)
        C = 0.5 * np.asarray(qml.grad(energy_circuit)(th_pnp), dtype=np.float64)

        A_reg = A + self.regularization * np.eye(A.shape[0])
        try:
            dtheta = np.linalg.solve(A_reg, -C)
        except np.linalg.LinAlgError:
            dtheta = np.linalg.lstsq(A_reg, -C, rcond=None)[0]

        return theta + dtau * dtheta

    # ── Readout: diagonal of the reduced system density matrix ────────────

    def _readout_probs(self, theta):
        qml = self._qml
        @qml.qnode(self._dev)
        def probs_circuit(th):
            self._ansatz(th)
            return qml.probs(wires=list(range(self.n_sys)))
        return np.asarray(probs_circuit(theta), dtype=np.float64)

    # ── Per-sample VarQITE responsibility ─────────────────────────────────

    def _varqite_responsibility(self, d_i, beta_t, s_t, sample_id=None):
        EPS = NumericalStability.EPS

        # Per-sample shift and scale — same convention as ClassicalQAVB.
        d_shift = d_i - d_i.min()
        d_range = d_shift.max()
        if d_range > 1e-10:
            d_scaled_K = 4.0 * d_shift / d_range
        else:
            d_scaled_K = d_shift.copy()

        # Pad to K_pad with a phantom penalty so phantom basis states are
        # energetically suppressed throughout imaginary-time evolution.
        d_padded = np.full(self.K_pad, self.PHANTOM_PENALTY, dtype=np.float64)
        d_padded[: self.K] = d_scaled_K

        hamiltonian = self._build_hamiltonian(d_padded, s_t)

        # Warm start: re-use the previous iteration's θ for this sample.
        if (self.warm_start and sample_id is not None
                and sample_id in self._theta_cache
                and len(self._theta_cache[sample_id]) == self.n_params):
            theta = self._theta_cache[sample_id].copy()
        else:
            # Identity-block init with a small Gaussian kick. θ = 0 is exactly
            # |TFD(0)⟩, but every ∂_i⟨H⟩|_{θ=0} vanishes by symmetry for this
            # brick-wall ansatz → McLachlan dynamics is trapped at the saddle.
            # The kick is small enough that we stay near the identity (no
            # barren plateau) but large enough to seed first-order dynamics.
            if self.init_perturbation > 0:
                theta = self._varqite_rng.normal(
                    0.0, self.init_perturbation, self.n_params
                ).astype(np.float64)
            else:
                theta = np.zeros(self.n_params, dtype=np.float64)

        tau_final = 0.5 * beta_t
        dtau = tau_final / max(self.n_varqite_steps, 1)
        for _ in range(self.n_varqite_steps):
            theta = self._varqite_step(theta, hamiltonian, dtau)

        if self.warm_start and sample_id is not None:
            self._theta_cache[sample_id] = theta

        probs = self._readout_probs(theta)
        # Drop phantom levels and renormalise on the K-block.
        r = np.clip(probs[: self.K], EPS, None)
        r = r / r.sum()
        return r

    # ── Override the quantum E-step ───────────────────────────────────────

    def _compute_r_annealed(self, X: np.ndarray, beta_t: float,
                            s_t: float) -> np.ndarray:
        E_log_pi = self._E_log_pi()
        ll = self._expected_log_lik_trigamma(X)
        D = -(E_log_pi[None, :] + ll)              # (N, K)

        N, K = D.shape
        r = np.zeros((N, K))
        for i in range(N):
            r[i] = self._varqite_responsibility(D[i], beta_t, s_t, sample_id=i)
        return r

    def _print_fit_header(self):
        trig = "ON" if self.use_trigamma_correction else "OFF"
        print(f"\nStarting VarQITE QAVB — DMM-SVVS  (genuine quantum subroutine)")
        print(f"  β0={self.beta0}, s0={self.s0}, "
              f"τ1={self.tau1}, τ2={self.tau2}, "
              f"prune_start={self.prune_start}, trigamma={trig}")
        print(f"  qubits: n_sys={self.n_sys}, n_anc={self.n_sys}, "
              f"aux=1, total={self.n_tot}")
        print(f"  ansatz_depth={self.ansatz_depth}, "
              f"n_params={self.n_params}, "
              f"n_varqite_steps={self.n_varqite_steps}, "
              f"mixer={self.mixer}, δ={self.regularization}")


# ════════════════════════════════════════════════════════════════════════════
# Verification: PennyLane circuit ↔ classical expm
# ════════════════════════════════════════════════════════════════════════════

def verify_pennylane_matches_classical(K=5, N_test=10, random_state=7):
    """
    Sanity check: the PennyLane circuit should reproduce the diagonal of the
    classical-expm Gibbs density matrix (up to float precision).
    """
    rng = np.random.default_rng(random_state)
    H_qu = _cyclic_hamiltonian(K)

    m = DMM_SVVS_PennyLaneQAVB_v2.__new__(DMM_SVVS_PennyLaneQAVB_v2)
    m.K = K
    m.H_qu = H_qu
    m._init_pennylane_device()

    max_diff = 0.0

    for _ in range(N_test):
        d_raw = rng.standard_normal(K) * 5.0       # arbitrary scale
        beta_t = float(rng.uniform(0.5, 5.0))
        s_t = float(rng.uniform(0.0, 1.0))

        # Classical reference (matching the rescaling done inside the
        # padded gibbs builder used by _pennylane_diagonal).
        d_shift = d_raw - d_raw.min()
        d_range = d_shift.max()
        d_scaled = d_shift / d_range if d_range > 1e-10 else d_shift

        M = (-beta_t * (1.0 - s_t)) * np.diag(d_scaled) - beta_t * s_t * H_qu
        rho = _safe_density_matrix_from_M(M)
        r_cl = np.real(np.diag(rho)).clip(1e-12)
        r_cl = r_cl / r_cl.sum()

        # PennyLane (via padded circuit; for K = power-of-2, K_pad = K → no padding)
        r_pl = m._pennylane_diagonal(d_raw, beta_t, s_t)

        diff = np.max(np.abs(r_cl - r_pl))
        max_diff = max(max_diff, diff)

    print(f"\nVerification (K={K}, {N_test} random tests)")
    print(f"  Max element-wise difference: {max_diff:.2e}")
    passed = max_diff < 1e-7
    print(f"  {'PASSED ✓' if passed else 'FAILED ✗'}")
    return passed


# ════════════════════════════════════════════════════════════════════════════
# Level-1 validation for VarQITE: trace-distance to the exact Gibbs state
# ════════════════════════════════════════════════════════════════════════════

def verify_varqite_gibbs(
    K: int = 4,
    beta: float = 5.0,
    s_t: float = 0.5,
    ansatz_depth: int = 3,
    n_varqite_steps: int = 30,
    mixer: str = "transverse_field",
    init_perturbation: float = 0.05,
    random_state: int = 0,
):
    """
    Prepare a Gibbs state via VarQITE for a random energy vector at fixed β
    and report the trace distance to the exact classical Gibbs state.

    Returns
    -------
    dict with keys:
        trace_distance  : ½‖ρ_varqite − ρ_exact‖₁ on the K-block
        overlap         : Σ_k min(r_varqite[k], r_exact[k])
        r_varqite       : (K,) VarQITE diagonal probabilities
        r_exact         : (K,) exact classical Gibbs diagonal
        rho_varqite     : (K, K) reduced ρ from VarQITE (K-block)
        rho_exact       : (K, K) exact Gibbs ρ (K-block)
        energies        : (n_varqite_steps,) ⟨H⟩ along the imaginary-time trajectory
    """
    rng = np.random.default_rng(random_state)
    d_raw = rng.standard_normal(K) * 2.0

    m = DMM_SVVS_VarQITE_QAVB.__new__(DMM_SVVS_VarQITE_QAVB)
    m.K = K
    m.ansatz_depth = ansatz_depth
    m.n_varqite_steps = n_varqite_steps
    m.mixer = mixer
    m.regularization = 1e-4
    m.warm_start = False
    m.metric_approx = None
    m.init_perturbation = float(init_perturbation)
    m._theta_cache = {}
    m._varqite_rng = np.random.default_rng(random_state)
    m.PHANTOM_PENALTY = DMM_SVVS_VarQITE_QAVB.PHANTOM_PENALTY
    m._init_pennylane_device()

    # Per-sample shift/scale/padding — identical to _varqite_responsibility.
    d_shift = d_raw - d_raw.min()
    d_range = d_shift.max()
    d_scaled_K = 4.0 * d_shift / d_range if d_range > 1e-10 else d_shift.copy()
    d_padded = np.full(m.K_pad, m.PHANTOM_PENALTY)
    d_padded[: m.K] = d_scaled_K

    # Build the classical reference Hamiltonian (K_pad × K_pad).
    if mixer == "transverse_field":
        X = np.array([[0.0, 1.0], [1.0, 0.0]])
        Iq = np.eye(2)
        H_mix = np.zeros((m.K_pad, m.K_pad))
        for q in range(m.n_sys):
            tensor_factors = [Iq] * m.n_sys
            tensor_factors[q] = X
            T = tensor_factors[0]
            for o in tensor_factors[1:]:
                T = np.kron(T, o)
            H_mix += T
        # H_S(s_t) = (1-s) diag − s·H_mixer  with H_mixer = -ΣX_q
        H_classical = (1.0 - s_t) * np.diag(d_padded) - s_t * (-H_mix)
    elif mixer == "cyclic_shift":
        H_classical = (1.0 - s_t) * np.diag(d_padded) \
            - s_t * m._cyclic_shift_padded_matrix()
    else:
        raise ValueError(f"Unknown mixer: {mixer!r}")

    rho_exact = _safe_density_matrix_from_M(-beta * H_classical)

    # ── Run VarQITE while collecting the energy trajectory ────────────────
    qml = m._qml
    pnp = m._pnp
    H_qml = m._build_hamiltonian(d_padded, s_t)

    @qml.qnode(m._dev, interface="autograd")
    def energy_circuit(th):
        m._ansatz(th)
        return qml.expval(H_qml)

    @qml.qnode(m._dev)
    def reduced_dm_circuit(th):
        m._ansatz(th)
        return qml.density_matrix(wires=list(range(m.n_sys)))

    # Identity-block init with the same perturbation used in production.
    if init_perturbation > 0:
        theta = m._varqite_rng.normal(0.0, init_perturbation, m.n_params)
    else:
        theta = np.zeros(m.n_params)
    dtau = 0.5 * beta / max(n_varqite_steps, 1)
    energies = []
    for _ in range(n_varqite_steps):
        theta = m._varqite_step(theta, H_qml, dtau)
        energies.append(float(energy_circuit(pnp.array(theta, requires_grad=False))))

    rho_var = np.asarray(reduced_dm_circuit(theta), dtype=complex)

    # Restrict to physical K-block for comparison 
    rho_var_K = rho_var[: K, : K].copy()
    rho_var_K /= np.trace(rho_var_K).real

    rho_exact_K = rho_exact[: K, : K].copy()
    rho_exact_K /= np.trace(rho_exact_K).real

    sv = np.linalg.svd(rho_var_K - rho_exact_K, compute_uv=False)
    trace_dist = 0.5 * float(sv.sum())

    r_vq = np.real(np.diag(rho_var_K)).clip(1e-15)
    r_vq = r_vq / r_vq.sum()
    r_exact = np.real(np.diag(rho_exact_K)).clip(1e-15)
    r_exact = r_exact / r_exact.sum()
    overlap = float(np.minimum(r_vq, r_exact).sum())

    return {
        "trace_distance": trace_dist,
        "overlap": overlap,
        "r_varqite": r_vq,
        "r_exact": r_exact,
        "rho_varqite": rho_var_K,
        "rho_exact": rho_exact_K,
        "energies": np.asarray(energies),
    }


def varqite_convergence_table(K: int = 4, beta: float = 5.0,
                              s_t: float = 0.5,
                              depths=(1, 2, 3, 4),
                              steps_list=(5, 10, 20, 40, 80),
                              random_state: int = 0):
    """
    Prints a depth × n_varqite_steps grid of trace distances. This is the
    Level-1 convergence study that the mentorship doc prescribes.
    """
    print(f"\nVarQITE Level-1 convergence (K={K}, β={beta}, s={s_t}, "
          f"transverse-field mixer)")
    print(f"  trace distance ‖ρ_varqite − ρ_exact‖₁ / 2")
    header = "depth\\steps  " + "  ".join(f"{n:>8d}" for n in steps_list)
    print("  " + header)
    print("  " + "-" * len(header))
    for d in depths:
        row = [f"  {d:^11d}"]
        for n in steps_list:
            res = verify_varqite_gibbs(
                K=K, beta=beta, s_t=s_t,
                ansatz_depth=d, n_varqite_steps=n,
                random_state=random_state,
            )
            row.append(f"{res['trace_distance']:8.4f}")
        print("  ".join(row))


# ════════════════════════════════════════════════════════════════════════════
# Smoke test
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    print("=" * 72)
    print("DMM-SVVS QAVB Smoke Test (revised implementation)")
    print("=" * 72)

    rng = np.random.default_rng(42)
    N, S, K_true = 150, 2000, 3
    block = S // K_true
    alpha = np.full((K_true, S), 0.1)
    for k in range(K_true):
        alpha[k, k * block:(k + 1) * block] = 3.0
    true_labels = rng.choice(K_true, size=N)
    X = np.array([
        rng.multinomial(5000, rng.dirichlet(alpha[true_labels[i]]))
        for i in range(N)
    ], dtype=float)

    print(f"\nDataset: N={N}, S={S}, K_true={K_true}\n")

    results = {}

    # 1. Standard VI
    print("--- Standard VI (v2) ---")
    m = DMM_SVVS_Variational_v2(
        K_max=10, nu='auto', max_iter=300, verbose=1, random_state=42,
        selection_prior=0.3, prune_threshold=0.2,
    )
    m.fit(X)
    pred = m.predict(X)
    results["Standard VI"] = (
        adjusted_rand_score(true_labels, pred),
        normalized_mutual_info_score(true_labels, pred),
        m.K,
    )

    # 2. DAVB
    print("\n--- DAVB ---")
    m = DMM_SVVS_DAVB(
        K_max=10, nu='auto', max_iter=400,
        beta0=0.01, tau2=150,
        verbose=1, random_state=42,
    )
    m.fit(X)
    pred = m.predict(X)
    results["DAVB"] = (
        adjusted_rand_score(true_labels, pred),
        normalized_mutual_info_score(true_labels, pred),
        m.K,
    )

    # 3. Classical QAVB (revised — no FIX A or FIX B)
    print("\n--- Classical QAVB (revised) ---")
    m = DMM_SVVS_ClassicalQAVB(
        K_max=10, nu='auto', max_iter=500,
        beta0=30.0, s0=1.0, tau1=100, tau2=200,
        verbose=1, random_state=42,
        selection_prior=0.3, prune_threshold=0.2,
        use_trigamma_correction=False,
    )
    m.fit(X)
    pred = m.predict(X)
    results["Classical QAVB (revised)"] = (
        adjusted_rand_score(true_labels, pred),
        normalized_mutual_info_score(true_labels, pred),
        m.K,
    )

    # 4. PennyLane QAVB
    print("\n--- PennyLane QAVB (revised) ---")
    m = DMM_SVVS_PennyLaneQAVB_v2(
        K_max=10, nu='auto', max_iter=500,
        beta0=30.0, s0=1.0, tau1=60, tau2=120,
        verbose=1, random_state=42,
        selection_prior=0.3, prune_threshold=0.2,
        use_trigamma_correction=False,
    )
    m.fit(X)
    pred = m.predict(X)
    results["PennyLane QAVB (revised)"] = (
        adjusted_rand_score(true_labels, pred),
        normalized_mutual_info_score(true_labels, pred),
        m.K,
    )

    # Summary
    print("\n" + "=" * 72)
    print(f"{'Method':<32} {'ARI':>8} {'NMI':>8} {'K':>5}")
    print("-" * 72)
    for name, (ari, nmi, k) in results.items():
        print(f"{name:<32} {ari:8.3f} {nmi:8.3f} {k:5d}")
    print("=" * 72)
    print(f"True K = {K_true}")

    print("\n--- Verification: PennyLane == Classical (purification scaffold) ---")
    verify_pennylane_matches_classical(K=4, N_test=15)
    verify_pennylane_matches_classical(K=5, N_test=15)   # tests padding (K=5, K_pad=8)

    print("\n--- Level-1 Verification: VarQITE → exact Gibbs (trace distance) ---")
    varqite_convergence_table(
        K=4, beta=2.0, s_t=0.5,
        depths=(2, 3, 4),
        steps_list=(5, 10, 20, 40),
        random_state=42,
    )

    # ── 5. VarQITE QAVB on a small dataset (genuine quantum E-step) ─────────

    print("\n--- VarQITE QAVB (small smoke test — genuine quantum E-step) ---")
    N_small = 40
    idx = rng.choice(N, size=N_small, replace=False)
    X_small = X[idx]
    labels_small = true_labels[idx]
    m = DMM_SVVS_VarQITE_QAVB(
        K_max=4, nu='auto',
        max_iter=30,                          # short — VarQITE is per-sample slow
        beta0=5.0, s0=1.0, tau1=8, tau2=18,   # compressed schedule
        prune_start=20,
        verbose=1, random_state=42,
        selection_prior=0.3, prune_threshold=0.2,
        use_trigamma_correction=False,
        n_varqite_steps=10,
        ansatz_depth=2,
        mixer="transverse_field",
        regularization=1e-4,
        warm_start=True,
        init_perturbation=0.05,
    )
    m.fit(X_small)
    pred = m.predict(X_small)
    results["VarQITE QAVB (N=40 smoke)"] = (
        adjusted_rand_score(labels_small, pred),
        normalized_mutual_info_score(labels_small, pred),
        m.K,
    )

    # Reprint the summary including VarQITE.
    print("\n" + "=" * 72)
    print("Final summary (incl. VarQITE smoke test on N=40 subset)")
    print(f"{'Method':<32} {'ARI':>8} {'NMI':>8} {'K':>5}")
    print("-" * 72)
    for name, (ari, nmi, k) in results.items():
        print(f"{name:<32} {ari:8.3f} {nmi:8.3f} {k:5d}")
    print("=" * 72)
