#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMM-SVVS with Quantum Annealing Variational Bayes — VERSION 4
====================================================================
**Variational Quantum Thermalizer (VQT)** as a NISQ-friendly
competitor to VarQITE for the per-sample Gibbs E-step.

Mechanism contrast (VarQITE vs. VQT)
------------------------------------
Both methods target the same Gibbs state ρ_β = exp(-βH)/Z. They differ
in *how* they find it:

  VarQITE                                  VQT 
  -------------                            -----------------
  • McLachlan imaginary-time dynamics      • Free-energy minimisation
  • Purified pure state on (sys, anc)      • Mixed state on sys ALONE
  • 2·n_sys + 1 qubits                     • n_sys qubits   (≈½)
  • QFI metric tensor (O(p²) grads)        • Plain param-shift (O(p))
  • McLachlan linear solve every step      • Adam on F = ⟨H⟩ - T·S
  • init perturbation needed (saddle)      • θ = 0 is NOT a saddle
  • depth on 2·n_sys qubits                • depth on n_sys qubits
                                             → half the parameter count

VQT's ansatz acts on the system register only because the entropy is
provided *classically* by the latent mixture p_φ(x). The mixed state
   ρ(θ,φ) = Σ_x p_φ(x) · U(θ)|x⟩⟨x|U(θ)†
is automatically PSD with unit trace; its spectrum is {p_φ(x)} so the
entropy S[ρ] = -Σ_x p_φ(x) ln p_φ(x) is exactly tractable for K ≤ a
few hundred. No quantum entropy estimation is needed.

References
----------
  Verdon G, Marks J, Nanda S, Leichenauer S, Hidary J. Quantum
      Hamiltonian-based models and the variational quantum thermalizer
      algorithm. arXiv:1910.02071 (2019).
  Miyahara H, Roychowdhury V. Quantum advantage in variational Bayes
      inference. PNAS 2023, 120(31):e2212660120.
  Dang T, Kumaishi K, Usui E, et al. Stochastic variational variable
      selection for high-dimensional microbiome data. Microbiome 2022,
      10:236.
"""

from __future__ import annotations

import os
import sys
from time import time

import numpy as np

# ── Import base + v2 infrastructure ─────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from DMM_SVVS_Variational_v2 import DMM_SVVS_Variational_v2, NumericalStability  # noqa: E402

# Re-export v2 building blocks so callers can `from v4 import *`.
from DMM_SVVS_Variational_QAVB_v2 import (  # noqa: E402
    _AnnealedDMMMixin,
    _annealing_schedule,
    _cyclic_hamiltonian,
    _safe_density_matrix_from_M,
    _diagonal_to_pauli_z_strings,
    DMM_SVVS_DAVB,
    DMM_SVVS_ClassicalQAVB,
    DMM_SVVS_PennyLaneQAVB_v2,
    DMM_SVVS_VarQITE_QAVB,
    verify_varqite_gibbs,
)


# ════════════════════════════════════════════════════════════════════════════
# Variational Quantum Thermalizer (VQT) for QAVB-DMM
# ════════════════════════════════════════════════════════════════════════════

class DMM_SVVS_VQT_QAVB(_AnnealedDMMMixin, DMM_SVVS_Variational_v2):
    """
    QAVB where the per-sample Gibbs density matrix is prepared by the
    Variational Quantum Thermalizer (VQT) — direct minimisation of the
    Gibbs–Helmholtz free energy

        F[ρ] = Tr(ρ H) − T · S[ρ],     T = 1/β_t

    over a parameterised mixed state

        ρ(θ, φ) = Σ_x p_φ(x) · U(θ) |x⟩⟨x| U(θ)†,

    where p_φ(x) = softmax(φ)_x is a classical categorical mixture over
    computational-basis states and U(θ) is a hardware-efficient circuit
    acting on the SYSTEM register alone (no ancilla, no auxiliary wire).

    Per-sample E-step
    -----------------
    1. Build the diagonal energy d_i from the SVVS posteriors.
    2. Per-sample shift, scale, pad with PHANTOM_PENALTY (match VarQITE).
    3. Build H_S(s_t) = (1 − s_t)·diag(d_padded) − s_t·H_mixer as a
       PennyLane Hamiltonian on n_sys system wires.
    4. Initialise (θ, φ) from the warm-start cache or from zeros.
    5. Run Adam on (θ, φ) with parameter-shift gradients on θ and the
       closed-form softmax gradient on φ. Stop early when F or the
       readout responsibility stabilises.
    6. Read out r_{i,·} = diag(ρ*) on the K-block, renormalise.
    7. Cache (θ*, φ*) for warm-starting the next iteration.

    Hyperparameters added on top of the parent
    ------------------------------------------
    ansatz_depth : int                                              (default 3)
        Layers L in U(θ). 3-4 for K=4, 4-6 for K=16.
    n_vqt_steps : int                                             (default 100)
        Maximum Adam steps per sample per iteration. Inner loop early-stops
        on free_energy_tol / r_early_stop_tol; n_vqt_steps is the cap.
    learning_rate : float                                          (default 0.05)
        Adam LR for θ updates.
    learning_rate_phi : float                                       (default 0.1)
        Adam LR for φ updates (typically larger; classical parameters are
        well-conditioned).
    update_strategy : str                                       (default "joint")
        "joint"       : one Adam step on (θ, φ) per inner step.
        "alternating" : alternate θ-only and φ-only Adam steps.
        "phi_first"   : optimise φ-only for n_phi_warmup steps before
                        starting joint updates (φ converges quickly because
                        its loss is convex with θ fixed).
    n_phi_warmup : int                                              (default 5)
        Used only when update_strategy == "phi_first".
    free_energy_tol : float                                       (default 1e-4)
        Stop when |F^(s) − F^(s-1)| < this.
    r_early_stop_tol : float                                       (default 1e-3)
        Stop when ||r^(s) − r^(s-1)||_∞ < this. (Mirrors v3_fast Layer 2.)
    enumerate_basis : bool                                          (default True)
        If True, enumerate all 2^n_sys basis states for the energy
        expectation. If False, Monte Carlo sample M_samples states from
        p_φ. Enumeration is unambiguously the right call at K ≤ 16.
    M_samples : int                                                 (default 100)
        Number of MC samples when enumerate_basis=False.
    warm_start : bool                                               (default True)
        Carry (θ, φ) across QAVB iterations per sample.

    Cost per inner step
    -------------------
    Energy: K_pad circuit evaluations (one per basis state).
    θ-gradient: 2·n_params · K_pad circuit evaluations.
    φ-gradient: free (uses the energies already computed).
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
        use_trigamma_correction=False,
        # VQT-specific
        ansatz_depth=3,
        n_vqt_steps=100,
        learning_rate=0.05,
        learning_rate_phi=0.1,
        update_strategy="joint",
        n_phi_warmup=5,
        free_energy_tol=1e-4,
        r_early_stop_tol=1e-3,
        enumerate_basis=True,
        M_samples=100,
        warm_start=True,
        mixer="transverse_field",
        device_name="default.qubit",
        adaptive_expressivity=True,
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

        self.ansatz_depth      = int(ansatz_depth)
        self._ansatz_depth_user = int(ansatz_depth)   # user-requested floor
        self.n_vqt_steps       = int(n_vqt_steps)
        self._n_vqt_steps_user  = int(n_vqt_steps)    # user-requested floor
        self.adaptive_expressivity = bool(adaptive_expressivity)
        self.learning_rate     = float(learning_rate)
        self.learning_rate_phi = float(learning_rate_phi)
        self.update_strategy   = str(update_strategy)
        self.n_phi_warmup      = int(n_phi_warmup)
        self.free_energy_tol   = float(free_energy_tol)
        self.r_early_stop_tol  = float(r_early_stop_tol)
        self.enumerate_basis   = bool(enumerate_basis)
        self.M_samples         = int(M_samples)
        self.warm_start        = bool(warm_start)
        self.mixer             = str(mixer)
        self.device_name       = str(device_name)

        # Warm-start caches per sample.
        self._theta_cache = {}
        self._phi_cache   = {}

        # Per-fit instrumentation (mirrors v3_fast style for paper figures).
        self._vqt_stats = {
            "inner_steps_per_iter":      [],
            "early_stops_per_iter":      [],
            "final_free_energy_per_iter": [],
        }

        # PRNG for MC sampling when enumerate_basis = False.
        self._vqt_rng = np.random.default_rng(random_state)

    # ── Re-use v2 helpers verbatim ────────────────────────────────────────
    _expected_log_lik_trigamma = DMM_SVVS_ClassicalQAVB._expected_log_lik_trigamma
    _cyclic_shift_padded_matrix = DMM_SVVS_VarQITE_QAVB._cyclic_shift_padded_matrix
    _dense_to_pauli_terms       = DMM_SVVS_VarQITE_QAVB._dense_to_pauli_terms

    # ── K-adaptive expressivity floor (depth + steps) ─────────────────────

    def _adaptive_depth_floor(self) -> int:
        """Minimum ansatz depth that can represent the mixer-dominated Gibbs
        state at high s_t on n_sys system qubits.

        The VQT mixed state ρ = Σ_x p_x U|x⟩⟨x|U† must span an entangled
        superposition across all n_sys qubits (the transverse-field driver's
        eigenstates). The brick-wall ansatz delivers one entangling layer per
        depth-unit, so it needs ≥ n_sys+1 layers. Empirically (probe at K=8,
        β=30, s_t∈[0.4,0.9]): depth=2 at n_sys=3 plateaus at trace dist ~0.29
        (a hard wall — more steps / larger lr do NOT help); depth=n_sys+1
        reaches ~0.004 with steps≥100.
        """
        return self.n_sys + 1

    def _adaptive_steps_floor(self) -> int:
        """Minimum Adam steps for the high-β free-energy descent to converge
        at depth ≥ n_sys+1 (n=40 reaches only ~0.06; n=100 reaches ~0.004).
        Scales gently with n_sys."""
        return 100 + 20 * max(0, self.n_sys - 2)

    def _apply_adaptive_expressivity(self):
        """Raise ansatz_depth / n_vqt_steps to the K-adaptive floors unless
        the user opted out (adaptive_expressivity=False). The user's requested
        values act as lower bounds, so explicitly asking for more is honoured.

        This is the fix for the end-to-end ARI gap vs QuBy-expm: the QAVB
        schedule spends its entire Phase I at high s_t (β=β0, s_t≫0), exactly
        the regime where an under-expressive ansatz produces a near-uniform
        readout and the M-step loses all cluster differentiation. Verified to
        cut the worst-case trace distance ~70× (0.34 → 0.005).
        """
        if not getattr(self, "adaptive_expressivity", False):
            return
        new_depth = max(self._ansatz_depth_user, self._adaptive_depth_floor())
        new_steps = max(self._n_vqt_steps_user, self._adaptive_steps_floor())
        if (getattr(self, "verbose", 0) >= 1
                and (new_depth != self.ansatz_depth
                     or new_steps != self.n_vqt_steps)):
            print(f"  [v4] adaptive expressivity (K={self.K}, "
                  f"n_sys={self.n_sys}): "
                  f"depth {self.ansatz_depth}→{new_depth}, "
                  f"n_vqt_steps {self.n_vqt_steps}→{new_steps}")
        self.ansatz_depth = new_depth
        self.n_vqt_steps  = new_steps

    # ── Device construction ───────────────────────────────────────────────

    def _init_pennylane_device(self):
        try:
            import pennylane as qml
            from pennylane import numpy as pnp
        except ImportError:
            raise ImportError("PennyLane required:  pip install pennylane>=0.30")

        self._qml = qml
        self._pnp = pnp
        self.n_sys = max(1, int(np.ceil(np.log2(max(self.K, 2)))))
        self.K_pad = 1 << self.n_sys

        self._apply_adaptive_expressivity()

        # VQT operates on the system register alone — no ancilla, no aux wire.
        self.n_tot = self.n_sys
        self.n_params = 2 * self.n_sys * self.ansatz_depth

        try:
            self._dev = qml.device(self.device_name, wires=self.n_tot)
            self._active_device = self.device_name
        except Exception:
            self._dev = qml.device("default.qubit", wires=self.n_tot)
            self._active_device = "default.qubit"

        # Caches are invalidated whenever the qubit count changes (pruning).
        self._theta_cache = {}
        self._phi_cache   = {}

    def _initialize_parameters(self, X, random_state):
        super()._initialize_parameters(X, random_state)
        self._init_pennylane_device()

    def _prune_empty_clusters(self):
        pruned = super()._prune_empty_clusters()
        if pruned:
            self._init_pennylane_device()
        return pruned

    # ── The variational ansatz on the system register ─────────────────────
    #
    # Brick-wall hardware-efficient ansatz: (RY, RZ) per qubit per layer,
    # followed by a CNOT entangling layer. For n_sys ≥ 3 the entangling
    # layer closes into a ring. n_sys = 1 has no entangler at all.
    # ----------------------------------------------------------------------

    def _ansatz(self, theta):
        qml = self._qml
        n_sys = self.n_sys
        idx = 0
        for _ in range(self.ansatz_depth):
            for q in range(n_sys):
                qml.RY(theta[idx], wires=q); idx += 1
                qml.RZ(theta[idx], wires=q); idx += 1
            if n_sys >= 2:
                for q in range(0, n_sys - 1, 2):
                    qml.CNOT(wires=[q, q + 1])
                for q in range(1, n_sys - 1, 2):
                    qml.CNOT(wires=[q, q + 1])
                if n_sys >= 3:
                    qml.CNOT(wires=[n_sys - 1, 0])

    def _prepare_basis_state(self, basis_index: int):
        """X-gates that prepare the computational-basis state |basis_index⟩.

        Wire convention follows PennyLane's `qml.matrix` / `qml.probs`:
        wire 0 is the MOST significant bit. This matches the convention
        used by `_diagonal_to_pauli_z_strings` in v2.
        """
        qml = self._qml
        for q in range(self.n_sys):
            if (basis_index >> (self.n_sys - 1 - q)) & 1:
                qml.PauliX(wires=q)

    # ── Hamiltonian construction (same Pauli decomposition as VarQITE) ────

    def _build_hamiltonian(self, d_padded: np.ndarray, s_t: float):
        """
        H_S(s_t) = (1−s_t)·diag(d_padded) − s_t·H_mixer

        Diagonal piece: Walsh–Hadamard expansion into Pauli-Z strings.
        Mixer piece:
          "transverse_field" : H_mixer = −Σ_q X_q  ⇒  contribution +s_t·X_q
          "cyclic_shift"     : padded K-block cyclic shift, decomposed by
                               brute-force Pauli-basis projection.
        """
        qml = self._qml
        n_sys = self.n_sys

        coeffs = []
        ops    = []

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

        if abs(s_t) > 1e-15:
            if self.mixer == "transverse_field":
                for q in range(n_sys):
                    coeffs.append(s_t)
                    ops.append(qml.PauliX(q))
            elif self.mixer == "cyclic_shift":
                H_mix = self._cyclic_shift_padded_matrix()
                for c, op in self._dense_to_pauli_terms(H_mix):
                    coeffs.append(-s_t * c)
                    ops.append(op)
            else:
                raise ValueError(f"Unknown mixer: {self.mixer!r}")

        if not coeffs:
            coeffs = [0.0]
            ops    = [qml.Identity(0)]

        return qml.Hamiltonian(coeffs, ops)

    # ── Energy expectation per basis state ────────────────────────────────

    def _make_energy_qnode(self, hamiltonian):
        """Build (and cache once per call site) the per-basis-state energy
        QNode. PennyLane re-compiles the QNode on construction, so we build
        it once per `_vqt_responsibility` invocation rather than per step.
        """
        qml = self._qml

        @qml.qnode(self._dev, interface="autograd")
        def energy_circuit(theta, basis_index):
            self._prepare_basis_state(int(basis_index))
            self._ansatz(theta)
            return qml.expval(hamiltonian)

        return energy_circuit

    def _make_probs_qnode(self):
        """Computational-basis probabilities — used for the readout."""
        qml = self._qml

        @qml.qnode(self._dev)
        def probs_circuit(theta, basis_index):
            self._prepare_basis_state(int(basis_index))
            self._ansatz(theta)
            return qml.probs(wires=list(range(self.n_sys)))

        return probs_circuit

    def _energy_per_basis_state(self, energy_qnode, theta, basis_indices=None):
        """E_x(θ) = ⟨x|U†HU|x⟩ for x ∈ basis_indices.

        Returns an array of length K_pad. Indices not in `basis_indices` are
        left as 0.0 — caller is responsible for using a consistent index set
        in the Monte Carlo case.
        """
        if basis_indices is None:
            basis_indices = range(self.K_pad)

        E = np.zeros(self.K_pad, dtype=np.float64)
        for x in basis_indices:
            E[int(x)] = float(energy_qnode(theta, x))
        return E

    # ── Free energy and its gradients ─────────────────────────────────────

    @staticmethod
    def _softmax_p(phi: np.ndarray) -> np.ndarray:
        z = phi - phi.max()
        e = np.exp(z)
        return e / e.sum()

    def _sample_phi(self, phi: np.ndarray, n: int) -> np.ndarray:
        """Draw n samples x ~ p_φ."""
        p = self._softmax_p(phi)
        return self._vqt_rng.choice(self.K_pad, size=int(n), p=p)

    def _free_energy(self, theta, phi, energy_qnode, beta_t,
                     basis_indices=None):
        """F(θ, φ) = ⟨H⟩ − T·S(p_φ).

        Returns (F, energies, p) so callers can re-use `energies` for the
        closed-form φ-gradient without recomputing them.
        """
        T_t = 1.0 / beta_t
        p   = self._softmax_p(phi)
        E   = self._energy_per_basis_state(energy_qnode, theta,
                                           basis_indices=basis_indices)
        energy_term  = float(np.sum(p * E))
        # +1e-15 guards against log(0) for never-sampled basis states.
        entropy_term = float(-np.sum(p * np.log(p + 1e-15)))
        F = energy_term - T_t * entropy_term
        return F, E, p

    def _grad_theta(self, theta, phi, energy_qnode, basis_indices=None):
        """dF/dθ_j = Σ_x p(x) · 0.5·(E_x(θ_j+π/2) − E_x(θ_j−π/2))

        Parameter-shift over θ. Cost: 2·p·|basis_indices| circuit evaluations.
        """
        p = self._softmax_p(phi)
        grad = np.zeros_like(theta)
        shift = np.pi / 2

        for j in range(len(theta)):
            th_p = theta.copy(); th_p[j] += shift
            th_m = theta.copy(); th_m[j] -= shift
            E_p = self._energy_per_basis_state(energy_qnode, th_p, basis_indices)
            E_m = self._energy_per_basis_state(energy_qnode, th_m, basis_indices)
            grad[j] = float(np.sum(p * 0.5 * (E_p - E_m)))
        return grad

    def _grad_phi(self, energies, phi, beta_t):
        """Closed-form softmax gradient of F.

        F = Σ_x p(x)·E_x + T·Σ_x p(x)·ln p(x),   p(x) = softmax(φ)_x.

        With dp(y)/dφ_x = p(y)·(δ_xy − p(x)):

            dF/dφ_x = Σ_y p(y)(δ_xy − p(x))·[E_y + T·ln p(y) + T]
                    = p(x)·[E_x + T·ln p(x) + T] − p(x)·Σ_y p(y)·[E_y + T·ln p(y) + T]
                    = p(x)·[E_x + T·ln p(x) − F_avg],

        where F_avg = Σ_y p(y)·[E_y + T·ln p(y)]. The +T constant inside the
        bracket cancels against Σ_y p(y)·T = T in the centring term. (The
        VQT guide's appendix prints the +T but it is a typo — it would
        introduce a spurious uniform offset that the softmax invariance to
        constants already absorbs, but the correct expression is cleaner.)

        Pure classical computation — uses the energies already evaluated for
        the θ-gradient pass.
        """
        T_t  = 1.0 / beta_t
        p    = self._softmax_p(phi)
        ln_p = np.log(p + 1e-15)
        bracket = energies + T_t * ln_p
        F_avg = float(np.sum(p * bracket))
        return p * (bracket - F_avg)

    # ── Readout: diag(ρ) on the K_pad register ────────────────────────────

    def _readout_responsibility(self, theta, phi, probs_qnode=None):
        """diag(ρ)_k = Σ_x p_φ(x) · |⟨k|U(θ)|x⟩|²

        Returns a length-K_pad vector (caller restricts to the K-block).
        """
        if probs_qnode is None:
            probs_qnode = self._make_probs_qnode()

        p = self._softmax_p(phi)
        diag = np.zeros(self.K_pad, dtype=np.float64)
        for x in range(self.K_pad):
            probs_x = np.asarray(probs_qnode(theta, x), dtype=np.float64)
            diag += p[x] * probs_x
        return diag

    # ── Per-sample VQT responsibility ─────────────────────────────────────

    def _vqt_responsibility(self, d_i, beta_t, s_t, sample_id=None):
        """Run free-energy minimisation for one sample.

        Returns
        -------
        r : (K,) responsibility vector on the physical K-block.
        stats : dict with keys
            steps_executed       : int — number of inner Adam steps actually run
            early_stopped        : bool
            final_free_energy    : float
        """
        EPS = NumericalStability.EPS

        # ── Per-sample shift + scale (matches VarQITE convention) ──────
        d_shift = d_i - d_i.min()
        d_range = d_shift.max()
        if d_range > 1e-10:
            d_scaled_K = 4.0 * d_shift / d_range
        else:
            d_scaled_K = d_shift.copy()

        d_padded = np.full(self.K_pad, self.PHANTOM_PENALTY, dtype=np.float64)
        d_padded[: self.K] = d_scaled_K

        hamiltonian   = self._build_hamiltonian(d_padded, s_t)
        energy_qnode  = self._make_energy_qnode(hamiltonian)
        probs_qnode   = self._make_probs_qnode()

        # ── Initialise (θ, φ) ──────────────────────────────────────────
        if (self.warm_start and sample_id is not None
                and sample_id in self._theta_cache
                and len(self._theta_cache[sample_id]) == self.n_params
                and len(self._phi_cache[sample_id])   == self.K_pad):
            theta = self._theta_cache[sample_id].copy()
            phi   = self._phi_cache[sample_id].copy()
        else:
            # θ=0 is the identity circuit (NOT a saddle for VQT — the mixer's
            # X terms drive the gradient even at the identity).
            # φ=0 ⇒ uniform p, matching the s=1 driver-ground-state.
            theta = np.zeros(self.n_params, dtype=np.float64)
            phi   = np.zeros(self.K_pad,     dtype=np.float64)

        # ── Adam state (two LRs: separate for θ and φ) ─────────────────
        m_th = np.zeros_like(theta); v_th = np.zeros_like(theta)
        m_ph = np.zeros_like(phi);   v_ph = np.zeros_like(phi)
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

        # ── Basis-index set for the energy sum ─────────────────────────
        if self.enumerate_basis:
            basis_idx = np.arange(self.K_pad)
        else:
            basis_idx = np.unique(self._sample_phi(phi, self.M_samples))

        F_prev = None
        r_prev = None
        steps_executed = 0
        early_stopped = False

        for step in range(1, self.n_vqt_steps + 1):
            F, energies, p = self._free_energy(theta, phi, energy_qnode,
                                               beta_t, basis_indices=basis_idx)

            # Decide which gradients we need this step.
            do_theta = True
            do_phi   = True
            if self.update_strategy == "alternating":
                do_theta = (step % 2 == 1)
                do_phi   = (step % 2 == 0)
            elif self.update_strategy == "phi_first":
                if step <= self.n_phi_warmup:
                    do_theta = False
                    do_phi   = True
                else:
                    do_theta = True
                    do_phi   = True
            elif self.update_strategy != "joint":
                raise ValueError(
                    f"Unknown update_strategy: {self.update_strategy!r}"
                )

            g_th = (self._grad_theta(theta, phi, energy_qnode,
                                     basis_indices=basis_idx)
                    if do_theta else None)
            g_ph = (self._grad_phi(energies, phi, beta_t)
                    if do_phi else None)

            # ── Adam updates ───────────────────────────────────────────
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

            # Re-sample basis indices in MC mode after φ has moved.
            if (not self.enumerate_basis) and (step % 5 == 0):
                basis_idx = np.unique(self._sample_phi(phi, self.M_samples))

            # ── Early stopping (every 2 steps after step 2) ────────────
            if step >= 2 and step % 2 == 0:
                if F_prev is not None and abs(F - F_prev) < self.free_energy_tol:
                    early_stopped = True
                    break
                r_curr = self._readout_responsibility(theta, phi, probs_qnode)
                if r_prev is not None:
                    delta = float(np.max(np.abs(r_curr - r_prev)))
                    if delta < self.r_early_stop_tol:
                        early_stopped = True
                        break
                r_prev = r_curr
            F_prev = F

        # ── Cache + final readout ─────────────────────────────────────
        if self.warm_start and sample_id is not None:
            self._theta_cache[sample_id] = theta
            self._phi_cache[sample_id]   = phi

        diag_full = self._readout_responsibility(theta, phi, probs_qnode)
        r = np.clip(diag_full[: self.K], EPS, None)
        r = r / r.sum()

        return r, {
            "steps_executed":    int(steps_executed),
            "early_stopped":     bool(early_stopped),
            "final_free_energy": float(F),
        }

    # ── Override the QAVB quantum E-step ──────────────────────────────────

    def _compute_r_annealed(self, X: np.ndarray, beta_t: float,
                            s_t: float) -> np.ndarray:
        E_log_pi = self._E_log_pi()
        ll = self._expected_log_lik_trigamma(X)
        D = -(E_log_pi[None, :] + ll)        # (N, K)

        N, K = D.shape
        r = np.zeros((N, K))

        inner_steps_total = 0
        early_stop_count  = 0
        final_F_total     = 0.0

        for i in range(N):
            r[i], stats = self._vqt_responsibility(
                D[i], beta_t, s_t, sample_id=i
            )
            inner_steps_total += stats["steps_executed"]
            early_stop_count  += int(stats["early_stopped"])
            final_F_total     += stats["final_free_energy"]

        self._vqt_stats["inner_steps_per_iter"].append(inner_steps_total)
        self._vqt_stats["early_stops_per_iter"].append(early_stop_count)
        self._vqt_stats["final_free_energy_per_iter"].append(final_F_total / max(N, 1))

        return r

    def _print_fit_header(self):
        trig = "ON" if self.use_trigamma_correction else "OFF"
        print(f"\nStarting VQT QAVB — DMM-SVVS  (free-energy minimisation)")
        print(f"  β0={self.beta0}, s0={self.s0}, "
              f"τ1={self.tau1}, τ2={self.tau2}, "
              f"prune_start={self.prune_start}, trigamma={trig}")
        print(f"  qubits: n_sys={self.n_sys}, n_anc=0, aux=0, total={self.n_tot}  "
              f"(VarQITE would use {2*self.n_sys + 1})")
        print(f"  ansatz_depth={self.ansatz_depth}, n_params={self.n_params}, "
              f"K_pad={self.K_pad}")
        print(f"  n_vqt_steps={self.n_vqt_steps}, lr_θ={self.learning_rate}, "
              f"lr_φ={self.learning_rate_phi}, strategy={self.update_strategy}")
        print(f"  mixer={self.mixer}, device={self._active_device}")


# ════════════════════════════════════════════════════════════════════════════
# Level-0 / Level-1 verification for VQT
# ════════════════════════════════════════════════════════════════════════════

def _build_classical_H_padded(d_padded: np.ndarray, s_t: float, n_sys: int,
                              mixer: str = "transverse_field") -> np.ndarray:
    """Reference K_pad × K_pad Hamiltonian for the VQT verification harness.

    Matches the convention used inside `_build_hamiltonian`:
        H = (1−s)·diag(d_padded) − s·H_mixer
        H_mixer  ("transverse_field") = −Σ_q X_q  ⇒  full term  +s·Σ_q X_q
                 ("cyclic_shift")     = padded cyclic shift + h.c.
    """
    K_pad = 1 << n_sys
    assert d_padded.shape == (K_pad,)

    if mixer == "transverse_field":
        X = np.array([[0.0, 1.0], [1.0, 0.0]])
        Iq = np.eye(2)
        H_mix = np.zeros((K_pad, K_pad))
        for q in range(n_sys):
            tensor_factors = [Iq] * n_sys
            tensor_factors[q] = X
            T = tensor_factors[0]
            for o in tensor_factors[1:]:
                T = np.kron(T, o)
            H_mix += T
        # H_mixer = -Σ X_q  ⇒  full Hamiltonian contribution = -s·(-Σ X_q) = +s·Σ X_q
        H = (1.0 - s_t) * np.diag(d_padded) - s_t * (-H_mix)
    elif mixer == "cyclic_shift":
        K = K_pad  # the helper builds it on the K-block; for verification we
                   # call with K_pad-length d_padded and let the K-block be K_pad.
        H_mix = np.zeros((K_pad, K_pad))
        idx = np.arange(K)
        nxt = (idx + 1) % K
        H_mix[idx, nxt] = 1.0
        H_mix[nxt, idx] = 1.0
        H = (1.0 - s_t) * np.diag(d_padded) - s_t * H_mix
    else:
        raise ValueError(f"Unknown mixer: {mixer!r}")
    return H


def verify_vqt_gibbs(
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
    """
    Level-0 + Level-1 sanity check for VQT.

    Sets up a random energy vector, runs VQT to convergence, and reports:
      • Level-0: VQT free energy vs the exact Gibbs free energy
                 F[ρ_β] = Tr(ρ_β H) − T·S[ρ_β].
                 The exact value is the global minimum; VQT should approach
                 it from above without crossing.
      • Level-1: trace distance ½‖ρ_VQT − ρ_exact‖₁ on the physical K-block,
                 plus diagonal overlap.

    Returns
    -------
    dict with:
        trace_distance, overlap, r_vqt, r_exact, rho_vqt, rho_exact,
        F_vqt_final, F_exact,
        (if return_trajectory) F_history, td_history, r_history
    """
    rng = np.random.default_rng(random_state)
    d_raw = rng.standard_normal(K) * 2.0

    # ── Build VQT object via __new__ to skip the SVVS parent's heavy init ──
    m = DMM_SVVS_VQT_QAVB.__new__(DMM_SVVS_VQT_QAVB)
    m.K = K
    m.ansatz_depth      = ansatz_depth
    m.n_vqt_steps       = n_vqt_steps
    m.learning_rate     = learning_rate
    m.learning_rate_phi = learning_rate_phi
    m.update_strategy   = update_strategy
    m.n_phi_warmup      = 5
    m.free_energy_tol   = 0.0          # disable early stopping for the curve
    m.r_early_stop_tol  = 0.0
    m.enumerate_basis   = True
    m.M_samples         = 100
    m.warm_start        = False
    m.mixer             = mixer
    m.device_name       = "default.qubit"
    m._vqt_rng          = np.random.default_rng(random_state)
    m._theta_cache      = {}
    m._phi_cache        = {}
    m.PHANTOM_PENALTY   = DMM_SVVS_VQT_QAVB.PHANTOM_PENALTY
    m._init_pennylane_device()

    # ── Per-sample shift + scale + pad (mirrors _vqt_responsibility) ──────
    d_shift = d_raw - d_raw.min()
    d_range = d_shift.max()
    d_scaled_K = 4.0 * d_shift / d_range if d_range > 1e-10 else d_shift.copy()
    d_padded = np.full(m.K_pad, m.PHANTOM_PENALTY, dtype=np.float64)
    d_padded[: m.K] = d_scaled_K

    # ── Build the classical reference ρ_β at temperature T = 1/β ──────────
    H_classical = _build_classical_H_padded(d_padded, s_t, m.n_sys, mixer=mixer)
    rho_exact_pad = _safe_density_matrix_from_M(-beta * H_classical)
    rho_exact_K = rho_exact_pad[: K, : K].copy()
    rho_exact_K = rho_exact_K / np.trace(rho_exact_K).real

    eig_e = np.linalg.eigvalsh(rho_exact_pad).clip(1e-15)
    S_exact = float(-np.sum(eig_e * np.log(eig_e)))
    F_exact = float(np.real(np.trace(rho_exact_pad @ H_classical)) - (1.0 / beta) * S_exact)

    # ── Run VQT, optionally collecting the full trajectory ───────────────
    hamiltonian  = m._build_hamiltonian(d_padded, s_t)
    energy_qnode = m._make_energy_qnode(hamiltonian)
    probs_qnode  = m._make_probs_qnode()

    theta = np.zeros(m.n_params, dtype=np.float64)
    phi   = np.zeros(m.K_pad,     dtype=np.float64)

    m_th = np.zeros_like(theta); v_th = np.zeros_like(theta)
    m_ph = np.zeros_like(phi);   v_ph = np.zeros_like(phi)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    F_history  = []
    td_history = []
    r_history  = []

    basis_idx = np.arange(m.K_pad)
    for step in range(1, n_vqt_steps + 1):
        F, energies, p = m._free_energy(theta, phi, energy_qnode, beta,
                                        basis_indices=basis_idx)

        g_th = m._grad_theta(theta, phi, energy_qnode, basis_indices=basis_idx)
        g_ph = m._grad_phi(energies, phi, beta)

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
            diag = m._readout_responsibility(theta, phi, probs_qnode)
            r_K = diag[: K] / max(diag[: K].sum(), 1e-15)
            r_history.append(r_K)

            # Reconstruct ρ_VQT on the K_pad register for trace distance.
            # ρ_VQT = Σ_x p(x) · U|x⟩⟨x|U†, but the projector-mixture form
            # makes computing the dense ρ K_pad²-cost; we use the convenient
            # closed form via the per-basis-state probability vectors.
            # Off-diagonal coherences require the full reduced density.
            qml = m._qml
            @qml.qnode(m._dev)
            def rho_circuit(theta, basis_index):
                m._prepare_basis_state(int(basis_index))
                m._ansatz(theta)
                return qml.density_matrix(wires=list(range(m.n_sys)))

            rho_pad = np.zeros((m.K_pad, m.K_pad), dtype=complex)
            for x in range(m.K_pad):
                rho_x = np.asarray(rho_circuit(theta, x), dtype=complex)
                rho_pad += p[x] * rho_x
            rho_K = rho_pad[: K, : K]
            rho_K = rho_K / np.trace(rho_K).real
            td = 0.5 * float(np.linalg.svd(rho_K - rho_exact_K,
                                           compute_uv=False).sum())
            td_history.append(td)

    # ── Final readout ─────────────────────────────────────────────────────
    p_final = m._softmax_p(phi)
    qml = m._qml
    @qml.qnode(m._dev)
    def rho_circuit_final(theta, basis_index):
        m._prepare_basis_state(int(basis_index))
        m._ansatz(theta)
        return qml.density_matrix(wires=list(range(m.n_sys)))

    rho_vqt_pad = np.zeros((m.K_pad, m.K_pad), dtype=complex)
    for x in range(m.K_pad):
        rho_x = np.asarray(rho_circuit_final(theta, x), dtype=complex)
        rho_vqt_pad += p_final[x] * rho_x
    rho_vqt_K = rho_vqt_pad[: K, : K]
    rho_vqt_K = rho_vqt_K / np.trace(rho_vqt_K).real

    r_vqt   = np.real(np.diag(rho_vqt_K)).clip(1e-15)
    r_vqt   = r_vqt / r_vqt.sum()
    r_exact = np.real(np.diag(rho_exact_K)).clip(1e-15)
    r_exact = r_exact / r_exact.sum()

    trace_dist = 0.5 * float(np.linalg.svd(rho_vqt_K - rho_exact_K,
                                           compute_uv=False).sum())
    overlap = float(np.minimum(r_vqt, r_exact).sum())

    F_vqt_final = (
        float(np.sum(p_final * m._energy_per_basis_state(energy_qnode, theta)))
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


def vqt_convergence_table(K: int = 4, beta: float = 5.0, s_t: float = 0.5,
                          depths=(2, 3, 4),
                          steps_list=(20, 50, 100, 200),
                          learning_rate: float = 0.05,
                          learning_rate_phi: float = 0.1,
                          random_state: int = 0):
    """Print a depth × n_vqt_steps grid of trace distances (VQT Level-1)."""
    print(f"\nVQT Level-1 convergence (K={K}, β={beta}, s={s_t}, "
          f"transverse-field mixer)")
    print(f"  trace distance ‖ρ_VQT − ρ_exact‖₁ / 2")
    header = "depth\\steps  " + "  ".join(f"{n:>8d}" for n in steps_list)
    print("  " + header)
    print("  " + "-" * len(header))
    for d in depths:
        row = [f"  {d:^11d}"]
        for n in steps_list:
            res = verify_vqt_gibbs(
                K=K, beta=beta, s_t=s_t,
                ansatz_depth=d, n_vqt_steps=n,
                learning_rate=learning_rate,
                learning_rate_phi=learning_rate_phi,
                random_state=random_state,
                return_trajectory=False,
            )
            row.append(f"{res['trace_distance']:8.4f}")
        print("  ".join(row))


# ════════════════════════════════════════════════════════════════════════════
# Level-3: mechanism-revealing trajectory  (VarQITE vs VQT, same problem)
# ════════════════════════════════════════════════════════════════════════════

def compare_varqite_vs_vqt_trajectory(
    K: int = 4,
    beta: float = 5.0,
    s_t: float = 0.5,
    varqite_depth: int = 3,
    varqite_steps: int = 40,
    vqt_depth: int = 3,
    vqt_steps: int = 200,
    mixer: str = "transverse_field",
    random_state: int = 0,
):
    """Side-by-side convergence trajectory for the two methods on the SAME
    random per-sample energy vector and the SAME schedule point.

    Returns a dict with `td_varqite`, `td_vqt` (trace-distance histories) and
    the final reduced density matrices for both methods. This is the figure
    described in §9.2 of the mentor's guide — *the* central visualisation of
    the paper's mechanism-comparison story.
    """
    # ── VQT trajectory (uses verify_vqt_gibbs's trajectory return) ────────
    res_vqt = verify_vqt_gibbs(
        K=K, beta=beta, s_t=s_t,
        ansatz_depth=vqt_depth, n_vqt_steps=vqt_steps,
        mixer=mixer, random_state=random_state,
        return_trajectory=True,
    )

    # ── VarQITE trajectory: we reuse the same d_raw → so it must be regen ─
    # by the same seed convention as verify_vqt_gibbs.
    rng = np.random.default_rng(random_state)
    d_raw = rng.standard_normal(K) * 2.0

    # Set up VarQITE via __new__ (matching verify_varqite_gibbs).
    m = DMM_SVVS_VarQITE_QAVB.__new__(DMM_SVVS_VarQITE_QAVB)
    m.K = K
    m.ansatz_depth     = varqite_depth
    m.n_varqite_steps  = varqite_steps
    m.mixer            = mixer
    m.regularization   = 1e-4
    m.warm_start       = False
    m.metric_approx    = None
    m.init_perturbation = 0.05
    m._theta_cache = {}
    m._varqite_rng = np.random.default_rng(random_state)
    m.PHANTOM_PENALTY = DMM_SVVS_VarQITE_QAVB.PHANTOM_PENALTY
    m._init_pennylane_device()

    d_shift = d_raw - d_raw.min()
    d_range = d_shift.max()
    d_scaled_K = 4.0 * d_shift / d_range if d_range > 1e-10 else d_shift.copy()
    d_padded = np.full(m.K_pad, m.PHANTOM_PENALTY)
    d_padded[: m.K] = d_scaled_K

    # Reference ρ_β on the K_pad register.
    H_classical = _build_classical_H_padded(d_padded, s_t, m.n_sys, mixer=mixer)
    rho_exact_pad = _safe_density_matrix_from_M(-beta * H_classical)
    rho_exact_K = rho_exact_pad[: K, : K].copy()
    rho_exact_K = rho_exact_K / np.trace(rho_exact_K).real

    qml = m._qml
    pnp = m._pnp
    H_qml = m._build_hamiltonian(d_padded, s_t)

    @qml.qnode(m._dev)
    def reduced_dm_circuit(th):
        m._ansatz(th)
        return qml.density_matrix(wires=list(range(m.n_sys)))

    theta = m._varqite_rng.normal(0.0, 0.05, m.n_params)
    dtau = 0.5 * beta / max(varqite_steps, 1)
    td_varqite = []
    for _ in range(varqite_steps):
        theta = m._varqite_step(theta, H_qml, dtau)
        rho_pad = np.asarray(reduced_dm_circuit(theta), dtype=complex)
        rho_K = rho_pad[: K, : K]
        rho_K = rho_K / np.trace(rho_K).real
        td_varqite.append(
            0.5 * float(np.linalg.svd(rho_K - rho_exact_K,
                                      compute_uv=False).sum())
        )

    return {
        "td_varqite":    np.asarray(td_varqite),
        "td_vqt":        res_vqt["td_history"],
        "F_vqt_history": res_vqt["F_history"],
        "rho_varqite":   reduced_dm_circuit(theta),
        "rho_vqt":       res_vqt["rho_vqt"],
        "rho_exact":     rho_exact_K,
        "F_exact":       res_vqt["F_exact"],
        "F_vqt_final":   res_vqt["F_vqt_final"],
    }


# ════════════════════════════════════════════════════════════════════════════
# Smoke test
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    print("=" * 72)
    print("DMM-SVVS QAVB v4 — VQT competitor smoke test")
    print("=" * 72)

    # ── Level-0/1: VQT prepares the exact Gibbs state at one schedule point ─
    print("\n--- Level-0/1: VQT free energy and trace distance ---")
    res = verify_vqt_gibbs(K=4, beta=2.0, s_t=0.5,
                           ansatz_depth=3, n_vqt_steps=150,
                           random_state=0)
    print(f"  F_exact      = {res['F_exact']:.6f}")
    print(f"  F_vqt_final  = {res['F_vqt_final']:.6f}  "
          f"(gap: {res['F_vqt_final'] - res['F_exact']:.2e})")
    print(f"  trace_dist   = {res['trace_distance']:.4e}")
    print(f"  diag overlap = {res['overlap']:.4f}")
    print(f"  r_vqt   = {np.round(res['r_vqt'],   4)}")
    print(f"  r_exact = {np.round(res['r_exact'], 4)}")

    print("\n--- Level-1 convergence table ---")
    vqt_convergence_table(K=4, beta=2.0, s_t=0.5,
                          depths=(2, 3), steps_list=(30, 80, 150),
                          random_state=0)

    # ── Level-3: mechanism comparison VarQITE vs VQT on same problem ──────
    print("\n--- Level-3: VarQITE vs VQT trajectory (same problem, K=4) ---")
    cmp = compare_varqite_vs_vqt_trajectory(
        K=4, beta=2.0, s_t=0.5,
        varqite_depth=3, varqite_steps=20,
        vqt_depth=3, vqt_steps=80,
        random_state=0,
    )
    print(f"  VarQITE final trace distance: {cmp['td_varqite'][-1]:.4e}  "
          f"({len(cmp['td_varqite'])} steps)")
    print(f"  VQT     final trace distance: {cmp['td_vqt'][-1]:.4e}  "
          f"({len(cmp['td_vqt'])} steps)")

    # ── End-to-end fit on a small synthetic compositional dataset ─────────
    print("\n--- End-to-end fit: VQT QAVB on N=40 synthetic DMM ---")
    rng = np.random.default_rng(42)
    N, S, K_true = 40, 800, 3
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

    m = DMM_SVVS_VQT_QAVB(
        K_max=4, nu='auto',
        max_iter=500,
        beta0=30.0, s0=1.0, tau1=100, tau2=200,
        prune_start=20,
        verbose=1, random_state=42,
        selection_prior=0.3, prune_threshold=0.2,
        use_trigamma_correction=False,
        ansatz_depth=2,
        n_vqt_steps=40,
        learning_rate=0.05,
        learning_rate_phi=0.1,
        update_strategy="joint",
        free_energy_tol=5e-4,
        r_early_stop_tol=2e-3,
        enumerate_basis=True,
        warm_start=True,
        mixer="transverse_field",
        device_name="lightning.qubit", # lightning.qubit, default.qubit
    )
    m.fit(X)
    pred = m.predict(X)
    ari = adjusted_rand_score(true_labels, pred)
    nmi = normalized_mutual_info_score(true_labels, pred)
    print(f"\n  VQT QAVB: ARI={ari:.3f}, NMI={nmi:.3f}, K_inferred={m.K}")

    # Per-fit instrumentation summary
    stats = m._vqt_stats
    if stats["inner_steps_per_iter"]:
        print(f"  Avg inner steps/iter (across QA phase): "
              f"{np.mean(stats['inner_steps_per_iter']):.1f}")
        print(f"  Avg early stops/iter (out of N={N}): "
              f"{np.mean(stats['early_stops_per_iter']):.1f}")
        print(f"  Final mean per-sample free energy: "
              f"{stats['final_free_energy_per_iter'][-1]:.4f}")

    print("\n" + "=" * 72)
    print("v4 smoke test complete")
    print("=" * 72)
