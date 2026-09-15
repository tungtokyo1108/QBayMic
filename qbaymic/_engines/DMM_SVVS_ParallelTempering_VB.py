#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMM_SVVS_ParallelTempering_VB.py — replica-exchange (parallel-tempering) VB
============================================================================
"""
from __future__ import annotations

import warnings
from time import time

import numpy as np
from sklearn.utils import check_array, check_random_state

from DMM_SVVS_Variational_v2 import DMM_SVVS_Variational_v2, NumericalStability

warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════════════════
# A single tempered VB replica
# ════════════════════════════════════════════════════════════════════════════

class _TemperedVBReplica(DMM_SVVS_Variational_v2):

    def __init__(self, beta, **kwargs):
        super().__init__(**kwargs)
        self.beta = float(beta)
        self._initialized = False
        self._last_free_energy = np.inf

    # ── tempered (deterministic-annealing) E-step ──────────────────────────

    def _tempered_logits(self, X):
        """Return (log_num (N,K), free_energy_density (N,)) at this β."""
        from scipy.special import logsumexp
        E_log_pi = self._E_log_pi()           # (K,)
        ll = self._expected_log_lik(X)        # (N, K)
        log_num = self.beta * (E_log_pi[None, :] + ll)   # (N, K)
        lse = logsumexp(log_num, axis=1)      # (N,)  log Σ_k [.]^β
        return log_num, lse

    def _update_r_tempered(self, X):
        EPS = NumericalStability.EPS
        log_num, lse = self._tempered_logits(X)
        self.r = np.exp(log_num - lse[:, None])
        self.r = np.maximum(self.r, EPS)
        self.r /= self.r.sum(axis=1, keepdims=True)
        return lse

    def tempered_free_energy(self, X):
        """
        The β-TEMPERED variational free energy that this replica minimises:

            F_β = −(1/β) Σ_i log Σ_k exp( β·(E[ln π_k] + E[ln p(x_i|k)]) ).
        """
        _, lse = self._tempered_logits(X)
        return float(-(1.0 / max(self.beta, 1e-12)) * lse.sum())

    def base_energy(self, X):
        
        from scipy.special import logsumexp
        E_log_pi = self._E_log_pi()
        ll = self._expected_log_lik(X)
        log_num = E_log_pi[None, :] + ll          # β=1 (no temperature power)
        return float(-logsumexp(log_num, axis=1).sum())

    def init_replica(self, X, random_state):
        """k-means initialisation (parent machinery), once, before sweeping."""
        self._initialize_parameters(X, random_state)
        self._initialized = True
        self._last_free_energy = self.tempered_free_energy(X)

    def reseed(self, X, random_state):
        
        self._initialize_parameters(X, random_state)
        self._last_free_energy = self.tempered_free_energy(X)

    def sweep_once(self, X, iteration, do_prune):
        
        self.n_iter = iteration
        self._clear_cache()

        self._update_r_tempered(X)          # tempered DAEM E-step
        self._update_f(X)                   # ── M-step on tempered r ──
        self._update_theta()
        self._update_xi_star()
        self._update_lambda_star(X)
        self._update_iota_star(X)

        if do_prune:
            self._prune_empty_clusters()

        self._clear_cache()
        F = self.tempered_free_energy(X)
        self._last_free_energy = F
        return F                            # energy = tempered free energy

    # ── exporting / importing a replica configuration for a swap ────────────
    def export_state(self):
        """Deep copy of the replica's variational state (a 'configuration')."""
        return dict(
            K=self.K,
            r=self.r.copy(),
            f=self.f.copy(),
            theta=self.theta.copy(),
            theta_prime=self.theta_prime.copy(),
            xi_star=self.xi_star.copy(),
            lambda_star=self.lambda_star.copy(),
            iota_star=self.iota_star.copy(),
            pruned=self._pruned_at_least_once,
            last_free_energy=self._last_free_energy,
        )

    def import_state(self, st):
        """Load a configuration exported by another replica (a swap)."""
        self.K = st["K"]
        self.r = st["r"].copy()
        self.f = st["f"].copy()
        self.theta = st["theta"].copy()
        self.theta_prime = st["theta_prime"].copy()
        self.xi_star = st["xi_star"].copy()
        self.lambda_star = st["lambda_star"].copy()
        self.iota_star = st["iota_star"].copy()
        self._pruned_at_least_once = st["pruned"]
        self._last_free_energy = st["last_free_energy"]
        self._clear_cache()


# ════════════════════════════════════════════════════════════════════════════
# Parallel-tempering / replica-exchange VB
# ════════════════════════════════════════════════════════════════════════════

class DMM_SVVS_ParallelTempering_VB:
    """
    Replica-exchange (parallel-tempering) variational Bayes for DMM-SVVS.
    """

    def __init__(self,
                 compute_budget=400,
                 n_replicas=8,
                 beta_min=0.30,
                 ladder=None,
                 swap_every=5,
                 # shared VB hyperparameters (identical across replicas)
                 K_max=5,
                 nu=1.0,
                 selection_prior=0.443,
                 prune_threshold=0.29,
                 prune_start=10,
                 prune_every=5,
                 zeta=1.0,
                 eta=1.0,
                 xi_1=1.0,
                 xi_2=1.0,
                 min_clusters=None,
                 random_state=42,
                 verbose=0):
        self.compute_budget = int(compute_budget)
        self.n_replicas = int(n_replicas)
        self.beta_min = float(beta_min)
        self.ladder = None if ladder is None else np.asarray(ladder, dtype=float)
        self.swap_every = int(swap_every)

        self.shared = dict(
            K_max=K_max, nu=nu, selection_prior=selection_prior,
            prune_threshold=prune_threshold, prune_start=prune_start,
            prune_every=prune_every, zeta=zeta, eta=eta, xi_1=xi_1, xi_2=xi_2,
            min_clusters=min_clusters, verbose=0,
        )
        self.random_state = random_state
        self.verbose = int(verbose)

        # filled by fit
        self.cold_ = None
        self.replicas_ = None
        self.ladder_ = None
        self.swap_acceptance_ = None
        self.overall_swap_acceptance_ = None
        self.best_elbo_ = None
        self.n_sweeps_per_replica_ = None
        self.n_energy_evals_ = None
        self.n_reseeds_ = None
        self.K = None
        self.weights_ = None

    # ── ladder construction ─────────────────────────────────────────────────
    def _build_ladder(self):
        if self.ladder is not None:
            lad = np.asarray(self.ladder, dtype=float)
            if abs(lad[0] - 1.0) > 1e-9:
                raise ValueError("ladder must start at β=1.0 (the cold replica)")
            return lad
        if self.n_replicas == 1:
            return np.array([1.0])
        # Geometric ladder 1.0 → beta_min (cold→hot). Geometric spacing gives
        # roughly constant swap acceptance per rung when the heat capacity is
        # roughly scale-free, the standard PT starting point (Earl–Deem 2005).
        return np.geomspace(1.0, self.beta_min, self.n_replicas)

    # ── fit ─────────────────────────────────────────────────────────────────
    def fit(self, X):
        X = check_array(X, dtype=np.float64)
        rng = check_random_state(self.random_state)

        ladder = self._build_ladder()
        M = len(ladder)
        self.ladder_ = ladder

        # Matched compute: total sweeps across replicas == compute_budget.
        n_sweeps = max(1, int(np.ceil(self.compute_budget / M)))
        self.n_sweeps_per_replica_ = n_sweeps
        self.n_energy_evals_ = n_sweeps * M

        # Build replicas. Each gets a DISTINCT k-means init seed (different
        # starting basins is the point of PT), derived from random_state.
        seeds = rng.randint(0, 2**31 - 1, size=M)
        replicas = []
        for m in range(M):
            rep = _TemperedVBReplica(beta=float(ladder[m]),
                                     max_iter=n_sweeps,
                                     random_state=int(seeds[m]),
                                     **self.shared)
            rep.init_replica(X, check_random_state(int(seeds[m])))
            replicas.append(rep)
        self.replicas_ = replicas

        # Per replica: U = common β=1 energy of its current config (for swaps),
        # plus a stagnation counter driving hot-replica reseeding.
        U = np.array([rep.base_energy(X) for rep in replicas])
        prev_F = np.array([rep._last_free_energy for rep in replicas])
        stagnation = np.zeros(M, dtype=int)
        swap_attempts = np.zeros(max(M - 1, 1))
        swap_accepts = np.zeros(max(M - 1, 1))
        n_reseeds = 0
        best_U_cold = np.inf

        if self.verbose >= 1:
            print(f"  [PT] M={M} replicas, ladder β={np.round(ladder, 3)}, "
                  f"{n_sweeps} sweeps/replica ({self.n_energy_evals_} total "
                  f"energy evals; budget {self.compute_budget})")

        STALL_ROUNDS = 3
        reseed_rng = np.random.RandomState(int(rng.randint(0, 2**31 - 1)))

        t0 = time()
        n_rounds = int(np.ceil(n_sweeps / self.swap_every))
        sweep_done = 0
        swap_parity = 0
        for rnd in range(n_rounds):
            # ── advance every replica by up to swap_every sweeps ──
            block = min(self.swap_every, n_sweeps - sweep_done)
            for _ in range(block):
                sweep_done += 1
                for m, rep in enumerate(replicas):
                    do_prune = (sweep_done >= rep.prune_start
                                and sweep_done % rep.prune_every == 0)
                    rep.sweep_once(X, sweep_done, do_prune)
            # Refresh the common (β=1) energy of every config after the block.
            for m, rep in enumerate(replicas):
                U[m] = rep.base_energy(X)
                F_now = rep._last_free_energy
                # stagnation in the replica's OWN tempered objective
                if F_now >= prev_F[m] - 1e-6 * (abs(prev_F[m]) + 1.0):
                    stagnation[m] += 1
                else:
                    stagnation[m] = 0
                prev_F[m] = F_now
            best_U_cold = min(best_U_cold, U[0])

            # ── Metropolis replica-exchange (alternate even/odd adjacent pairs)
            # Swap acceptance uses the COMMON β=1 energy U at the two configs:
            #   log A = (β_i − β_j)·(U_i − U_j)   (Earl–Deem 2005).
            for m in range(swap_parity, M - 1, 2):
                bi, bj = ladder[m], ladder[m + 1]
                swap_attempts[m] += 1
                if replicas[m].K != replicas[m + 1].K:
                    continue
                log_A = (bi - bj) * (U[m] - U[m + 1])
                if log_A >= 0.0 or reseed_rng.random_sample() < np.exp(min(log_A, 0.0)):
                    si = replicas[m].export_state()
                    sj = replicas[m + 1].export_state()
                    replicas[m].import_state(sj)
                    replicas[m + 1].import_state(si)
                    U[m], U[m + 1] = U[m + 1], U[m]
                    prev_F[m], prev_F[m + 1] = prev_F[m + 1], prev_F[m]
                    stagnation[m], stagnation[m + 1] = 0, 0
                    swap_accepts[m] += 1
            swap_parity ^= 1

            # ── Hot-replica reseeding (protect the cold chain, m∈{0,1}) ──
            if rnd < n_rounds - 1:                  # never reseed on the last round
                for m in range(2, M):
                    if stagnation[m] >= STALL_ROUNDS:
                        new_seed = reseed_rng.randint(0, 2**31 - 1)
                        replicas[m].reseed(X, check_random_state(int(new_seed)))
                        U[m] = replicas[m].base_energy(X)
                        prev_F[m] = replicas[m]._last_free_energy
                        stagnation[m] = 0
                        n_reseeds += 1

            if self.verbose >= 2:
                print(f"    round {rnd+1}/{n_rounds}: cold U={U[0]:.1f} "
                      f"K={replicas[0].K} reseeds={n_reseeds} t={time()-t0:.1f}s")

        # Finalise the cold replica as the reported result.
        cold = replicas[0]
        cold._prune_empty_clusters()
        cold.weights_ = cold._compute_weights()
        self.cold_ = cold
        self.K = cold.K
        self.weights_ = cold.weights_
        self.best_elbo_ = float(-best_U_cold)
        self.n_reseeds_ = n_reseeds
        with np.errstate(invalid="ignore", divide="ignore"):
            self.swap_acceptance_ = np.where(swap_attempts > 0,
                                             swap_accepts / np.maximum(swap_attempts, 1),
                                             np.nan)
        tot_att = swap_attempts.sum()
        self.overall_swap_acceptance_ = (float(swap_accepts.sum() / tot_att)
                                         if tot_att > 0 else float("nan"))

        if self.verbose >= 1:
            print(f"  [PT] done: cold K={self.K}, best ELBO={self.best_elbo_:.1f}, "
                  f"swap acc per pair={np.round(self.swap_acceptance_, 2)}, "
                  f"overall={self.overall_swap_acceptance_:.2f}, "
                  f"time={time()-t0:.1f}s")
        return self

    # ── prediction (delegate to the cold β=1 replica) ───────────────────────
    def predict(self, X):
        if self.cold_ is None:
            raise RuntimeError("call fit() before predict()")
        return self.cold_.predict(X)

def tune_ladder(X,
                compute_budget,
                shared,
                n_replicas=8,
                beta_min_grid=(0.50, 0.30, 0.15, 0.08, 0.04, 0.02, 0.01),
                swap_every=5,
                target_lo=0.20,
                target_hi=0.40,
                tune_seeds=(0, 1, 2),
                verbose=True):

    if verbose:
        print(f"  [tune] selecting β_min for {n_replicas}-rung ladder, "
              f"target swap acc ∈ [{target_lo:.0%}, {target_hi:.0%}]")
    diagnostics = []
    chosen = None
    for bmin in beta_min_grid:
        accs = []
        for s in tune_seeds:
            pt = DMM_SVVS_ParallelTempering_VB(
                compute_budget=compute_budget, n_replicas=n_replicas,
                beta_min=bmin, swap_every=swap_every, random_state=s,
                verbose=0, **shared)
            pt.fit(X)
            accs.append(pt.overall_swap_acceptance_)
        mean_acc = float(np.nanmean(accs))
        diagnostics.append(dict(beta_min=bmin, mean_swap_acc=mean_acc,
                                ladder=pt.ladder_.tolist()))
        in_window = target_lo <= mean_acc <= target_hi
        if verbose:
            mark = "  ← in window" if in_window else ""
            print(f"  [tune] β_min={bmin:.2f}: mean swap acc={mean_acc:.2f}{mark}")
        if in_window and chosen is None:
            chosen = pt.ladder_.copy()
    if chosen is None:
        # No grid point landed in-window: take the one closest to the window
        # centre (most-efficient available ladder).
        centre = 0.5 * (target_lo + target_hi)
        best = min(diagnostics, key=lambda d: abs(d["mean_swap_acc"] - centre))
        chosen = np.asarray(best["ladder"], dtype=float)
        if verbose:
            print(f"  [tune] no grid β_min hit the window; using closest "
                  f"(β_min={best['beta_min']:.2f}, acc={best['mean_swap_acc']:.2f})")
    return chosen, diagnostics


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    from sklearn.metrics import adjusted_rand_score
    from data_generators import generate_high_overlap_clusters

    print("Smoke-test — DMM_SVVS_ParallelTempering_VB on Case-1 (σ≈0.29)")
    print("=" * 72)
    X, y, _ = generate_high_overlap_clusters(
        N=400, S=5000, K=3, seed=0, separation=0.2, imbalance=0.0,
        zero_inflation=0.8, signal_fraction=0.15)

    shared = dict(K_max=5, nu=1.0, selection_prior=0.443, prune_threshold=0.29,
                  prune_start=10, prune_every=5)

    pt = DMM_SVVS_ParallelTempering_VB(
        compute_budget=400, n_replicas=8, beta_min=0.30, swap_every=5,
        random_state=0, verbose=1, **shared)
    pt.fit(X)
    ari = adjusted_rand_score(y, pt.predict(X))
    print(f"\n  cold replica ARI={ari:.4f}  K={pt.K}  "
          f"(true K=3)  budget audit: {pt.n_energy_evals_} evals "
          f"({pt.n_sweeps_per_replica_}×{len(pt.ladder_)})")
