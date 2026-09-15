#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMM-SVVS Variational Inference 
"""

import numpy as np
from scipy.special import digamma, logsumexp, expit
from sklearn.utils import check_array, check_random_state
from sklearn import cluster
from time import time


class NumericalStability:
    EPS     = 1e-10
    MAX_EXP = 500

    @staticmethod
    def safe_log(x):
        return np.log(np.maximum(x, NumericalStability.EPS))

    @staticmethod
    def safe_exp(x):
        return np.exp(np.clip(x, -NumericalStability.MAX_EXP, NumericalStability.MAX_EXP))


class DMM_SVVS_Variational_v2:
    """
    Dirichlet Multinomial Mixture with Variational Variable Selection 

    Parameters
    ----------
    K_max : int
        Maximum / initial number of clusters (truncation level).
    nu : float or 'auto'
        DP concentration parameter for stick-breaking.
        'auto'  → nu = 1/K_max  (recommended; weakly prefers fewer clusters)
        Small (0.01–0.1) → favours fewer clusters.
        Large  (1–10)    → spreads mass across many clusters.
    zeta : float
        Prior concentration for cluster-specific Dirichlet (alpha_k).
    eta : float
        Prior concentration for background Dirichlet (beta).
    xi_1, xi_2 : float
        Beta(xi_1, xi_2) prior on per-OTU selection probabilities ε_j.
        Default 1.0 / 1.0 → uniform (neutral) prior.  [FIX 4]
    selection_prior : float
        Initial warm-start value for f (per-sample selection probability).
        Does NOT set the Beta hyperparameters (that is xi_1/xi_2).
    tol : float
        Relative ELBO convergence tolerance (default 1e-4, tighter than v1). [FIX 3]
    max_iter : int
        Maximum CAVI iterations.
    prune_threshold : float
        A cluster is kept only if BOTH its stick-breaking weight > prune_threshold
        AND its sample count > min_cluster_size.  [FIX 1]
    min_clusters : int or None
        Minimum number of clusters to keep at any pruning step.
        None → max(2, K_max // 5).
    prune_start : int
        Iteration at which pruning begins (default 10, earlier than v1's 50). [FIX 3]
    prune_every : int
        Prune every this many iterations (default 5, more frequent than v1). [FIX 3]
    verbose : int
    random_state : int
    """

    def __init__(self,
                 K_max=10,
                 nu='auto',
                 zeta=1.0,
                 eta=1.0,
                 xi_1=1.0,
                 xi_2=1.0,
                 selection_prior=0.3,
                 tol=1e-4,
                 max_iter=500,
                 prune_threshold=0.02,
                 min_clusters=None,
                 prune_start=10,
                 prune_every=5,
                 verbose=1,
                 random_state=42):

        self.K_max           = K_max
        self.nu_input        = nu          # store raw input; resolved in fit()
        self.zeta            = zeta
        self.eta             = eta
        self.xi_1            = xi_1
        self.xi_2            = xi_2
        self.selection_prior = selection_prior
        self.tol             = tol
        self.max_iter        = max_iter
        self.prune_threshold = prune_threshold
        self.min_clusters    = min_clusters
        self.prune_start     = prune_start
        self.prune_every     = prune_every
        self.verbose         = verbose
        self.random_state    = random_state

        # Runtime state
        self.N = self.S = self.K = None
        self.r = self.f = None
        self.theta = self.theta_prime = None
        self.xi_star = self.lambda_star = self.iota_star = None
        self.elbo_history = []
        self.converged = False
        self.n_iter = 0
        self._cache = {}
        self._pruned_at_least_once = False

    # ─────────────────────────────────────────────────────────────────────
    # Initialization
    # ─────────────────────────────────────────────────────────────────────

    def _resolve_nu(self):
        """
        Resolve nu from 'auto'  
        """
        if self.nu_input == 'auto':
            return 1.0 / self.K_max
        nu = float(self.nu_input)
        if nu > 2.0 and self.verbose >= 1:
            print(f"  [Warning] nu={nu:.1f} is large — stick-breaking will spread mass "
                  f"across all {self.K_max} clusters and resist pruning. "
                  f"Consider nu='auto' or nu<1 for real data.")
        return nu

    def _initialize_parameters(self, X, random_state):
        self.N, self.S = X.shape
        self.K = self.K_max
        self.nu = self._resolve_nu()

        # Resolve min_clusters
        self._min_clusters = (
            max(2, self.K_max // 5)
            if self.min_clusters is None
            else int(self.min_clusters)
        )

        self._min_cluster_size = max(1.0, self.N / (5.0 * self.K_max))

        if self.verbose >= 1:
            print(f"Initializing: N={self.N}, S={self.S}, K_max={self.K_max}")
            print(f"  ν={self.nu:.4f}, ζ={self.zeta}, η={self.eta}, "
                  f"ξ=({self.xi_1},{self.xi_2}), min_K={self._min_clusters}, "
                  f"min_cluster_size={self._min_cluster_size:.1f}")

        # Responsibilities from k-means
        self.r = self._init_responsibilities_kmeans(X, random_state)

        # Feature selection warm-start at selection_prior  
        self.f = np.full((self.N, self.S), self.selection_prior)

        # Stick-breaking: symmetric init
        self.theta       = np.ones(self.K)
        self.theta_prime = np.ones(self.K) * self.nu

        # Beta prior on feature selection probabilities
        # xi_1 = xi_2 = 1.0  →  Beta(1,1) = Uniform[0,1]
        self.xi_star = np.empty((self.S, 2))
        self.xi_star[:, 0] = self.xi_1
        self.xi_star[:, 1] = self.xi_2

        # Cluster-specific Dirichlet parameters: init from k-means statistics
        self.lambda_star = np.zeros((self.K, self.S))
        for k in range(self.K):
            mask = self.r[:, k] > 0.1
            if mask.sum() > 0:
                mu = X[mask].mean(axis=0) + 0.5
                self.lambda_star[k] = mu / mu.sum() * (self.zeta * self.S)
            else:
                self.lambda_star[k] = self.zeta
        self.lambda_star = np.maximum(self.lambda_star, 0.1)

        # Background Dirichlet parameters
        mu_all = X.mean(axis=0) + 0.5
        self.iota_star = mu_all / mu_all.sum() * (self.eta * self.S)
        self.iota_star = np.maximum(self.iota_star, 0.1)

        self._cache = {}
        self._pruned_at_least_once = False

    def _init_responsibilities_kmeans(self, X, random_state):
        N = X.shape[0]
        if self.K == 1:
            return np.ones((N, 1))
        try:
            X_log  = np.log1p(X)
            kmeans = cluster.KMeans(
                n_clusters=min(self.K, N),
                n_init=10, max_iter=100,
                random_state=random_state
            )
            labels = kmeans.fit_predict(X_log)
            r = np.zeros((N, self.K))
            r[np.arange(N), labels] = 1.0
        except Exception as e:
            if self.verbose >= 1:
                print(f"  K-means failed ({e}), using random init")
            r = random_state.rand(N, self.K)
            r /= r.sum(axis=1, keepdims=True)
        return r

    # ─────────────────────────────────────────────────────────────────────
    # Cached expectations
    # ─────────────────────────────────────────────────────────────────────

    def _E_log_alpha(self):
        """(K, S) array: E[log α_kj] = ψ(λ_kj) − ψ(Σ_j λ_kj)  — all K at once."""
        if 'Ela' not in self._cache:
            s = self.lambda_star.sum(axis=1, keepdims=True)  # (K, 1)
            self._cache['Ela'] = digamma(self.lambda_star) - digamma(s)
        return self._cache['Ela']

    def _E_log_beta(self):
        if 'Elb' not in self._cache:
            self._cache['Elb'] = digamma(self.iota_star) - digamma(self.iota_star.sum())
        return self._cache['Elb']

    def _E_log_pi(self):
        """
        E[log π_k] via vectorized cumsum — O(K).
        """
        if 'Elpi' not in self._cache:
            st  = self.theta + self.theta_prime
            elg  = digamma(self.theta)        - digamma(st)   # (K,)
            el1g = digamma(self.theta_prime)  - digamma(st)   # (K,)
            cum  = np.concatenate([[0.0], np.cumsum(el1g[:-1])])
            self._cache['Elpi'] = elg + cum
        return self._cache['Elpi']

    def _clear_cache(self):
        self._cache = {}

    # ─────────────────────────────────────────────────────────────────────
    # CAVI update steps 
    # ─────────────────────────────────────────────────────────────────────

    def _expected_log_lik(self, X):
        """
        (N, K) expected log-likelihood via exact E[log α] form.

        E[log p(x_i | z_i=k)] = Σ_j [ f_j·x_ij·E[log α_kj]
                                      + (1−f_j)·x_ij·E[log β_j] ]
        combined[k,j] = f_j·E[log α_kj] + (1−f_j)·E[log β_j]
        ll = X @ combined.T                            (N,K) via one matmul
        """
        E_log_a  = self._E_log_alpha()   # (K, S)
        E_log_b  = self._E_log_beta()    # (S,)
        f_avg    = self.f.mean(axis=0)   # (S,) — per-feature mean over samples
        combined = (f_avg[None, :] * E_log_a
                    + (1.0 - f_avg)[None, :] * E_log_b[None, :])  # (K, S)
        return X @ combined.T            # (N, K)

    def _update_r(self, X):
        EPS      = NumericalStability.EPS
        E_log_pi = self._E_log_pi()          # (K,)
        ll       = self._expected_log_lik(X) # (N, K)

        log_r  = E_log_pi[None, :] + ll
        log_r -= logsumexp(log_r, axis=1, keepdims=True)
        self.r = np.exp(log_r)
        self.r = np.maximum(self.r, EPS)
        self.r /= self.r.sum(axis=1, keepdims=True)

    def _update_f(self, X):
        """
        Per-feature selection update via exact E[log α] matmul.

        log_odds_j = E[log(ξ1_j/ξ2_j)]
                   + Σ_i Σ_k r_ik · x_ij · E[log α_kj]   (sel term)
                   − Σ_i       x_ij · E[log β_j]          (bg  term)
        """
        EPS = NumericalStability.EPS

        xi_sum = self.xi_star.sum(axis=1)
        E_xi1  = digamma(self.xi_star[:, 0]) - digamma(xi_sum)  # (S,)
        E_xi2  = digamma(self.xi_star[:, 1]) - digamma(xi_sum)  # (S,)

        E_log_alpha = self._E_log_alpha()   # (K, S)
        E_log_beta  = self._E_log_beta()    # (S,)

        # r-weighted E[log α_kj]: (N,K)@(K,S) = (N,S)
        el_alpha_ni = self.r @ E_log_alpha  # (N, S)

        # x_ij-weighted sums over i → (S,)
        log_ps_raw = (X * el_alpha_ni).sum(axis=0)           # (S,)
        log_pu_raw = (X * E_log_beta[None, :]).sum(axis=0)   # (S,)

        log_odds = np.clip(E_xi1 + log_ps_raw - E_xi2 - log_pu_raw, -500, 500)
        # f shape must stay (N, S) for xi_star update compatibility
        f_vec   = np.clip(expit(log_odds), EPS, 1 - EPS)     # (S,)
        self.f  = np.broadcast_to(f_vec, (self.N, self.S)).copy()

    def _update_theta(self):
        r_k              = self.r.sum(axis=0)
        self.theta       = np.maximum(1.0 + r_k, NumericalStability.EPS)
        cumsum_r         = np.cumsum(r_k)
        self.theta_prime = np.maximum(
            self.nu + (cumsum_r[-1] - cumsum_r), NumericalStability.EPS
        )
        self.theta_prime[-1] = max(self.nu, NumericalStability.EPS)

    def _update_xi_star(self):
        self.xi_star[:, 0] = np.maximum(self.xi_1 + self.f.sum(axis=0),
                                        NumericalStability.EPS)
        self.xi_star[:, 1] = np.maximum(self.xi_2 + (1 - self.f).sum(axis=0),
                                        NumericalStability.EPS)

    def _update_lambda_star(self, X):
        """
        Exact conjugate CAVI update:
          λ_kj = ζ + Σ_i r_ik · f_j · x_ij  =  ζ + r^T @ (f * X)
        """
        f_avg = self.f.mean(axis=0)          # (S,) per-feature mean
        fX    = f_avg[None, :] * X           # (N, S)
        self.lambda_star = np.maximum(
            self.zeta + self.r.T @ fX,       # (K, N) @ (N, S) = (K, S)
            0.1
        )

    def _update_iota_star(self, X):
        """
        Exact conjugate CAVI update:
          ι_j = η + Σ_i (1 − f_j) · x_ij
        """
        f_avg = self.f.mean(axis=0)   # (S,)
        self.iota_star = np.maximum(
            self.eta + ((1.0 - f_avg)[None, :] * X).sum(axis=0),
            0.1
        )

    # ─────────────────────────────────────────────────────────────────────
    # ELBO
    # ─────────────────────────────────────────────────────────────────────

    def _compute_elbo(self, X):
        """Fast ELBO using the same matmul expected log-likelihood."""
        EPS      = NumericalStability.EPS
        E_log_pi = self._E_log_pi()
        ll       = self._expected_log_lik(X)   # (N, K) — one matmul

        elbo  = float(np.sum(self.r * (E_log_pi[None, :] + ll)))
        elbo -= float(np.sum(self.r * np.log(np.maximum(self.r, EPS))))
        return elbo

    # ─────────────────────────────────────────────────────────────────────
    # Pruning 
    # ─────────────────────────────────────────────────────────────────────

    def _compute_weights(self):
        """Stick-breaking mixing weights E[π_k] via log-cumprod."""
        E_gamma  = self.theta / (self.theta + self.theta_prime)
        log_w    = np.log(np.maximum(E_gamma, NumericalStability.EPS))
        log_1mg  = np.log(np.maximum(1 - E_gamma, NumericalStability.EPS))
        log_w   += np.concatenate([[0.0], np.cumsum(log_1mg[:-1])])
        return np.exp(log_w)

    def _prune_empty_clusters(self):
        """
        Prune clusters that are genuinely empty by BOTH criteria.  

        A cluster is kept if:
          weight  > prune_threshold          (stick-breaking support)
          AND  sample count > min_cluster_size  (data support)
        """
        weights       = self._compute_weights()
        cluster_sizes = self.r.sum(axis=0)

        # AND logic: must pass both tests   [FIX 1 — was OR]
        keep = (weights > self.prune_threshold) & \
               (cluster_sizes > self._min_cluster_size)

        # Hard floor: always keep at least _min_clusters
        n_keep = keep.sum()
        if n_keep < self._min_clusters:
            # Rescue the top-_min_clusters clusters by weight
            top_idx       = np.argsort(weights)[-self._min_clusters:]
            keep          = np.zeros(self.K, dtype=bool)
            keep[top_idx] = True
            n_keep        = self._min_clusters

        if n_keep < self.K:
            if self.verbose >= 1:
                removed = self.K - n_keep
                print(f"  Pruning: {self.K} → {n_keep} clusters "
                      f"(removed {removed}; min_K={self._min_clusters})")
            self.K             = n_keep
            self.r             = self.r[:, keep]
            self.theta         = self.theta[keep]
            self.theta_prime   = self.theta_prime[keep]
            self.lambda_star   = self.lambda_star[keep]
            self.r            /= self.r.sum(axis=1, keepdims=True)
            self._clear_cache()
            self._pruned_at_least_once = True
            return True
        return False

    # ─────────────────────────────────────────────────────────────────────
    # Main fit loop  
    # ─────────────────────────────────────────────────────────────────────

    def fit(self, X):
        """
        Fit via coordinate ascent variational inference.
        """
        X            = check_array(X, dtype=np.float64)
        random_state = check_random_state(self.random_state)

        self._initialize_parameters(X, random_state)

        if self.verbose >= 1:
            print(f"\nStarting CAVI  (v2)")
            print("=" * 70)

        t0 = time()

        for iteration in range(1, self.max_iter + 1):
            self.n_iter = iteration
            self._clear_cache()

            # ── CAVI updates ──────────────────────────────────────────────
            self._update_r(X)
            self._update_f(X)
            self._update_theta()
            self._update_xi_star()
            self._update_lambda_star(X)
            self._update_iota_star(X)

            # ── Pruning: start earlier and run more frequently  [FIX 3] ──
            if iteration >= self.prune_start and iteration % self.prune_every == 0:
                self._prune_empty_clusters()

            # ── ELBO + convergence check ──────────────────────────────────
            if iteration % 10 == 0:
                elbo = self._compute_elbo(X)
                self.elbo_history.append(elbo)

                if self.verbose >= 1:
                    print(f"Iter {iteration:4d}: ELBO = {elbo:14.2f}, "
                          f"K = {self.K}, Time = {time()-t0:.1f}s")

                # Only check convergence after at least one pruning pass  [FIX 3]
                if self._pruned_at_least_once and len(self.elbo_history) >= 3:
                    recent  = self.elbo_history[-3:]
                    changes = [abs(recent[i] - recent[i-1]) /
                               (abs(recent[i]) + 1e-10)
                               for i in range(1, len(recent))]
                    if all(c < self.tol for c in changes):
                        self.converged = True
                        if self.verbose >= 1:
                            print(f"\n✓ Converged at iteration {iteration}")
                        break

        # Final pruning pass
        self._prune_empty_clusters()
        self.weights_ = self._compute_weights()

        if self.verbose >= 1:
            print(f"\nFinal results:")
            print(f"  Number of clusters : {self.K}")
            print(f"  Mixing weights     : {np.round(self.weights_, 4)}")
            print(f"  Total time         : {time()-t0:.2f}s")
            print(f"  Converged          : {self.converged}")

        return self

    # ─────────────────────────────────────────────────────────────────────
    # Prediction and inspection
    # ─────────────────────────────────────────────────────────────────────

    def predict(self, X):
        X     = check_array(X, dtype=np.float64)
        N_new = X.shape[0]
        f_new = np.tile(self.f.mean(axis=0), (N_new, 1))
        N_orig, f_orig, r_orig = self.N, self.f, self.r
        self.N, self.f, self.r = N_new, f_new, np.zeros((N_new, self.K))
        self._clear_cache()
        self._update_r(X)
        labels = self.r.argmax(axis=1)
        self.N, self.f, self.r = N_orig, f_orig, r_orig
        return labels

    def get_selected_features(self, threshold=0.5):
        selected = {}
        for i in range(self.N):
            idx = np.where(self.f[i] > threshold)[0]
            if len(idx) > 0:
                selected[i] = {'indices': idx.tolist(),
                                'probabilities': self.f[i, idx].tolist()}
        return selected

    def get_cluster_signatures(self, threshold=0.5):
        signatures = {}
        for k in range(self.K):
            mask = self.r[:, k] > 0.5
            if mask.sum() > 0:
                avg_sel   = self.f[mask].mean(axis=0)
                important = np.where(avg_sel > threshold)[0]
                if len(important) > 0:
                    lam_sum = self.lambda_star[k].sum()
                    alpha_k = self.lambda_star[k] / lam_sum
                    signatures[k] = {
                        'features':       important.tolist(),
                        'selection_probs': avg_sel[important].tolist(),
                        'alpha_values':    alpha_k[important].tolist(),
                        'n_samples':       int(mask.sum()),
                    }
        return signatures


if __name__ == "__main__":
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    print("Smoke-test — DMM_SVVS_Variational_v2")
    print("=" * 70)

    rng = np.random.default_rng(42)
    N, S, K_true = 150, 200, 3

    # Block-structured DMM data (moderate difficulty)
    alpha = np.full((K_true, S), 0.1)
    block = S // K_true
    for k in range(K_true):
        alpha[k, k*block:(k+1)*block] = 3.0
    true_labels = rng.choice(K_true, size=N)
    X = np.array([rng.multinomial(5000, rng.dirichlet(alpha[true_labels[i]]))
                  for i in range(N)], dtype=float)

    model = DMM_SVVS_Variational_v2(
        K_max=10, nu='auto', max_iter=300, verbose=1, random_state=42
    )
    model.fit(X)

    pred = model.predict(X)
    print(f"\nARI = {adjusted_rand_score(true_labels, pred):.3f}")
    print(f"NMI = {normalized_mutual_info_score(true_labels, pred):.3f}")
    print(f"K estimated = {model.K}  (true K = {K_true})")
