#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Search Hyperparameter Optimisation for DMM_SVVS_VarQITE_QAVB_JAX (v3.1)
================================================================================

"""

import json
import time as _time
import warnings

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from DMM_SVVS_Variational_QAVB_v3_1fast import DMM_SVVS_VarQITE_QAVB_JAX

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Default search ranges  (same defaults as random_search_qavb_v2.py)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_K_MAX_RANGE           = (3, 15)        # int,   uniform
DEFAULT_NU_RANGE              = (0.01, 2.0)    # float, log-uniform
DEFAULT_SELECTION_PRIOR_RANGE = (0.05, 0.95)   # float, uniform
DEFAULT_PRUNE_THRESHOLD_RANGE = (1e-4, 0.15)   # float, log-uniform
DEFAULT_TAU1_RANGE            = (30, 120)      # int,   uniform (quantum phase end)
DEFAULT_TAU2_DELTA_RANGE      = (30, 150)      # int,   uniform (tau2 = tau1 + delta)

# Fixed parameters that are NOT tuned during the search.
# use_trigamma_correction is fixed to False (user memory).
FIXED_PARAMS = dict(
    zeta                    = 1.0,
    eta                     = 1.0,
    xi_1                    = 1.0,
    xi_2                    = 1.0,
    beta0                   = 30.0,
    s0                      = 1.0,
    tol                     = 1e-4,
    max_iter                = 400,    # search budget; refit uses larger
    prune_start             = 10,
    prune_every             = 5,
    min_clusters            = None,
    verbose                 = 0,
    use_trigamma_correction = False,  # FIXED per user memory
    # ── VarQITE-specific (match the class defaults) ──────────────────────
    n_varqite_steps         = 40,
    ansatz_depth            = 3,
    mixer                   = "transverse_field",
    regularization          = 1e-4,
    warm_start              = True,
    metric_approx           = None,
    init_perturbation       = 0.05,
    # ── v3_fast Layers 2-4 (inherited) ──────────────────────────────────
    r_early_stop_tol        = None,   # ignored on JAX path
    metric_refresh          = 1,
    dedup_n_clusters        = None,
    # ── v3.1 JAX knobs ───────────────────────────────────────────────────
    sample_batch_size       = None,
    jit_warmup              = True,
    phantom_penalty         = 10.0,
    decouple_phantom_mixer  = True,
    dtau_max                = 0.2,
    n_estep_restarts        = 2,
)


# ─────────────────────────────────────────────────────────────────────────────
# Sampling helpers  
# ─────────────────────────────────────────────────────────────────────────────

def _sample_int(rng, low, high):
    """Sample an integer uniformly from [low, high] (both ends inclusive)."""
    return int(rng.integers(int(low), int(high) + 1))


def _sample_uniform(rng, low, high):
    """Sample a float uniformly from [low, high)."""
    return float(rng.uniform(float(low), float(high)))


def _sample_loguniform(rng, low, high):
    """
    Sample a float log-uniformly from [low, high).
    """
    log_low  = np.log(float(low))
    log_high = np.log(float(high))
    return float(np.exp(rng.uniform(log_low, log_high)))


def _sample_config(rng,
                   K_max_range,
                   nu_range,
                   selection_prior_range,
                   prune_threshold_range,
                   tau1_range,
                   tau2_delta_range):
    """
    Draw one random configuration from the given search ranges.

    tau2 is sampled as tau1 + delta with delta drawn from tau2_delta_range,
    guaranteeing tau2 > tau1 (thermal phase always follows quantum phase).

    Returns
    -------
    dict with keys: K_max, nu, selection_prior, prune_threshold, tau1, tau2
    """
    K_max           = _sample_int(rng, *K_max_range)
    nu              = _sample_loguniform(rng, *nu_range)
    selection_prior = _sample_uniform(rng, *selection_prior_range)
    prune_threshold = _sample_loguniform(rng, *prune_threshold_range)
    tau1            = _sample_int(rng, *tau1_range)
    delta           = _sample_int(rng, *tau2_delta_range)
    tau2            = tau1 + delta

    return {
        "K_max":           K_max,
        "nu":              nu,
        "selection_prior": selection_prior,
        "prune_threshold": prune_threshold,
        "tau1":            tau1,
        "tau2":            tau2,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single trial
# ─────────────────────────────────────────────────────────────────────────────

def _run_trial(X, true_labels, config, trial_seed):
    
    params = {**FIXED_PARAMS, **config, "random_state": trial_seed}
    t0     = _time.time()
    error  = None

    try:
        model = DMM_SVVS_VarQITE_QAVB_JAX(**params)
        model.fit(X)
        pred  = model.predict(X)
        ari   = float(adjusted_rand_score(true_labels, pred))
        nmi   = float(normalized_mutual_info_score(true_labels, pred))
        K_est = int(model.K)
    except Exception as exc:
        ari, nmi, K_est = -1.0, -1.0, -1
        error = str(exc)

    elapsed = round(_time.time() - t0, 2)

    return {
        "ari":         ari,
        "nmi":         nmi,
        "K_estimated": K_est,
        "elapsed":     elapsed,
        "config":      config,
        "trial_seed":  trial_seed,
        "error":       error,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main random-search function
# ─────────────────────────────────────────────────────────────────────────────

def random_search(X,
                  true_labels,
                  n_trials                = 50,
                  K_max_range             = DEFAULT_K_MAX_RANGE,
                  nu_range                = DEFAULT_NU_RANGE,
                  selection_prior_range   = DEFAULT_SELECTION_PRIOR_RANGE,
                  prune_threshold_range   = DEFAULT_PRUNE_THRESHOLD_RANGE,
                  tau1_range              = DEFAULT_TAU1_RANGE,
                  tau2_delta_range        = DEFAULT_TAU2_DELTA_RANGE,
                  master_seed             = 42,
                  verbose                 = True):
    
    X           = np.asarray(X, dtype=float)
    true_labels = np.asarray(true_labels)
    master_rng  = np.random.default_rng(int(master_seed))

    # ── Header ────────────────────────────────────────────────────────────
    if verbose:
        print(f"\nDMM-SVVS VarQITE_QAVB_JAX (v3.1) — Random Hyperparameter "
              f"Search  (scored by ARI)")
        print("=" * 100)
        print(f"  n_trials      = {n_trials},   master_seed = {master_seed}")
        print(f"  K_max            : {K_max_range}   [uniform int]")
        print(f"  nu               : {nu_range}   [log-uniform, DP concentration]")
        print(f"  selection_prior  : {selection_prior_range}   [uniform float]")
        print(f"  prune_threshold  : {prune_threshold_range}   [log-uniform]")
        print(f"  tau1             : {tau1_range}   [uniform int, quantum phase end]")
        print(f"  tau2             : tau1 + delta, delta ∈ {tau2_delta_range}  "
              f"[thermal phase length]")
        fixed_display = {k: v for k, v in FIXED_PARAMS.items() if k != "verbose"}
        print(f"  Fixed: {json.dumps(fixed_display, default=str)}")
        print("=" * 100)
        _print_row_header()

    all_results = []
    best_ari    = -np.inf
    best_result = None
    best_config = None

    for trial in range(1, n_trials + 1):
        config     = _sample_config(
                         master_rng,
                         K_max_range,
                         nu_range,
                         selection_prior_range,
                         prune_threshold_range,
                         tau1_range,
                         tau2_delta_range)
        trial_seed = int(master_rng.integers(0, 2**31))

        result = _run_trial(X, true_labels, config, trial_seed)
        all_results.append(result)

        is_best = result["ari"] > best_ari
        if is_best:
            best_ari    = result["ari"]
            best_result = result
            best_config = config

        if verbose:
            _print_trial_row(trial, result, is_best)

    # Sort by ARI descending; break ties by NMI
    all_results.sort(key=lambda r: (r["ari"], r["nmi"]), reverse=True)

    if verbose:
        print("=" * 100)
        print(f"\n  Best ARI    : {best_ari:.4f}")
        print(f"  Best NMI    : {best_result['nmi']:.4f}")
        print(f"  Best K_est  : {best_result['K_estimated']}")
        print(f"  Best seed   : {best_result['trial_seed']}")
        print(f"  Best config :")
        for k, v in best_config.items():
            if isinstance(v, float):
                print(f"      {k:<22s} = {v:.6f}")
            else:
                print(f"      {k:<22s} = {v}")
        print(f"      {'random_state':<22s} = {best_result['trial_seed']}")
        print(f"      {'use_trigamma_correction':<22s} = False  (fixed)")

    return {
        "best_config":  best_config,
        "best_result":  best_result,
        "all_results":  all_results,
        "model_class":  "VarQITE_QAVB_JAX_v3_1",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Final refit with best config
# ─────────────────────────────────────────────────────────────────────────────

def refit_best(X,
               true_labels,
               best_result,
               n_restarts   = 5,
               max_iter     = 600,
               verbose      = True):
    
    X           = np.asarray(X, dtype=float)
    true_labels = np.asarray(true_labels)

    best_config = best_result["config"]
    trial_seed  = best_result["trial_seed"]

    # Safety clamp: a very large prune_threshold from a wide search range can
    # delete all clusters on refit.  Cap at 0.15 (same convention as v2).
    safe_config = dict(best_config)
    if safe_config.get("prune_threshold", 0.0) > 0.15:
        if verbose:
            old_val = safe_config["prune_threshold"]
            print(f"  [refit_best] Clamping prune_threshold {old_val:.6f} → 0.15")
        safe_config["prune_threshold"] = 0.15

    # Build restart seeds: original trial seed first, then derived extras.
    rng_extra   = np.random.default_rng(trial_seed + 1)
    n_extra     = max(n_restarts - 1, 0)
    extra_seeds = rng_extra.integers(0, 2**31, size=n_extra).tolist()
    all_seeds   = [trial_seed] + extra_seeds   # length == n_restarts

    if verbose:
        print(f"\nFinal refit (VarQITE_QAVB_JAX v3.1) — "
              f"n_restarts={n_restarts},  max_iter={max_iter}")
        print("=" * 80)
        print(f"  Config:")
        for k, v in safe_config.items():
            if isinstance(v, float):
                print(f"      {k:<22s} = {v:.6f}")
            else:
                print(f"      {k:<22s} = {v}")
        print(f"      {'use_trigamma_correction':<22s} = False  (fixed)")
        print()
        print(f"  {'Restart':>8}  {'Seed':>12}  {'ARI':>7}  "
              f"{'NMI':>7}  {'K':>4}  Note")
        print("  " + "-" * 64)

    best_model = None
    best_ari   = -np.inf
    best_nmi   = -np.inf

    for idx, seed in enumerate(all_seeds):
        params = {
            **FIXED_PARAMS,
            **safe_config,
            "max_iter":     max_iter,
            "random_state": seed,
            "verbose":      0,
        }
        try:
            m     = DMM_SVVS_VarQITE_QAVB_JAX(**params)
            m.fit(X)
            pred  = m.predict(X)
            ari_r = float(adjusted_rand_score(true_labels, pred))
            nmi_r = float(normalized_mutual_info_score(true_labels, pred))

            note = "← original trial seed" if idx == 0 else ""
            if verbose:
                marker = " *" if ari_r > best_ari else "  "
                print(f"  {idx+1:>8d}  {seed:>12d}  {ari_r:>7.4f}  "
                      f"{nmi_r:>7.4f}  {m.K:>4d}  {note}{marker}")

            # Keep the restart with the highest ARI; break ties with NMI
            if ari_r > best_ari or (ari_r == best_ari and nmi_r > best_nmi):
                best_ari   = ari_r
                best_nmi   = nmi_r
                best_model = m

        except Exception as exc:
            if verbose:
                print(f"  {idx+1:>8d}  {seed:>12d}  FAILED: {exc}")

    if best_model is None:
        raise RuntimeError("All restarts failed — check config and data.")

    if verbose:
        pred = best_model.predict(X)
        ari  = adjusted_rand_score(true_labels, pred)
        nmi  = normalized_mutual_info_score(true_labels, pred)
        print(f"\n  Best refit ARI  = {ari:.4f}")
        print(f"  Best refit NMI  = {nmi:.4f}")
        print(f"  K estimated     = {best_model.K}")

    return best_model


# ─────────────────────────────────────────────────────────────────────────────
# Summary table helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_row_header():
    print(f"  {'#':>5}  {'ARI':>7}  {'NMI':>7}  {'K_est':>5}  {'sec':>6}  "
          f"{'K_max':>5}  {'nu':>8}  {'sel':>6}  {'prune':>8}  "
          f"{'tau1':>5}  {'tau2':>5}  {'seed':>12}")
    print("  " + "-" * 98)


def _print_trial_row(trial, result, is_best):
    c      = result["config"]
    marker = " ◀ best" if is_best else ""
    err    = f"  ERROR: {result['error']}" if result["error"] else ""
    print(f"  {trial:5d}  {result['ari']:7.4f}  {result['nmi']:7.4f}  "
          f"{result['K_estimated']:5d}  {result['elapsed']:6.1f}  "
          f"{c['K_max']:5d}  {c['nu']:8.4f}  "
          f"{c['selection_prior']:6.3f}  {c['prune_threshold']:8.5f}  "
          f"{c['tau1']:5d}  {c['tau2']:5d}  "
          f"{result['trial_seed']:>12d}{marker}{err}")


def print_top_k(search_result, top_k=10):
    """
    Print a ranked summary table of the top-k configurations by ARI.

    Parameters
    ----------
    search_result : dict  — return value of random_search()
    top_k         : int   — number of configurations to show
    """
    results = search_result["all_results"][:top_k]
    n_shown = len(results)

    print(f"\nTop-{n_shown} configurations — VarQITE_QAVB_JAX v3.1 (by ARI):")
    print(f"  {'Rank':>4}  {'ARI':>7}  {'NMI':>7}  {'K_est':>5}  "
          f"{'K_max':>5}  {'nu':>8}  {'sel':>6}  {'prune':>8}  "
          f"{'tau1':>5}  {'tau2':>5}  {'seed':>12}")
    print("  " + "-" * 94)
    for rank, r in enumerate(results, 1):
        c = r["config"]
        print(f"  {rank:4d}  {r['ari']:7.4f}  {r['nmi']:7.4f}  "
              f"{r['K_estimated']:5d}  "
              f"{c['K_max']:5d}  {c['nu']:.6f}  "
              f"{c['selection_prior']:.6f}  {c['prune_threshold']:.6f}  "
              f"{c['tau1']:5d}  {c['tau2']:5d}  {r['trial_seed']:>12d}")


def save_results(search_result, path=None):
    """
    Save all trial results to a JSON file for later analysis.

    Parameters
    ----------
    search_result : dict  — return value of random_search()
    path          : str or None
        Output file path.  Defaults to random_search_varqite_qavb_jax_v3_1_results.json
    """
    if path is None:
        path = "random_search_varqite_qavb_jax_v3_1_results.json"

    with open(path, "w") as fh:
        json.dump(search_result["all_results"], fh, indent=2, default=str)
    print(f"  Results saved to {path}  ({len(search_result['all_results'])} trials)")


# ─────────────────────────────────────────────────────────────────────────────
# Demo — run when executed directly
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Generate synthetic block-structured DMM data ──────────────────────
    rng_data = np.random.default_rng(0)
    N, S, K_true = 80, 150, 3

    alpha_true = np.full((K_true, S), 0.1)
    block = S // K_true
    for k in range(K_true):
        alpha_true[k, k * block:(k + 1) * block] = 3.0
    true_labels = rng_data.choice(K_true, size=N)
    X = np.array(
        [rng_data.multinomial(3000, rng_data.dirichlet(alpha_true[true_labels[i]]))
         for i in range(N)],
        dtype=float,
    )
    print(f"Synthetic data: N={N}, S={S}, K_true={K_true}, "
          f"class sizes={np.bincount(true_labels).tolist()}")

    # ── Search ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RANDOM SEARCH: DMM_SVVS_VarQITE_QAVB_JAX (v3.1)")
    print("=" * 60)
    results = random_search(
        X                     = X,
        true_labels           = true_labels,
        n_trials              = 8,
        K_max_range           = (3, 8),
        nu_range              = (0.01, 2.0),
        selection_prior_range = (0.05, 0.95),
        prune_threshold_range = (1e-4, 0.15),
        tau1_range            = (20, 60),
        tau2_delta_range      = (30, 80),
        master_seed           = 42,
        verbose               = True,
    )
    print_top_k(results, top_k=5)
    save_results(results)

    best_model = refit_best(
        X           = X,
        true_labels = true_labels,
        best_result = results["best_result"],
        n_restarts  = 2,
        max_iter    = 300,
        verbose     = True,
    )

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Final result")
    print("=" * 60)
    pred = best_model.predict(X)
    ari  = adjusted_rand_score(true_labels, pred)
    nmi  = normalized_mutual_info_score(true_labels, pred)
    print(f"  VarQITE_QAVB_JAX v3.1   ARI={ari:.4f}  NMI={nmi:.4f}  "
          f"K={best_model.K}  (true K = {K_true})")
