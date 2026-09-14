#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Search Hyperparameter Optimisation for DMM_SVVS_VarQITE_QAVB_JAX (v3.1)
================================================================================

Mirrors `random_search_qavb_v2.py` in structure and API, but targets the JAX
+ JIT + vmap VarQITE model — DMM_SVVS_VarQITE_QAVB_JAX from
DMM_SVVS_Variational_QAVB_v3_1 — and maximises ARI against ground-truth
labels by random search over the same six clustering-relevant hyperparameters.

The boolean toggle use_trigamma_correction is FIXED to False throughout the
search (per the user's memory: trigamma correction off for QAVB).

Hyperparameters searched
------------------------
  K_max            int      [low, high]          uniform integer
  nu               float    (0, ∞)               log-uniform float
  selection_prior  float    (0, 1)               uniform float
  prune_threshold  float    (0, 1)               log-uniform float
  tau1             int      [low, high]          uniform integer (quantum phase end)
  tau2             int      tau1 + delta         uniform integer (thermal phase end)

Hyperparameters held fixed
--------------------------
  use_trigamma_correction = False   (per user memory)
  zeta, eta, xi_1, xi_2   = 1.0
  beta0                   = 30.0
  s0                      = 1.0
  tol                     = 1e-4
  max_iter                = 400     (search budget; refit_best uses 600)
  prune_start             = 10
  prune_every             = 5
  min_clusters            = None
  # VarQITE-specific
  n_varqite_steps         = 40
  ansatz_depth            = 3
  mixer                   = "cyclic_shift"  (avoids the transverse-field
                                             phantom-coupling problem at any K)
  regularization          = 1e-4
  warm_start              = True
  init_perturbation       = 0.05
  metric_approx           = None
  # v3_fast layers 2-4 (inherited)
  metric_refresh          = 1       (exact per-step McLachlan, no caching)
  r_early_stop_tol        = None    (ignored on the JAX path anyway)
  dedup_n_clusters        = "auto"  (centroid-VarQITE for speed on large N)
  sample_batch_size       = None    (one vmap over all N)
  jit_warmup              = True
  # Padding-fidelity knobs (Fix A + Fix B from the supervisor's Rung-1 guide)
  phantom_penalty         = 10.0    (was 1000 — softens stiffness on padded K)
  decouple_phantom_mixer  = True    (inert for cyclic_shift; projects
                                     transverse-field onto K-block if used)
  # Integrator + multi-restart knobs (Fix C + Fix D from the gap audit)
  dtau_max                = 0.2     (clamps Euler step; auto-bumps
                                     n_varqite_steps to ≥75 at β=30)
  n_estep_restarts        = 3       (multi-restart McLachlan; per E-step,
                                     keeps lowest-⟨H⟩ trajectory. Drop to 1
                                     for a faster but lower-quality search.)

Usage
-----
    from random_search_qavb_v3_1 import random_search, refit_best, print_top_k

    results = random_search(
        X           = X,
        true_labels = true_labels,
        n_trials    = 30,
        master_seed = 42,
        verbose     = True,
    )

    print_top_k(results, top_k=10)

    best_model = refit_best(
        X           = X,
        true_labels = true_labels,
        best_result = results["best_result"],
        n_restarts  = 3,
    )

Returns
-------
random_search() returns a dict with:
    best_config  : dict  — hyperparameters of the best trial
    best_result  : dict  — full record of the best trial (ari, nmi, K, ...)
    all_results  : list  — every trial record sorted by ARI descending
    model_class  : str   — "VarQITE_QAVB_JAX_v3_1"
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
    # ── Padding-fidelity knobs (Fix A + Fix B; supervisor's Rung-1 guide) ─
    # Fix A: phantom_penalty=10 (was 1e3) restores max-prob 1.0000 on every
    # padded K in {3,5,6,7} at depth=3, n_steps=80 in the unit probe
    # (probe_padding_rung0_rung1.py). Non-monotonic in penalty — 10 is the
    # specific operating point, not "softer is always better."
    # Fix B: decouple_phantom_mixer=True is inert for the cyclic_shift mixer
    # (v2 already zeros phantom rows/cols at line 1169-1177) but is the
    # supervisor's recommended setting; if a trial switches mixer to
    # transverse_field, the K-block projection then matters.
    phantom_penalty         = 10.0,
    decouple_phantom_mixer  = True,
    # ── Integrator-stability knob (Fix C — Issue #4 from the gap audit) ──
    # dtau_max=0.2 clamps the McLachlan Euler step size. Without it, at
    # beta0=30 and the user-requested n_varqite_steps=40 above, dtau =
    # (30/2)/40 = 0.375, which exceeds the stability threshold ~0.4 found
    # in probe_v3_1_dtau_sweep.py and the trajectory overshoots / collapses
    # to ~uniform. With the clamp, n_varqite_steps is internally bumped to
    # ceil((beta0/2)/dtau_max) = ceil(15/0.2) = 75 at beta0=30 — the user
    # n_varqite_steps above (40) is a LOWER BOUND. Pass dtau_max=None to
    # disable the clamp.
    dtau_max                = 0.2,
    # ── Multi-restart E-step (Fix D — Issue #2 from the gap audit) ───────
    # n_estep_restarts=3 runs the McLachlan trajectory 3 times per E-step
    # from different theta_0 perturbations and selects per sample the one
    # with the lowest <H>(theta_final). On Rung-4 hard data, R=3 lifts
    # median ARI 0.492 -> 0.574 (BEATS QuBy-expm 0.556). R=5 plateaus.
    # R=1 disables the fix (legacy). Wall-time cost is ~3x per fit at R=3.
    #
    # For a FAST search prioritising hyperparameter coverage over per-trial
    # quality, set this to 1. For a HIGH-QUALITY search at ~3x the cost,
    # leave it at 3. Recommended for the difficult-regime search.
    n_estep_restarts        = 2,
)


# ─────────────────────────────────────────────────────────────────────────────
# Sampling helpers  (identical to v2 — kept local so both files stand alone)
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

    Log-uniform sampling allocates equal probability mass per decade,
    appropriate for scale parameters such as nu and prune_threshold.
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
    """
    Fit one DMM_SVVS_VarQITE_QAVB_JAX configuration and return ARI, NMI,
    K_estimated, elapsed.

    On any exception (numerical failure, invalid config) the trial returns
    ARI = NMI = -1 with the error message attached.

    Parameters
    ----------
    X            : (N, S) float array
    true_labels  : (N,)   int array
    config       : dict   — sampled hyperparameters
    trial_seed   : int    — random_state for this trial

    Returns
    -------
    dict with keys: ari, nmi, K_estimated, elapsed, config, trial_seed, error
    """
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
    """
    Random search over DMM_SVVS_VarQITE_QAVB_JAX hyperparameters,
    maximising ARI.

    use_trigamma_correction is fixed to False throughout the search.

    Parameters
    ----------
    X : np.ndarray, shape (N, S)
        Count data matrix.
    true_labels : np.ndarray, shape (N,)
        Ground-truth cluster labels for ARI evaluation.
    n_trials : int
        Number of random configurations to evaluate.
    K_max_range : tuple (int_low, int_high)
        Truncation level range.  Both ends inclusive.
    nu_range : tuple (float_low, float_high)
        DP concentration parameter range.  Sampled log-uniformly.
    selection_prior_range : tuple (float_low, float_high)
        Initial feature-selection warm-start.  Sampled uniformly.
    prune_threshold_range : tuple (float_low, float_high)
        Cluster deletion threshold.  Sampled log-uniformly.
    tau1_range : tuple (int_low, int_high)
        Quantum-annealing phase length, in iterations.
    tau2_delta_range : tuple (int_low, int_high)
        Additional iterations after tau1 for the thermal-annealing phase.
        Actual tau2 = tau1 + delta.
    master_seed : int
        Seed for the search RNG — makes the entire run reproducible.
    verbose : bool
        Print a live per-trial progress table if True.

    Returns
    -------
    dict with keys:
        best_config   : dict  — hyperparameters of the best trial
        best_result   : dict  — full result record of the best trial
        all_results   : list  — all trial records sorted by ARI descending
        model_class   : str   — "VarQITE_QAVB_JAX_v3_1"
    """
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
    """
    Re-fit DMM_SVVS_VarQITE_QAVB_JAX using the best configuration found by
    random_search, running multiple independent restarts and keeping the
    one with the highest ARI.

    The original trial_seed that produced the best ARI during the search is
    always used as one of the restart seeds, guaranteeing the search result
    is reproduced at minimum.

    Parameters
    ----------
    X            : np.ndarray, shape (N, S)
    true_labels  : np.ndarray, shape (N,)
    best_result  : dict
        The dict under results["best_result"] returned by random_search().
        Must contain keys "config" and "trial_seed".
    n_restarts   : int
        Total number of independent random restarts.
    max_iter     : int
        Maximum CAVI iterations per restart (higher than search budget).
    verbose      : bool

    Returns
    -------
    Fitted DMM_SVVS_VarQITE_QAVB_JAX instance with the highest ARI across
    all restarts.
    """
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
