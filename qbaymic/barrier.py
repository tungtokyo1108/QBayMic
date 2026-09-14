"""
Barrier diagnostic — the signal fraction σ that predicts recoverability.
========================================================================
The central quantity of the paper is the *signal fraction*

    σ = B_signal / B ,

where ``B`` is the clustering barrier height (a Jensen–Shannon divergence
between cluster profiles, scaled by the total counts) and ``B_signal`` is the
part of that barrier due to genuine between-cluster separation rather than
finite-sample noise. σ ∈ [0, 1] orders clustering *difficulty* independently of
the raw barrier height, and it predicts the regime:

    σ ≲ 0.2         unrecoverable  — no method succeeds
    σ ≈ 0.25–0.45   advantage band — the quantum E-step separates from classical
    σ ≳ 0.45        signal-dominated — all methods succeed (small quantum margin)

This module exposes the diagnostic for both settings used in the paper:

* :func:`signal_fraction` — for SYNTHETIC data whose generator profiles are
  known, σ is computed exactly (population JS for the signal, empirical JS for
  the barrier).
* :func:`signal_fraction_estimate` — for REAL data, where the true profiles are
  unknown, σ is estimated by debiasing the empirical barrier with a within-group
  split-half sampling null.

Both return a dict with ``sigma``, the barrier decomposition (``B``,
``B_signal``, ``B_noise``), and the total counts.
"""
from __future__ import annotations

from . import _engines  # noqa: F401  (installs the engine path shim)

import itertools
import numpy as np

from compute_barrier_snr import barrier_snr as _barrier_snr_synthetic


def signal_fraction(N, S, K, separation, zero_inflation,
                    signal_fraction=0.15, seed=0):
    """Exact signal fraction σ for a synthetic instance.

    Computes σ for data drawn from the paper's Dirichlet–multinomial generator,
    using the *known* population profiles for the signal term. Use this to place
    a synthetic configuration on the difficulty axis before running any method.

    Parameters
    ----------
    N, S, K : int
        Number of samples, taxa, and clusters.
    separation : float in [0, 1]
        Between-cluster separation knob (0 = identical clusters, 1 = far apart).
    zero_inflation : float in [0, 1]
        Fraction of independently zeroed entries.
    signal_fraction : float, default=0.15
        Fraction of informative taxa.
    seed : int, default=0
        Generator seed.

    Returns
    -------
    dict with keys ``sigma``, ``B``, ``B_signal``, ``B_noise``, ``Ntot``.
    """
    d = _barrier_snr_synthetic(N=N, S=S, K=K, sep=separation,
                               zi=zero_inflation, f=signal_fraction, seed=seed)
    return dict(sigma=float(d["sigma"]), B=float(d["B"]),
                B_signal=float(d["B_signal"]), B_noise=float(d["B_noise"]),
                Ntot=float(d["Ntot"]))


# ── real-data estimator (split-half debiasing), self-contained ───────────────
def _norm(v):
    v = np.asarray(v, dtype=np.float64)
    s = v.sum()
    return v / s if s > 0 else np.full_like(v, 1.0 / len(v))


def _js_lambda(p, q, lam):
    """λ-weighted Jensen–Shannon divergence (nats)."""
    p, q = np.asarray(p), np.asarray(q)
    m = lam * p + (1 - lam) * q
    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))
    return lam * _kl(p, m) + (1 - lam) * _kl(q, m)


def _split_half_js(Xk, rng, n_rep=20):
    """Within-group sampling null: split a cluster's samples into halves, pool
    counts, and measure JS between the halves — the JS expected between two
    draws of the *same* distribution (pure finite-sample noise)."""
    n = Xk.shape[0]
    if n < 2:
        return 0.0
    idx = np.arange(n)
    vals = []
    for _ in range(n_rep):
        rng.shuffle(idx)
        h = n // 2
        A = Xk[idx[:h]].sum(0)
        B = Xk[idx[h:]].sum(0)
        if A.sum() == 0 or B.sum() == 0:
            continue
        lam = A.sum() / (A.sum() + B.sum())
        vals.append(_js_lambda(_norm(A), _norm(B), lam))
    return float(np.mean(vals)) if vals else 0.0


def _barrier_pair(Xa, Xb, rng):
    a, b = Xa.sum(0), Xb.sum(0)
    Na, Nb = float(a.sum()), float(b.sum())
    Ntot = Na + Nb
    lam = Na / Ntot if Ntot > 0 else 0.5
    js_raw = _js_lambda(_norm(a), _norm(b), lam)      # inflated by sampling
    B = Ntot * js_raw
    bias = 0.5 * (_split_half_js(Xa, rng) + _split_half_js(Xb, rng))
    B_signal = Ntot * max(js_raw - bias, 0.0)         # debiased separation
    return dict(Ntot=Ntot, B=B, B_signal=B_signal, B_noise=B - B_signal,
                sigma=(B_signal / B) if B > 0 else 0.0)


def signal_fraction_estimate(X, labels, seed=0):
    """Estimate σ for a REAL count dataset given a (possibly approximate) label
    assignment, by debiasing the empirical barrier with a split-half null.

    For K > 2 groups the reported σ is the *minimum* over all pairwise σ values —
    the hardest pair gates difficulty. Use a preliminary partition (e.g. k-means
    on relative abundances, or one greedy-VB pass) when true labels are unknown;
    σ measures the separability of the *fitted* groups, so it need only be
    approximately correct.

    Parameters
    ----------
    X : ndarray (n_samples, n_taxa)
        Integer count matrix.
    labels : array-like (n_samples,)
        Group assignment.
    seed : int, default=0
        Seed for the split-half resampling null.

    Returns
    -------
    dict with keys ``sigma`` (== σ_min over pairs), ``sigma_pairs`` (list of
    per-pair σ), ``B``, ``B_signal``, ``B_noise`` (for the hardest pair),
    and ``K``.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    K = len(classes)
    if K < 2:
        raise ValueError("need at least two groups to compute σ")
    rng = np.random.default_rng(seed)
    pairs = []
    for a, b in itertools.combinations(classes, 2):
        Xa, Xb = X[labels == a], X[labels == b]
        if Xa.shape[0] < 2 or Xb.shape[0] < 2:
            continue
        pairs.append(_barrier_pair(Xa, Xb, rng))
    if not pairs:
        raise ValueError("no group pair had ≥2 samples on both sides")
    hardest = min(pairs, key=lambda d: d["sigma"])
    return dict(sigma=float(hardest["sigma"]),
                sigma_pairs=[float(p["sigma"]) for p in pairs],
                B=float(hardest["B"]), B_signal=float(hardest["B_signal"]),
                B_noise=float(hardest["B_noise"]), K=int(K))


def regime(sigma):
    """Name the recoverability regime for a signal fraction ``sigma``."""
    if sigma < 0.20:
        return "unrecoverable"
    if sigma <= 0.45:
        return "advantage band"
    return "signal-dominated"
