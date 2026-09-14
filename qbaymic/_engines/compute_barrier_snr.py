#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_barrier_snr.py — self-contained Barrier / SNR calculator
================================================================
Computes the "Barrier / SNR" quantities used in the report (§2.2 and the §6
re-diagnosis), for any generate_high_overlap_clusters config:

    B          = N^tot · JS_lambda          (Thm 1 — barrier height)
    B_signal   = N^tot · JS_signal          (Thm 2 / Prop 3 — separation signal)
    B_noise    = B - B_signal               (estimation-noise term)
    N^tot      = post-zero-inflation pair counts
    sigma      = B_signal / B               (Thm 4 — difficulty axis = SNR ratio)
    SNR        = sigma / (1 - sigma)        = B_signal / B_noise

FULLY SELF-CONTAINED: the data generator and all information-theory helpers are
inlined here; this file imports nothing from the other validation scripts.
Run directly for the three report cases, or import barrier_snr() for any config.
"""
from __future__ import annotations
import sys
import numpy as np

EPS = 1e-300


# ── data generator (verbatim DGP from the repo / the v2 document) ─────────
def generate_high_overlap_clusters(N=200, S=100, K=4, seed=42,
                                    separation=1.0, imbalance=0.5,
                                    zero_inflation=0.15, signal_fraction=0.15):
    rng = np.random.default_rng(seed)
    separation      = float(np.clip(separation,      0.0, 1.0))
    imbalance       = float(np.clip(imbalance,       0.0, 1.0))
    zero_inflation  = float(np.clip(zero_inflation,  0.0, 0.99))
    signal_fraction = float(np.clip(signal_fraction, 1e-4, 1.0))

    conc = 10.0 ** (1.0 - 2.0 * imbalance)
    raw_pi = rng.dirichlet(np.ones(K) * conc)
    true_labels = rng.choice(K, size=N, p=raw_pi)
    for k in range(K):
        if (true_labels == k).sum() == 0:
            true_labels[rng.integers(N)] = k

    ALPHA_SIGNAL_MAX = 1.8
    ALPHA_CROSS      = 0.20
    ALPHA_BG         = 0.05
    alpha_signal = ALPHA_CROSS + (ALPHA_SIGNAL_MAX - ALPHA_CROSS) * separation

    S_signal = max(K, int(round(S * signal_fraction)))
    block    = S_signal // K
    alpha = np.full((K, S), ALPHA_BG)
    for k in range(K):
        start = k * block
        end   = start + block if k < K - 1 else S_signal
        alpha[k, start:end] = alpha_signal
        for kp in range(K):
            if kp != k:
                s2 = kp * block
                e2 = s2 + block if kp < K - 1 else S_signal
                alpha[k, s2:e2] = ALPHA_CROSS

    lib_r = 5
    lib_mean = 8_000
    lib_p = lib_r / (lib_r + lib_mean)
    lib_sizes = rng.negative_binomial(n=lib_r, p=lib_p, size=N)
    lib_sizes = np.maximum(lib_sizes, 100)

    X = np.zeros((N, S), dtype=np.float64)
    for i in range(N):
        k = true_labels[i]
        p_i = rng.dirichlet(alpha[k])
        X[i] = rng.multinomial(int(lib_sizes[i]), p_i)

    if zero_inflation > 0.0:
        zi_mask = rng.uniform(size=(N, S)) < zero_inflation
        X[zi_mask] = 0.0
    return X, true_labels


# ── information-theory helpers ────────────────────────────────────────────
def _norm(p):
    p = np.asarray(p, dtype=np.float64)
    s = p.sum()
    return p / s if s > 0 else p


def _shannon_H(p):
    p = _norm(p); m = p > 0
    return float(-np.sum(p[m] * np.log(p[m])))


def js_lambda(p_a, p_b, lam):
    """lambda-weighted Jensen-Shannon divergence (nats)."""
    p_a = _norm(p_a); p_b = _norm(p_b)
    p_M = lam * p_a + (1.0 - lam) * p_b
    return _shannon_H(p_M) - lam * _shannon_H(p_a) - (1.0 - lam) * _shannon_H(p_b)


def _population_profiles(S, K, sep, f):
    """Population cluster mean frequencies p_bar_k = alpha_k / alpha_0."""
    ALPHA_SIGNAL_MAX = 1.8; ALPHA_CROSS = 0.20; ALPHA_BG = 0.05
    alpha_signal = ALPHA_CROSS + (ALPHA_SIGNAL_MAX - ALPHA_CROSS) * sep
    S_signal = max(K, int(round(S * f)))
    block = S_signal // K
    alpha = np.full((K, S), ALPHA_BG)
    for k in range(K):
        start = k * block
        end = start + block if k < K - 1 else S_signal
        alpha[k, start:end] = alpha_signal
        for kp in range(K):
            if kp != k:
                s2 = kp * block
                e2 = s2 + block if kp < K - 1 else S_signal
                alpha[k, s2:e2] = ALPHA_CROSS
    pbar = alpha / alpha.sum(axis=1, keepdims=True)
    rho = alpha_signal / ALPHA_CROSS
    return pbar, rho


# ── the calculator ────────────────────────────────────────────────────────
def barrier_snr(N, S, K, sep, zi, f, seed=0):
    """Barrier / SNR quantities for one config (the two most-populated clusters)."""
    X, y = generate_high_overlap_clusters(
        N=N, S=S, K=K, seed=seed, separation=sep, imbalance=0.0,
        zero_inflation=zi, signal_fraction=f)
    sizes = np.bincount(y, minlength=K)
    a, b = np.argsort(sizes)[::-1][:2]

    # Empirical centers + counts on the pair.
    Xa = X[y == a].sum(axis=0); Xb = X[y == b].sum(axis=0)
    Na = float(Xa.sum()); Nb = float(Xb.sum())
    Ntot = Na + Nb
    lam = Na / Ntot
    p_a = _norm(Xa); p_b = _norm(Xb)

    B = Ntot * js_lambda(p_a, p_b, lam)                  # Thm 1 (height)

    pbar, rho = _population_profiles(S, K, sep, f)
    B_signal = Ntot * js_lambda(pbar[a], pbar[b], 0.5)   # Thm 2 / Prop 3
    B_noise = B - B_signal
    sigma = B_signal / B if B > 0 else 0.0               # Thm 4 (axis)
    SNR = sigma / (1.0 - sigma) if sigma < 1 else float("inf")

    return dict(N=N, S=S, K=K, sep=sep, zi=zi, f=f, rho=rho,
                B=B, B_signal=B_signal, B_noise=B_noise, Ntot=Ntot,
                sigma=sigma, SNR=SNR, sparsity=float((X == 0).mean()))


# The three report cases.
CASES = [
    ("Case 1  K=3 ZI=.80 sep=.2", dict(N=400, S=5000, K=3, sep=0.2, zi=0.80, f=0.15)),
    ("Case 2  K=4 ZI=.50 sep=.2", dict(N=400, S=5000, K=4, sep=0.2, zi=0.50, f=0.15)),
    ("Case 3  K=3 ZI=.80 sep=.8", dict(N=200, S=2000, K=3, sep=0.8, zi=0.80, f=0.15)),
]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 92)
    print("Barrier / SNR quantities  (Part A — exact; for report §2.2 and §6)")
    print("=" * 92)
    print(f"  {'case':<26} {'B':>9} {'B_signal':>9} {'B_noise':>9} "
          f"{'N^tot':>10} {'sigma':>6} {'SNR':>6}")
    print("  " + "-" * 84)
    rows = []
    for tag, cfg in CASES:
        d = barrier_snr(**cfg)
        rows.append((tag, d))
        print(f"  {tag:<26} {d['B']:>9,.0f} {d['B_signal']:>9,.0f} "
              f"{d['B_noise']:>9,.0f} {d['Ntot']:>10,.0f} "
              f"{d['sigma']:>6.3f} {d['SNR']:>6.2f}")

    print("\n--- markdown rows for report §6 (sigma | B_signal | SNR) ---")
    for tag, d in rows:
        print(f"| {tag} | {d['sigma']:.3f} | {d['B_signal']:,.0f} | {d['SNR']:.2f} |")


if __name__ == "__main__":
    main()
