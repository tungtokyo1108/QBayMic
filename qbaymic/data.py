"""
Synthetic microbiome count data with controlled difficulty.
============================================================
A single Dirichlet–multinomial mixture generator whose difficulty is set by two
interpretable knobs — ``separation`` (between-cluster signal) and
``zero_inflation`` (sparsity that erodes it). Together with a fixed informative
-taxon fraction these place an instance anywhere on the signal-fraction axis σ
(see :mod:`qbaymic.barrier`), spanning the unrecoverable, advantage, and
signal-dominated regimes.
"""
from __future__ import annotations

from . import _engines  # noqa: F401  (installs the engine path shim)

import numpy as np

from data_generators import generate_high_overlap_clusters as _gen


def make_counts(n_samples=400, n_taxa=5000, n_clusters=3, separation=0.2,
                zero_inflation=0.80, signal_fraction=0.15, imbalance=0.0,
                seed=0):
    """Generate a synthetic count matrix with known cluster labels.

    Parameters
    ----------
    n_samples, n_taxa, n_clusters : int
        Matrix dimensions and the true number of clusters.
    separation : float in [0, 1], default=0.2
        Between-cluster signal (0 = identical clusters, 1 = well separated).
        The default 0.2, with the defaults below, gives σ ≈ 0.29 — the advantage
        band where the quantum E-step separates from the classical baselines.
    zero_inflation : float in [0, 1], default=0.80
        Fraction of independently zeroed entries (16S sparsity).
    signal_fraction : float, default=0.15
        Fraction of informative taxa.
    imbalance : float in [0, 1], default=0.0
        Cluster-size imbalance (0 = balanced).
    seed : int, default=0
        Generator seed.

    Returns
    -------
    X : ndarray (n_samples, n_taxa) of integer counts.
    y : ndarray (n_samples,) of true cluster labels in {0, ..., n_clusters-1}.
    """
    X, y, _ = _gen(N=n_samples, S=n_taxa, K=n_clusters, seed=seed,
                   separation=separation, imbalance=imbalance,
                   zero_inflation=zero_inflation,
                   signal_fraction=signal_fraction)
    return np.asarray(X), np.asarray(y)
