"""
The σ recoverability diagnostic on real data.
=============================================
Run:  python examples/03_sigma_diagnostic.py

The signal fraction σ predicts, *before* clustering, whether a dataset sits in
the regime where the quantum E-step helps. On synthetic data σ is exact; on real
data it is estimated from a (possibly approximate) partition by debiasing the
empirical barrier with a within-group split-half sampling null. This example
demonstrates the real-data estimator using a synthetic matrix with a k-means
partition standing in for "unknown labels".
"""
import os
import sys
# run directly from the repo without installing (pip install -e . also works)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.cluster import KMeans
from qbaymic import make_counts, signal_fraction, signal_fraction_estimate, regime

# A synthetic instance well inside the advantage band.
X, y = make_counts(n_samples=400, n_taxa=2000, n_clusters=3,
                   separation=0.4, zero_inflation=0.80, seed=0)

# Exact σ (synthetic path, uses the known generator profiles).
exact = signal_fraction(N=400, S=2000, K=3, separation=0.4, zero_inflation=0.80)
print(f"exact σ (synthetic)      = {exact['sigma']:.3f}  ({regime(exact['sigma'])})")

# Real-data path: pretend we do NOT know the labels — get a partition from
# k-means on relative abundances, then estimate σ from it.
rel = X / np.clip(X.sum(1, keepdims=True), 1, None)
partition = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(rel)
est = signal_fraction_estimate(X, partition)
print(f"estimated σ (real path)  = {est['sigma']:.3f}  "
      f"(σ_min over {est['K']} groups; pairwise = "
      f"{[round(s, 2) for s in est['sigma_pairs']]})")

print("\nThe estimate uses only counts + a partition — no ground-truth labels —")
print("so it can be run on any real dataset to decide whether QBayMic will help.")
