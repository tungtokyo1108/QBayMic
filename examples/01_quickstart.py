"""
Quickstart — generate data, compute the σ diagnostic, fit, evaluate.
====================================================================
Run:  python examples/01_quickstart.py

Shows the full loop in a few lines: make a synthetic count matrix in the
advantage band, check its signal fraction σ, fit the exact quantum method, and
score the recovered clustering against the known labels.
"""
import os
import sys
# run directly from the repo without installing (pip install -e . also works)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qbaymic import QBayMic, make_counts, signal_fraction, regime
from sklearn.metrics import adjusted_rand_score

# 1. Synthetic microbiome counts in the advantage band (σ ≈ 0.29).
X, y = make_counts(n_samples=400, n_taxa=5000, n_clusters=3,
                   separation=0.2, zero_inflation=0.80, seed=0)
print(f"data: {X.shape[0]} samples x {X.shape[1]} taxa, "
      f"{len(set(y))} true clusters")

# 2. The recoverability diagnostic (no clustering run needed).
sig = signal_fraction(N=400, S=5000, K=3, separation=0.2, zero_inflation=0.80)
print(f"signal fraction σ = {sig['sigma']:.3f}  ->  {regime(sig['sigma'])}")

# 3. Fit the exact quantum E-step and score it.
model = QBayMic(method="ed", K_max=4, random_state=0).fit(X)
ari = adjusted_rand_score(y, model.labels_)
print(f"ED:  inferred K = {model.K_},  ARI = {ari:.3f}")
