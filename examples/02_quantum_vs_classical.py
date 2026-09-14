"""
Quantum vs. classical in the advantage band.
=============================================
Run:  python examples/02_quantum_vs_classical.py

Reproduces the paper's central finding on one synthetic instance: at σ ≈ 0.29
the classical methods (greedy VB, parallel tempering, the annealing-only DAVB
control) fail to recover the clusters, while the quantum E-steps (ED, VarQITE,
VQT) succeed. Because this uses a single random seed it is illustrative; the
paper reports the reliability distribution over 100 seeds.
"""
import os
import sys
import time
# run directly from the repo without installing (pip install -e . also works)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qbaymic import QBayMic, make_counts, signal_fraction, regime
from sklearn.metrics import adjusted_rand_score

X, y = make_counts(n_samples=400, n_taxa=5000, n_clusters=3,
                   separation=0.2, zero_inflation=0.80, seed=0)
sig = signal_fraction(N=400, S=5000, K=3, separation=0.2, zero_inflation=0.80)
print(f"instance: σ = {sig['sigma']:.3f} ({regime(sig['sigma'])}), "
      f"true K = {len(set(y))}\n")

print(f"{'method':<10}{'type':<12}{'K':>4}{'ARI':>8}{'seconds':>9}")
print("-" * 43)
for method in ("greedy_vb", "pt", "davb", "ed", "varqite", "vqt"):
    t0 = time.time()
    m = QBayMic(method=method, K_max=4, random_state=0).fit(X)
    ari = adjusted_rand_score(y, m.labels_)
    kind = "quantum" if m.is_quantum else "classical"
    print(f"{method:<10}{kind:<12}{m.K_:>4}{ari:>8.3f}{time.time()-t0:>9.1f}")

print("\nExpected: classical arms ~0 ARI; quantum arms recover structure.")
print("(Single seed — see the paper for the R=100 reliability distribution.)")
