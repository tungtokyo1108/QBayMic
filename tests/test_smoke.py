"""
Smoke tests — every public entry point runs on a tiny instance.
===============================================================
Run:  python -m pytest tests/          (or: python tests/test_smoke.py)

Kept small and fast so it is CI-runnable. Verifies the API surface, not the
scientific results (those are in the paper): each of the six methods fits and
returns valid labels, and the σ diagnostic runs on both the synthetic and
real-data paths.
"""
import os
import sys

import numpy as np

# allow running the file directly (python tests/test_smoke.py) as well as pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qbaymic import (QBayMic, METHODS, make_counts,
                     signal_fraction, signal_fraction_estimate, regime)


def _tiny():
    # small enough that even the circuit methods finish in seconds
    return make_counts(n_samples=90, n_taxa=300, n_clusters=3,
                       separation=0.4, seed=0)


def test_make_counts_shapes():
    X, y = _tiny()
    assert X.shape == (90, 300)
    assert set(np.unique(y)) <= {0, 1, 2}


def test_signal_fraction_synthetic():
    d = signal_fraction(N=90, S=300, K=3, separation=0.4, zero_inflation=0.80)
    assert 0.0 <= d["sigma"] <= 1.0
    assert d["B"] >= d["B_signal"] >= 0.0
    assert regime(d["sigma"]) in ("unrecoverable", "advantage band",
                                  "signal-dominated")


def test_signal_fraction_estimate_realpath():
    X, y = _tiny()
    d = signal_fraction_estimate(X, y)
    assert 0.0 <= d["sigma"] <= 1.0
    assert d["K"] == 3
    assert len(d["sigma_pairs"]) == 3  # C(3,2)


def test_all_methods_fit_predict():
    X, y = _tiny()
    for method in METHODS:
        m = QBayMic(method=method, K_max=4, random_state=0, max_iter=120).fit(X)
        labels = m.predict(X)
        assert labels.shape == (X.shape[0],)
        assert m.K_ >= 1
        assert m.is_quantum == (method in ("ed", "varqite", "vqt"))


def test_bad_method_raises():
    try:
        QBayMic(method="not_a_method")
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unknown method")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nall smoke tests passed")
