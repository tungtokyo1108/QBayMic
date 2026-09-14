"""
QBayMic — Quantum Bayesian clustering of microbiome count data.
===============================================================
A Dirichlet–multinomial mixture with stochastic variable selection whose E-step
can be run with a classical or a quantum engine, and a barrier diagnostic (σ)
that predicts when the quantum E-step provides an advantage.

Quickstart
----------
>>> from qbaymic import QBayMic, make_counts, signal_fraction
>>> X, y = make_counts(separation=0.2)          # advantage-band synthetic data
>>> signal_fraction(N=400, S=5000, K=3,
...                  separation=0.2, zero_inflation=0.80)["sigma"]  # ≈ 0.29
>>> labels = QBayMic(method="vqt", K_max=4).fit_predict(X)

Public API
----------
- ``QBayMic``                 unified sklearn-style clustering estimator
- ``make_counts``             synthetic count data with known labels
- ``signal_fraction``         exact σ for synthetic instances
- ``signal_fraction_estimate``  σ estimated from real data + a partition
- ``regime``                  name the recoverability regime for a σ value
- ``METHODS``                 the six available E-step methods
"""
from .models import QBayMic, METHODS
from .data import make_counts
from .barrier import signal_fraction, signal_fraction_estimate, regime

__all__ = ["QBayMic", "METHODS", "make_counts",
           "signal_fraction", "signal_fraction_estimate", "regime"]
__version__ = "1.0.0"
