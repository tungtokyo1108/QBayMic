"""
QBayMic — a unified, scikit-learn-style interface to the six clustering methods.
================================================================================
QBayMic (Quantum Bayesian Microbiome clustering) fits a Dirichlet–multinomial
mixture with stochastic variable selection (DMM-SVVS) using one of six E-step
engines that share the *same* variational objective and differ only in how the
E-step responsibilities are computed:

    method          E-step                                       type
    --------------  -------------------------------------------  -----------
    "greedy_vb"     coordinate-ascent variational Bayes          classical
    "pt"            parallel-tempering VB (multi-temperature)    classical
    "davb"          deterministic annealing VB (s=0 control)     classical control
    "ed"            exact quantum Gibbs E-step (diagonalisation) quantum (exact)
    "varqite"       variational imaginary-time evolution         quantum (circuit)
    "vqt"           variational quantum thermalizer              quantum (circuit)

All six expose the same ``fit(X)`` / ``predict(X)`` interface and return hard
cluster labels; the number of clusters ``K`` is inferred by pruning from an
over-specified ``K_max``.

Basic use
---------
>>> from qbaymic import QBayMic
>>> model = QBayMic(method="vqt", K_max=4, random_state=0)
>>> model.fit(X)                    # X : (n_samples, n_taxa) integer count matrix
>>> labels = model.predict(X)       # or model.labels_
>>> model.K_                        # inferred number of clusters

The default hyperparameters are the shared configuration used throughout the
paper; override any of them via keyword arguments. The quantum methods
("ed", "varqite", "vqt") additionally accept the annealing-schedule parameters
(``beta0``, ``s0``, ``tau1``, ``tau2``) and, for the circuit methods, the ansatz
depth and optimisation budget.

See ``examples/`` for runnable scripts, and :func:`qbaymic.barrier.signal_fraction`
for the σ recoverability diagnostic that predicts when the quantum advantage
appears.
"""
from __future__ import annotations

# Importing the internal engine package installs the path shim that lets the
# vendored, validated engine modules resolve their flat cross-imports.
from . import _engines  # noqa: F401

import numpy as np

# ── the six E-step engines (validated implementations, imported by the shim) ──
from DMM_SVVS_Variational_v2 import DMM_SVVS_Variational_v2
from DMM_SVVS_ParallelTempering_VB import (
    DMM_SVVS_ParallelTempering_VB, tune_ladder)
from DMM_SVVS_Variational_QAVB_v2 import DMM_SVVS_PennyLaneQAVB_v2
from DMM_SVVS_Variational_QAVB_v3_1fast import DMM_SVVS_VarQITE_QAVB_JAX
from DMM_SVVS_Variational_QAVB_v4_2fast import DMM_SVVS_VQT_QAVB_JAX_Fast


METHODS = ("greedy_vb", "pt", "davb", "ed", "varqite", "vqt")

_QUANTUM = ("ed", "varqite", "vqt")
_CIRCUIT = ("varqite", "vqt")

# Shared DMM-SVVS prior / schedule defaults (the paper's reference configuration).
_SHARED_DEFAULTS = dict(
    K_max=4,
    nu=3.2888120853,               # Dirichlet concentration prior
    selection_prior=0.7115435126,  # feature-selection Beta prior
    prune_threshold=0.1352268850,  # empty-cluster prune cutoff
    prune_start=10,
    prune_every=5,
    max_iter=400,
)
# Annealing schedule shared by the quantum family (ED / VarQITE / VQT) and DAVB.
_SCHEDULE_DEFAULTS = dict(beta0=30.0, s0=1.0, tau1=100, tau2=230,
                          mixer="transverse_field")
# Circuit-method (VarQITE / VQT) ansatz + optimisation defaults.
_CIRCUIT_DEFAULTS = dict(ansatz_depth=3, n_steps=40, learning_rate=0.05)


class QBayMic:
    """Dirichlet–multinomial mixture clustering with a selectable E-step engine.

    Parameters
    ----------
    method : {"greedy_vb", "pt", "davb", "ed", "varqite", "vqt"}
        Which E-step engine to use. See the module docstring for the mapping.
    K_max : int, default=4
        Maximum number of clusters; the effective ``K`` is inferred by pruning.
        Over-specify by one or two over the expected number of clusters.
    random_state : int, default=0
        Seed for the fit initialisation.
    nu, selection_prior, prune_threshold, prune_start, prune_every, max_iter
        DMM-SVVS prior and optimisation controls (shared by all methods).
    beta0, s0, tau1, tau2, mixer
        Annealing-schedule parameters; used by "ed", "varqite", "vqt", "davb".
        ``s0=0`` (as in "davb") removes the quantum mixer — the annealing-only
        control. ``mixer`` is "transverse_field" (default) or "cyclic_shift".
    ansatz_depth, n_steps, learning_rate
        Circuit ansatz depth and optimisation budget; used by "varqite", "vqt".
    verbose : int, default=0
        Verbosity passed to the engine.
    **engine_kwargs
        Any additional keyword forwarded verbatim to the underlying engine.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Hard cluster assignment after :meth:`fit`.
    K_ : int
        Inferred number of clusters (after pruning).
    responsibilities_ : ndarray of shape (n_samples, K_)
        Soft cluster responsibilities.
    estimator_ : object
        The underlying fitted engine instance (for advanced inspection).
    """

    def __init__(self, method="ed", K_max=4, random_state=0,
                 nu=None, selection_prior=None, prune_threshold=None,
                 prune_start=None, prune_every=None, max_iter=None,
                 beta0=None, s0=None, tau1=None, tau2=None, mixer=None,
                 ansatz_depth=None, n_steps=None, learning_rate=None,
                 verbose=0, **engine_kwargs):
        method = str(method).lower()
        if method not in METHODS:
            raise ValueError(
                f"method must be one of {METHODS}, got {method!r}")
        self.method = method
        self.random_state = int(random_state)
        self.verbose = int(verbose)
        self.engine_kwargs = engine_kwargs

        # resolve shared / schedule / circuit params against the defaults
        def pick(name, val, table):
            return table[name] if val is None else val

        self.K_max = int(pick("K_max", K_max, _SHARED_DEFAULTS))
        self.nu = pick("nu", nu, _SHARED_DEFAULTS)
        self.selection_prior = pick("selection_prior", selection_prior,
                                    _SHARED_DEFAULTS)
        self.prune_threshold = pick("prune_threshold", prune_threshold,
                                    _SHARED_DEFAULTS)
        self.prune_start = pick("prune_start", prune_start, _SHARED_DEFAULTS)
        self.prune_every = pick("prune_every", prune_every, _SHARED_DEFAULTS)
        self.max_iter = pick("max_iter", max_iter, _SHARED_DEFAULTS)

        self.beta0 = pick("beta0", beta0, _SCHEDULE_DEFAULTS)
        self.s0 = pick("s0", s0, _SCHEDULE_DEFAULTS)
        self.tau1 = pick("tau1", tau1, _SCHEDULE_DEFAULTS)
        self.tau2 = pick("tau2", tau2, _SCHEDULE_DEFAULTS)
        self.mixer = pick("mixer", mixer, _SCHEDULE_DEFAULTS)

        self.ansatz_depth = pick("ansatz_depth", ansatz_depth, _CIRCUIT_DEFAULTS)
        self.n_steps = pick("n_steps", n_steps, _CIRCUIT_DEFAULTS)
        self.learning_rate = pick("learning_rate", learning_rate,
                                  _CIRCUIT_DEFAULTS)

        self.estimator_ = None
        self.labels_ = None
        self.responsibilities_ = None
        self.K_ = None

    # ── construction of the underlying engine ────────────────────────────────
    def _shared(self):
        return dict(K_max=self.K_max, nu=self.nu,
                    selection_prior=self.selection_prior,
                    prune_threshold=self.prune_threshold,
                    prune_start=self.prune_start, prune_every=self.prune_every)

    def _build(self, X):
        m = self.method
        common = dict(max_iter=self.max_iter, random_state=self.random_state,
                      verbose=self.verbose)
        sched = dict(beta0=self.beta0, s0=self.s0, tau1=self.tau1,
                     tau2=self.tau2, use_trigamma_correction=False)

        if m == "greedy_vb":
            return DMM_SVVS_Variational_v2(**common, **self._shared(),
                                           **self.engine_kwargs)

        if m == "pt":
            # tune the temperature ladder once on this X (paper protocol)
            ladder, _ = tune_ladder(
                X, compute_budget=self.max_iter, shared=self._shared(),
                n_replicas=self.engine_kwargs.pop("n_replicas", 8),
                swap_every=5, tune_seeds=(0, 1, 2), verbose=False)
            return DMM_SVVS_ParallelTempering_VB(
                compute_budget=self.max_iter, ladder=ladder, swap_every=5,
                random_state=self.random_state, verbose=self.verbose,
                **self._shared(), **self.engine_kwargs)

        # The exact-diagonalisation engine (ED, and DAVB via s0=0) uses a fixed
        # transverse-field mixer and does not take a `mixer` kwarg; only the
        # circuit methods (VarQITE, VQT) expose the mixer as a design variable.
        if m == "davb":  # annealing-only control: exact engine with s0=0
            return DMM_SVVS_PennyLaneQAVB_v2(
                **common, **{**sched, "s0": 0.0},
                **self._shared(), **self.engine_kwargs)

        if m == "ed":    # exact quantum Gibbs E-step
            return DMM_SVVS_PennyLaneQAVB_v2(
                **common, **sched,
                **self._shared(), **self.engine_kwargs)

        if m == "varqite":
            return DMM_SVVS_VarQITE_QAVB_JAX(
                **common, **sched, mixer=self.mixer,
                n_varqite_steps=self.n_steps, ansatz_depth=self.ansatz_depth,
                warm_start=True, jit_warmup=True, **self._shared(),
                **self.engine_kwargs)

        if m == "vqt":
            return DMM_SVVS_VQT_QAVB_JAX_Fast(
                **common, **sched, mixer=self.mixer,
                n_vqt_steps=self.n_steps, ansatz_depth=self.ansatz_depth,
                learning_rate=self.learning_rate, warm_start=True,
                enumerate_basis=True, jit_warmup=True, **self._shared(),
                **self.engine_kwargs)

        raise ValueError(self.method)  # unreachable (validated in __init__)

    # ── scikit-learn-style API ───────────────────────────────────────────────
    def fit(self, X, y=None):
        """Fit the mixture to the integer count matrix ``X`` (n_samples, n_taxa).

        ``y`` is ignored (accepted for scikit-learn compatibility). Returns
        ``self`` so calls can be chained.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2-D (n_samples, n_taxa)")
        self.estimator_ = self._build(X)
        self.estimator_.fit(X)
        self.labels_ = np.asarray(self.estimator_.predict(X))
        self.K_ = int(getattr(self.estimator_, "K", len(np.unique(self.labels_))))
        r = getattr(self.estimator_, "r", None)
        # PT wraps its cold replica; expose responsibilities when available.
        if r is None and hasattr(self.estimator_, "cold_"):
            r = getattr(self.estimator_.cold_, "r", None)
        self.responsibilities_ = None if r is None else np.asarray(r)
        return self

    def predict(self, X):
        """Return hard cluster labels for ``X``. Requires a prior :meth:`fit`."""
        if self.estimator_ is None:
            raise RuntimeError("call fit() before predict()")
        return np.asarray(self.estimator_.predict(np.asarray(X, dtype=np.float64)))

    def fit_predict(self, X, y=None):
        """Convenience: :meth:`fit` then return ``labels_``."""
        return self.fit(X).labels_

    @property
    def is_quantum(self):
        """True for the quantum E-step methods ("ed", "varqite", "vqt")."""
        return self.method in _QUANTUM

    def __repr__(self):
        return (f"QBayMic(method={self.method!r}, K_max={self.K_max}, "
                f"random_state={self.random_state})")
